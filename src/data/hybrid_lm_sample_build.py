"""Build hybrid LM rows from text (tokenizers + RNG); used by prepare_hybrid_lm_dataset (multiprocessing-safe)."""

from __future__ import annotations

import sys
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Filled in each worker by pool_init
_GPT2_TOK: Any = None
_AE_TOK: Any = None


def _take_prefix_by_tokens(
    text: str,
    tokenizer,
    n: int,
    add_special_tokens: bool = False,
) -> tuple[str, str] | None:
    if n == 0:
        return "", text
    enc = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
        truncation=False,
    )
    om = enc.get("offset_mapping")
    if om is None:
        raise RuntimeError(
            f"Tokenizer {type(tokenizer).__name__} has no offset_mapping; use a Fast tokenizer."
        )
    ids = enc["input_ids"]
    if len(ids) < n:
        return None
    end_char = om[n - 1][1]
    return text[:end_char], text[end_char:]


def try_build_sample(
    text: str,
    rng: np.random.Generator,
    gpt2_tok,
    ae_tok,
    prompt_min: int,
    prompt_max: int,
    completion_min: int,
    completion_max: int,
    max_latent_steps: int,
    ae_max_length: int,
) -> dict | None:
    Lp = int(rng.integers(prompt_min, prompt_max + 1))
    K = int(rng.integers(0, max_latent_steps + 1))
    Lc = int(rng.integers(completion_min, completion_max + 1))

    pr = _take_prefix_by_tokens(text, gpt2_tok, Lp)
    if pr is None:
        return None
    prompt_text, rest = pr

    ae_need = K * ae_max_length
    if ae_need > 0:
        ae_enc = ae_tok(
            rest,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        om = ae_enc.get("offset_mapping")
        if om is None:
            raise RuntimeError("AE tokenizer must support offset_mapping (Fast tokenizer).")
        ids_full = ae_enc["input_ids"]
        if len(ids_full) < ae_need:
            return None
        enc_ae = ids_full[:ae_need]
        end_char = om[ae_need - 1][1]
        rest = rest[end_char:]
        latent_steps = []
        for k in range(K):
            chunk = enc_ae[k * ae_max_length : (k + 1) * ae_max_length]
            latent_steps.append(ae_tok.decode(chunk, skip_special_tokens=False))
    else:
        latent_steps = []

    g2_enc = gpt2_tok(
        rest,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    om2 = g2_enc.get("offset_mapping")
    if om2 is None:
        raise RuntimeError("GPT-2 tokenizer must support offset_mapping (Fast tokenizer).")
    ids2 = g2_enc["input_ids"]
    if len(ids2) < Lc:
        return None
    end_c = om2[Lc - 1][1]
    completion_text = rest[:end_c]

    return {
        "task": prompt_text,
        "latent_steps": latent_steps,
        "labels": completion_text,
    }


def _row_rng_seed(base_seed: int, row_index: int) -> int:
    ss = np.random.SeedSequence(
        [base_seed & 0xFFFFFFFF, row_index & 0xFFFFFFFF, (base_seed >> 16) & 0xFFFFFFFF]
    )
    return int(ss.generate_state(1, dtype=np.uint64)[0] & 0x7FFFFFFFFFFFFFFF)


def pool_init(gpt2_name: str, ae_backbone: str) -> None:
    global _GPT2_TOK, _AE_TOK
    _GPT2_TOK = AutoTokenizer.from_pretrained(gpt2_name)
    _AE_TOK = AutoTokenizer.from_pretrained(ae_backbone)


def _worker_one(args: Tuple) -> Optional[dict]:
    (
        row_index,
        text,
        base_seed,
        tries_per_paragraph,
        prompt_min,
        prompt_max,
        completion_min,
        completion_max,
        max_latent_steps,
        ae_max_length,
    ) = args
    rng = np.random.default_rng(_row_rng_seed(base_seed, row_index))
    for _ in range(tries_per_paragraph):
        s = try_build_sample(
            text,
            rng,
            _GPT2_TOK,
            _AE_TOK,
            prompt_min,
            prompt_max,
            completion_min,
            completion_max,
            max_latent_steps,
            ae_max_length,
        )
        if s is not None:
            return s
    return None


def enough_accepted_for_train_cap(num_accepted: int, num_train_samples: Optional[int], num_val_samples: Optional[int]) -> bool:
    if num_train_samples is None:
        return False
    v_cfg = num_val_samples if num_val_samples is not None else 2000
    n_val = min(v_cfg, max(1, num_accepted // 10))
    return num_accepted - n_val >= num_train_samples


def build_samples_sequential(
    paragraphs: List[str],
    rng: np.random.Generator,
    gpt2_tok,
    ae_tok,
    prompt_min: int,
    prompt_max: int,
    completion_min: int,
    completion_max: int,
    max_latent_steps: int,
    ae_max_length: int,
    tries_per_paragraph: int,
    num_train_samples: Optional[int],
    num_val_samples: Optional[int],
) -> tuple[List[dict], bool]:
    rows: List[dict] = []
    early_stop = False
    for text in tqdm(paragraphs, desc="Building samples"):
        if not isinstance(text, str) or not text.strip():
            continue
        s = text.strip()
        for _ in range(tries_per_paragraph):
            sample = try_build_sample(
                s,
                rng,
                gpt2_tok,
                ae_tok,
                prompt_min,
                prompt_max,
                completion_min,
                completion_max,
                max_latent_steps,
                ae_max_length,
            )
            if sample is not None:
                rows.append(sample)
                if enough_accepted_for_train_cap(len(rows), num_train_samples, num_val_samples):
                    early_stop = True
                break
        if early_stop:
            break
    return rows, early_stop


def build_samples_parallel(
    paragraphs: List[str],
    *,
    base_seed: int,
    gpt2_tokenizer_name: str,
    ae_backbone_name: str,
    prompt_min: int,
    prompt_max: int,
    completion_min: int,
    completion_max: int,
    max_latent_steps: int,
    ae_max_length: int,
    tries_per_paragraph: int,
    num_workers: int,
    chunk_size: int,
    num_train_samples: Optional[int],
    num_val_samples: Optional[int],
) -> tuple[List[dict], bool]:
    """Process paragraphs in chunks with a process pool; early stop between chunks."""
    rows: List[dict] = []
    early_stop = False
    n_par = len(paragraphs)

    mp_ctx = get_context("spawn" if sys.platform == "win32" else "fork")
    with mp_ctx.Pool(
        num_workers,
        initializer=pool_init,
        initargs=(gpt2_tokenizer_name, ae_backbone_name),
        maxtasksperchild=512,
    ) as pool:
        offset = 0
        with tqdm(total=n_par, desc="Building samples (parallel)") as pbar:
            while offset < n_par:
                end = min(offset + chunk_size, n_par)
                tasks: List[Tuple] = []
                for i in range(offset, end):
                    t = paragraphs[i]
                    if isinstance(t, str) and t.strip():
                        tasks.append(
                            (
                                i,
                                t.strip(),
                                base_seed,
                                tries_per_paragraph,
                                prompt_min,
                                prompt_max,
                                completion_min,
                                completion_max,
                                max_latent_steps,
                                ae_max_length,
                            )
                        )
                if tasks:
                    inner_cs = max(4, min(128, len(tasks) // max(num_workers * 4, 1) or 4))
                    chunk_results = pool.map(_worker_one, tasks, chunksize=inner_cs)
                    for sample in chunk_results:
                        if sample is not None:
                            rows.append(sample)
                            if enough_accepted_for_train_cap(
                                len(rows), num_train_samples, num_val_samples
                            ):
                                early_stop = True
                                break
                pbar.update(end - offset)
                offset = end
                if early_stop:
                    break

    return rows, early_stop
