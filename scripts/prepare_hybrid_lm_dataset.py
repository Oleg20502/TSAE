#!/usr/bin/env python3
"""Build hybrid latent LM data from a preprocess YAML (same style as prepare_dataset.py).

Requires Fast tokenizers (return_offsets_mapping). Discards samples if any segment is too short.

Example:
  conda activate rmt-a100
  python scripts/prepare_hybrid_lm_dataset.py --configs configs/preprocess/hybrid_wiki_k4.yaml
  # Set data.preprocessed_dir in train config to data.preprocessed_dir from this YAML.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import _chunk_batched, _split_paragraphs_batched
from src.utils.config import merge_bottleneck_configs


def _take_prefix_by_tokens(
    text: str,
    tokenizer,
    n: int,
    add_special_tokens: bool = False,
) -> tuple[str, str] | None:
    """First ``n`` tokens as text prefix; returns (prefix, remainder) or None if too short."""
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


def _enough_accepted_for_train_cap(num_accepted: int, dc) -> bool:
    """True once len(rows) is large enough that post-shuffle split can fill ``num_train_samples``."""
    if dc.num_train_samples is None:
        return False
    v_cfg = dc.num_val_samples if dc.num_val_samples is not None else 2000
    n_val = min(v_cfg, max(1, num_accepted // 10))
    return num_accepted - n_val >= dc.num_train_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare hybrid latent LM dataset from YAML (data section; same pattern as prepare_dataset.py)."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="YAML config file(s); merged by section. Uses cfg.data (Bottleneck DataConfig + hybrid fields).",
    )
    args = parser.parse_args()

    cfg = merge_bottleneck_configs(*args.configs)
    dc = cfg.data

    if not dc.preprocessed_dir:
        raise SystemExit("data.preprocessed_dir must be set in the config (output directory).")
    if dc.ae_config_path is None:
        raise SystemExit("data.ae_config_path must be set (bottleneck YAML for AE tokenizer + max_length).")
    if dc.max_latent_steps is None:
        raise SystemExit("data.max_latent_steps must be set.")

    out = Path(dc.preprocessed_dir)
    out.mkdir(parents=True, exist_ok=True)

    ae_exp = merge_bottleneck_configs(dc.ae_config_path)
    ae_max_length = int(ae_exp.model.max_length)
    ae_tok = AutoTokenizer.from_pretrained(ae_exp.model.backbone_name)
    gpt2_tok = AutoTokenizer.from_pretrained(dc.gpt2_tokenizer_name)

    rng = np.random.default_rng(dc.seed)

    num_proc = dc.prepare_num_proc or 1
    batch_size = dc.preprocess_batch_size or 2000
    mode = (dc.preprocess_mode or "paragraphs").strip().lower()

    print(f"Dataset: {dc.dataset_name} / {dc.dataset_config}")
    print(f"Output: {out}")
    print(f"Preprocess mode: {mode}")
    print(f"max_latent_steps: {dc.max_latent_steps}, ae_max_length (from AE config): {ae_max_length}")
    max_samples = dc.max_samples
    print(f"Max samples (documents before split/chunk): {max_samples or 'null'}")

    print("Loading dataset from HuggingFace...")
    load_kw: dict = {}
    if dc.cache_dir:
        load_kw["cache_dir"] = dc.cache_dir
    ds = load_dataset(
        dc.dataset_name,
        dc.dataset_config,
        split="train",
        trust_remote_code=True,
        **load_kw,
    )
    n_total = len(ds)
    print(f"Loaded {n_total:,} rows.")

    n_take = min(max_samples, n_total) if max_samples is not None else n_total
    ds = ds.shuffle(seed=dc.seed).select(range(n_take))
    print(f"Subsampled to {n_take:,} documents.")

    if mode == "chunks":
        chunk_size = dc.chunk_size_tokens
        if not chunk_size:
            raise ValueError("preprocess_mode is 'chunks' but chunk_size_tokens is not set in config")
        print(f"Chunking by GPT-2 tokens: chunk_size={chunk_size}, num_proc={num_proc}, batch_size={batch_size}")
        ds = _chunk_batched(
            ds,
            text_column=dc.text_column,
            chunk_size_tokens=chunk_size,
            tokenizer_name=dc.gpt2_tokenizer_name,
            batch_size=batch_size,
            num_proc=num_proc if num_proc > 1 else None,
            drop_incomplete=dc.drop_incomplete_chunks,
        )
    elif mode == "paragraphs":
        print(f"Splitting into paragraphs: num_proc={num_proc}, batch_size={batch_size}")
        ds = _split_paragraphs_batched(
            ds,
            text_column=dc.text_column,
            batch_size=batch_size,
            num_proc=num_proc if num_proc > 1 else None,
        )
    else:
        print("Using dataset as-is (no paragraph/chunk split).")
    print(f"finished splitting paragraphs")
    paragraphs = ds[dc.text_column]
    print(f"Text rows after preprocessing: {len(paragraphs):,}")

    if dc.num_train_samples is not None:
        print(
            "Early exit: will stop once enough accepted samples exist for "
            f"num_train_samples={dc.num_train_samples} (same val rule as post-shuffle split)."
        )

    rows: list[dict] = []
    early_stop = False
    for text in tqdm(paragraphs, desc="Building samples"):
        if not isinstance(text, str) or not text.strip():
            continue
        s = text.strip()
        for _ in range(dc.tries_per_paragraph):
            sample = try_build_sample(
                s,
                rng,
                gpt2_tok,
                ae_tok,
                dc.prompt_min_len,
                dc.prompt_max_len,
                dc.completion_min_len,
                dc.completion_max_len,
                dc.max_latent_steps,
                ae_max_length,
            )
            if sample is not None:
                rows.append(sample)
                if _enough_accepted_for_train_cap(len(rows), dc):
                    early_stop = True
                break
        if early_stop:
            break

    if not rows:
        raise SystemExit("No samples produced; increase max_samples, tries_per_paragraph, or text length.")

    print(f"Accepted samples: {len(rows):,}" + (" (early stop)" if early_stop else ""))
    full_ds = Dataset.from_list(rows)
    full_ds = full_ds.shuffle(seed=dc.seed)

    n_val = min(dc.num_val_samples or 2000, max(1, len(full_ds) // 10))
    n_train = len(full_ds) - n_val
    if dc.num_train_samples is not None:
        n_train = min(n_train, dc.num_train_samples)
    if n_train + n_val > len(full_ds):
        n_train = len(full_ds) - n_val
    train_ds = full_ds.select(range(n_train))
    val_ds = full_ds.select(range(n_train, n_train + n_val))
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    train_dir = out / "train"
    val_dir = out / "validation"
    train_ds.save_to_disk(str(train_dir))
    val_ds.save_to_disk(str(val_dir))

    preprocess_config_path = out / "preprocess_config.yaml"
    with open(preprocess_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    meta = {
        "dataset_name": dc.dataset_name,
        "dataset_config": dc.dataset_config,
        "text_column": dc.text_column,
        "preprocess_mode": mode,
        "early_stop_after_accepted": early_stop,
        "max_latent_steps": dc.max_latent_steps,
        "ae_max_length": ae_max_length,
        "ae_backbone": ae_exp.model.backbone_name,
        "gpt2_tokenizer_name": dc.gpt2_tokenizer_name,
        "prompt_min_len": dc.prompt_min_len,
        "prompt_max_len": dc.prompt_max_len,
        "completion_min_len": dc.completion_min_len,
        "completion_max_len": dc.completion_max_len,
        "seed": dc.seed,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {train_dir} and {val_dir}.")
    print(f"Saved preprocessing config to {preprocess_config_path}.")
    print("In your train config set: data.preprocessed_dir:", str(out.resolve()))


if __name__ == "__main__":
    main()
