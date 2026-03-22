#!/usr/bin/env python3
"""Build hybrid latent LM data: for each paragraph, sample L_p, K, L_c and cut GPT-2 | K×AE | GPT-2.

Requires Fast tokenizers (return_offsets_mapping). Discards samples if any segment is too short.

Example:
  conda activate rmt-a100
  python scripts/prepare_hybrid_lm_dataset.py \\
    --output_dir outputs/hybrid_lm/wiki_k4 \\
    --max_latent_steps 4 \\
    --ae_config_path configs/.../config.yaml \\
    --max_samples 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import _split_paragraphs_batched
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
        # Slice token ids from one full encode of ``rest`` so substring re-tokenization cannot drift.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hybrid latent LM dataset (sequential GPT-2 / AE / GPT-2).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--dataset_config", type=str, default="20231101.en")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--max_latent_steps", type=int, required=True)
    parser.add_argument(
        "--ae_config_path",
        type=str,
        required=True,
        help="Bottleneck experiment YAML (backbone tokenizer + model.max_length).",
    )
    parser.add_argument("--gpt2_tokenizer_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--max_samples", type=int, default=100_000, help="Max documents before paragraph split")
    parser.add_argument("--num_train_samples", type=int, default=None, help="Cap train rows after build (None = all)")
    parser.add_argument("--num_val_samples", type=int, default=10_000)
    parser.add_argument("--prompt_min_len", type=int, default=32)
    parser.add_argument("--prompt_max_len", type=int, default=96)
    parser.add_argument("--completion_min_len", type=int, default=1)
    parser.add_argument("--completion_max_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_num_proc", type=int, default=None)
    parser.add_argument("--preprocess_batch_size", type=int, default=2000)
    parser.add_argument("--tries_per_paragraph", type=int, default=8, help="Random retries per paragraph")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ae_exp = merge_bottleneck_configs(args.ae_config_path)
    ae_max_length = int(ae_exp.model.max_length)
    ae_tok = AutoTokenizer.from_pretrained(ae_exp.model.backbone_name)
    gpt2_tok = AutoTokenizer.from_pretrained(args.gpt2_tokenizer_name)

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.dataset_name} / {args.dataset_config} ...")
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        trust_remote_code=True,
    )
    n_total = len(ds)
    n_take = min(args.max_samples, n_total) if args.max_samples is not None else n_total
    ds = ds.shuffle(seed=args.seed).select(range(n_take))
    print(f"Subsampled to {n_take:,} documents.")

    num_proc = args.prepare_num_proc or 1
    batch_size = args.preprocess_batch_size or 2000
    print("Splitting paragraphs ...")
    para_ds = _split_paragraphs_batched(
        ds,
        text_column=args.text_column,
        batch_size=batch_size,
        num_proc=num_proc if num_proc > 1 else None,
    )
    paragraphs = para_ds[args.text_column]
    print(f"Paragraphs: {len(paragraphs):,}")

    rows: list[dict] = []
    for text in tqdm(paragraphs, desc="Building samples"):
        if not isinstance(text, str) or not text.strip():
            continue
        s = text.strip()
        for _ in range(args.tries_per_paragraph):
            sample = try_build_sample(
                s,
                rng,
                gpt2_tok,
                ae_tok,
                args.prompt_min_len,
                args.prompt_max_len,
                args.completion_min_len,
                args.completion_max_len,
                args.max_latent_steps,
                ae_max_length,
            )
            if sample is not None:
                rows.append(sample)
                break

    if not rows:
        raise SystemExit("No samples produced; increase max_samples, tries_per_paragraph, or text length.")

    print(f"Accepted samples: {len(rows):,}")
    full_ds = Dataset.from_list(rows)
    full_ds = full_ds.shuffle(seed=args.seed)

    n_val = min(args.num_val_samples, max(1, len(full_ds) // 10))
    n_train = len(full_ds) - n_val
    if args.num_train_samples is not None:
        n_train = min(n_train, args.num_train_samples)
    if n_train + n_val > len(full_ds):
        n_train = len(full_ds) - n_val
    train_ds = full_ds.select(range(n_train))
    val_ds = full_ds.select(range(n_train, n_train + n_val))
    print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

    train_dir = out / "train"
    val_dir = out / "validation"
    train_ds.save_to_disk(str(train_dir))
    val_ds.save_to_disk(str(val_dir))

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "text_column": args.text_column,
        "max_latent_steps": args.max_latent_steps,
        "ae_max_length": ae_max_length,
        "ae_backbone": ae_exp.model.backbone_name,
        "gpt2_tokenizer_name": args.gpt2_tokenizer_name,
        "prompt_min_len": args.prompt_min_len,
        "prompt_max_len": args.prompt_max_len,
        "completion_min_len": args.completion_min_len,
        "completion_max_len": args.completion_max_len,
        "seed": args.seed,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {train_dir} and {val_dir}; meta.json written.")
    print("Training: set data.preprocessed_dir to:", str(out.resolve()))


if __name__ == "__main__":
    main()
