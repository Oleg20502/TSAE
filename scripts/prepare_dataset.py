#!/usr/bin/env python3
"""Prepare a text dataset (download, paragraph split, shuffle, split) and save to disk.

Run once before training. Training then loads from preprocessed_dir and skips heavy preprocessing.

Example:
  python scripts/prepare_dataset.py --configs configs/preprocess/fineweb_10bt_ml_64.yaml
  # Then set data.preprocessed_dir in your train/dataset config to the preprocessed_dir from the preprocess config.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from datasets import load_dataset

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import merge_bottleneck_configs
from src.data.datasets import _split_paragraphs_batched, _chunk_batched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dataset (download, split paragraphs, shuffle, train/val split) and save to disk."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="YAML config files (same as training; data section used)",
    )
    args = parser.parse_args()

    cfg = merge_bottleneck_configs(*args.configs)
    dc = cfg.data

    output_dir = dc.preprocessed_dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_proc = dc.prepare_num_proc or 1
    batch_size = dc.preprocess_batch_size or 2000
    mode = (dc.preprocess_mode or "paragraphs").strip().lower()

    print(f"Dataset: {dc.dataset_name} / {dc.dataset_config}")
    print(f"Output: {out}")
    print(f"Preprocess mode: {mode}")
    print(f"Train samples: {dc.num_train_samples}, Val samples: {dc.num_val_samples}")
    max_samples = dc.max_samples
    print(f"Max samples to process (before split/chunk): {max_samples or 'null'}")
    if mode == "chunks":
        chunk_size = dc.chunk_size_tokens
        if not chunk_size:
            raise ValueError("preprocess_mode is 'chunks' but chunk_size_tokens is not set in config")
        chunk_batch_size = dc.preprocess_batch_size or 500
        print(f"Chunking: chunk_size_tokens={chunk_size}, num_proc={num_proc or 1}, batch_size={chunk_batch_size}")
    else:
        print(f"Paragraph split: num_proc={num_proc or 1}, batch_size={batch_size}")

    # Load full split from HF (streaming not used; we subsample first to limit work)
    print("Loading dataset from HuggingFace...")
    ds = load_dataset(
        dc.dataset_name,
        dc.dataset_config,
        split="train",
        trust_remote_code=True,
    )
    n_total = len(ds)
    print(f"Loaded {n_total:,} rows.")

    # Subsample before expensive map (None = no limit)
    n_take = min(max_samples, n_total) if max_samples is not None else n_total
    ds = ds.shuffle(seed=dc.seed).select(range(n_take))
    print(f"Subsampled to {n_take:,} documents.")

    if mode == "chunks":
        print("Chunking by GPT-2 tokens...")
        ds = _chunk_batched(
            ds,
            text_column=dc.text_column,
            chunk_size_tokens=dc.chunk_size_tokens,
            tokenizer_name=dc.gpt2_tokenizer_name,
            batch_size=batch_size,
            num_proc=num_proc,
            drop_incomplete=dc.drop_incomplete_chunks,
        )
    elif mode == "paragraphs":
        print("Splitting into paragraphs...")
        ds = _split_paragraphs_batched(
            ds,
            text_column=dc.text_column,
            batch_size=batch_size,
            num_proc=num_proc,
        )
    else:
        print("Using dataset as-is (no split).")

    n_after = len(ds)
    print(f"After preprocessing: {n_after:,} text rows.")

    # Shuffle and split train / validation
    ds = ds.shuffle(seed=dc.seed)
    n_train = dc.num_train_samples or (n_after - (dc.num_val_samples or 2000))
    n_val = dc.num_val_samples or 2000
    total = n_train + n_val
    if total > n_after:
        total = n_after
        n_val = min(n_val, total // 10)
        n_train = total - n_val
    train_ds = ds.select(range(n_train))
    val_ds = ds.select(range(n_train, n_train + n_val))
    print(f"Train: {len(train_ds):,}, Validation: {len(val_ds):,}")

    # Save to disk for fast loading in training
    train_dir = out / "train"
    val_dir = out / "validation"
    train_ds.save_to_disk(str(train_dir))
    val_ds.save_to_disk(str(val_dir))

    # Save preprocessing config to the same folder for reproducibility
    preprocess_config_path = out / "preprocess_config.yaml"
    with open(preprocess_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Saved to {train_dir} and {val_dir}.")
    print(f"Saved preprocessing config to {preprocess_config_path}.")
    print("In your train config set: data.preprocessed_dir:", output_dir)


if __name__ == "__main__":
    main()
