#!/usr/bin/env python3
"""Build hybrid latent LM data from a preprocess YAML (same style as prepare_dataset.py).

Requires Fast tokenizers (return_offsets_mapping). Discards samples if any segment is too short.

Sample building uses multiprocessing (fork on Linux, spawn on Windows) when prepare_num_proc != 1.
Set prepare_num_proc: 1 in YAML to force single-process (legacy RNG order).

Example:
  conda activate rmt-a100
  python scripts/prepare_hybrid_lm_dataset.py --configs configs/preprocess/hybrid_wiki_k4.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import _chunk_batched, _split_paragraphs_batched
from src.data.hybrid_lm_sample_build import build_samples_parallel, build_samples_sequential
from src.utils.config import merge_bottleneck_configs


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
    ae_backbone = ae_exp.model.backbone_name

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

    paragraphs = list(ds[dc.text_column])
    print(f"Text rows after preprocessing: {len(paragraphs):,}")

    if dc.num_train_samples is not None:
        print(
            "Early exit: will stop once enough accepted samples exist for "
            f"num_train_samples={dc.num_train_samples} (between parallel chunks)."
        )

    use_parallel = dc.prepare_num_proc is None or dc.prepare_num_proc > 1
    if dc.prepare_num_proc == 1:
        use_parallel = False

    if use_parallel:
        if dc.prepare_num_proc is not None and dc.prepare_num_proc > 1:
            build_workers = dc.prepare_num_proc
        else:
            build_workers = max(1, (os.cpu_count() or 8) - 1)
        if dc.hybrid_build_chunk_size is not None:
            chunk_sz = dc.hybrid_build_chunk_size
        else:
            chunk_sz = max(4096, min(65536, len(paragraphs) // max(1, build_workers * 2) or 65536))
            if dc.num_train_samples is not None:
                chunk_sz = min(chunk_sz, 16384)
        print(
            f"Parallel sample build: workers={build_workers}, chunk_size={chunk_sz} "
            f"(set data.hybrid_build_chunk_size / prepare_num_proc to tune)"
        )
        rows, early_stop = build_samples_parallel(
            paragraphs,
            base_seed=dc.seed,
            gpt2_tokenizer_name=dc.gpt2_tokenizer_name,
            ae_backbone_name=ae_backbone,
            prompt_min=dc.prompt_min_len,
            prompt_max=dc.prompt_max_len,
            completion_min=dc.completion_min_len,
            completion_max=dc.completion_max_len,
            max_latent_steps=dc.max_latent_steps,
            ae_max_length=ae_max_length,
            tries_per_paragraph=dc.tries_per_paragraph,
            num_workers=build_workers,
            chunk_size=chunk_sz,
            num_train_samples=dc.num_train_samples,
            num_val_samples=dc.num_val_samples,
        )
    else:
        print("Single-process sample build (prepare_num_proc: 1).")
        gpt2_tok = AutoTokenizer.from_pretrained(dc.gpt2_tokenizer_name)
        ae_tok = AutoTokenizer.from_pretrained(ae_backbone)
        rng = np.random.default_rng(dc.seed)
        rows, early_stop = build_samples_sequential(
            paragraphs,
            rng=rng,
            gpt2_tok=gpt2_tok,
            ae_tok=ae_tok,
            prompt_min=dc.prompt_min_len,
            prompt_max=dc.prompt_max_len,
            completion_min=dc.completion_min_len,
            completion_max=dc.completion_max_len,
            max_latent_steps=dc.max_latent_steps,
            ae_max_length=ae_max_length,
            tries_per_paragraph=dc.tries_per_paragraph,
            num_train_samples=dc.num_train_samples,
            num_val_samples=dc.num_val_samples,
        )

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
        "parallel_sample_build": use_parallel,
        "max_latent_steps": dc.max_latent_steps,
        "ae_max_length": ae_max_length,
        "ae_backbone": ae_backbone,
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
