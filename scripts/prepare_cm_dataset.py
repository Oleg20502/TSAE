#!/usr/bin/env python3
"""Prepare CM training sequences from FineWeb and save to disk.

Run this once before training the Concept Model.  For every FineWeb document
the script tokenises with GPT-2, cuts into fixed-length chunks, and groups
consecutive chunks from the same document into sequences of ``n_chunks``
chunks.  The result is saved as HuggingFace datasets (train / validation) to
``data.preprocessed_dir``.

Usage
-----
    python scripts/prepare_cm_dataset.py \\
        --config configs/preprocess/cm_fineweb_preprocess.yaml

Then set ``data.preprocessed_dir`` in your CM training config to the value of
``data.preprocessed_dir`` from the preprocess config.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.concept_datasets import build_cm_sequences
from src.utils.config import load_concept_config_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CM training sequences from FineWeb and save to disk."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Path(s) to ConceptExperimentConfig YAML file(s).",
    )
    args = parser.parse_args()

    cfg = load_concept_config_from_paths(args.config)
    dc  = cfg.data

    output_dir = dc.preprocessed_dir
    if not output_dir:
        raise ValueError("data.preprocessed_dir must be set in the config.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== CM Dataset Preparation ===")
    print(f"  Source  : {dc.dataset_name} / {dc.dataset_config}")
    print(f"  n_chunks: {dc.n_chunks}  chunk_size_tokens: {dc.chunk_size_tokens}")
    print(f"  Output  : {out}")
    print(f"  Max docs: {dc.max_docs or 'all'}")
    print(f"  Val     : {dc.num_val_samples:,} sequences")
    print()

    splits = build_cm_sequences(dc)

    train_dir = out / "train"
    val_dir   = out / "validation"
    print(f"Saving train ({len(splits['train']):,} rows) → {train_dir}")
    splits["train"].save_to_disk(str(train_dir))
    print(f"Saving validation ({len(splits['validation']):,} rows) → {val_dir}")
    splits["validation"].save_to_disk(str(val_dir))

    # Save the config used for reproducibility
    config_out = out / "preprocess_config.yaml"
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Config saved to {config_out}")
    print()
    print(f"Done.  Set data.preprocessed_dir: {output_dir}  in your training config.")


if __name__ == "__main__":
    main()
