#!/usr/bin/env python3
"""Training script for the Concept Model.

Loads a frozen bottleneck AE from a checkpoint, then trains a Concept Model
that predicts the next AE latent in a sequence of consecutive text chunks.

Usage:
    accelerate launch scripts/train_concept_model.py --config configs/train/cm_fineweb.yaml
    accelerate launch scripts/train_concept_model.py --config configs/train/cm_fineweb.yaml \
        --resume_from_checkpoint latest
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.concept_collators import ChunkGroupDataset, CMCollator
from src.data.datasets import load_text_dataset
from src.models.bottleneck_ae import load_bottleneck_model
from src.models.concept_model import build_concept_model
from src.trainers.concept_trainer import ConceptTrainer
from src.utils.config import (
    DataConfig,
    load_concept_config_from_paths,
)


def _save_concept_config(cfg, path: Path) -> None:
    """Dump the ConceptExperimentConfig to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Train Concept Model in AE latent space")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to YAML config file(s). Multiple files are merged (later overrides earlier).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume from, or 'latest' to auto-detect.",
    )
    args = parser.parse_args()

    cfg = load_concept_config_from_paths(args.config)
    print(f"=== Concept Model Config ===\n{cfg}\n")

    mc  = cfg.model
    tc  = cfg.train
    dc  = cfg.data

    Path(tc.output_dir).mkdir(parents=True, exist_ok=True)
    _save_concept_config(cfg, Path(tc.output_dir) / "config.yaml")

    # ------------------------------------------------------------------
    # Load frozen AE
    # ------------------------------------------------------------------
    print(f"Loading frozen AE from {mc.ae_checkpoint_path} ...")
    ae_model, ae_tokenizer, ae_cfg = load_bottleneck_model(
        config_paths=[mc.ae_config_path],
        checkpoint_path=mc.ae_checkpoint_path,
        device="cpu",  # Accelerate will move to correct device
    )
    ae_encoder = ae_model.encoder
    ae_decoder = ae_model.decoder

    n_latent_tokens = ae_cfg.model.n_latent_tokens
    d_ae            = ae_cfg.model.d_model
    ae_max_length   = ae_cfg.model.max_length

    print(
        f"Frozen AE: d_ae={d_ae}  n_latent_tokens={n_latent_tokens}"
        f"  max_length={ae_max_length}"
    )

    # ------------------------------------------------------------------
    # Build Concept Model
    # ------------------------------------------------------------------
    concept_model = build_concept_model(mc, d_ae=d_ae, n_latent_tokens=n_latent_tokens)

    n_cm = sum(p.numel() for p in concept_model.parameters())
    n_ae = sum(p.numel() for p in ae_encoder.parameters()) + sum(
        p.numel() for p in ae_decoder.parameters()
    )
    print(
        f"Parameters:\n"
        f"  Concept Model: {n_cm:,} trainable\n"
        f"  Frozen AE encoder+decoder: {n_ae:,}\n"
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Reuse AE data loading — points at same preprocessed FineWeb chunks.
    # We construct a minimal DataConfig for load_text_dataset.
    data_cfg = DataConfig(
        preprocessed_dir=dc.preprocessed_dir,
        text_column=dc.text_column,
        num_val_samples=dc.num_val_samples,
        seed=dc.seed,
    )
    datasets = load_text_dataset(data_cfg)

    train_ds = ChunkGroupDataset(
        datasets["train"], n_chunks=dc.n_chunks, text_column=dc.text_column
    )
    eval_ds = ChunkGroupDataset(
        datasets["validation"], n_chunks=dc.n_chunks, text_column=dc.text_column
    )

    print(
        f"Dataset: {len(train_ds):,} train sequences  "
        f"{len(eval_ds):,} eval sequences  "
        f"(N={dc.n_chunks} chunks each)"
    )

    collator = CMCollator(tokenizer=ae_tokenizer, max_length=ae_max_length)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = ConceptTrainer(
        concept_model=concept_model,
        ae_encoder=ae_encoder,
        ae_decoder=ae_decoder,
        n_latent_tokens=n_latent_tokens,
        lambda_mse=mc.lambda_mse,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        train_config=tc,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_ema_dir = tc.output_dir + "/final"
    trainer.save_model(final_ema_dir)
    print(f"EMA Concept Model saved to {final_ema_dir}")

    if tc.ema_decay and tc.ema_decay > 0.0:
        final_raw_dir = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw_dir)
        print(f"Non-EMA Concept Model saved to {final_raw_dir}")


if __name__ == "__main__":
    main()
