#!/usr/bin/env python3
"""Training script for the Bottleneck autoencoder without a repr encoder."""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collators import ARDecoderCollator
from src.data.datasets import load_text_dataset
from src.eval.reconstruction_metrics import compute_metrics
from src.models.bottleneck_ae import build_ae_components
from src.trainers import BottleneckTrainer
from src.utils.config import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Train Bottleneck autoencoder (no repr encoder)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config containing model, train, and data sections",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume from, or 'latest' to auto-detect",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"=== Config ===\n{cfg}\n")

    tc = cfg.train
    Path(tc.output_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg, Path(tc.output_dir) / "config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or 0

    encoder, decoder, latent_aug, lambda_sem = build_ae_components(
        cfg, vocab_size, pad_token_id
    )
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in decoder.parameters()
    )
    print(f"Parameters: {n_params:,} trainable")

    datasets = load_text_dataset(cfg.data)
    collator = ARDecoderCollator(tokenizer, cfg.model.max_length, cfg.data.text_column)

    trainer = BottleneckTrainer(
        encoder=encoder,
        decoder=decoder,
        repr_encoder=None,
        latent_aug=latent_aug,
        lambda_sem=lambda_sem,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        train_config=tc,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = tc.output_dir + "/final"
    trainer.save_model(final_dir, tokenizer=tokenizer)
    print(f"Model saved to {final_dir}")

    if tc.ema_decay and tc.ema_decay > 0.0:
        final_raw_dir = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw_dir, tokenizer=tokenizer)
        print(f"Non-EMA model saved to {final_raw_dir}")


if __name__ == "__main__":
    main()
