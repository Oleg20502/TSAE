#!/usr/bin/env python3
"""Training script for the Bottleneck autoencoder."""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer, TrainingArguments

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import merge_bottleneck_configs
from src.data.datasets import load_text_dataset
from src.data.collators import ARDecoderCollator
from src.models.bottleneck_ae import build_bottleneck_model
from src.eval.reconstruction_metrics import compute_metrics
from src.trainers import BottleneckTrainer, preprocess_logits_for_metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Bottleneck autoencoder")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more YAML config files (later overrides earlier)",
    )
    args = parser.parse_args()

    # Load config
    cfg = merge_bottleneck_configs(*args.configs)
    print(f"=== Config ===\n{cfg}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or 0

    # Model
    model = build_bottleneck_model(cfg, vocab_size, pad_token_id)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Data
    datasets = load_text_dataset(cfg.data)
    collator = ARDecoderCollator.from_data_config(tokenizer, cfg.data)

    # Training arguments
    tc = cfg.train
    training_args = TrainingArguments(
        output_dir=tc.output_dir,
        num_train_epochs=tc.epochs,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.lr,
        weight_decay=tc.weight_decay,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=tc.warmup_steps,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="steps",
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        max_grad_norm=tc.max_grad_norm,
        fp16=tc.fp16,
        bf16=tc.bf16,
        dataloader_num_workers=tc.dataloader_num_workers,
        seed=tc.seed,
        report_to=tc.report_to,
        remove_unused_columns=False,  # we pass custom columns
    )

    # Trainer
    trainer = BottleneckTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        ema_decay=tc.ema_decay,
    )

    # Train
    trainer.train()

    # Save final EMA and non-EMA models
    final_ema_dir = tc.output_dir + "/final"
    trainer.save_model(final_ema_dir)
    tokenizer.save_pretrained(final_ema_dir)
    print(f"EMA model saved to {final_ema_dir}")

    if tc.ema_decay is not None and tc.ema_decay > 0.0:
        final_raw_dir = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw_dir)
        tokenizer.save_pretrained(final_raw_dir)
        print(f"Non-EMA model saved to {final_raw_dir}")


if __name__ == "__main__":
    main()
