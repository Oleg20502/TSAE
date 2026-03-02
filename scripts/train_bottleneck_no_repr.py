#!/usr/bin/env python3
"""Training script for the Bottleneck autoencoder."""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer, TrainingArguments

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import merge_bottleneck_configs, BottleneckExperimentConfig
from src.backbones.simcse_repr import SimCSEReprEncoder
from src.models.bottleneck_encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.models.decoder import AutoRegressiveDecoder
from src.models.bottleneck_ae import BottleneckAE
from src.data.datasets import load_text_dataset
from src.data.collators import ARDecoderCollator
from src.eval.reconstruction_metrics import compute_metrics
from src.trainers import BottleneckTrainer, preprocess_logits_for_metrics


# ---------------------------------------------------------------------------
# Build model from config
# ---------------------------------------------------------------------------

def build_rae_model(
    cfg: BottleneckExperimentConfig,
    vocab_size: int,
    pad_token_id: int,
) -> BottleneckAE:
    mc = cfg.model

    # # Representation backbone (frozen, semantic loss target only)
    # repr_encoder = SimCSEReprEncoder(model_name=mc.backbone_name)

    # Bottleneck encoder (standalone transformer)
    encoder = BottleneckEncoder(
        vocab_size=vocab_size,
        d_model=mc.d_model,
        d_latent=mc.d_latent,
        n_latent_tokens=mc.n_latent_tokens,
        n_layers=mc.encoder_layers,
        n_heads=mc.encoder_heads,
        d_ff=mc.encoder_ff_dim,
        max_length=mc.max_length,
        dropout=mc.encoder_dropout,
        pad_token_id=pad_token_id,
    )

    decoder = AutoRegressiveDecoder(
        vocab_size=vocab_size,
        d_model=mc.d_model,
        n_layers=mc.decoder_layers,
        n_heads=mc.decoder_heads,
        d_ff=mc.decoder_ff_dim,
        max_length=mc.max_length,
        dropout=mc.decoder_dropout,
        pad_token_id=pad_token_id,
    )

    # Latent augmentation
    # latent_aug = LatentAugmentation(
    #     noise_std=mc.noise_std,
    #     feature_dropout_p=mc.feature_dropout_p,
    # )

    # Full model
    model = BottleneckAE(
        encoder=encoder,
        decoder=decoder,
        repr_encoder=None,
        latent_aug=None,
        lambda_sem=mc.lambda_sem,
        freeze_repr=mc.freeze_repr,
    )
    return model


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
    model = build_rae_model(cfg, vocab_size, pad_token_id)
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
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(tc.output_dir + "/final")
    tokenizer.save_pretrained(tc.output_dir + "/final")
    print(f"Model saved to {tc.output_dir}/final")


if __name__ == "__main__":
    main()
