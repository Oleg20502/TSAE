#!/usr/bin/env python3
"""Train the Bottleneck autoencoder without a sentence repr encoder (recon-only).

Same stack as ``train_bottleneck.py``, but ``repr_encoder`` and ``sem_proj`` are
omitted; only reconstruction loss is used (``sent_emb`` is always absent).
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collators import build_collator
from src.data.datasets import load_text_dataset
from src.eval.reconstruction_metrics import compute_metrics
from src.models.bottleneck_ae import BottleneckAE, build_ae_components, load_ae_weights
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

    ae_tokenizer = AutoTokenizer.from_pretrained(cfg.model.ae_tokenizer_name)
    if ae_tokenizer.pad_token_id is None:
        if ae_tokenizer.eos_token is not None:
            ae_tokenizer.pad_token = ae_tokenizer.eos_token
        elif ae_tokenizer.bos_token is not None:
            ae_tokenizer.pad_token = ae_tokenizer.bos_token

    embedder_tokenizer = AutoTokenizer.from_pretrained(cfg.model.repr_encoder_name)
    vocab_size = ae_tokenizer.vocab_size
    pad_token_id = ae_tokenizer.pad_token_id if ae_tokenizer.pad_token_id is not None else 0

    encoder, decoder, latent_aug, lambda_sem = build_ae_components(
        cfg, vocab_size, pad_token_id
    )
    autoencoder = BottleneckAE(
        encoder,
        decoder,
        sem_proj=None,
        latent_aug=latent_aug,
        lambda_sem=lambda_sem,
    )

    if tc.init_from_checkpoint:
        print(f"Initializing BottleneckAE from: {tc.init_from_checkpoint}")
        load_ae_weights(tc.init_from_checkpoint, autoencoder)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(
        f"Parameters: encoder {n_enc:,} trainable; decoder {n_dec:,} trainable; "
        f"total {n_enc + n_dec:,} trainable"
    )

    datasets = load_text_dataset(cfg.data)
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    
    collator = build_collator(
        ae_tokenizer,
        embedder_tokenizer,
        cfg,
    )

    trainer = BottleneckTrainer(
        autoencoder=autoencoder,
        repr_encoder=None,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        train_config=tc,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = tc.output_dir + "/final"
    trainer.save_model(final_dir, tokenizer=ae_tokenizer)
    print(f"EMA model saved to {final_dir}")

    if tc.ema_decay and tc.ema_decay > 0.0:
        final_raw_dir = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw_dir, tokenizer=ae_tokenizer)
        print(f"Non-EMA model saved to {final_raw_dir}")


if __name__ == "__main__":
    main()
