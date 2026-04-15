#!/usr/bin/env python3
"""Training script for the Bottleneck autoencoder."""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.collators import build_collator
from src.data.datasets import load_text_dataset
from src.eval.reconstruction_metrics import compute_metrics
from src.models.bottleneck_ae import build_repr_encoder, build_ae_components, build_sem_proj, load_ae_weights, BottleneckAE
from src.trainers import BottleneckTrainer
from src.utils.config import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description="Train Bottleneck autoencoder")
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

    repr_encoder = build_repr_encoder(cfg.model.repr_encoder_name)
    sem_proj = build_sem_proj(encoder.d_model, repr_encoder.sent_dim)

    autoencoder = BottleneckAE(
        encoder,
        decoder,
        sem_proj=sem_proj,
        latent_aug=latent_aug,
        lambda_sem=lambda_sem,
    )

    if tc.init_from_checkpoint:
        print(f"Initializing BottleneckAE from: {tc.init_from_checkpoint}")
        load_ae_weights(tc.init_from_checkpoint, autoencoder)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    n_sem_proj = sum(p.numel() for p in sem_proj.parameters())
    n_repr = sum(p.numel() for p in repr_encoder.parameters()) if repr_encoder else 0
    print(f"Parameters:\nEncoder: {n_enc:,} trainable; Decoder: {n_dec:,} trainable; SemProj: {n_sem_proj:,} trainable; Repr: {n_repr:,} frozen.")
    print(f"Total: {n_enc + n_dec + n_sem_proj:,} trainable / {n_enc + n_dec + n_sem_proj + n_repr:,} all parameters")
    
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
        repr_encoder=repr_encoder,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        train_config=tc,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_ema_dir = tc.output_dir + "/final"
    trainer.save_model(final_ema_dir, tokenizer=ae_tokenizer)
    print(f"EMA model saved to {final_ema_dir}")

    if tc.ema_decay and tc.ema_decay > 0.0:
        final_raw_dir = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw_dir, tokenizer=ae_tokenizer)
        print(f"Non-EMA model saved to {final_raw_dir}")


if __name__ == "__main__":
    main()
