#!/usr/bin/env python3
"""Train hybrid latent reasoning model (GPT-2 + frozen bottleneck AE).

GSM8K: default dataset from HuggingFace + ``HybridLatentCollator``.
LM / Wikipedia: set ``data.preprocessed_dir`` to ``prepare_hybrid_lm_dataset.py`` output + ``GeneralHybridLatentCollator`` (auto).

Usage:
    conda activate rmt-a100
    accelerate launch scripts/train_hybrid_latent_model.py --config configs/train/hybrid_gsm8k.yaml
    accelerate launch scripts/train_hybrid_latent_model.py --config configs/train/hybrid_wikipedia_k4.yaml
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hybrid_latent_collators import GeneralHybridLatentCollator, HybridLatentCollator
from src.data.hybrid_latent_datasets import load_hybrid_latent_dataset
from src.models.bottleneck_ae import load_bottleneck_model
from src.models.hybrid_latent_model import HybridLatentReasoningGPT2
from src.trainers.hybrid_latent_trainer import HybridLatentTrainer, load_hybrid_latent_weights
from src.utils.config import load_hybrid_latent_config_from_paths


def _save_cfg(cfg, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Hybrid latent reasoning training")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="YAML config path(s); later files override earlier.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint dir to resume, or 'latest'.",
    )
    args = parser.parse_args()

    cfg = load_hybrid_latent_config_from_paths(args.config)
    print(f"=== Hybrid latent config ===\n{cfg}\n")

    mc = cfg.model
    tc = cfg.train
    dc = cfg.data

    Path(tc.output_dir).mkdir(parents=True, exist_ok=True)
    _save_cfg(cfg, Path(tc.output_dir) / "config.yaml")

    print(f"Loading frozen AE from {mc.ae_checkpoint_path} ...")
    ae_model, ae_tokenizer, ae_cfg = load_bottleneck_model(
        config_paths=[mc.ae_config_path],
        checkpoint_path=mc.ae_checkpoint_path,
        device="cpu",
    )
    ae_encoder = ae_model.encoder
    ae_decoder = ae_model.decoder
    n_latent_tokens = ae_cfg.model.n_latent_tokens
    d_ae = ae_cfg.model.d_model
    ae_max_length = ae_cfg.model.max_length

    model = HybridLatentReasoningGPT2(
        d_ae=d_ae,
        n_latent_tokens=n_latent_tokens,
        pretrained_name=mc.pretrained_gpt2,
        max_seq_len=mc.max_model_seq_len,
    )

    if tc.init_from_checkpoint:
        if args.resume_from_checkpoint is not None:
            print("Skipping init_from_checkpoint because --resume_from_checkpoint is set.")
        else:
            print(f"Warm-start hybrid model from {tc.init_from_checkpoint}")
            load_hybrid_latent_weights(tc.init_from_checkpoint, model)

    from transformers import AutoTokenizer

    gpt2_tok = AutoTokenizer.from_pretrained(dc.gpt2_tokenizer_name)

    datasets = load_hybrid_latent_dataset(dc)
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['validation'])}")
    
    train_ds = datasets["train"]
    eval_ds = datasets["validation"]

    if dc.preprocessed_dir:
        collator = GeneralHybridLatentCollator(
            gpt2_tok=gpt2_tok,
            ae_tok=ae_tokenizer,
            ae_max_length=ae_max_length,
            n_latent_tokens=n_latent_tokens,
            cfg=dc,
        )
    else:
        collator = HybridLatentCollator(
            gpt2_tok=gpt2_tok,
            ae_tok=ae_tokenizer,
            ae_max_length=ae_max_length,
            n_latent_tokens=n_latent_tokens,
            cfg=dc,
        )

    trainer = HybridLatentTrainer(
        model=model,
        ae_encoder=ae_encoder,
        ae_decoder=ae_decoder,
        ae_tokenizer=ae_tokenizer,
        n_latent_tokens=n_latent_tokens,
        model_cfg=mc,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        train_config=tc,
        end_of_thinking_phrase=dc.end_of_thinking_phrase,
        max_cot_steps=dc.max_cot_steps,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_ema = tc.output_dir + "/final"
    trainer.save_model(final_ema)
    print(f"Saved EMA hybrid model to {final_ema}")

    if tc.ema_decay and tc.ema_decay > 0.0:
        final_raw = tc.output_dir + "/final_raw"
        trainer.save_non_ema_model(final_raw)
        print(f"Saved raw weights to {final_raw}")


if __name__ == "__main__":
    main()
