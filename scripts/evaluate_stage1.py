#!/usr/bin/env python3
"""Evaluate a trained Stage-1 RAE-text autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import merge_configs, ExperimentConfig
from src.backbones.simcse_repr import SimCSEReprEncoder
from src.models.detail_encoder import DetailEncoder
from src.models.decoder import AutoRegressiveDecoder
from src.models.rae_text import RAEText
from src.data.datasets import load_text_dataset
from src.data.collators import ARDecoderCollator
from src.eval.reconstruction_metrics import (
    token_accuracy,
    mean_edit_distance,
    perplexity_from_loss,
)
from src.eval.semantic_metrics import cosine_sim_batch
from scripts.train_stage1 import build_model


@torch.no_grad()
def evaluate(
    model: RAEText,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_samples: int = 1000,
    num_qualitative: int = 20,
) -> dict:
    """Run full evaluation on the given dataloader."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_sem = 0.0
    n_batches = 0
    all_tok_acc = []
    all_edit_dist = []
    all_cos_sim = []
    qualitative = []

    bos_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 101
    eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 102
    n_seen = 0

    for batch in dataloader:
        if n_seen >= max_samples:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass for loss
        out = model(**batch)
        total_loss += out["loss"].item()
        total_recon += out["l_recon"].item()
        total_sem += out["l_sem"].item()
        n_batches += 1

        # Greedy decode for metrics
        gen_ids = model.generate_greedy(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            max_length=batch["input_ids"].size(1),
        )

        # Token accuracy (compare against labels where not -100)
        labels = batch["labels"]
        min_len = min(gen_ids.size(1), labels.size(1))
        acc = token_accuracy(gen_ids[:, :min_len], labels[:, :min_len])
        all_tok_acc.append(acc)

        # Decode to text
        orig_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_edit_dist.append(mean_edit_distance(gen_texts, orig_texts))

        # Semantic similarity
        orig_emb, _ = model.repr_encoder.encode(batch["input_ids"], batch["attention_mask"])

        # Re-tokenize generated texts for embedding
        gen_enc = tokenizer(
            gen_texts,
            max_length=batch["input_ids"].size(1),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        gen_emb, _ = model.repr_encoder.encode(gen_enc["input_ids"], gen_enc["attention_mask"])
        sims = cosine_sim_batch(orig_emb, gen_emb)
        all_cos_sim.extend(sims.cpu().tolist())

        # Collect qualitative examples
        if len(qualitative) < num_qualitative:
            for orig, gen in zip(orig_texts, gen_texts):
                if len(qualitative) >= num_qualitative:
                    break
                qualitative.append({"original": orig, "reconstruction": gen})

        n_seen += batch["input_ids"].size(0)

    # Aggregate
    cos_tensor = torch.tensor(all_cos_sim)
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "recon_loss": total_recon / max(n_batches, 1),
        "sem_loss": total_sem / max(n_batches, 1),
        "perplexity": perplexity_from_loss(total_recon / max(n_batches, 1)),
        "token_accuracy": sum(all_tok_acc) / max(len(all_tok_acc), 1),
        "edit_distance": sum(all_edit_dist) / max(len(all_edit_dist), 1),
        "sem_cos_mean": cos_tensor.mean().item() if len(cos_tensor) > 0 else 0.0,
        "sem_cos_median": cos_tensor.median().item() if len(cos_tensor) > 0 else 0.0,
        "num_samples": n_seen,
    }
    return {"metrics": metrics, "qualitative": qualitative}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 RAE autoencoder")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="YAML config files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = merge_configs(*args.configs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or 0

    # Model
    device = torch.device(args.device)
    model = build_model(cfg, vocab_size, pad_token_id)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    state_dict_path = ckpt_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        # Try safetensors
        import glob
        safetensor_files = glob.glob(str(ckpt_path / "*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file
            state_dict = load_file(safetensor_files[0])
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_path}")
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Data
    datasets = load_text_dataset(cfg.data)
    collator = ARDecoderCollator.from_data_config(tokenizer, cfg.data)
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=cfg.eval.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    # Evaluate
    results = evaluate(
        model=model,
        dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_samples=cfg.eval.max_samples,
        num_qualitative=cfg.eval.num_qualitative,
    )

    # Print results
    print("\n=== Metrics ===")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Qualitative Examples (first {len(results['qualitative'])}) ===")
    for i, ex in enumerate(results["qualitative"]):
        print(f"\n--- Example {i+1} ---")
        print(f"  Original:       {ex['original'][:200]}")
        print(f"  Reconstruction: {ex['reconstruction'][:200]}")

    # Save to JSON
    out_path = Path(cfg.train.output_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
