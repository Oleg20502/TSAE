#!/usr/bin/env python3
# Legacy script, should be deprecated and removed. Left for future rebuilding
"""Evaluate a trained RAE-text autoencoder."""


import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import merge_configs
from src.data.datasets import load_text_dataset
from src.data.collators import ARDecoderCollator

from src.eval.evaluation import evaluate


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
    model = build_rae_model(cfg, vocab_size, pad_token_id)

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
    collator = ARDecoderCollator(tokenizer, cfg.model.max_length, cfg.data.text_column)
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
