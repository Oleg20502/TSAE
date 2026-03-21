#!/usr/bin/env python3
"""
Evaluate sentence embedding models on MTEB STS + Probing tasks.

Usage examples:

Bottleneck AE (latent embeddings):
  python scripts/eval_mteb.py \
    --model-type bottleneck \
    --config configs/model/bottleneck_bert.yaml \
    --checkpoint outputs/bottleneck_ae/final/model.safetensors \
    --output-dir mteb_results/vavae/ \
    --batch-size 32
    

Sentence-Transformers model:
  python scripts/eval_mteb.py \
    --model-type st \
    --st-model sentence-transformers/all-MiniLM-L6-v2 \
    --output-dir mteb_results/all-MiniLM-L6-v2/ \
    --batch-size 32
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones.repr_embedder import STReprEncoder
from src.models.bottleneck_ae import BottleneckAE, load_bottleneck_model, build_repr_encoder
from src.utils.config import load_config


TASK_TYPES = ["STS", "Classification"]

# ---------------------------------------------------------------------
# MTEB v2 encoder wrapper (implements AbsEncoder protocol)
# ---------------------------------------------------------------------

class EncoderWrapper(AbsEncoder):
    """Wrap BottleneckAE or  as an MTEB-compatible encoder.

    Uses mean-pooled latent tokens as the sentence embedding.
    Inherits similarity / similarity_pairwise from AbsEncoder.
    """

    def __init__(
        self,
        model: BottleneckAE | STReprEncoder,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_length: int = 16,
        force_pool: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length
        self.mteb_model_meta = ModelMeta.create_empty()

        self.force_pool = force_pool

    @torch.no_grad()
    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]

        all_embs: list[np.ndarray] = []
        batch_size = kwargs.get("batch_size", 32)

        for start in range(0, len(sentences), batch_size):
            batch_texts = sentences[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            embs = self.model.encode(enc["input_ids"], enc["attention_mask"])
            if self.force_pool:
                embs = embs.mean(dim=1)  # (B, d_latent)

            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MTEB STS + Probing evaluation on BottleneckAE or other sentence embedding model."
    )

    p.add_argument(
        "--task-type",
        choices=TASK_TYPES,
        default="STS",
        help="MTEB task type to evaluate.",
    )

    p.add_argument(
        "--model-type",
        choices=["bottleneck", "st"],
        required=True,
    )

    p.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config file for BottleneckAE (repeatable; later files override earlier ones).",
    )
    p.add_argument(
        "--checkpoint",
        help="Path to BottleneckAE checkpoint (.bin or .safetensors).",
    )

    p.add_argument(
        "--st-model",
        help="Sentence-Transformers model name or HF hub id.",
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=16,
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--output-dir",
        default="mteb_results",
        help="Directory to write per-task JSON results.",
    )
    p.add_argument(
        "--no-repr",
        action="store_true",
        help="Use the BottleneckAE without the representation encoder.",
    )
    p.add_argument(
        "--use-legacy-repr",
        action="store_true",
        help="Use CLSReprEncoder as the representation encoder.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.model_type == "bottleneck":
        if not args.config or not args.checkpoint:
            raise SystemExit("--config and --checkpoint are required for model-type=bottleneck")

        model, tokenizer, cfg = load_bottleneck_model(
            config_paths=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

        wrapped_model = EncoderWrapper(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_length=cfg.model.max_length,
            force_pool = True,
        )
        model_name = "BottleneckAE"

    else:  # sentence-transformers
        if not args.st_model:
            raise SystemExit("--st-model is required for model-type=sentence-transformers")

        model_name = args.st_model       

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = build_repr_encoder(model_name, args.use_legacy_repr)

        wrapped_model = EncoderWrapper(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_length=args.max_length,
            force_pool = False,
        )


    # ---------------- MTEB tasks & evaluation ----------------
    print(f"Loading MTEB tasks ({args.task_type}) ...")

    if args.task_type == "STS":
        tasks = mteb.get_tasks(
            task_types=["STS"],
            languages=["eng"]
        )
    elif args.task_type == "Classification":
        tasks = mteb.get_tasks(
            tasks=[
                "Banking77Classification.v2",
                "HUMETweetSentimentExtractionClassification",
                "PoemSentimentClassification.v2",
                "ToxicChatClassification.v2",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "TweetSentimentExtractionClassification.v2",
                "MultiHateClassification",
            ],
            languages=["eng"]
        )
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    print(f"Running evaluation for model: {model_name}")
    result = mteb.evaluate(
        wrapped_model,
        tasks=tasks,
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="always",
    )

    # ---------------- Save & summarise results ----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_result in result.task_results:
        out_path = output_dir / f"{args.task_type} / {task_result.task_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(task_result.model_dump_json(indent=2))
        print(f"  {task_result.task_name}: saved to {out_path}")

    per_task: dict[str, float] = {}
    for tr in result.task_results:
        per_task[tr.task_name] = tr.get_score()

    scores = np.array(list(per_task.values()))
    aggregate = {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
    }

    summary = {"tasks": per_task, "aggregate": aggregate}
    summary_path = output_dir / f"{args.task_type} / summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(result.task_results)} task results written to {output_dir}/")
    for name, score in per_task.items():
        print(f"  {name}: {score:.4f}")
    print(f"\n  mean={aggregate['mean']:.4f}  median={aggregate['median']:.4f}  std={aggregate['std']:.4f}")


if __name__ == "__main__":
    main()
