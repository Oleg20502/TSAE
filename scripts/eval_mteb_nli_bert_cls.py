#!/usr/bin/env python3
"""
Temporary file.
Evaluate nli-bert-base via CLS pooling (not standard pooling for it).

Usage example:
  python scripts/eval_mteb_nli_bert_cls.py \
    --config configs/model/bottleneck_bert.yaml \
    --output-dir mteb_results/nli-bert-cls/ \
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

from src.backbones.repr_embedder import CLSReprEncoder
from src.utils.config import merge_bottleneck_configs


# ---------------------------------------------------------------------
# MTEB v2 encoder wrapper (implements AbsEncoder protocol)
# ---------------------------------------------------------------------

class CLSEncoderWrapper(AbsEncoder):
    """Wrap Encoder an MTEB-compatible encoder.

    Inherits similarity / similarity_pairwise from AbsEncoder.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_length: int = 128,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length
        self.mteb_model_meta = ModelMeta.create_empty()

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

            embs, _ = self.model.encode(enc["input_ids"], enc["attention_mask"])

            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run MTEB STS + Probing evaluation on BottleneckAE or other sentence embedding model."
    )

    p.add_argument(
        "--config",
        action="append",
        default=None,
        help="YAML config file for BottleneckAE (repeatable; later files override earlier ones).",
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

    return p.parse_args()


def main():
    args = parse_args()

    if not args.config:
        raise SystemExit("--config is required")
    
    config_paths = args.config
    cfg = merge_bottleneck_configs(*config_paths)
    model_name = "sentence-transformers/nli-bert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = CLSReprEncoder(model_name=model_name)
    wrapped_model = CLSEncoderWrapper(
        model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=cfg.model.max_length,
    )

    # ---------------- MTEB tasks & evaluation ----------------
    print(f"Loading MTEB tasks (STS + Probing) ...")
    tasks = mteb.get_tasks(task_types=["STS", "Probing"], languages=["eng"])

    print(f"Running MTEB evaluation for model: {model_name}")
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
        out_path = output_dir / f"{task_result.task_name}.json"
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
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(result.task_results)} task results written to {output_dir}/")
    for name, score in per_task.items():
        print(f"  {name}: {score:.4f}")
    print(f"\n  mean={aggregate['mean']:.4f}  median={aggregate['median']:.4f}  std={aggregate['std']:.4f}")


if __name__ == "__main__":
    main()
