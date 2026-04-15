#!/usr/bin/env python3
"""Evaluate a hybrid latent checkpoint with greedy ``HybridLatentReasoningGPT2.generate``.

Loads the same YAML as training, restores hybrid weights, caches ``z_end`` from the frozen
AE encoder (same as the trainer), then runs generation on the validation split (unbatched).

Usage:
    conda activate rmt-a100
    python scripts/eval_hybrid_latent_generate.py \\
        --config configs/train/hybrid_gsm8k.yaml \\
        --checkpoint outputs/hybrid_latent_5/nl_1_sl_1.0/final/model.safetensors

    # Or pass a directory that contains ``model.safetensors``:
    python scripts/eval_hybrid_latent_generate.py --config ... --checkpoint outputs/.../final
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hybrid_latent_collators import HybridLatentCollator
from src.data.hybrid_latent_datasets import load_hybrid_latent_dataset
from src.models.bottleneck_ae import load_bottleneck_model
from src.models.hybrid_latent_model import HybridLatentReasoningGPT2
from src.trainers.hybrid_latent_trainer import load_hybrid_latent_weights
from src.utils.config import load_hybrid_latent_config_from_paths


def _resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        p = os.path.join(path, "model.safetensors")
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(
            f"No model.safetensors under directory: {path}"
        )
    return path


def _row_task_labels(row: Any, dc) -> Tuple[str, str]:
    """Task / gold answer strings (same keys as hybrid datasets / collator)."""
    if isinstance(row, dict):
        task = row.get(dc.task_column) or row.get("task") or ""
        labels = row.get(dc.labels_column) or row.get("labels") or ""
    else:
        task = getattr(row, dc.task_column, None) or getattr(row, "task", None) or ""
        labels = getattr(row, dc.labels_column, None) or getattr(row, "labels", None) or ""
    return str(task), str(labels)


def _last_number(s: str) -> Optional[str]:
    s = s.replace(",", "")
    found = re.findall(r"-?\d+(?:\.\d+)?", s)
    return found[-1] if found else None


@torch.no_grad()
def _cache_end_thinking_latent(
    model: HybridLatentReasoningGPT2,
    ae_encoder: torch.nn.Module,
    ae_tokenizer,
    phrase: str,
    device: torch.device,
) -> None:
    ae_max_len = int(ae_encoder.max_length)
    enc = ae_tokenizer(
        phrase,
        max_length=ae_max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    z = ae_encoder(ids, am)
    model.set_end_thinking_latent(z)


def _encode_prompt_batch(
    collator: HybridLatentCollator,
    gpt2_tok,
    task: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_gpt = gpt2_tok.pad_token_id or gpt2_tok.eos_token_id or 0
    pt = gpt2_tok(task, add_special_tokens=False)["input_ids"]
    p_tensor, p_mask = collator._pad_gpt2(pt, collator.P, pad_gpt, padding_side="left")
    prompt_token_ids = p_tensor.unsqueeze(0).to(device)
    prompt_attention_mask = p_mask.unsqueeze(0).to(device)
    trig = torch.tensor([collator.trigger_ids], dtype=torch.long, device=device)
    return prompt_token_ids, prompt_attention_mask, trig


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid latent eval via generate()")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Training YAML path(s); later overrides earlier.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="model.safetensors / pytorch_model.bin or directory containing model.safetensors",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("validation", "train"),
        help="Which split from load_hybrid_latent_dataset to score.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of examples (after dataset subsetting from config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: cuda if available).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=("bf16", "fp16", "fp32"),
        help="Compute dtype for the hybrid model on GPU (ignored on CPU -> fp32).",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="If set, append one JSON object per line (task, gold, pred, em, n_latent_chunks).",
    )
    args = parser.parse_args()

    cfg = load_hybrid_latent_config_from_paths(args.config)
    mc = cfg.model
    dc = cfg.data

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  checkpoint: {ckpt_path}")

    print(f"Loading AE from {mc.ae_checkpoint_path} ...")
    ae_model, ae_tokenizer, ae_cfg = load_bottleneck_model(
        config_paths=[mc.ae_config_path],
        checkpoint_path=mc.ae_checkpoint_path,
        device="cpu",
    )
    ae_encoder = ae_model.encoder
    ae_encoder.eval()
    for p in ae_encoder.parameters():
        p.requires_grad = False

    n_latent_tokens = ae_cfg.model.n_latent_tokens
    d_ae = ae_cfg.model.d_model
    ae_max_length = ae_cfg.model.max_length

    model = HybridLatentReasoningGPT2(
        d_ae=d_ae,
        n_latent_tokens=n_latent_tokens,
        pretrained_name=mc.pretrained_gpt2,
        max_seq_len=mc.max_model_seq_len,
    )
    load_hybrid_latent_weights(ckpt_path, model, device="cpu")
    model.to(device)
    model.eval()

    if device.type == "cuda":
        if args.dtype == "bf16" and torch.cuda.is_bf16_supported():
            model.to(dtype=torch.bfloat16)
        elif args.dtype == "fp16":
            model.to(dtype=torch.float16)
        else:
            model.to(dtype=torch.float32)
    else:
        model.to(dtype=torch.float32)

    # Keep AE encoder in fp32 like HybridLatentTrainer (z_end is cast inside set_end_thinking_latent).
    ae_encoder = ae_encoder.to(device)

    _cache_end_thinking_latent(
        model, ae_encoder, ae_tokenizer, dc.end_of_thinking_phrase, device
    )

    gpt2_tok = AutoTokenizer.from_pretrained(dc.gpt2_tokenizer_name)
    bos_id = gpt2_tok.bos_token_id
    if bos_id is None:
        bos_id = gpt2_tok.eos_token_id or 50256
    eos_id = gpt2_tok.eos_token_id
    if eos_id is None:
        eos_id = bos_id

    collator = HybridLatentCollator(
        gpt2_tok=gpt2_tok,
        ae_tok=ae_tokenizer,
        ae_max_length=ae_max_length,
        n_latent_tokens=n_latent_tokens,
        cfg=dc,
    )

    datasets = load_hybrid_latent_dataset(dc)
    ds = datasets[args.split]
    n_total = len(ds)
    n_eval = n_total if args.max_samples is None else min(n_total, args.max_samples)

    max_latent = dc.max_cot_steps
    max_ans = dc.max_answer_tokens

    em = 0
    em_num = 0
    n_latent_hist: Dict[int, int] = {}

    pred_f = open(args.predictions_path, "a", encoding="utf-8") if args.predictions_path else None
    try:
        for i in tqdm(range(n_eval), desc="generate eval"):
            row = ds[i]
            task, gt_answer = _row_task_labels(row, dc)

            prompt_ids, prompt_m, trig = _encode_prompt_batch(collator, gpt2_tok, task, device)

            gen_ids, n_chunks = model.generate(
                prompt_ids,
                prompt_m,
                trig,
                max_latent_chunks=max_latent,
                max_answer_tokens=max_ans,
                bos_token_id=int(bos_id),
                eos_token_id=int(eos_id),
            )
            pred = gpt2_tok.decode(gen_ids[0].tolist(), skip_special_tokens=True)

            ok = pred == gt_answer
            em += int(ok)
            em_num += 1
            n_latent_hist[n_chunks] = n_latent_hist.get(n_chunks, 0) + 1

            if pred_f is not None:
                rec = {
                    "task": task,
                    "gold": gt_answer,
                    "pred": pred,
                    "exact": ok,
                    "n_latent_chunks": n_chunks,
                }
                pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if pred_f is not None:
            pred_f.close()

    acc = em / max(em_num, 1)
    print(f"=== Results ({n_eval} examples, split={args.split}) ===")
    print(f"Answer accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
