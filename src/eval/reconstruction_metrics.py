"""Reconstruction quality metrics for Stage-1 evaluation."""

from __future__ import annotations

from typing import Dict, List

import editdistance
import torch


def token_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor, ignore_index: int = -100) -> float:
    """Fraction of correctly predicted tokens (ignoring padding).

    Args:
        pred_ids:   (B, T) predicted token ids.
        target_ids: (B, T) ground-truth token ids (-100 for padding).

    Returns:
        Accuracy in [0, 1].
    """
    mask = target_ids != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (pred_ids == target_ids) & mask
    return (correct.sum().float() / mask.sum().float()).item()


def mean_edit_distance(pred_texts: List[str], target_texts: List[str]) -> float:
    """Average character-level edit distance between predicted and target texts.

    Args:
        pred_texts:   list of decoded predicted strings.
        target_texts: list of decoded target strings.

    Returns:
        Mean edit distance (lower is better).
    """
    if len(pred_texts) == 0:
        return 0.0
    total = sum(editdistance.eval(p, t) for p, t in zip(pred_texts, target_texts))
    return total / len(pred_texts)


def perplexity_from_loss(loss: float) -> float:
    """Convert mean cross-entropy loss to perplexity."""
    return torch.exp(torch.tensor(loss)).item()


def compute_reconstruction_metrics(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    pred_texts: List[str],
    target_texts: List[str],
    ce_loss: float,
) -> Dict[str, float]:
    """Compute all reconstruction metrics in one call.

    Returns:
        Dict with keys: token_accuracy, edit_distance, perplexity.
    """
    return {
        "token_accuracy": token_accuracy(pred_ids, target_ids),
        "edit_distance": mean_edit_distance(pred_texts, target_texts),
        "perplexity": perplexity_from_loss(ce_loss),
    }
