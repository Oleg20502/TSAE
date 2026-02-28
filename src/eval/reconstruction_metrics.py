"""Reconstruction quality metrics for autoencoder evaluation."""

from typing import Dict

import numpy as np
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


def perplexity_from_loss(loss: float) -> float:
    """Convert mean cross-entropy loss to perplexity."""
    return torch.exp(torch.tensor(loss)).item()


# ---
# Metrics for Bottleneck AE
# ---


def reconstruction_accuracy(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -100,
) -> float:
    """Token-level reconstruction accuracy (next-token setup).

    Args:
        predictions: (B, T) predicted token ids.
        label_ids: (B, T) target token ids with padding = ignore_index.

    Returns:
        Accuracy in [0, 1].
    """
    pred_ids = predictions[:, :-1]          # (B, T-1)
    shift_labels = label_ids[:, 1:]         # (B, T-1)

    mask = shift_labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (pred_ids.astype(shift_labels.dtype) == shift_labels) & mask
    return float(correct.sum() / mask.sum())


def exact_match_accuracy(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -100,
) -> float:
    """Sequence-level exact match: fraction of sequences where all non-padding tokens match.

    Uses next-token alignment: predictions[:, :-1] vs label_ids[:, 1:].

    Args:
        predictions: (B, T) predicted token ids.
        label_ids: (B, T) target token ids with padding = ignore_index.

    Returns:
        Fraction of sequences with exact match in [0, 1].
    """
    pred_ids = np.array(predictions[:, :-1])
    shift_labels = np.array(label_ids[:, 1:])
    B = pred_ids.shape[0]
    valid_mask = shift_labels != ignore_index
    n_valid = 0
    n_exact = 0
    for b in range(B):
        mask_b = valid_mask[b]
        if not mask_b.any():
            continue
        n_valid += 1
        if np.all((pred_ids[b][mask_b] == shift_labels[b][mask_b])):
            n_exact += 1
    if n_valid == 0:
        return 0.0
    return float(n_exact / n_valid)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute reconstruction_accuracy and exact_match for Trainer.

    Expects predictions to be (B, T) predicted token ids (e.g. from
    preprocess_logits_for_metrics = argmax(logits, dim=-1)).
    """
    predictions = np.array(eval_pred.predictions)
    label_ids = np.array(eval_pred.label_ids)
    ignore_index = -100
    return {
        "token_accuracy": reconstruction_accuracy(
            predictions, label_ids, ignore_index=ignore_index
        ),
        "sequence_accuracy": exact_match_accuracy(
            predictions, label_ids, ignore_index=ignore_index
        ),
    }

