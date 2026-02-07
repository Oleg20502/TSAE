"""Semantic consistency loss (cosine similarity)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def semantic_consistency_loss(
    pred_sem: torch.Tensor,
    target_sem: torch.Tensor,
) -> torch.Tensor:
    """Cosine-similarity based semantic consistency loss.

    Computes ``1 - mean(cos_sim(pred, target))`` so that perfect alignment
    gives 0 and orthogonal vectors give 1.

    Args:
        pred_sem:   (B, D) predicted semantic embedding (from decoder's sem head).
        target_sem: (B, D) target semantic embedding (from repr encoder).

    Returns:
        Scalar loss in [0, 2].
    """
    pred_norm = F.normalize(pred_sem, dim=-1)
    tgt_norm = F.normalize(target_sem, dim=-1)
    cosine = (pred_norm * tgt_norm).sum(dim=-1)  # (B,)
    return 1.0 - cosine.mean()
