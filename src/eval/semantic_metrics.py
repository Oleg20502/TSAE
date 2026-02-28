"""Semantic preservation metrics for Stage-1 evaluation."""

from typing import Dict, List

import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_sim_batch(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> torch.Tensor:
    """Element-wise cosine similarity between two embedding batches.

    Args:
        emb_a: (B, D)
        emb_b: (B, D)

    Returns:
        (B,) cosine similarities.
    """
    return F.cosine_similarity(emb_a, emb_b, dim=-1)


def compute_semantic_metrics(
    cosine_similarities: torch.Tensor | List[float],
) -> Dict[str, float]:
    """Aggregate semantic similarity scores.

    Args:
        cosine_similarities: (N,) tensor or list of per-sample cosine sims.

    Returns:
        Dict with mean, median, min, max cosine similarity.
    """
    if isinstance(cosine_similarities, list):
        cosine_similarities = torch.tensor(cosine_similarities)

    return {
        "sem_cos_mean": cosine_similarities.mean().item(),
        "sem_cos_median": cosine_similarities.median().item(),
        "sem_cos_min": cosine_similarities.min().item(),
        "sem_cos_max": cosine_similarities.max().item(),
    }
