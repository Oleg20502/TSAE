"""Semantic preservation metrics for Stage-1 evaluation."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from src.backbones.base_repr import BaseTextReprEncoder


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


@torch.no_grad()
def semantic_preservation_score(
    encoder: BaseTextReprEncoder,
    original_ids: torch.Tensor,
    original_mask: torch.Tensor,
    reconstructed_ids: torch.Tensor,
    reconstructed_mask: torch.Tensor,
) -> float:
    """Mean cosine similarity between embeddings of originals and reconstructions.

    Args:
        encoder:            the representation encoder (SimCSE / TSDAE).
        original_ids:       (B, T) original token ids.
        original_mask:       (B, T) attention mask.
        reconstructed_ids:  (B, T') reconstructed token ids.
        reconstructed_mask: (B, T') attention mask.

    Returns:
        Mean cosine similarity in [-1, 1] (higher is better).
    """
    orig_emb, _ = encoder.encode(original_ids, original_mask)
    recon_emb, _ = encoder.encode(reconstructed_ids, reconstructed_mask)
    sims = cosine_sim_batch(orig_emb, recon_emb)
    return sims.mean().item()


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
