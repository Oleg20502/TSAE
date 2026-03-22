"""Best-checkpoint selection (HuggingFace Trainer–style).

Used by custom trainers to pick which saved checkpoint has the best eval metric
and to exempt that directory from rotation when ``save_total_limit`` is set.
"""

from __future__ import annotations

import math
from typing import Optional


def default_greater_is_better(metric_name: str) -> bool:
    """Infer optimization direction from the metric name (HF-compatible heuristics).

    Unknown metrics default to *lower is better* (loss-like), matching HF.
    """
    name = metric_name.lower()
    if any(k in name for k in ("loss", "error", "perplexity", "mse", "ce", "cross_entropy")):
        return False
    if any(
        k in name
        for k in (
            "accuracy",
            "f1",
            "precision",
            "recall",
            "bleu",
            "rouge",
            "auc",
            "map",
        )
    ):
        return True
    return False


def metric_improves(
    current: float,
    best: Optional[float],
    greater_is_better: bool,
) -> bool:
    """Whether ``current`` is strictly better than ``best``."""
    if best is None:
        return True
    if greater_is_better:
        return current > best
    return current < best


def is_valid_metric_value(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))
