"""Token-level reconstruction loss (cross-entropy)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_cross_entropy_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum and count of per-position CE (shifted next-token), ignoring *ignore_index*.

    Returns:
        ``(ce_sum, n_valid)`` with ``n_valid`` a 0-dim long tensor.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    device = shift_logits.device
    if not (shift_labels != ignore_index).any():
        return (
            torch.zeros((), device=device, dtype=torch.float32),
            torch.zeros((), device=device, dtype=torch.long),
        )
    ce_sum = F.cross_entropy(
        shift_logits.float().reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    n_valid = (shift_labels != ignore_index).sum()
    return ce_sum, n_valid


def reconstruction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute token-level cross-entropy between decoder logits and target labels.

    The logits and labels are shifted so that ``logits[:, :-1]`` predicts
    ``labels[:, 1:]`` (standard next-token setup).  The caller is expected
    to have already placed ``ignore_index`` on padding positions in *labels*.

    Args:
        logits: (B, T, V) raw logits from the decoder.
        labels: (B, T) target token ids with padding set to *ignore_index*.
        ignore_index: label value to ignore in CE (default -100).

    Returns:
        Scalar mean cross-entropy loss.
    """
    ce_sum, n_valid = reconstruction_cross_entropy_stats(logits, labels, ignore_index)
    if n_valid.item() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return (ce_sum / n_valid.float()).to(dtype=logits.dtype)
