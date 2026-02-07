"""Token-level reconstruction loss (cross-entropy)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
