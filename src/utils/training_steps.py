"""Training loop step counts (distributed + gradient accumulation)."""

from __future__ import annotations

import math


def optimizer_steps_per_epoch(
    num_train_samples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_processes: int,
) -> int:
    """Optimizer steps per process per epoch under DDP + gradient accumulation.

    Matches PyTorch ``DistributedSampler`` with ``drop_last=False`` (Accelerate
    default): each rank sees ``ceil(num_train_samples / num_processes)`` samples;
    micro-batches per rank are
    ``ceil(samples_per_process / per_device_batch_size)``; optimizer steps are
    micro-batches divided by ``gradient_accumulation_steps``.

    Use this when the progress bar advances once per ``accelerator.sync_gradients``
    (i.e. once per real optimizer step), not once per forward.
    """
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")
    samples_per_process = math.ceil(num_train_samples / num_processes)
    micro_batches_per_process = math.ceil(samples_per_process / per_device_batch_size)
    return math.ceil(micro_batches_per_process / gradient_accumulation_steps)
