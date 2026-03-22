"""Tests for distributed training step accounting."""

import math

from src.utils.training_steps import optimizer_steps_per_epoch


def test_optimizer_steps_per_epoch_matches_ddp_sampler_single_gpu():
    n, b, g, p = 1000, 32, 4, 1
    samples = math.ceil(n / p)
    micro = math.ceil(samples / b)
    expected = math.ceil(micro / g)
    assert optimizer_steps_per_epoch(n, b, g, p) == expected


def test_optimizer_steps_per_epoch_four_gpu_fineweb_scale():
    # Example from FineWeb-scale run: 4 GPUs, local batch 32, grad accum 4
    n = 638_949_308
    assert optimizer_steps_per_epoch(n, 32, 4, 4) == 1_247_948
