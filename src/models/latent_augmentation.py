"""Latent augmentation: noise injection and feature dropout for robustness."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAugmentation(nn.Module):
    """Applies augmentations to the latent vector during training.

    Two augmentation strategies (applied sequentially):

    1. **Gaussian noise** – adds ``noise_std * randn_like(z)`` to the latent.
    2. **Feature dropout** – randomly zeroes out individual dimensions of
       the latent vector with probability ``feature_dropout_p``.

    Both augmentations are **only active during training** (``self.training``).
    During evaluation the input is returned unchanged.

    Args:
        noise_std:         Standard deviation of additive Gaussian noise (0 = disabled).
        feature_dropout_p: Probability of dropping each latent dimension (0 = disabled).
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        feature_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.feature_dropout_p = feature_dropout_p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, L, D) latent tensor (typically L=1 for the bottleneck AE).

        Returns:
            Augmented latent tensor of the same shape.
        """
        if not self.training:
            return z

        # 1) Gaussian noise
        if self.noise_std > 0.0:
            z = z + self.noise_std * torch.randn_like(z)

        # 2) Feature dropout (drop individual dimensions)
        if self.feature_dropout_p > 0.0:
            z = F.dropout(z, p=self.feature_dropout_p, training=True)

        return z
