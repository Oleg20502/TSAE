"""Latent augmentation: noise injection and feature dropout for robustness."""

import math

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
        normalize_latent: bool = False,
        sigma_type: str = "abs", # "abs" or "rel"
    ):
        super().__init__()
        self.noise_std = noise_std
        self.feature_dropout_p = feature_dropout_p
        self.normalize_latent = normalize_latent
        
        if sigma_type == "abs":
            self.sigma_noise = self._abs_sigma_noise
        elif sigma_type == "rel":
            self.sigma_noise = self._rel_sigma_noise
        else:
            raise ValueError(f"Invalid sigma_type: {sigma_type}")
    
    def _abs_sigma_noise(self, z: torch.Tensor) -> torch.Tensor:
        return self.noise_std * torch.randn_like(z)

    def _rel_sigma_noise(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = z.norm(dim=-1, keepdim=True)
        d = z.shape[-1]
        return self.noise_std * z_norm * torch.randn_like(z) / math.sqrt(d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, L, D) latent tensor

        Returns:
            Augmented latent tensor of the same shape.
        """
        if not self.training:
            return z

        # 1) Gaussian noise
        if self.noise_std > 0.0:
            z = z + self.sigma_noise(z)

        # 2) Feature dropout (drop individual dimensions)
        if self.feature_dropout_p > 0.0:
            z = F.dropout(z, p=self.feature_dropout_p, training=True)

        if self.normalize_latent:
            z = F.normalize(z, p=2, dim=-1)

        return z
