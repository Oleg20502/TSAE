"""Abstract base class for text representation encoders."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BaseTextReprEncoder(ABC, nn.Module):
    """Interface that every representation backbone must implement.

    Subclasses provide a pretrained (or trainable) encoder that returns
    both sentence-level and token-level representations.
    """

    @property
    @abstractmethod
    def sent_dim(self) -> int:
        """Dimensionality of the sentence-level embedding."""
        ...

    @property
    @abstractmethod
    def tok_dim(self) -> int:
        """Dimensionality of each token-level hidden state."""
        ...

    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of tokenized text.

        Args:
            input_ids:      (B, T) token ids.
            attention_mask:  (B, T) 1/0 mask.

        Returns:
            sent_emb: (B, D_s) sentence-level semantic embedding.
            tok_emb:  (B, T, D_t) token-level hidden states.
        """
        ...
