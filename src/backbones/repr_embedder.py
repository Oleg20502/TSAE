"""Classes for text representation encoders."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import AutoModel



class BaseTextReprEncoder(ABC, nn.Module):
    """Abstract base class for text representation encoders.
    Interface that every representation backbone must implement.

    Subclasses provide a pretrained (or trainable) encoder that returns
    both sentence-level and token-level representations.
    """

    @abstractmethod
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of tokenized text.

        Args:
            input_ids:      (B, T) token ids.
            attention_mask:  (B, T) 1/0 mask.
            model_kwargs:     other kwargs

        Returns:
            sent_emb: (B, D_s) sentence-level semantic embedding.
            tok_emb:  (B, T, D_t) token-level hidden states.
        """
        ...



class STReprEncoder(BaseTextReprEncoder):
    def __init__(
        self,
        model_name: str,
        force_normalize: bool = False,
    ):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self._sent_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_kwargs,
    ):
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        out = self.model(features, **model_kwargs)
        sent_emb = out["sentence_embedding"]

        return sent_emb
    
    @property
    def sent_dim(self) -> int:
        return self._sent_dim



class CLSReprEncoder(BaseTextReprEncoder):
    """Representaion embedder via CLS pooling.

    Uses CLS pooling for the sentence embedding and the full last-hidden-state
    for token-level representations.
    """

    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size: int = self.model.config.hidden_size
        self._sent_dim = hidden_size
        self._tok_dim = hidden_size

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    @property
    def tok_dim(self) -> int:
        return self._tok_dim

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok_emb = outputs.last_hidden_state          # (B, T, H)
        sent_emb = tok_emb[:, 0, :]                  # CLS pooling  (B, H)
        return sent_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias so the module is callable in the standard nn.Module way."""
        return self.encode(input_ids, attention_mask)


class AllReprEncoder(BaseTextReprEncoder):
    """
    Wraps a HuggingFace mean pooling embedder model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size: int = self.model.config.hidden_size
        self._sent_dim = hidden_size
        self._tok_dim = hidden_size

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    @property
    def tok_dim(self) -> int:
        return self._tok_dim

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok_emb = outputs.last_hidden_state          # (B, T, H)
        sent_emb = tok_emb.mean(dim=1)              # (B, H)
        return sent_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias so the module is callable in the standard nn.Module way."""
        return self.encode(input_ids, attention_mask)
