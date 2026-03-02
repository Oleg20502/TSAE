"""SimCSE backbone wrapper for sentence + token embeddings."""

from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from src.backbones.base_repr import BaseTextReprEncoder


class SimCSEReprEncoder(BaseTextReprEncoder):
    """Wraps a HuggingFace SimCSE model.

    Uses CLS pooling for the sentence embedding and the full last-hidden-state
    for token-level representations.
    """

    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden_size: int = self.model.config.hidden_size
        self._sent_dim = hidden_size
        self._tok_dim = hidden_size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    @property
    def tok_dim(self) -> int:
        return self._tok_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok_emb = outputs.last_hidden_state          # (B, T, H)
        sent_emb = tok_emb[:, 0, :]                  # CLS pooling  (B, H)
        return sent_emb, tok_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias so the module is callable in the standard nn.Module way."""
        return self.encode(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def get_tokenizer(model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"):
        return AutoTokenizer.from_pretrained(model_name)


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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    @property
    def tok_dim(self) -> int:
        return self._tok_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok_emb = outputs.last_hidden_state          # (B, T, H)
        sent_emb = tok_emb.mean(dim=1)              # (B, H)
        return sent_emb, tok_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias so the module is callable in the standard nn.Module way."""
        return self.encode(input_ids, attention_mask)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def get_tokenizer(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        return AutoTokenizer.from_pretrained(model_name)
