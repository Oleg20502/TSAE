"""Test tensor shapes through all model components."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones.repr_embedder import BaseTextReprEncoder
from src.models.decoder import AutoRegressiveDecoder


# ---------------------------------------------------------------------------
# Dummy backbone for fast tests (avoids downloading real model)
# ---------------------------------------------------------------------------

class DummyReprEncoder(BaseTextReprEncoder):
    """Lightweight dummy encoder for shape testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self._sent_dim = hidden_size
        self._tok_dim = hidden_size
        self.embedding = nn.Embedding(1000, hidden_size)

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    @property
    def tok_dim(self) -> int:
        return self._tok_dim

    def encode(self, input_ids, attention_mask):
        tok_emb = self.embedding(input_ids)       # (B, T, H)
        sent_emb = tok_emb.mean(dim=1)            # (B, H) naive pooling
        return sent_emb, tok_emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, H = 4, 32, 64  # batch, seq len, hidden
D_SEM = 32
D_DET = 32
M = 8  # detail tokens
VOCAB = 1000
D_DEC = 32
N_HEADS = 4


@pytest.fixture
def dummy_encoder():
    return DummyReprEncoder(hidden_size=H)


@pytest.fixture
def decoder():
    return AutoRegressiveDecoder(
        vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
        d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
    )


@pytest.fixture
def sample_batch():
    input_ids = torch.randint(1, VOCAB, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    # Simulate some padding
    attention_mask[:, -5:] = 0
    input_ids[attention_mask == 0] = 0

    decoder_input_ids = torch.randint(1, VOCAB, (B, T))
    decoder_attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = torch.randint(1, VOCAB, (B, T))
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackboneShapes:
    def test_encode_shapes(self, dummy_encoder):
        ids = torch.randint(0, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        sent_emb, tok_emb = dummy_encoder.encode(ids, mask)
        assert sent_emb.shape == (B, H)
        assert tok_emb.shape == (B, T, H)


class TestDecoderShapes:
    def test_output_shapes(self, decoder):
        latent_tokens = torch.randn(B, 1 + M, D_DEC)
        dec_ids = torch.randint(0, VOCAB, (B, T))
        dec_mask = torch.ones(B, T, dtype=torch.long)

        logits, dec_hidden = decoder(latent_tokens, dec_ids, dec_mask)
        assert logits.shape == (B, T, VOCAB)
        assert dec_hidden.shape == (B, D_DEC)

    def test_no_mask(self, decoder):
        latent_tokens = torch.randn(B, 1 + M, D_DEC)
        dec_ids = torch.randint(0, VOCAB, (B, T))

        logits, dec_hidden = decoder(latent_tokens, dec_ids)
        assert logits.shape == (B, T, VOCAB)
        assert dec_hidden.shape == (B, D_DEC)

