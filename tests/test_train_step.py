"""Test that a single training step runs and the loss decreases."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones.base_repr import BaseTextReprEncoder
from src.models.detail_encoder import DetailEncoder
from src.models.decoder import AutoRegressiveDecoder
from src.models.rae_text import RAEText


# ---------------------------------------------------------------------------
# Dummy backbone (same as test_shapes)
# ---------------------------------------------------------------------------

class DummyReprEncoder(BaseTextReprEncoder):
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
        tok_emb = self.embedding(input_ids)
        sent_emb = tok_emb.mean(dim=1)
        return sent_emb, tok_emb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, H = 4, 32, 64
D_SEM = 32
D_DET = 32
M = 8
VOCAB = 1000
D_DEC = 32
N_HEADS = 4


def _build_rae_model():
    encoder = DummyReprEncoder(hidden_size=H)
    detail_enc = DetailEncoder(
        input_dim=H, d_det=D_DET, n_tokens=M, n_layers=2, n_heads=N_HEADS,
    )
    decoder = AutoRegressiveDecoder(
        vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
        d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
    )
    model = RAEText(
        repr_encoder=encoder,
        detail_encoder=detail_enc,
        decoder=decoder,
        d_sem=D_SEM,
        lambda_sem=0.2,
        freeze_repr=False,
    )
    return model


def _make_batch():
    input_ids = torch.randint(1, VOCAB, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    decoder_input_ids = torch.randint(1, VOCAB, (B, T))
    decoder_attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = torch.randint(1, VOCAB, (B, T))
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

class TestTrainStep:
    def test_loss_is_finite(self):
        model = _build_rae_model()
        batch = _make_batch()
        out = model(**batch)
        assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"

    def test_backward_runs(self):
        model = _build_rae_model()
        batch = _make_batch()
        out = model(**batch)
        out["loss"].backward()
        # Check that gradients exist for trainable params
        n_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert n_grads > 0, "No gradients computed"

    def test_loss_decreases(self):
        """Run a few optimiser steps and verify the loss trend is downward."""
        model = _build_rae_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        # The loss after 10 steps should be lower than the initial loss
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_frozen_repr_no_grad(self):
        """When freeze_repr=True the backbone should have no gradients."""
        model = _build_rae_model()
        # Rebuild with freeze
        model_frozen = RAEText(
            repr_encoder=model.repr_encoder,
            detail_encoder=model.detail_encoder,
            decoder=model.decoder,
            d_sem=D_SEM,
            lambda_sem=0.2,
            freeze_repr=True,
        )
        batch = _make_batch()
        out = model_frozen(**batch)
        out["loss"].backward()

        for name, p in model_frozen.repr_encoder.named_parameters():
            assert not p.requires_grad, f"repr_encoder.{name} should be frozen"
            assert p.grad is None, f"repr_encoder.{name} should have no grad"
