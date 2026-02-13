"""Tests for the Bottleneck autoencoder: shapes, augmentation, forward pass, and training step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones.base_repr import BaseTextReprEncoder
from src.models.bottleneck_encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.models.decoder import AutoRegressiveDecoder
from src.models.bottleneck_ae import BottleneckAE


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

B, T, H = 4, 32, 64  # batch, seq len, hidden / backbone dim
D_LATENT = 32
N_LATENT = 4  # n_latent_tokens for tests
VOCAB = 1000
D_DEC = 32
N_HEADS = 4


@pytest.fixture
def encoder():
    return BottleneckEncoder(
        vocab_size=VOCAB, d_model=D_LATENT, d_latent=D_LATENT,
        n_latent_tokens=N_LATENT,
        n_layers=2, n_heads=N_HEADS, d_ff=D_LATENT * 2,
        max_length=T, pad_token_id=0,
    )


@pytest.fixture
def decoder():
    return AutoRegressiveDecoder(
        vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
        d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
    )


@pytest.fixture
def dummy_repr():
    return DummyReprEncoder(hidden_size=H)


@pytest.fixture
def bottleneck_model(encoder, decoder, dummy_repr):
    return BottleneckAE(
        encoder=encoder,
        decoder=decoder,
        repr_encoder=dummy_repr,
        latent_aug=LatentAugmentation(noise_std=0.1, feature_dropout_p=0.1),
        lambda_sem=0.2,
        freeze_repr=False,  # don't freeze dummy
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
# Tests: BottleneckEncoder
# ---------------------------------------------------------------------------

class TestBottleneckEncoderShapes:
    def test_output_shape(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        z = encoder(ids, mask)
        assert z.shape == (B, N_LATENT, D_LATENT)

    def test_with_padding(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        mask[:, -10:] = 0
        ids[mask == 0] = 0
        z = encoder(ids, mask)
        assert z.shape == (B, N_LATENT, D_LATENT)

    def test_no_mask(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        z = encoder(ids, attention_mask=None)
        assert z.shape == (B, N_LATENT, D_LATENT)

    def test_different_d_latent_and_n_tokens(self):
        """When d_model != d_latent and n_latent_tokens > 1, shapes are correct."""
        enc = BottleneckEncoder(
            vocab_size=VOCAB, d_model=64, d_latent=32,
            n_latent_tokens=3,
            n_layers=1, n_heads=4, d_ff=128,
            max_length=T, pad_token_id=0,
        )
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        z = enc(ids, mask)
        assert z.shape == (B, 3, 32)


# ---------------------------------------------------------------------------
# Tests: LatentAugmentation
# ---------------------------------------------------------------------------

class TestLatentAugmentation:
    def test_no_op_defaults(self):
        """With default params (0.0, 0.0), output should equal input."""
        aug = LatentAugmentation()
        aug.train()
        z = torch.randn(B, 1, D_LATENT)
        z_out = aug(z)
        assert torch.equal(z, z_out)

    def test_noise_changes_output(self):
        aug = LatentAugmentation(noise_std=1.0)
        aug.train()
        z = torch.randn(B, 1, D_LATENT)
        z_out = aug(z)
        # With std=1.0, output should differ
        assert not torch.equal(z, z_out)

    def test_eval_mode_is_identity(self):
        """In eval mode, augmentation should be a no-op even with high params."""
        aug = LatentAugmentation(noise_std=1.0, feature_dropout_p=0.9)
        aug.eval()
        z = torch.randn(B, 1, D_LATENT)
        z_out = aug(z)
        assert torch.equal(z, z_out)

    def test_feature_dropout_zeros_some_dims(self):
        aug = LatentAugmentation(feature_dropout_p=0.5)
        aug.train()
        torch.manual_seed(0)
        z = torch.ones(B, 1, D_LATENT)
        z_out = aug(z)
        # Some elements should be zeroed (with high probability at p=0.5)
        assert (z_out == 0).any(), "Expected some zeroed dimensions from feature dropout"


# ---------------------------------------------------------------------------
# Tests: BottleneckAE shapes
# ---------------------------------------------------------------------------

class TestBottleneckAEShapes:
    def test_encode_latent(self, bottleneck_model):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        z, sent_emb = bottleneck_model.encode_latent(ids, mask)
        assert z.shape == (B, N_LATENT, D_LATENT)
        assert sent_emb.shape == (B, H)

    def test_forward_output_keys(self, bottleneck_model, sample_batch):
        out = bottleneck_model(**sample_batch)
        assert "loss" in out
        assert "logits" in out
        assert "l_recon" in out
        assert "l_sem" in out
        assert "sent_emb" in out
        assert "z" in out

    def test_forward_shapes(self, bottleneck_model, sample_batch):
        out = bottleneck_model(**sample_batch)
        assert out["logits"].shape == (B, T, VOCAB)
        assert out["z"].shape == (B, N_LATENT, D_LATENT)
        assert out["loss"].dim() == 0  # scalar

    def test_generate_greedy(self, bottleneck_model):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        gen = bottleneck_model.generate_greedy(
            ids, mask, bos_token_id=101, eos_token_id=102, max_length=16,
        )
        assert gen.shape[0] == B
        assert gen.shape[1] <= 16
        assert gen[:, 0].eq(101).all()  # starts with BOS


# ---------------------------------------------------------------------------
# Tests: BottleneckAE training step
# ---------------------------------------------------------------------------

def _build_bottleneck_model():
    encoder = BottleneckEncoder(
        vocab_size=VOCAB, d_model=D_LATENT, d_latent=D_LATENT,
        n_latent_tokens=N_LATENT,
        n_layers=2, n_heads=N_HEADS, d_ff=D_LATENT * 2,
        max_length=T, pad_token_id=0,
    )
    decoder = AutoRegressiveDecoder(
        vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
        d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
    )
    repr_enc = DummyReprEncoder(hidden_size=H)
    return BottleneckAE(
        encoder=encoder,
        decoder=decoder,
        repr_encoder=repr_enc,
        latent_aug=LatentAugmentation(noise_std=0.05, feature_dropout_p=0.1),
        lambda_sem=0.2,
        freeze_repr=False,
    )


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


class TestBottleneckTrainStep:
    def test_loss_is_finite(self):
        model = _build_bottleneck_model()
        batch = _make_batch()
        out = model(**batch)
        assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"

    def test_backward_runs(self):
        model = _build_bottleneck_model()
        batch = _make_batch()
        out = model(**batch)
        out["loss"].backward()
        # Check that gradients exist for trainable params
        n_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert n_grads > 0, "No gradients computed"

    def test_loss_decreases(self):
        """Run a few optimiser steps and verify the loss trend is downward."""
        model = _build_bottleneck_model()
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
        model = _build_bottleneck_model()
        # Rebuild with freeze
        model_frozen = BottleneckAE(
            encoder=model.encoder,
            decoder=model.decoder,
            repr_encoder=model.repr_encoder,
            latent_aug=model.latent_aug,
            lambda_sem=0.2,
            freeze_repr=True,
        )
        batch = _make_batch()
        out = model_frozen(**batch)
        out["loss"].backward()

        for name, p in model_frozen.repr_encoder.named_parameters():
            assert not p.requires_grad, f"repr_encoder.{name} should be frozen"
            assert p.grad is None, f"repr_encoder.{name} should have no grad"

    def test_augmentation_disabled_in_eval(self):
        """Augmentation should not alter latent in eval mode."""
        model = _build_bottleneck_model()
        model.eval()
        batch = _make_batch()

        z1, _ = model.encode_latent(batch["input_ids"], batch["attention_mask"])
        z_aug = model.latent_aug(z1)
        assert torch.equal(z1, z_aug), "Augmentation should be identity in eval mode"
