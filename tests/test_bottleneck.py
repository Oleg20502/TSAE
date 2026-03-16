"""Tests for the Bottleneck autoencoder: shapes, augmentation, forward pass, and training step."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones.repr_embedder import BaseTextReprEncoder
from src.models.bottleneck_ae import BottleneckAE
from src.models.decoder import AutoRegressiveDecoder, ParallelLatentDecoder
from src.models.encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.trainers.bottleneck_trainer import _TrainableCore


# ---------------------------------------------------------------------------
# Dummy backbone (avoids downloading a real model)
# ---------------------------------------------------------------------------

class DummyReprEncoder(BaseTextReprEncoder):
    """Lightweight dummy encoder for shape testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self._sent_dim = hidden_size
        self.embedding = nn.Embedding(1000, hidden_size)

    @property
    def sent_dim(self) -> int:
        return self._sent_dim

    def encode(self, input_ids, attention_mask):
        tok_emb = self.embedding(input_ids)   # (B, T, H)
        return tok_emb.mean(dim=1)            # (B, H)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B, T, H = 4, 32, 64
D_MODEL = 32
N_LATENT = 4
VOCAB = 1000
D_DEC = 32
N_HEADS = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    return BottleneckEncoder(
        vocab_size=VOCAB, d_model=D_MODEL,
        n_latent_tokens=N_LATENT,
        n_layers=2, n_heads=N_HEADS, d_ff=D_MODEL * 2,
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
def bottleneck_model(encoder, decoder):
    """Lightweight inference-only BottleneckAE."""
    return BottleneckAE(encoder=encoder, decoder=decoder)


@pytest.fixture
def trainable_core(encoder, decoder, dummy_repr):
    """_TrainableCore with a semantic projection head."""
    sem_proj = nn.Linear(D_MODEL, H)
    return _TrainableCore(
        encoder=encoder,
        decoder=decoder,
        latent_aug=LatentAugmentation(noise_std=0.1, feature_dropout_p=0.1),
        sem_proj=sem_proj,
        lambda_sem=0.2,
    )


@pytest.fixture
def sample_batch():
    input_ids = torch.randint(1, VOCAB, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
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
# Helpers used by non-fixture tests
# ---------------------------------------------------------------------------

def _make_batch():
    return {
        "input_ids": torch.randint(1, VOCAB, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "decoder_input_ids": torch.randint(1, VOCAB, (B, T)),
        "decoder_attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(1, VOCAB, (B, T)),
    }


def _build_trainable_core(decoder_type="autoregressive"):
    encoder = BottleneckEncoder(
        vocab_size=VOCAB, d_model=D_MODEL,
        n_latent_tokens=N_LATENT,
        n_layers=2, n_heads=N_HEADS, d_ff=D_MODEL * 2,
        max_length=T, pad_token_id=0,
    )
    if decoder_type == "autoregressive":
        dec = AutoRegressiveDecoder(
            vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
            d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
        )
    else:
        dec = ParallelLatentDecoder(
            vocab_size=VOCAB, d_model=D_DEC, n_layers=2, n_heads=N_HEADS,
            d_ff=D_DEC * 2, max_length=T, pad_token_id=0,
        )
    sem_proj = nn.Linear(D_MODEL, H)
    return _TrainableCore(
        encoder=encoder, decoder=dec,
        latent_aug=LatentAugmentation(noise_std=0.05, feature_dropout_p=0.1),
        sem_proj=sem_proj,
        lambda_sem=0.2,
    )


# ---------------------------------------------------------------------------
# Tests: BottleneckEncoder
# ---------------------------------------------------------------------------

class TestBottleneckEncoderShapes:
    def test_output_shape(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        z = encoder(ids, mask)
        assert z.shape == (B, N_LATENT, D_MODEL)

    def test_with_padding(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        mask[:, -10:] = 0
        ids[mask == 0] = 0
        z = encoder(ids, mask)
        assert z.shape == (B, N_LATENT, D_MODEL)

    def test_no_mask(self, encoder):
        ids = torch.randint(1, VOCAB, (B, T))
        z = encoder(ids, attention_mask=None)
        assert z.shape == (B, N_LATENT, D_MODEL)


# ---------------------------------------------------------------------------
# Tests: LatentAugmentation
# ---------------------------------------------------------------------------

class TestLatentAugmentation:
    def test_no_op_defaults(self):
        aug = LatentAugmentation()
        aug.train()
        z = torch.randn(B, 1, D_MODEL)
        assert torch.equal(z, aug(z))

    def test_noise_changes_output(self):
        aug = LatentAugmentation(noise_std=1.0)
        aug.train()
        z = torch.randn(B, 1, D_MODEL)
        assert not torch.equal(z, aug(z))

    def test_eval_mode_is_identity(self):
        aug = LatentAugmentation(noise_std=1.0, feature_dropout_p=0.9)
        aug.eval()
        z = torch.randn(B, 1, D_MODEL)
        assert torch.equal(z, aug(z))

    def test_feature_dropout_zeros_some_dims(self):
        aug = LatentAugmentation(feature_dropout_p=0.5)
        aug.train()
        torch.manual_seed(0)
        z = torch.ones(B, 1, D_MODEL)
        assert (aug(z) == 0).any(), "Expected some zeroed dimensions"


# ---------------------------------------------------------------------------
# Tests: BottleneckAE (inference-only module)
# ---------------------------------------------------------------------------

class TestBottleneckAEShapes:
    def test_encode_latent(self, bottleneck_model):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        z = bottleneck_model.encode(ids, mask)
        assert z.shape == (B, N_LATENT, D_MODEL)

    def test_generate_greedy(self, bottleneck_model):
        ids = torch.randint(1, VOCAB, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        gen = bottleneck_model.generate_greedy(
            ids, mask, bos_token_id=101, eos_token_id=102, max_length=16,
        )
        assert gen.shape[0] == B
        assert gen.shape[1] <= 16
        assert gen[:, 0].eq(101).all()


# ---------------------------------------------------------------------------
# Tests: _TrainableCore forward pass
# ---------------------------------------------------------------------------

class TestTrainableCoreForward:
    def test_output_keys_with_sem(self, trainable_core, sample_batch, dummy_repr):
        sent_emb = dummy_repr.encode(sample_batch["input_ids"], sample_batch["attention_mask"])
        out = trainable_core(**sample_batch, sent_emb=sent_emb)
        assert "loss" in out
        assert "logits" in out
        assert "l_recon" in out
        assert "l_sem" in out

    def test_output_keys_no_sem(self, encoder, decoder):
        core = _TrainableCore(
            encoder=encoder, decoder=decoder,
            latent_aug=LatentAugmentation(), sem_proj=None, lambda_sem=0.0,
        )
        out = core(**_make_batch())
        assert "loss" in out and "l_recon" in out
        assert "l_sem" not in out

    def test_output_shapes(self, trainable_core, sample_batch, dummy_repr):
        sent_emb = dummy_repr.encode(sample_batch["input_ids"], sample_batch["attention_mask"])
        out = trainable_core(**sample_batch, sent_emb=sent_emb)
        assert out["logits"].shape == (B, T, VOCAB)
        assert out["loss"].dim() == 0

    def test_loss_is_finite(self):
        core = _build_trainable_core()
        batch = _make_batch()
        repr_enc = DummyReprEncoder(hidden_size=H)
        sent_emb = repr_enc.encode(batch["input_ids"], batch["attention_mask"])
        out = core(**batch, sent_emb=sent_emb)
        assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"

    def test_loss_is_finite_parallel_decoder(self):
        core = _build_trainable_core(decoder_type="parallel")
        batch = _make_batch()
        repr_enc = DummyReprEncoder(hidden_size=H)
        sent_emb = repr_enc.encode(batch["input_ids"], batch["attention_mask"])
        out = core(**batch, sent_emb=sent_emb)
        assert torch.isfinite(out["loss"]), f"Loss is not finite (parallel): {out['loss']}"


# ---------------------------------------------------------------------------
# Tests: training step
# ---------------------------------------------------------------------------

class TestBottleneckTrainStep:
    def test_backward_runs(self, trainable_core, sample_batch, dummy_repr):
        sent_emb = dummy_repr.encode(sample_batch["input_ids"], sample_batch["attention_mask"])
        out = trainable_core(**sample_batch, sent_emb=sent_emb)
        out["loss"].backward()
        n_grads = sum(
            1 for p in trainable_core.parameters() if p.requires_grad and p.grad is not None
        )
        assert n_grads > 0, "No gradients computed"

    def test_loss_decreases(self):
        core = _build_trainable_core()
        optimizer = torch.optim.Adam(core.parameters(), lr=1e-3)
        repr_enc = DummyReprEncoder(hidden_size=H)
        batch = _make_batch()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            sent_emb = repr_enc.encode(batch["input_ids"], batch["attention_mask"])
            out = core(**batch, sent_emb=sent_emb)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_augmentation_disabled_in_eval(self, trainable_core, sample_batch):
        trainable_core.eval()
        z = trainable_core.encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        z_aug = trainable_core.latent_aug(z)
        assert torch.equal(z, z_aug), "Augmentation should be identity in eval mode"
