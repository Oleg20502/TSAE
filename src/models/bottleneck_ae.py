"""Bottleneck autoencoder: lightweight inference module and training component builders."""

from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.backbones.repr_embedder import BaseTextReprEncoder, CLSReprEncoder, STReprEncoder
from src.models.decoder import AutoRegressiveDecoder, ParallelLatentDecoder
from src.models.encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.utils.config import BottleneckModelConfig, BottleneckExperimentConfig, load_config_from_paths
from src.losses.reconstruction import reconstruction_loss
from src.losses.semantic import semantic_consistency_loss


class BottleneckAE(nn.Module):
    """Autoencoder module: encoder + decoder + latent_aug + sem_proj.

    repr_encoder is intentionally absent: it is frozen, contributes no
    gradients, and must not be saved to checkpoints.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sem_proj: Optional[nn.Module] = None,
        latent_aug: Optional[nn.Module] = None,
        lambda_sem: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_aug = latent_aug
        self.sem_proj = sem_proj
        self.lambda_sem = lambda_sem or 0.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        sent_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode → augment → decode → compute loss.

        Args:
            sent_emb: pre-computed sentence embedding from repr_encoder (detached).
                      When None the semantic loss term is skipped.
        """
        z = self.encoder(input_ids, attention_mask)       # (B, L, D)
        z_aug = self.latent_aug(z)                        # no-op during eval

        logits = self.decoder(
            latent_tokens=z_aug,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        l_recon = reconstruction_loss(logits, labels)

        if sent_emb is not None and self.sem_proj is not None:
            z_sem = self.sem_proj(z.mean(dim=1))          # (B, D_s)
            l_sem = semantic_consistency_loss(z_sem, sent_emb)
            loss = l_recon + self.lambda_sem * l_sem
            return {
                "loss": loss,
                "logits": logits,
                "l_recon": l_recon.detach(),
                "l_sem": l_sem.detach(),
            }
        else:
            return {
                "loss": l_recon,
                "logits": logits,
                "l_recon": l_recon.detach(),
            }

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to latent z of shape (B, n_latent_tokens, d_model)."""
        return self.encoder(input_ids, attention_mask)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding from the latent.

        Args:
            input_ids:       (B, T) encoder input token ids.
            attention_mask:  (B, T) encoder attention mask.
            bos_token_id:    beginning-of-sequence token id.
            eos_token_id:    end-of-sequence token id.
            max_length:      maximum number of decoding steps.

        Returns:
            generated: (B, T') generated token ids.
        """
        z = self.encode(input_ids, attention_mask)
        B = z.size(0)
        device = z.device

        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits = self.decoder(latent_tokens=z, decoder_input_ids=generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token.squeeze(-1) == eos_token_id).all():
                break

        return generated


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_repr_encoder(
    backbone_name: str,
    use_legacy_repr: bool = False,
) -> BaseTextReprEncoder:
        if use_legacy_repr:
            return CLSReprEncoder(model_name=backbone_name)
        else:
            return STReprEncoder(model_name=backbone_name)


def build_sem_proj(
    d_model: int,
    sent_dim: int,
) -> nn.Linear:
    return nn.Linear(d_model, sent_dim)

def build_encoder(
    cfg: BottleneckModelConfig,
    vocab_size: int,
    pad_token_id: int,
) -> BottleneckEncoder:
    return BottleneckEncoder(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_latent_tokens=cfg.n_latent_tokens,
        n_layers=cfg.encoder_layers,
        n_heads=cfg.encoder_heads,
        d_ff=cfg.encoder_ff_dim,
        max_length=cfg.max_length,
        dropout=cfg.encoder_dropout,
        pad_token_id=pad_token_id,
        normalize_latent=cfg.normalize_latent,
    )


def build_decoder(
    cfg: BottleneckModelConfig,
    vocab_size: int,
    pad_token_id: int,
) -> nn.Module:
    if cfg.decoder_type == "autoregressive":
        return AutoRegressiveDecoder(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.decoder_layers,
            n_heads=cfg.decoder_heads,
            d_ff=cfg.decoder_ff_dim,
            max_length=cfg.max_length,
            dropout=cfg.decoder_dropout,
            pad_token_id=pad_token_id,
        )
    elif cfg.decoder_type == "parallel":
        return ParallelLatentDecoder(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.decoder_layers,
            n_heads=cfg.decoder_heads,
            d_ff=cfg.decoder_ff_dim,
            max_length=cfg.max_length,
            dropout=cfg.decoder_dropout,
            pad_token_id=pad_token_id,
        )
    else:
        raise ValueError(f"Unknown decoder_type: {cfg.decoder_type}")


def build_ae_components(
    cfg: BottleneckExperimentConfig,
    vocab_size: int,
    pad_token_id: int,
) -> Tuple[BottleneckEncoder, nn.Module, LatentAugmentation, float]:
    """Build all components required for training.

    Returns:
        (encoder, decoder, latent_aug, lambda_sem)
    """
    mc = cfg.model
    encoder = build_encoder(mc, vocab_size, pad_token_id)
    decoder = build_decoder(mc, vocab_size, pad_token_id)

    latent_aug = LatentAugmentation(
        noise_std=mc.noise_std,
        feature_dropout_p=mc.feature_dropout_p,
        normalize_latent=mc.normalize_latent,
        sigma_type=mc.sigma_type,
    )

    return encoder, decoder, latent_aug, mc.lambda_sem


# ---------------------------------------------------------------------------
# Inference loader
# ---------------------------------------------------------------------------

def load_ae_weights(
    checkpoint_path: str,
    autoencoder: BottleneckAE,
    device: str = "cpu",
) -> None:
    """Load encoder+decoder weights from a safetensors (or .pt) checkpoint in-place.

    Use this for warm-starting: the models are already built with the correct
    architecture; this function just populates their parameters from a previous
    checkpoint.  Optimizer / scheduler state is NOT restored.
    """
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(checkpoint_path, device=device)
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state = {k: v for k, v in state.items() if not k.startswith("repr_encoder")}

    missing, unexpected = autoencoder.load_state_dict(state, strict=False)
    if unexpected:
        print("Warning: unexpected keys in checkpoint:", unexpected)
    if (
        missing
        and hasattr(autoencoder.decoder, "lm_head")
        and hasattr(autoencoder.decoder, "tok_emb")
        and "decoder.lm_head.weight" in missing
    ):
        autoencoder.decoder.lm_head.weight = autoencoder.decoder.tok_emb.weight
        missing = [k for k in missing if k != "decoder.lm_head.weight"]
    if missing:
        print("Warning: missing keys (not loaded):", missing)


def load_bottleneck_model(
    config_paths: List[str],
    checkpoint_path: str,
    device: str,
):
    """Build and load BottleneckAE model from config and checkpoint.

    repr_encoder, latent_aug, and sem_proj are not loaded — they are
    training-only components that are not stored in checkpoints.
    """
    cfg = load_config_from_paths(config_paths)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or 0

    encoder = build_encoder(cfg.model, vocab_size, pad_token_id)
    decoder = build_decoder(cfg.model, vocab_size, pad_token_id)
    # FIXME: the output should be sent_dim size. But no sent_dim in the config
    sem_proj = build_sem_proj(encoder.d_model, encoder.d_model)

    model = BottleneckAE(
        encoder=encoder,
        decoder=decoder,
        sem_proj=sem_proj,
        latent_aug=None,
        lambda_sem=cfg.model.lambda_sem
    )

    load_ae_weights(checkpoint_path, model, device=device)

    return model, tokenizer, cfg