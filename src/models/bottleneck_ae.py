"""Bottleneck autoencoder: lightweight inference module and training component builders."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.backbones.repr_embedder import BaseTextReprEncoder, CLSReprEncoder, STReprEncoder
from src.models.decoder import AutoRegressiveDecoder, ParallelLatentDecoder
from src.models.encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.utils.config import BottleneckExperimentConfig, load_config_from_paths


class BottleneckAE(nn.Module):
    """Inference-only bottleneck autoencoder.

    Holds only encoder and decoder — the components needed at inference time.
    Training-time components (repr_encoder, latent_aug, sem_proj) are the
    responsibility of BottleneckTrainer and are never stored here.
    """

    def __init__(self, encoder: BottleneckEncoder, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode text to bottleneck latent z of shape (B, n_latent_tokens, d_model)."""
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
        """Greedy autoregressive decoding from the bottleneck latent.

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

def build_ae_components(
    cfg: BottleneckExperimentConfig,
    vocab_size: int,
    pad_token_id: int,
    build_repr_encoder: bool = True,
    use_legacy_repr: bool = False,
) -> Tuple[BottleneckEncoder, nn.Module, Optional[BaseTextReprEncoder], LatentAugmentation, float]:
    """Build all components required for training.

    Returns:
        (encoder, decoder, repr_encoder, latent_aug, lambda_sem)
        repr_encoder is None when build_repr_encoder=False.
    """
    mc = cfg.model

    repr_encoder: Optional[BaseTextReprEncoder] = None
    if build_repr_encoder:
        if use_legacy_repr:
            repr_encoder = CLSReprEncoder(model_name=mc.backbone_name)
        else:
            repr_encoder = STReprEncoder(model_name=mc.backbone_name)

    encoder = BottleneckEncoder(
        vocab_size=vocab_size,
        d_model=mc.d_model,
        n_latent_tokens=mc.n_latent_tokens,
        n_layers=mc.encoder_layers,
        n_heads=mc.encoder_heads,
        d_ff=mc.encoder_ff_dim,
        max_length=mc.max_length,
        dropout=mc.encoder_dropout,
        pad_token_id=pad_token_id,
        normalize_latent=mc.normalize_latent,
    )

    if mc.decoder_type == "autoregressive":
        decoder: nn.Module = AutoRegressiveDecoder(
            vocab_size=vocab_size,
            d_model=mc.d_model,
            n_layers=mc.decoder_layers,
            n_heads=mc.decoder_heads,
            d_ff=mc.decoder_ff_dim,
            max_length=mc.max_length,
            dropout=mc.decoder_dropout,
            pad_token_id=pad_token_id,
        )
    elif mc.decoder_type == "parallel":
        decoder = ParallelLatentDecoder(
            vocab_size=vocab_size,
            d_model=mc.d_model,
            n_layers=mc.decoder_layers,
            n_heads=mc.decoder_heads,
            d_ff=mc.decoder_ff_dim,
            max_length=mc.max_length,
            dropout=mc.decoder_dropout,
            pad_token_id=pad_token_id,
        )
    else:
        raise ValueError(f"Unknown decoder_type: {mc.decoder_type}")

    latent_aug = LatentAugmentation(
        noise_std=mc.noise_std,
        feature_dropout_p=mc.feature_dropout_p,
        normalize_latent=mc.normalize_latent,
        sigma_type=mc.sigma_type,
    )

    return encoder, decoder, repr_encoder, latent_aug, mc.lambda_sem


# ---------------------------------------------------------------------------
# Inference loader
# ---------------------------------------------------------------------------

def load_bottleneck_model(
    config_paths: List[str],
    checkpoint_path: str,
    device: str,
    use_legacy_repr: bool = False,
):
    """Rebuild a lightweight BottleneckAE (encoder + decoder) from config and checkpoint.

    repr_encoder, latent_aug, and sem_proj are not loaded — they are
    training-only components that are not stored in checkpoints.
    """
    cfg = load_config_from_paths(config_paths)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id or 0

    encoder, decoder, _, _, _ = build_ae_components(
        cfg, vocab_size, pad_token_id, build_repr_encoder=False
    )
    model = BottleneckAE(encoder, decoder)

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(checkpoint_path, device=device)
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Only load encoder/decoder keys; sem_proj and any other training keys are ignored
    enc_dec_state = {
        k: v for k, v in state.items() if k.startswith(("encoder.", "decoder."))
    }
    missing, unexpected = model.load_state_dict(enc_dec_state, strict=False)
    if unexpected:
        print("Warning: unexpected keys in checkpoint:", unexpected)
    # Re-tie lm_head weight when checkpoint omitted the duplicate key (safetensors)
    if (
        missing
        and hasattr(model.decoder, "lm_head")
        and hasattr(model.decoder, "tok_emb")
        and "decoder.lm_head.weight" in missing
    ):
        model.decoder.lm_head.weight = model.decoder.tok_emb.weight
        missing = [k for k in missing if k != "decoder.lm_head.weight"]
    if missing:
        print("Warning: missing keys (not loaded):", missing)

    return model, tokenizer, cfg
