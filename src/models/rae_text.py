"""RAE-Text: deterministic latent autoencoder for text (Stage A, first family)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.backbones.base_repr import BaseTextReprEncoder
from src.models.detail_encoder import DetailEncoder
from src.models.decoder import AutoRegressiveDecoder
from src.losses.reconstruction import reconstruction_loss
from src.losses.semantic import semantic_consistency_loss


class RAEText(nn.Module):
    """Representation-Autoencoder for Text.

    Composes:
        repr_encoder  – frozen (or partially frozen) SimCSE / TSDAE backbone
        sem_proj      – projects sentence embedding to d_sem
        detail_encoder – compresses token states to M detail latent tokens
        decoder       – Transformer decoder cross-attending to latent tokens
        sem_head      – maps pooled decoder state back to semantic space

    The ``forward`` method returns a dict containing a ``"loss"`` key so the
    model can be used directly with ``transformers.Trainer``.
    """

    def __init__(
        self,
        repr_encoder: BaseTextReprEncoder,
        detail_encoder: DetailEncoder,
        decoder: AutoRegressiveDecoder,
        d_sem: int = 256,
        lambda_sem: float = 0.2,
        freeze_repr: bool = True,
    ):
        super().__init__()
        self.repr_encoder = repr_encoder
        self.detail_encoder = detail_encoder
        self.decoder = decoder
        self.lambda_sem = lambda_sem

        # Semantic projection: backbone sent_dim -> d_sem
        self.sem_proj = nn.Linear(repr_encoder.sent_dim, d_sem)

        # If detail dim != decoder dim, add a projection on the latent side
        d_det = detail_encoder.d_det
        d_dec = decoder.d_model
        if d_sem != d_dec or d_det != d_dec:
            # We project the concatenated latent sequence to decoder dim
            # sem tokens are d_sem, detail tokens are d_det.
            # We handle them separately then concat.
            self._need_latent_proj = True
            self.sem_to_dec = nn.Linear(d_sem, d_dec) if d_sem != d_dec else nn.Identity()
            self.det_to_dec = nn.Linear(d_det, d_dec) if d_det != d_dec else nn.Identity()
        else:
            self._need_latent_proj = False
            self.sem_to_dec = nn.Identity()
            self.det_to_dec = nn.Identity()

        # Semantic head: decoder pooled state -> repr_encoder.sent_dim
        self.sem_head = nn.Linear(d_dec, repr_encoder.sent_dim)

        # Freeze backbone
        if freeze_repr:
            for p in self.repr_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce latent tokens from raw token ids.

        Returns:
            z:        (B, 1+M, d_dec) concatenated latent sequence.
            sent_emb: (B, D_s) original sentence embedding (for sem loss).
            z_det:    (B, M, d_det) detail latents (before projection).
        """
        sent_emb, tok_emb = self.repr_encoder.encode(input_ids, attention_mask)

        z_sem = self.sem_proj(sent_emb).unsqueeze(1)                  # (B, 1, d_sem)
        z_det = self.detail_encoder(tok_emb, attention_mask)          # (B, M, d_det)

        # Project to decoder dim and concatenate
        z_sem_dec = self.sem_to_dec(z_sem)                            # (B, 1, d_dec)
        z_det_dec = self.det_to_dec(z_det)                            # (B, M, d_dec)
        z = torch.cat([z_sem_dec, z_det_dec], dim=1)                  # (B, 1+M, d_dec)

        return z, sent_emb, z_det

    # ------------------------------------------------------------------
    # Forward (HF Trainer compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> decode -> compute loss.

        Returns a dict with ``"loss"`` as the first key (HF Trainer requirement).
        """
        # Encode
        z, sent_emb, z_det = self.encode_latents(input_ids, attention_mask)

        # Decode
        logits, dec_hidden = self.decoder(
            latent_tokens=z,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # Losses
        l_recon = reconstruction_loss(logits, labels)

        pred_sem = self.sem_head(dec_hidden)            # (B, D_s)
        l_sem = semantic_consistency_loss(pred_sem, sent_emb)

        loss = l_recon + self.lambda_sem * l_sem

        return {
            "loss": loss,
            "logits": logits,
            "l_recon": l_recon.detach(),
            "l_sem": l_sem.detach(),
            "sent_emb": sent_emb.detach(),
            "z": z.detach(),
        }

    # ------------------------------------------------------------------
    # Greedy generation (for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding from the autoencoder.

        Args:
            input_ids:      (B, T) encoder input.
            attention_mask:  (B, T).
            bos_token_id:   beginning-of-sequence token id.
            eos_token_id:   end-of-sequence token id.
            max_length:     maximum decoding steps.

        Returns:
            generated: (B, T') generated token ids.
        """
        z, _, _ = self.encode_latents(input_ids, attention_mask)
        B = z.size(0)
        device = z.device

        # Start with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits, _ = self.decoder(
                latent_tokens=z,
                decoder_input_ids=generated,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token.squeeze(-1) == eos_token_id).all():
                break

        return generated
