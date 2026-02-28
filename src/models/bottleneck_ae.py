"""Bottleneck autoencoder: latent token sequence with independent encoder and decoder transformers."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.backbones.base_repr import BaseTextReprEncoder
from src.models.bottleneck_encoder import BottleneckEncoder
from src.models.latent_augmentation import LatentAugmentation
from src.models.decoder import AutoRegressiveDecoder
from src.losses.reconstruction import reconstruction_loss
from src.losses.semantic import semantic_consistency_loss


class BottleneckAE(nn.Module):
    """Bottleneck Autoencoder for Text.

    Composes:
        encoder           – standalone Transformer that compresses text to
                            n_latent_tokens (B, n_latent_tokens, d_latent)
        repr_encoder      – frozen SimCSE backbone used *only* as a semantic
                            target for the semantic loss
        latent_aug        – optional noise / feature-dropout augmentation
                            applied to the latent during training
        decoder           – Transformer decoder that reconstructs text from
                            the (augmented) latent via cross-attention
        sem_proj          – projects mean-pooled latent to the repr_encoder's
                            sentence-embedding space for the semantic loss

    The ``forward`` method returns a dict containing a ``"loss"`` key so the
    model can be used directly with ``transformers.Trainer``.
    """

    def __init__(
        self,
        encoder: BottleneckEncoder,
        decoder: AutoRegressiveDecoder,
        repr_encoder: BaseTextReprEncoder | None = None,
        latent_aug: LatentAugmentation | None = None,
        lambda_sem: float = 0.2,
        freeze_repr: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.repr_encoder = repr_encoder
        self.latent_aug = latent_aug or LatentAugmentation()  # no-op defaults
        self.lambda_sem = lambda_sem

        d_latent = encoder.d_latent
        d_dec = decoder.d_model

        # Project latent to decoder dim if they differ
        if d_latent != d_dec:
            self.latent_to_dec = nn.Linear(d_latent, d_dec)
        else:
            self.latent_to_dec = nn.Identity()

        # Semantic projection: latent -> repr_encoder sentence dim
        if self.repr_encoder:
            self.sem_proj = nn.Linear(d_latent, repr_encoder.sent_dim)

        # Freeze the representation backbone (used only for the semantic target)
        if freeze_repr and self.repr_encoder:
            for p in self.repr_encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the representaion embedder semantic target.

        Returns:
            sent_emb: (B, D_s) sentence embedding (detached target).
        """
        sent_emb = None
        if self.repr_encoder:
            with torch.no_grad():
                sent_emb, _ = self.repr_encoder.encode(input_ids, attention_mask)

        return sent_emb

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the bottleneck latent and the representaion embedder semantic target.

        Returns:
            z:        (B, n_latent_tokens, d_latent) encoder latent sequence.
        """
        # Encoder produces latent from raw token ids
        z = self.encoder(input_ids, attention_mask)  # (B, n_latent_tokens, d_latent)

        return z

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
        """Full forward pass: encode -> augment -> decode -> compute loss.

        Returns a dict with ``"loss"`` as the first key (HF Trainer requirement).
        """
        # Encode
        z = self.encode(input_ids, attention_mask)
        sent_emb = self.embed(input_ids, attention_mask)

        # Latent augmentation (only during training)
        z_aug = self.latent_aug(z)

        # Project latent to decoder dim
        z_dec = self.latent_to_dec(z_aug)  # (B, n_latent_tokens, d_dec)

        # Decode
        logits, _dec_hidden = self.decoder(
            latent_tokens=z_dec,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # ----- Losses -----
        # Reconstruction loss (cross-entropy)
        l_recon = reconstruction_loss(logits, labels)

        # Semantic loss (cosine similarity between projected latent and SimCSE embedding)
        if self.repr_encoder:
            z_pooled = z.mean(dim=1)  # (B, n_latent_tokens, d_latent) -> (B, d_latent)
            z_sem = self.sem_proj(z_pooled)  # (B, D_s)
            l_sem = semantic_consistency_loss(z_sem, sent_emb)
            loss = l_recon + self.lambda_sem * l_sem

            return {
                "loss": loss,
                "logits": logits,
                "l_recon": l_recon.detach(),
                "l_sem": l_sem.detach(),
                "sent_emb": sent_emb.detach(),
                "z": z.detach(),
            }
        else:
            loss = l_recon 
            return {
                "loss": loss,
                "logits": logits,
                "l_recon": l_recon.detach(),
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
        """Greedy autoregressive decoding from the bottleneck autoencoder.

        Args:
            input_ids:      (B, T) encoder input.
            attention_mask:  (B, T).
            bos_token_id:   beginning-of-sequence token id.
            eos_token_id:   end-of-sequence token id.
            max_length:     maximum decoding steps.

        Returns:
            generated: (B, T') generated token ids.
        """
        z = self.encode(input_ids, attention_mask)
        z_dec = self.latent_to_dec(z)  # (B, 1, d_dec)

        B = z.size(0)
        device = z.device

        # Start with BOS
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits, _ = self.decoder(
                latent_tokens=z_dec,
                decoder_input_ids=generated,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token.squeeze(-1) == eos_token_id).all():
                break

        return generated
