"""Detail encoder: compresses token-level hidden states into M detail latent tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class DetailEncoder(nn.Module):
    """Learned-query cross-attention that compresses variable-length token
    hidden states into a fixed set of M detail latent tokens.

    Architecture (per layer):
        queries  = learned latent tokens  (M, d_det)
        keys/values = projected token hidden states  (T, d_det)
        output   = cross-attention result  (M, d_det)
    """

    def __init__(
        self,
        input_dim: int,
        d_det: int = 256,
        n_tokens: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_det = d_det

        # Projection from backbone hidden size to detail dim
        self.input_proj = nn.Linear(input_dim, d_det)

        # Learned query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_det) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=d_det,
                            num_heads=n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(d_det),
                        "ffn": nn.Sequential(
                            nn.Linear(d_det, d_det * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_det * 4, d_det),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(d_det),
                    }
                )
            )

    def forward(
        self,
        tok_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tok_emb:        (B, T, H) token-level hidden states from backbone.
            attention_mask:  (B, T) with 1 for real tokens, 0 for padding.

        Returns:
            detail_latents: (B, M, d_det)
        """
        B = tok_emb.size(0)

        # Project token embeddings to detail dimension
        kv = self.input_proj(tok_emb)  # (B, T, d_det)

        # Build key_padding_mask: True where padded (nn.MHA convention)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.eq(0)  # (B, T), True = ignore

        # Expand learned queries for the batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, M, d_det)

        for layer in self.layers:
            # Cross-attention: queries attend to token KV
            attn_out, _ = layer["cross_attn"](
                query=queries,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
            )
            queries = layer["norm1"](queries + attn_out)
            queries = layer["norm2"](queries + layer["ffn"](queries))

        return queries  # (B, M, d_det)
