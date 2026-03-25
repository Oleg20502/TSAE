"""Bottleneck encoder: compresses text into latent tokens via self-attn + cross-attn transformer."""

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single encoder block
# ---------------------------------------------------------------------------

class BottleneckEncoderBlock(nn.Module):
    """One encoder block: self-attn on text tokens -> cross-attn (latent queries -> text KV) -> FFN.

    The self-attention refines text token representations.
    The cross-attention lets learned latent queries gather information
    from the (refined) text tokens.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention on text tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: latent (query) attends to text tokens (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward (applied to each latent query position)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,
        latent_query: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_tokens:          (B, T, D) text token embeddings.
            latent_query:         (B, L, D) latent query tokens (L = n_latent_tokens).
            text_key_padding_mask: (B, T) True where padded (ignored positions).

        Returns:
            text_tokens: (B, T, D) refined text tokens (after self-attention).
            latent_query: (B, L, D) updated latent query tokens.
        """
        # 1) Self-attention on text tokens
        sa_out, _ = self.self_attn(
            query=text_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask,
        )
        text_tokens = self.norm1(text_tokens + sa_out)

        # 2) Cross-attention: latent query attends to text tokens
        ca_out, _ = self.cross_attn(
            query=latent_query,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_key_padding_mask,
        )
        latent_query = self.norm2(latent_query + ca_out)

        # 3) FFN on the latent query
        latent_query = self.norm3(latent_query + self.ffn(latent_query))

        return text_tokens, latent_query


# ---------------------------------------------------------------------------
# Full bottleneck encoder
# ---------------------------------------------------------------------------

class BottleneckEncoder(nn.Module):
    """Transformer encoder that compresses text into a sequence of latent tokens.

    Has its own token and positional embeddings (independent from any backbone).
    Uses N blocks of self-attention on text + cross-attention for n_latent_tokens
    learned latent queries to gather information from the text.

    Returns the latent sequence of shape ``(B, n_latent_tokens, d_model)``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_latent_tokens: int = 1,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        max_length: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        normalize_latent: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latent_tokens = n_latent_tokens
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        
        # Token + positional embeddings (encoder's own)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Learned latent query tokens (1, n_latent_tokens, d_model)
        self.latent_query = nn.Parameter(torch.randn(1, n_latent_tokens, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [BottleneckEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=not normalize_latent)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name == "latent_query":
                continue  # already initialised
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, T) token ids.
            attention_mask:  (B, T) 1 for real tokens, 0 for padding.

        Returns:
            latent: (B, n_latent_tokens, d_model) compressed latent sequence.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embed text tokens
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        text_tokens = self.tok_emb(input_ids) + self.pos_emb(positions)
        text_tokens = self.emb_dropout(text_tokens)  # (B, T, d_model)

        # Key padding mask for self-attention (True = ignore)
        text_kp_mask = None
        if attention_mask is not None:
            text_kp_mask = attention_mask.eq(0)  # (B, T)

        # Expand learned latent queries for the batch (B, n_latent_tokens, d_model)
        latent_query = self.latent_query.expand(B, -1, -1)

        # Run through encoder blocks
        for block in self.blocks:
            text_tokens, latent_query = block(text_tokens, latent_query, text_kp_mask)

        # Final layer norm
        latent = self.ln_f(latent_query)         # (B, n_latent_tokens, d_model)

        return latent
