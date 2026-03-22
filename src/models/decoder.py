"""Transformer decoder with causal self-attention and cross-attention to latent tokens."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """One transformer decoder block: causal self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to latent tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
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
        x: torch.Tensor,
        latent_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, D)  decoder token embeddings.
            latent_tokens:  (B, L, D)  latent tokens (z_sem + z_det).
            causal_mask:    (T, T)     additive causal mask.
            self_key_padding_mask: (B, T) True where padded.
        """
        # 1) Causal self-attention
        sa_out, _ = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=causal_mask,
            key_padding_mask=self_key_padding_mask,
        )
        x = self.norm1(x + sa_out)

        # 2) Cross-attention to latent tokens
        ca_out, _ = self.cross_attn(query=x, key=latent_tokens, value=latent_tokens)
        x = self.norm2(x + ca_out)

        # 3) FFN
        x = self.norm3(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class AutoRegressiveDecoder(nn.Module):
    """Autoregressive Transformer decoder conditioned on latent tokens via cross-attention.

    Returns logits over the vocabulary and a pooled hidden state for the
    semantic consistency head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        max_length: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # LM head (weight-tied to token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask (T, T): True = mask out (same type as key_padding_mask for MHA)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        latent_tokens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_tokens:          (B, L, D_lat) concatenated [z_sem ; z_det].
            decoder_input_ids:      (B, T) token ids for teacher-forced decoding.
            decoder_attention_mask:  (B, T) 1 for real, 0 for pad.

        Returns:
            logits:     (B, T, V)
            dec_hidden: (B, D) mean-pooled decoder hidden state (for sem head).
        """
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device

        # Embed decoder tokens
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(decoder_input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Causal mask
        causal_mask = self._make_causal_mask(T, device)

        # Key padding mask for self-attention (True = ignore)
        self_kp_mask = None
        if decoder_attention_mask is not None:
            self_kp_mask = decoder_attention_mask.eq(0)

        # Run through blocks
        for block in self.blocks:
            x = block(x, latent_tokens, causal_mask, self_kp_mask)

        x = self.ln_f(x)

        # LM logits
        logits = self.lm_head(x)  # (B, T, V)

        return logits


class ParallelLatentDecoder(nn.Module):
    """Non-autoregressive decoder that reconstructs all tokens from the latent in one pass.

    Positions are parameterised only by their
    positional embeddings and attend to the latent tokens; no token inputs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        max_length: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        # Positional embeddings for output queries (no token embeddings)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # Reuse DecoderBlock but without causal masking (we pass an all-false mask)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        causal_mask = torch.zeros(self.max_length, self.max_length, dtype=torch.bool) # No causality
        self.register_buffer("causal_mask", causal_mask)

        self.ln_f = nn.LayerNorm(d_model)

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_tokens:          (B, L, D_lat) encoder latent sequence.
            decoder_input_ids:      (B, T) token ids, not used here.
            decoder_attention_mask: (B, T) 1 for real, 0 for pad, not used here.

        Returns:
            logits:     (B, T, V)
            dec_hidden: (B, D) mean-pooled decoder hidden state.
        """
        B = latent_tokens.shape[0]
        device = latent_tokens.device

        # Output queries: purely positional
        positions = torch.arange(self.max_length, device=device).unsqueeze(0)  # (1, max_length)
        x = self.pos_emb(positions)
        x = self.emb_dropout(x)
        x = x.expand(B, self.max_length, -1)  # (B, max_length, D)

        for block in self.blocks:
            x = block(x, latent_tokens, self.causal_mask)

        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, V)

        return logits
