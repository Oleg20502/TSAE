"""Concept Model: transformer that predicts the next AE latent in latent space.

Supports two variants:
  - ``ConceptModel``:     custom scratch transformer with block-level causal masking.
  - ``ConceptModelGPT2``: wraps a pretrained HuggingFace causal-LM backbone
                          (e.g. GPT-2) with input/output projections.
                          Uses standard token-level causal masking from the backbone.

Both variants accept latent sequences of shape ``(B, N*n, d_ae)`` and return
predicted latent sequences of the same shape.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import ConceptModelConfig

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean centering)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputed rotary (RoPE) sin/cos tables.

    Stores tables up to ``max_seq_len`` positions. The cache is automatically
    extended if a longer sequence is encountered.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        half = self.head_dim // 2
        theta = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, theta)            # (seq_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)    # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to ``x`` of shape ``(..., seq_len, head_dim)``."""
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len * 2)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: two parallel projections gated by SiLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Block-level causal mask
# ---------------------------------------------------------------------------

def make_block_causal_mask(
    n_chunks: int,
    n_latent: int,
    device: torch.device,
) -> torch.Tensor:
    """Additive attention mask enforcing block-level causality.

    Position ``i`` (in chunk block ``b_i = i // n_latent``) may attend to
    position ``j`` if ``b_i >= b_j`` (i.e., same block or a previous block).

    Returns:
        mask: ``(1, 1, N*n, N*n)`` float tensor — 0 for allowed positions,
              ``-inf`` for masked positions.
    """
    total = n_chunks * n_latent
    block_ids = torch.arange(total, device=device) // n_latent  # (total,)
    # [i, j]: masked when block_ids[i] < block_ids[j]
    mask = torch.where(
        block_ids.unsqueeze(0) <= block_ids.unsqueeze(1),  # (total, total)
        torch.zeros(total, total, device=device),
        torch.full((total, total), float("-inf"), device=device),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N*n, N*n)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope  # shared across all blocks
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_h)


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         ``(B, T, d_model)``
            attn_mask: additive mask ``(1, 1, T, T)`` or compatible broadcastable shape.
                       Passed directly to ``F.scaled_dot_product_attention`` as
                       ``attn_mask``, so 0 = attend, ``-inf`` = masked.
        """
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x))  # (B, H, T, D_h)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q = self.rope(q, T)
        k = self.rope(k, T)

        # Uses Flash Attention automatically on supported hardware/dtypes.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, D_h)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Single transformer block (custom CM)
# ---------------------------------------------------------------------------


class ConceptModelBlock(nn.Module):
    """One pre-norm transformer block: Norm → MHA (RoPE) → Norm → SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, n_heads, rope, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         ``(B, T, d_model)``
            attn_mask: additive mask ``(1, 1, T, T)`` or compatible broadcastable shape.
                       Passed directly to ``F.scaled_dot_product_attention`` as
                       ``attn_mask``, so 0 = attend, ``-inf`` = masked.
        """
        x = x + self.attention(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Custom Concept Model
# ---------------------------------------------------------------------------

class ConceptModel(nn.Module):
    """Scratch transformer for latent-space language modelling.

    Input and output are in AE latent space (``d_ae`` dimensions). When
    ``d_model != d_ae`` linear projections are added. Block-level causal
    masking is used so that all ``n`` tokens within the same latent block
    attend to each other freely, but only to tokens in earlier blocks.
    """

    def __init__(
        self,
        d_ae: int,
        n_latent_tokens: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.d_ae           = d_ae
        self.n_latent_tokens = n_latent_tokens

        # self.input_norm  = RMSNorm(d_ae)  # encoder uses LayerNorm
        self.input_norm = nn.LayerNorm(d_ae)
        self.input_proj  = nn.Linear(d_ae, d_model, bias=False) if d_ae != d_model else nn.Identity()

        rope = RotaryEmbedding(d_model // n_heads, base=rope_base, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList([
            ConceptModelBlock(d_model, n_heads, d_ff, rope, dropout)
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, d_ae, bias=False) if d_ae != d_model else nn.Identity()
        # self.output_norm = RMSNorm(d_ae)  # encoder uses LayerNorm
        self.output_norm = nn.LayerNorm(d_ae)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N*n, d_ae)`` — flattened latent sequence.

        Returns:
            ``(B, N*n, d_ae)`` — predicted next-latent sequence.
        """
        B, T, _ = x.shape
        n = self.n_latent_tokens
        n_chunks = T // n

        x = self.input_norm(x)
        x = self.input_proj(x)

        mask = make_block_causal_mask(n_chunks, n, x.device)  # (1, 1, T, T)

        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


# ---------------------------------------------------------------------------
# GPT-2 (or any HF causal-LM) wrapper
# ---------------------------------------------------------------------------

class CustomGPT2Model(GPT2Model):
    """
    Custom GPT-2 model adapted to use block-level causal masking.
    Intended to use as backbone for the Concept Model.    
    """
    
    def forward(
        self,
        inputs_embeds, # (B, N*n, d_ae)
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):

        seq_len = inputs_embeds.shape[-2]

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(
            seq_len, device=inputs_embeds.device
        ) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + (seq_len,) + (hidden_states.size(-1),)

        for block in self.h:
            hidden_states = block(
                hidden_states,
                past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.post_init()
    
    def forwrd(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)


class ConceptModelGPT2(nn.Module):
    """Wraps a pretrained HuggingFace causal language model as a Concept Model.

    The pretrained backbone is used as-is and fully fine-tuned. Its token
    embedding is replaced by a linear projection from ``d_ae`` → ``d_backbone``.
    The LM head is replaced by a linear projection back to ``d_ae``.

    Because HuggingFace GPT-2 (and most causal-LM implementations) apply a
    built-in triangular causal mask internally, this variant uses standard
    **token-level** causal masking rather than block-level latent masking.
    """

    def __init__(
        self,
        d_ae: int,
        pretrained_name: str = "gpt2",
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        from transformers import AutoConfig

        self.d_ae = d_ae

        backbone_cfg = AutoConfig.from_pretrained(pretrained_name)
        d_backbone = backbone_cfg.hidden_size

        model = CustomGPT2LMHeadModel(backbone_cfg)

        pretrained = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        missing, unexpected = model.load_state_dict(pretrained.state_dict(), strict=False)

        if max_seq_len is not None and max_seq_len > backbone_cfg.max_position_embeddings:
            backbone_cfg.max_position_embeddings = max_seq_len

        self.input_norm  = nn.LayerNorm(d_ae) # encoder uses LayerNorm
        self.input_proj  = nn.Linear(d_ae, d_backbone, bias=False)

        self.backbone    = model.transformer

        self.output_proj = nn.Linear(d_backbone, d_ae, bias=False)
        self.output_norm = nn.LayerNorm(d_ae) # encoder uses LayerNorm

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, N*n, d_ae)`` — flattened latent sequence.

        Returns:
            ``(B, N*n, d_ae)`` — predicted next-latent sequence.
        """

        B, T, _ = x.shape
        n = self.n_latent_tokens
        n_chunks = T // n

        x = self.input_norm(x)
        embeds = self.input_proj(x)           # (B, T, d_backbone)

        attention_mask = make_block_causal_mask(n_chunks, n, x.device)  # (1, 1, T, T)

        out = self.backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state  # (B, T, d_backbone)
        out = self.output_proj(out)           # (B, T, d_ae)
        out = self.output_norm(out)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_concept_model(cfg: ConceptModelConfig, d_ae: int, n_latent_tokens: int) -> nn.Module:
    """Instantiate the correct Concept Model variant from config.

    Args:
        cfg:             ConceptModelConfig.
        d_ae:            Latent dimension of the frozen AE (``d_model`` in AE config).
        n_latent_tokens: Number of latent tokens per chunk in the AE.

    Returns:
        An ``nn.Module`` with signature ``forward(x: Tensor) -> Tensor``
        where ``x`` and the output have shape ``(B, N*n, d_ae)``.
    """
    if cfg.cm_type == "custom":
        return ConceptModel(
            d_ae=d_ae,
            n_latent_tokens=n_latent_tokens,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.ff_dim,
            rope_base=cfg.rope_base,
            dropout=cfg.dropout,
        )
    else:
        return ConceptModelGPT2(
            d_ae=d_ae,
            pretrained_name=cfg.cm_type,
        )


def load_concept_weights(
    checkpoint_path: str,
    concept_model: nn.Module,
    device: str = "cpu",
) -> None:
    """Load Concept Model weights from safetensors or PyTorch ``.pt`` in-place.

    Checkpoints written by ``ConceptTrainer`` use keys prefixed with
    ``concept_model.``; those prefixes are stripped before
    ``load_state_dict``. If no such keys exist, the full dict is loaded
    (useful for raw state exports).

    Does not restore optimizer / scheduler — use ``resume_from_checkpoint``
    for full training state.
    """
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(checkpoint_path, device=device)
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    prefix = "concept_model."
    if any(k.startswith(prefix) for k in state):
        inner = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    else:
        inner = dict(state)

    missing, unexpected = concept_model.load_state_dict(inner, strict=False)
    if unexpected:
        print("Warning: unexpected keys in checkpoint:", unexpected)
    if missing:
        print("Warning: missing keys (not loaded):", missing)
