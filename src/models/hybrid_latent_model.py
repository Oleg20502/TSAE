"""Hybrid latent reasoning: GPT-2 backbone over mixed token + AE latent positions (block-causal)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from src.models.concept_model import make_variable_block_causal_mask


class HybridLatentReasoningGPT2(nn.Module):
    """Pretrained GPT-2 with token embeddings for text and projections for AE latents.

    Sequence layout (per sample, before right-padding to batch max):

    ``[prompt tokens | trigger | learned_start (n) | cot_1 (n) | ... | cot_K (n)
       | BOS | answer... | EOS]``

    Block attention: each token is its own block; each latent chunk of ``n`` slots is one block.
    """

    def __init__(
        self,
        d_ae: int,
        n_latent_tokens: int,
        pretrained_name: str = "openai-community/gpt2",
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.d_ae = d_ae
        self.n_latent_tokens = n_latent_tokens

        self.lm = GPT2LMHeadModel.from_pretrained(pretrained_name)
        cfg = self.lm.config
        self.d_model = cfg.n_embd
        if max_seq_len is not None and max_seq_len > cfg.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds GPT-2 position embeddings "
                f"({cfg.max_position_embeddings}); extend wpe or use a larger model."
            )

        self.z_start = nn.Parameter(torch.zeros(n_latent_tokens, d_ae))
        nn.init.normal_(self.z_start, std=0.02)

        self.latent_input_norm = nn.LayerNorm(d_ae)
        self.latent_input_proj = nn.Linear(d_ae, self.d_model, bias=False)
        self.latent_out_proj = nn.Linear(self.d_model, d_ae, bias=False)
        self.latent_out_norm = nn.LayerNorm(d_ae)
        nn.init.xavier_uniform_(self.latent_input_proj.weight)
        nn.init.xavier_uniform_(self.latent_out_proj.weight)

        self.register_buffer("_z_end", torch.zeros(1, n_latent_tokens, d_ae), persistent=False)

    @property
    def transformer(self):
        return self.lm.transformer

    @property
    def lm_head(self):
        return self.lm.lm_head

    def set_end_thinking_latent(self, z: torch.Tensor) -> None:
        """Cache AE encoding of the end-of-thinking phrase, shape ``(1, n, d_ae)``."""
        self._z_end = z.detach().to(dtype=self._z_end.dtype, device=self._z_end.device)

    def get_end_thinking_latent(self) -> torch.Tensor:
        return self._z_end

    def build_block_ids(
        self,
        P: int,
        n_trig: int,
        n_cot_steps: int,
        A: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Block id per position for one row (``(T,)``), monotonic non-decreasing."""
        n = self.n_latent_tokens
        ids: List[int] = []
        bid = 0
        for _ in range(P):
            ids.append(bid)
            bid += 1
        for _ in range(n_trig):
            ids.append(bid)
            bid += 1
        for _ in range(1 + n_cot_steps):
            for _i in range(n):
                ids.append(bid)
            bid += 1
        for _ in range(A):
            ids.append(bid)
            bid += 1
        return torch.tensor(ids, device=device, dtype=torch.long)

    def build_inputs_embeds_and_masks(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        trigger_token_ids: torch.Tensor,
        cot_latents: torch.Tensor,
        cot_valid: torch.Tensor,
        answer_token_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Assemble flat ``inputs_embeds``, ``block_ids``, ``seq_attention_mask``, and index metadata.

        Args:
            prompt_token_ids: ``(B, P_max)``
            prompt_attention_mask: ``(B, P_max)`` 1 = real
            trigger_token_ids: ``(B, L_trig)``
            cot_latents: ``(B, K_max, n, d_ae)`` — padded; invalid steps may be zero
            cot_valid: ``(B, K_max)`` bool
            answer_token_ids: ``(B, A_max)`` includes BOS + body + EOS
            answer_attention_mask: ``(B, A_max)``
        """
        B, P_max = prompt_token_ids.shape
        L_trig = trigger_token_ids.size(1)
        K_max = cot_latents.size(1)
        n = self.n_latent_tokens
        A_max = answer_token_ids.size(1)
        device = prompt_token_ids.device
        dtype = self.lm.transformer.wte.weight.dtype

        lat_span = (1 + K_max) * n
        T = P_max + L_trig + lat_span + A_max

        gpt_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        is_latent = torch.zeros(B, T, dtype=torch.bool, device=device)
        latent_vals = torch.zeros(B, T, self.d_ae, device=device, dtype=cot_latents.dtype)

        meta: Dict[str, Any] = {
            "P_max": P_max,
            "L_trig": L_trig,
            "K_max": K_max,
            "A_max": A_max,
            "T": T,
            "n": n,
            "prompt_start": 0,
            "trigger_start": P_max,
            "latent_start": P_max + L_trig,
            "answer_start": P_max + L_trig + lat_span,
        }

        gpt_ids[:, :P_max] = prompt_token_ids

        trig0 = P_max
        gpt_ids[:, trig0 : trig0 + L_trig] = trigger_token_ids

        lat0 = trig0 + L_trig
        z0 = self.z_start.unsqueeze(0).expand(B, -1, -1)
        latent_vals[:, lat0 : lat0 + n] = z0
        is_latent[:, lat0 : lat0 + n] = True

        for k in range(K_max):
            off = lat0 + n + k * n
            latent_vals[:, off : off + n] = cot_latents[:, k]
            is_latent[:, off : off + n] = True

        ans0 = lat0 + (1 + K_max) * n
        gpt_ids[:, ans0 : ans0 + A_max] = answer_token_ids

        tok_e = self.lm.transformer.wte(gpt_ids)
        lat_e = self.latent_input_proj(self.latent_input_norm(latent_vals.to(dtype)))
        inputs_embeds = torch.where(is_latent.unsqueeze(-1), lat_e, tok_e)

        block_row = self.build_block_ids(P_max, L_trig, K_max, A_max, device)
        block_ids = block_row.unsqueeze(0).expand(B, -1)

        seq_attention_mask = torch.zeros(B, T, device=device, dtype=dtype)
        seq_attention_mask[:, :P_max] = prompt_attention_mask.to(dtype)
        seq_attention_mask[:, trig0 : trig0 + L_trig] = 1.0
        seq_attention_mask[:, lat0 : lat0 + (1 + K_max) * n] = 1.0
        seq_attention_mask[:, ans0 : ans0 + A_max] = answer_attention_mask.to(dtype)

        return inputs_embeds, block_ids, seq_attention_mask, meta

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_4d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.lm.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_4d,
        ).last_hidden_state
        lm_logits = self.lm_head(hidden)
        z_pred = self.latent_out_norm(self.latent_out_proj(hidden))
        return {"hidden": hidden, "lm_logits": lm_logits, "latent_pred": z_pred}

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        trigger_token_ids: torch.Tensor,
        end_latent_mse_threshold: float,
        max_latent_chunks: int = 64,
        max_answer_tokens: int = 256,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
    ) -> Tuple[torch.Tensor, int]:
        """Two-phase generation: latent blocks until MSE vs ``_z_end`` < threshold, then greedy tokens.

        Uses full-sequence forwards (no KV cache) for clarity.

        Args:
            prompt_token_ids: ``(1, P)``
            prompt_attention_mask: ``(1, P)``
            trigger_token_ids: ``(1, L_trig)``
            end_latent_mse_threshold: stop latent phase when mean MSE below this
            max_latent_chunks: safety cap on latent blocks after start
            max_answer_tokens: cap after BOS
            bos_token_id / eos_token_id: GPT-2 specials (defaults: often ``50256`` for BOS/EOS in byte-level GPT-2)

        Returns:
            ``(generated_token_ids (1, T_out), n_latent_chunks_emitted)``
        """
        self.eval()
        device = prompt_token_ids.device
        dtype = self.lm.transformer.wte.weight.dtype
        z_end = self.get_end_thinking_latent().to(device=device, dtype=dtype)

        pieces_embeds: List[torch.Tensor] = []
        pieces_block_sizes: List[int] = []

        emb = self.lm.transformer.wte

        P = int(prompt_attention_mask.sum().item())
        gpt_ids = prompt_token_ids[:, :P]
        pieces_embeds.append(emb(gpt_ids))
        pieces_block_sizes.extend([1] * P)

        L_trig = trigger_token_ids.size(1)
        pieces_embeds.append(emb(trigger_token_ids))
        pieces_block_sizes.extend([1] * L_trig)

        z0 = self.latent_input_proj(self.latent_input_norm(self.z_start.unsqueeze(0)))
        pieces_embeds.append(z0)
        pieces_block_sizes.append(self.n_latent_tokens)

        def block_ids_from_sizes(sizes: List[int], dev: torch.device) -> torch.Tensor:
            ids: List[int] = []
            b = 0
            for sz in sizes:
                ids.extend([b] * sz)
                b += 1
            return torch.tensor(ids, device=dev, dtype=torch.long)

        n = self.n_latent_tokens
        n_latent_emitted = 0

        for _ in range(max_latent_chunks):
            full_e = torch.cat(pieces_embeds, dim=1)
            T = full_e.size(1)
            b_ids = block_ids_from_sizes(pieces_block_sizes, device).unsqueeze(0)
            mask = make_variable_block_causal_mask(
                b_ids,
                dtype=dtype,
                merge_key_padding_mask=torch.ones(1, T, device=device, dtype=dtype),
            )
            out = self.forward(full_e, mask)
            z_last = out["latent_pred"][0, T - n : T]
            mse = F.mse_loss(z_last.float(), z_end[0].float()).item()
            if mse < end_latent_mse_threshold:
                break
            z_next_ae = z_last.unsqueeze(0).to(dtype=dtype)
            next_chunk = self.latent_input_proj(self.latent_input_norm(z_next_ae))
            pieces_embeds.append(next_chunk)
            pieces_block_sizes.append(n)
            n_latent_emitted += 1

        bos = emb(torch.tensor([[bos_token_id]], device=device))
        pieces_embeds.append(bos)
        pieces_block_sizes.append(1)

        gen_ids: List[int] = []
        for _step in range(max_answer_tokens):
            full_e = torch.cat(pieces_embeds, dim=1)
            T = full_e.size(1)
            b_ids = block_ids_from_sizes(pieces_block_sizes, device).unsqueeze(0)
            mask = make_variable_block_causal_mask(
                b_ids,
                dtype=dtype,
                merge_key_padding_mask=torch.ones(1, T, device=device, dtype=dtype),
            )
            out = self.forward(full_e, mask)
            logits = out["lm_logits"][0, -1]
            nxt = int(logits.argmax(dim=-1).item())
            gen_ids.append(nxt)
            if nxt == eos_token_id:
                break
            pieces_embeds.append(emb(torch.tensor([[nxt]], device=device)))
            pieces_block_sizes.append(1)

        return torch.tensor([gen_ids], device=device, dtype=torch.long), n_latent_emitted
