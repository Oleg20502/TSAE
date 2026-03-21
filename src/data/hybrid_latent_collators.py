"""Batching for hybrid latent training (GPT-2 + AE tokens per CoT step)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from src.data.hybrid_latent_datasets import parse_cot_steps
from src.utils.config import HybridLatentDataConfig


@dataclass
class HybridLatentCollator:
    """Tokenize task / CoT steps / answer; fixed shapes from config and ``n_latent_tokens`` (AE width in time)."""

    gpt2_tok: PreTrainedTokenizerBase
    ae_tok: PreTrainedTokenizerBase
    ae_max_length: int
    n_latent_tokens: int
    cfg: HybridLatentDataConfig

    def __post_init__(self) -> None:
        dc = self.cfg
        self.P = dc.max_prompt_tokens
        self.K = dc.max_cot_steps
        self.A = dc.max_answer_tokens
        enc = self.gpt2_tok(dc.reasoning_trigger, add_special_tokens=False)
        self.trigger_ids: List[int] = enc["input_ids"]
        self.L_trig = len(self.trigger_ids)
        self.T = self.P + self.L_trig + (1 + self.K) * self.n_latent_tokens + self.A
        self.answer_start = self.P + self.L_trig + (1 + self.K) * self.n_latent_tokens

    def _pad_gpt2(self, ids: List[int], max_len: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = ids[:max_len]
        mask = torch.zeros(max_len, dtype=torch.float32)
        L = len(ids)
        if L > 0:
            mask[:L] = 1.0
        if L < max_len:
            ids = ids + [pad_id] * (max_len - L)
        return torch.tensor(ids, dtype=torch.long), mask

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        pad_gpt = self.gpt2_tok.pad_token_id or self.gpt2_tok.eos_token_id or 0
        bos = self.gpt2_tok.bos_token_id
        if bos is None:
            bos = self.gpt2_tok.eos_token_id or 50256
        eos = self.gpt2_tok.eos_token_id
        if eos is None:
            eos = bos

        ae_pad = self.ae_tok.pad_token_id or 0
        ae_bos = self.ae_tok.cls_token_id or self.ae_tok.bos_token_id or 101

        B = len(batch)
        prompt_ids = torch.zeros(B, self.P, dtype=torch.long)
        prompt_m = torch.zeros(B, self.P, dtype=torch.float32)
        trig = torch.tensor([self.trigger_ids] * B, dtype=torch.long)
        cot_in = torch.zeros(B, self.K, self.ae_max_length, dtype=torch.long)
        cot_m = torch.zeros(B, self.K, self.ae_max_length, dtype=torch.float32)
        cot_valid = torch.zeros(B, self.K, dtype=torch.bool)
        cot_dec_in = torch.full((B, self.K, self.ae_max_length), ae_pad, dtype=torch.long)
        cot_lbl = torch.zeros(B, self.K, self.ae_max_length, dtype=torch.long)
        ans_ids = torch.zeros(B, self.A, dtype=torch.long)
        ans_m = torch.zeros(B, self.A, dtype=torch.float32)

        end_enc = self.ae_tok(
            self.cfg.end_of_thinking_phrase,
            max_length=self.ae_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        end_ids = end_enc["input_ids"].squeeze(0)
        end_m = end_enc["attention_mask"].squeeze(0).float()
        end_lbl = end_ids.clone()
        end_lbl[end_ids == ae_pad] = -100
        end_dec = torch.full_like(end_ids, ae_pad)
        end_dec[0] = ae_bos
        if end_ids.size(0) > 1:
            end_dec[1:] = end_ids[:-1]

        for i, row in enumerate(batch):
            task = row["task"] or ""
            steps = parse_cot_steps(row["cot"] or "")
            if len(steps) == 0:
                steps = ["0"]
            steps = steps[: self.K]

            pt = self.gpt2_tok(task, add_special_tokens=False)["input_ids"]
            p_tensor, p_mask = self._pad_gpt2(pt, self.P, pad_gpt)
            prompt_ids[i] = p_tensor
            prompt_m[i] = p_mask

            for k, st in enumerate(steps):
                enc = self.ae_tok(
                    st,
                    max_length=self.ae_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].squeeze(0)
                am = enc["attention_mask"].squeeze(0).float()
                cot_in[i, k] = ids
                cot_m[i, k] = am
                cot_valid[i, k] = True
                lbl = ids.clone()
                lbl[ids == ae_pad] = -100
                cot_lbl[i, k] = lbl
                cot_dec_in[i, k, 0] = ae_bos
                cot_dec_in[i, k, 1:] = ids[:-1]

            ans = row["labels"] or ""
            at = self.gpt2_tok.encode(ans, add_special_tokens=False)
            full = [bos] + at + [eos]
            a_tensor, a_mask = self._pad_gpt2(full, self.A, pad_gpt)
            ans_ids[i] = a_tensor
            ans_m[i] = a_mask

        lm_labels = torch.full((B, self.T), -100, dtype=torch.long)
        for b in range(B):
            valid_a = int(ans_m[b].sum().item())
            for t in range(max(valid_a - 1, 0)):
                lm_labels[b, self.answer_start + t] = ans_ids[b, t + 1]

        return {
            "prompt_token_ids": prompt_ids,
            "prompt_attention_mask": prompt_m,
            "trigger_token_ids": trig,
            "cot_ae_input_ids": cot_in,
            "cot_ae_attention_mask": cot_m,
            "cot_valid": cot_valid,
            "cot_decoder_input_ids": cot_dec_in,
            "cot_labels": cot_lbl,
            "answer_token_ids": ans_ids,
            "answer_attention_mask": ans_m,
            "lm_labels": lm_labels,
            "end_phrase_input_ids": end_ids.unsqueeze(0).expand(B, -1).clone(),
            "end_phrase_attention_mask": end_m.unsqueeze(0).expand(B, -1).clone(),
            "end_decoder_input_ids": end_dec.unsqueeze(0).expand(B, -1).clone(),
            "end_labels": end_lbl.unsqueeze(0).expand(B, -1).clone(),
        }
