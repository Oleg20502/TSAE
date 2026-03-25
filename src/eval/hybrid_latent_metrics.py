"""Accuracy metrics for hybrid latent model validation (answer LM + AE decoder on CoT)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _answer_exact_match_per_row(
    lm_logits: torch.Tensor,
    answer_token_ids: torch.Tensor,
    answer_attention_mask: torch.Tensor,
    answer_start: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row exact match: greedy LM predictions vs targets after BOS.

    Returns:
        (matches, counts) each ``(B,)`` float — count is 1 if row has ≥2 valid answer
        tokens (BOS + ≥1 target), else 0 (excluded from mean).
    """
    B, A = answer_token_ids.shape
    device = lm_logits.device
    matches = torch.zeros(B, device=device)
    counts = torch.zeros(B, device=device)
    pred_ids = lm_logits.argmax(dim=-1)
    for b in range(B):
        valid = int(answer_attention_mask[b].sum().item())
        if valid < 2:
            continue
        counts[b] = 1.0
        n_pred = valid - 1
        pred_seq = pred_ids[b, answer_start : answer_start + n_pred]
        tgt_seq = answer_token_ids[b, 1:valid]
        L = min(pred_seq.numel(), tgt_seq.numel())
        matches[b] = float((pred_seq[:L] == tgt_seq[:L]).all().item())
    return matches, counts


def _decoder_shifted_token_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Next-token setup: ``logits[:, :-1]`` vs ``labels[:, 1:]``.

    Returns:
        (n_correct, n_total) scalars on same device.
    """
    pred = logits.argmax(dim=-1)
    sp = pred[:, :-1]
    sl = labels[:, 1:]
    m = sl != ignore_index
    if not m.any():
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    correct = ((sp == sl) & m).sum().to(torch.float32)
    total = m.sum().to(torch.float32)
    return correct, total


def _decoder_shifted_exact_per_row(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-row: all non-ignored next-token positions correct."""
    pred = logits.argmax(dim=-1)
    sp = pred[:, :-1]
    sl = labels[:, 1:]
    m = sl != ignore_index
    match = (sp == sl) | (~m)
    return match.all(dim=1)


def accumulate_hybrid_eval_batch(
    lm_logits: torch.Tensor,
    latent_pred: torch.Tensor,
    meta: Dict,
    batch: Dict[str, torch.Tensor],
    ae_decoder: torch.nn.Module,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """Compute additive statistics for one eval batch (sum numerators/denominators).

    Keys are raw counts for later division after ``accelerator.reduce``.
    """
    device = lm_logits.device
    answer_start = int(meta["answer_start"])
    lat0 = int(meta["latent_start"])
    n = int(meta["n"])
    K_max = int(batch["cot_ae_input_ids"].size(1))
    cot_valid = batch["cot_valid"]

    out: Dict[str, torch.Tensor] = {}

    m_ans, c_ans = _answer_exact_match_per_row(
        lm_logits,
        batch["answer_token_ids"],
        batch["answer_attention_mask"],
        answer_start,
    )
    out["answer_em_sum"] = m_ans.sum()
    out["answer_em_den"] = c_ans.sum()

    cot_tok_cor = torch.tensor(0.0, device=device)
    cot_tok_tot = torch.tensor(0.0, device=device)
    step_em_sums = []
    step_em_dens = []

    full_ok = torch.ones(lm_logits.size(0), dtype=torch.bool, device=device)

    for j in range(K_max):
        mask = cot_valid[:, j]
        if not mask.any():
            step_em_sums.append(torch.tensor(0.0, device=device))
            step_em_dens.append(torch.tensor(0.0, device=device))
            continue

        pred_j = latent_pred[:, lat0 + j * n : lat0 + (j + 1) * n]
        dec_in = batch["cot_decoder_input_ids"][mask, j]
        lbl = batch["cot_labels"][mask, j]
        logits = ae_decoder(latent_tokens=pred_j[mask], decoder_input_ids=dec_in)
        logits_f = logits.float()

        c_cor, c_tot = _decoder_shifted_token_stats(logits_f, lbl, ignore_index)
        cot_tok_cor = cot_tok_cor + c_cor
        cot_tok_tot = cot_tok_tot + c_tot

        row_ok = _decoder_shifted_exact_per_row(logits_f, lbl, ignore_index)
        step_em_sums.append(row_ok.to(torch.float32).sum())
        step_em_dens.append(mask.sum().to(torch.float32))

        em_b = torch.zeros(mask.size(0), dtype=torch.bool, device=device)
        em_b[mask] = row_ok
        full_ok = full_ok & (~mask | em_b)

    has_any_cot = cot_valid.any(dim=1)
    out["cot_full_em_sum"] = (full_ok & has_any_cot).to(torch.float32).sum()
    out["cot_full_em_den"] = has_any_cot.to(torch.float32).sum()

    out["cot_tok_cor"] = cot_tok_cor
    out["cot_tok_tot"] = cot_tok_tot

    for j in range(K_max):
        out[f"cot_step_{j + 1}_em_sum"] = step_em_sums[j]
        out[f"cot_step_{j + 1}_em_den"] = step_em_dens[j]

    return out
