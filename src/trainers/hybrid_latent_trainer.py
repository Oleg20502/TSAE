"""Accelerate trainer: hybrid GPT-2 + frozen AE (latent reasoning + answer LM)."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_model, save_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, get_constant_schedule_with_warmup

from src.losses.reconstruction import reconstruction_loss
from src.models.concept_model import make_variable_block_causal_mask
from src.models.hybrid_latent_model import HybridLatentReasoningGPT2
from src.utils.best_checkpoint import (
    default_greater_is_better,
    is_valid_metric_value,
    metric_improves,
)
from src.utils.config import HybridLatentModelConfig, TrainConfig
from src.utils.training_steps import optimizer_steps_per_epoch


class HybridLatentTrainer:
    def __init__(
        self,
        model: HybridLatentReasoningGPT2,
        ae_encoder: torch.nn.Module,
        ae_decoder: torch.nn.Module,
        ae_tokenizer: PreTrainedTokenizerBase,
        n_latent_tokens: int,
        model_cfg: HybridLatentModelConfig,
        train_dataset,
        eval_dataset,
        data_collator,
        train_config: TrainConfig,
        end_of_thinking_phrase: str,
    ):
        self.cfg = train_config
        self.model_cfg = model_cfg
        self._n_latent = n_latent_tokens
        self._lambda_mse = model_cfg.lambda_mse
        self._lambda_ans = model_cfg.lambda_answer_ce
        self._ae_tokenizer = ae_tokenizer
        self._end_phrase = end_of_thinking_phrase

        self._core = model

        mixed_precision = (
            "fp16" if train_config.fp16 else ("bf16" if train_config.bf16 else "no")
        )
        log_with = (
            train_config.report_to
            if train_config.report_to not in ("none", "no", "", None)
            else None
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_dir=train_config.output_dir,
        )

        optimizer = AdamW(
            self._core.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            collate_fn=data_collator,
            num_workers=train_config.dataloader_num_workers,
            shuffle=True,
        )
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=train_config.batch_size,
            collate_fn=data_collator,
            num_workers=train_config.dataloader_num_workers,
        )

        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=train_config.warmup_steps
        )

        (
            self._core,
            self._optimizer,
            self._train_dl,
            self._eval_dl,
            self._scheduler,
        ) = self.accelerator.prepare(self._core, optimizer, train_dl, eval_dl, scheduler)

        try:
            n_train = len(train_dataset)  # type: ignore[arg-type]
        except TypeError:
            n_train = None
        if n_train is not None:
            self._steps_per_epoch = optimizer_steps_per_epoch(
                n_train,
                train_config.batch_size,
                train_config.gradient_accumulation_steps,
                self.accelerator.num_processes,
            )
        else:
            self._steps_per_epoch = math.ceil(
                len(self._train_dl) / train_config.gradient_accumulation_steps
            )

        device = self.accelerator.device
        self._ae_encoder = ae_encoder.to(device)
        self._ae_encoder.eval()
        for p in self._ae_encoder.parameters():
            p.requires_grad = False

        self._ae_decoder = ae_decoder.to(device)
        self._ae_decoder.eval()
        for p in self._ae_decoder.parameters():
            p.requires_grad = False

        self._cache_end_thinking_latent()

        self._ema: Optional[ExponentialMovingAverage] = None
        if train_config.ema_decay and train_config.ema_decay > 0.0:
            params = [p for p in self._core.parameters() if p.requires_grad]
            if params:
                self._ema = ExponentialMovingAverage(params, decay=train_config.ema_decay)

        self._saved_checkpoints: List[str] = []
        self._best_metric: Optional[float] = None
        self._best_model_checkpoint: Optional[str] = None
        self._last_eval_metrics: Optional[Dict[str, float]] = None
        self._last_eval_step: int = -1

    @torch.no_grad()
    def _cache_end_thinking_latent(self) -> None:
        device = self.accelerator.device
        mc = self.accelerator.unwrap_model(self._core)
        # Must match encoder.pos_emb length (BottleneckEncoder.max_length); longer
        # sequences OOB the position embedding table (see tok_emb + pos_emb in encoder).
        ae_max_len = int(self._ae_encoder.max_length)
        enc = self._ae_tokenizer(
            self._end_phrase,
            max_length=ae_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        z = self._ae_encoder(ids, am)
        mc.set_end_thinking_latent(z)

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = batch["prompt_token_ids"].size(0)
        device = batch["prompt_token_ids"].device
        dtype = self.accelerator.unwrap_model(self._core).lm.transformer.wte.weight.dtype

        cot_in = batch["cot_ae_input_ids"]
        cot_m = batch["cot_ae_attention_mask"]
        K_max = cot_in.size(1)
        T_ae = cot_in.size(2)
        flat_in = cot_in.view(B * K_max, T_ae)
        flat_m = cot_m.view(B * K_max, T_ae)
        with torch.no_grad():
            z_flat = self._ae_encoder(flat_in, flat_m)
        cot_latents = z_flat.view(B, K_max, self._n_latent, z_flat.size(-1))

        end_ids = batch["end_phrase_input_ids"]
        end_m = batch["end_phrase_attention_mask"]
        with torch.no_grad():
            z_end = self._ae_encoder(end_ids[:1], end_m[:1])

        m = self.accelerator.unwrap_model(self._core)
        inputs_embeds, block_ids, seq_m, meta = m.build_inputs_embeds_and_masks(
            batch["prompt_token_ids"],
            batch["prompt_attention_mask"],
            batch["trigger_token_ids"],
            cot_latents,
            batch["cot_valid"],
            batch["answer_token_ids"],
            batch["answer_attention_mask"],
        )

        attn_4d = make_variable_block_causal_mask(
            block_ids,
            dtype=dtype,
            merge_key_padding_mask=seq_m,
        )

        out = m(inputs_embeds, attn_4d)
        lm_logits = out["lm_logits"]
        latent_pred = out["latent_pred"]
        V = lm_logits.size(-1)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch["lm_labels"][:, 1:].contiguous()
        loss_lm = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        lat0 = meta["latent_start"]
        n = meta["n"]
        cot_valid = batch["cot_valid"]

        mse_terms: List[torch.Tensor] = []
        ce_terms: List[torch.Tensor] = []

        z_end_b = z_end.expand(B, -1, -1).to(dtype=latent_pred.dtype)

        for j in range(K_max + 1):
            pred_j = latent_pred[:, lat0 + j * n : lat0 + (j + 1) * n]
            if j < K_max:
                tgt = cot_latents[:, j]
                mask = cot_valid[:, j]
            else:
                tgt = z_end_b
                mask = cot_valid[:, K_max - 1] if K_max > 0 else torch.ones(B, dtype=torch.bool, device=device)

            if not mask.any():
                continue
            mse_terms.append(F.mse_loss(pred_j[mask], tgt[mask]))

            if j < K_max:
                dec_in = batch["cot_decoder_input_ids"][mask, j]
                lbl = batch["cot_labels"][mask, j]
            else:
                dec_in = batch["end_decoder_input_ids"][mask]
                lbl = batch["end_labels"][mask]

            logits = self._ae_decoder(latent_tokens=pred_j[mask], decoder_input_ids=dec_in)
            ce_terms.append(reconstruction_loss(logits, lbl))

        mse_loss = torch.stack(mse_terms).mean() if mse_terms else torch.tensor(0.0, device=device)
        ce_lat = torch.stack(ce_terms).mean() if ce_terms else torch.tensor(0.0, device=device)

        loss = self._lambda_ans * loss_lm + ce_lat + self._lambda_mse * mse_loss

        return {
            "loss": loss,
            "l_lm": loss_lm.detach(),
            "l_latent_ce": ce_lat.detach(),
            "l_mse": mse_loss.detach(),
        }

    def train(self, resume_from_checkpoint: Optional[str] = None):
        cfg = self.cfg
        self.accelerator.init_trackers("hybrid_latent", config=asdict(cfg))

        global_step = 0
        start_epoch = 0
        resume_steps_in_epoch = 0

        if resume_from_checkpoint is not None:
            if resume_from_checkpoint == "latest":
                resume_from_checkpoint = self._find_latest_checkpoint()
            if resume_from_checkpoint is not None:
                global_step = self.load_checkpoint(resume_from_checkpoint)
                start_epoch = global_step // self._steps_per_epoch
                resume_steps_in_epoch = global_step % self._steps_per_epoch

        log_loss_sum = log_lm_sum = log_lce_sum = log_mse_sum = log_gnorm_sum = 0.0
        log_micro_steps = 0
        last_grad_norm = 0.0

        for epoch in range(start_epoch, cfg.epochs):
            self._core.train()

            if epoch == start_epoch and resume_steps_in_epoch > 0:
                batches_to_skip = resume_steps_in_epoch * cfg.gradient_accumulation_steps
                train_dl_epoch = self.accelerator.skip_first_batches(
                    self._train_dl, batches_to_skip
                )
                steps_this_epoch = self._steps_per_epoch - resume_steps_in_epoch
            else:
                train_dl_epoch = self._train_dl
                steps_this_epoch = self._steps_per_epoch

            epoch_bar = tqdm(
                total=steps_this_epoch,
                desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                disable=not self.accelerator.is_main_process,
                dynamic_ncols=True,
            )

            for batch in train_dl_epoch:
                with self.accelerator.accumulate(self._core):
                    outputs = self._forward(batch)
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self._core.parameters(), cfg.max_grad_norm
                        )
                        last_grad_norm = grad_norm.item()

                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()

                log_loss_sum += loss.detach().item()
                log_lm_sum += outputs["l_lm"].item()
                log_lce_sum += outputs["l_latent_ce"].item()
                log_mse_sum += outputs["l_mse"].item()
                log_micro_steps += 1

                if self.accelerator.sync_gradients:
                    epoch_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})
                    epoch_bar.update(1)

                    log_gnorm_sum += last_grad_norm
                    if self._ema is not None:
                        self._ema.update()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        self._log_train(
                            global_step,
                            epoch,
                            log_loss_sum,
                            log_lm_sum,
                            log_lce_sum,
                            log_mse_sum,
                            log_gnorm_sum,
                            log_micro_steps,
                        )
                        log_loss_sum = log_lm_sum = log_lce_sum = log_mse_sum = log_gnorm_sum = 0.0
                        log_micro_steps = 0

                    if global_step % cfg.eval_steps == 0:
                        self.evaluate(step=global_step)
                        self._core.train()

                    if global_step % cfg.save_steps == 0:
                        ckpt_dir = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}"
                        )
                        self.save_checkpoint(ckpt_dir, global_step=global_step, epoch=epoch)
                        self._maybe_update_best_model_checkpoint(ckpt_dir, global_step)
                        self._rotate_checkpoints()

            epoch_bar.close()
            resume_steps_in_epoch = 0

        self.accelerator.end_training()

    def _log_train(
        self,
        step: int,
        epoch: int,
        loss_sum: float,
        lm_sum: float,
        lce_sum: float,
        mse_sum: float,
        gnorm_sum: float,
        n: int,
    ):
        n = max(n, 1)
        log_dict = {
            "train/loss": loss_sum / n,
            "train/l_lm": lm_sum / n,
            "train/l_latent_ce": lce_sum / n,
            "train/l_mse": mse_sum / n,
            "train/grad_norm": gnorm_sum / n,
            "train/lr": self._scheduler.get_last_lr()[0],
            "epoch": epoch,
        }
        self.accelerator.log(log_dict, step=step)

        if self.accelerator.is_main_process:
            tqdm.write(
                f"epoch={epoch + 1}  step={step}"
                f"  loss={loss_sum/n:.4f}  l_lm={lm_sum/n:.4f}"
                f"  l_latent_ce={lce_sum/n:.4f}  l_mse={mse_sum/n:.4f}"
                f"  gnorm={gnorm_sum/n:.3f}  lr={self._scheduler.get_last_lr()[0]:.2e}"
            )

    def evaluate(self, step: Optional[int] = None) -> Dict[str, float]:
        self._core.eval()

        if self._ema is not None:
            self._ema.store()
            self._ema.copy_to()

        device = self.accelerator.device
        loss_acc = torch.zeros(1, device=device)
        lm_acc = torch.zeros(1, device=device)
        lce_acc = torch.zeros(1, device=device)
        mse_acc = torch.zeros(1, device=device)
        count = torch.zeros(1, device=device)

        eval_bar = tqdm(
            self._eval_dl,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True,
            leave=False,
        )

        try:
            for batch in eval_bar:
                with torch.no_grad():
                    outputs = self._forward(batch)
                loss_acc += outputs["loss"].detach()
                lm_acc += outputs["l_lm"].detach()
                lce_acc += outputs["l_latent_ce"].detach()
                mse_acc += outputs["l_mse"].detach()
                count += 1
        finally:
            if self._ema is not None:
                self._ema.restore()

        loss_acc = self.accelerator.reduce(loss_acc, reduction="sum")
        lm_acc = self.accelerator.reduce(lm_acc, reduction="sum")
        lce_acc = self.accelerator.reduce(lce_acc, reduction="sum")
        mse_acc = self.accelerator.reduce(mse_acc, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum")

        eval_loss = (loss_acc / count).item()
        eval_lm = (lm_acc / count).item()
        eval_lce = (lce_acc / count).item()
        eval_mse = (mse_acc / count).item()
        metrics: Dict[str, float] = {
            "eval/loss": eval_loss,
            "eval/l_lm": eval_lm,
            "eval/l_latent_ce": eval_lce,
            "eval/l_mse": eval_mse,
            "eval/ppl_lm": math.exp(min(eval_lm, 20)),
        }

        self.accelerator.log(metrics, step=step)

        if self.accelerator.is_main_process:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            tqdm.write(f"[eval step={step}]  {parts}")

        self._last_eval_metrics = dict(metrics)
        self._last_eval_step = step if step is not None else -1

        return metrics

    def save_checkpoint(
        self, output_dir: str, global_step: int = 0, epoch: int = 0, save_ema: bool = True
    ):
        self.accelerator.wait_for_everyone()
        os.makedirs(output_dir, exist_ok=True)

        self.accelerator.save_state(output_dir)

        if self.accelerator.is_main_process:
            if self._ema is not None:
                torch.save(
                    self._ema.state_dict(),
                    os.path.join(output_dir, "ema_state.pt"),
                )

            trainer_state = {
                "global_step": global_step,
                "epoch": epoch,
                "best_metric": self._best_metric,
                "best_model_checkpoint": self._best_model_checkpoint,
            }
            with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f, indent=2)

            if save_ema and self._ema is not None:
                self._ema.store()
                self._ema.copy_to()
            try:
                core = self.accelerator.unwrap_model(self._core)
                save_model(
                    core,
                    os.path.join(output_dir, "model.safetensors"),
                )
            finally:
                if save_ema and self._ema is not None:
                    self._ema.restore()

            self._saved_checkpoints.append(output_dir)

    def save_model(self, output_dir: str):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            if self._ema is not None:
                self._ema.store()
                self._ema.copy_to()
            try:
                core = self.accelerator.unwrap_model(self._core)
                save_model(
                    core,
                    os.path.join(output_dir, "model.safetensors"),
                )
            finally:
                if self._ema is not None:
                    self._ema.restore()

    def save_non_ema_model(self, output_dir: str):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            core = self.accelerator.unwrap_model(self._core)
            save_model(
                core,
                os.path.join(output_dir, "model.safetensors"),
            )

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        self.accelerator.load_state(checkpoint_dir)

        if self._ema is not None:
            ema_path = os.path.join(checkpoint_dir, "ema_state.pt")
            if os.path.exists(ema_path):
                self._ema.load_state_dict(
                    torch.load(ema_path, map_location="cpu", weights_only=True)
                )

        with open(os.path.join(checkpoint_dir, "trainer_state.json")) as f:
            state = json.load(f)
        global_step = state["global_step"]
        self._best_metric = state.get("best_metric")
        self._best_model_checkpoint = state.get("best_model_checkpoint")

        self._saved_checkpoints = sorted(
            [
                str(d)
                for d in Path(self.cfg.output_dir).glob("checkpoint-*")
                if d.is_dir() and (d / "trainer_state.json").exists()
            ],
            key=lambda d: int(Path(d).name.split("-")[1]),
        )

        if self.accelerator.is_main_process:
            tqdm.write(f"Resumed from {checkpoint_dir}  (global_step={global_step})")

        return global_step

    def _find_latest_checkpoint(self) -> Optional[str]:
        candidates = sorted(
            [
                d
                for d in Path(self.cfg.output_dir).glob("checkpoint-*")
                if d.is_dir() and (d / "trainer_state.json").exists()
            ],
            key=lambda d: int(d.name.split("-")[1]),
        )
        return str(candidates[-1]) if candidates else None

    def _checkpoint_paths_equal(self, a: str, b: str) -> bool:
        try:
            return os.path.abspath(a) == os.path.abspath(b)
        except OSError:
            return a == b

    def _maybe_update_best_model_checkpoint(self, checkpoint_dir: str, global_step: int) -> None:
        cfg = self.cfg
        if not cfg.metric_for_best_model:
            return
        if self._last_eval_step != global_step or self._last_eval_metrics is None:
            return

        key = cfg.metric_for_best_model
        if key not in self._last_eval_metrics:
            if self.accelerator.is_main_process:
                tqdm.write(
                    f"Warning: metric_for_best_model={key!r} not in eval metrics "
                    f"{list(self._last_eval_metrics.keys())}"
                )
            return

        value = float(self._last_eval_metrics[key])
        if not is_valid_metric_value(value):
            return

        greater = cfg.greater_is_better
        if greater is None:
            greater = default_greater_is_better(key)

        if metric_improves(value, self._best_metric, greater):
            self._best_metric = value
            self._best_model_checkpoint = checkpoint_dir
            if self.accelerator.is_main_process:
                tqdm.write(
                    f"*** New best model {key}={value:.6f} — checkpoint: {checkpoint_dir}"
                )

    def _rotate_checkpoints(self):
        limit = self.cfg.save_total_limit
        if not limit or limit <= 0:
            return
        best = self._best_model_checkpoint
        while len(self._saved_checkpoints) > limit:
            remove_idx = None
            for i, ckpt in enumerate(self._saved_checkpoints):
                if best and self._checkpoint_paths_equal(ckpt, best):
                    continue
                remove_idx = i
                break
            if remove_idx is None:
                break
            oldest = self._saved_checkpoints.pop(remove_idx)
            if self.accelerator.is_main_process and os.path.isdir(oldest):
                shutil.rmtree(oldest)


def load_hybrid_latent_weights(checkpoint_path: str, model: HybridLatentReasoningGPT2, device: str = "cpu") -> None:
    if checkpoint_path.endswith(".safetensors"):
        missing, unexpected = load_model(
            model, checkpoint_path, strict=False, device=device
        )
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)

    if unexpected:
        print("Warning: unexpected keys in checkpoint:", unexpected)
    if missing:
        print("Warning: missing keys (not loaded):", missing)
