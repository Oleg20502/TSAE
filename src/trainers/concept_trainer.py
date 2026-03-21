"""Accelerate-based trainer for the Concept Model.

Structural notes:
- ``_CMTrainableCore`` wraps only the Concept Model (CM); the frozen AE encoder
  and decoder live outside it, matching the repr_encoder pattern in
  ``BottleneckTrainer``.
- The frozen AE decoder is called inside ``_forward()`` so CE-loss gradients
  flow through it back to the CM via normal autograd. Because decoder parameters
  have ``requires_grad=False`` they accumulate no ``.grad``; only the CM learns.
- The frozen AE encoder is called under ``torch.no_grad()`` — its output is
  treated as a fixed target / input.
- EMA, Accelerate preparation, checkpointing, and resume logic mirror
  ``BottleneckTrainer`` exactly.
"""

import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm
from transformers import get_constant_schedule_with_warmup

from src.losses.reconstruction import reconstruction_loss
from src.utils.best_checkpoint import (
    default_greater_is_better,
    is_valid_metric_value,
    metric_improves,
)
from src.utils.config import TrainConfig


# ---------------------------------------------------------------------------
# Trainable module (Concept Model only)
# ---------------------------------------------------------------------------

class _CMTrainableCore(nn.Module):
    """Thin wrapper around the Concept Model so Accelerate can prepare it.

    Only the CM parameters are inside this module; the frozen AE encoder and
    decoder live in the trainer and are never included in optimizer states or
    saved checkpoints.
    """

    def __init__(self, concept_model: nn.Module):
        super().__init__()
        self.concept_model = concept_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a flattened latent sequence through the CM.

        Args:
            x: ``(B, N*n, d_ae)`` flattened latent sequence.

        Returns:
            ``(B, N*n, d_ae)`` predicted latent sequence.
        """
        return self.concept_model(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ConceptTrainer:
    """Lightweight Accelerate-based trainer for the Concept Model.

    The Concept Model is trained to predict the next AE latent in a sequence.
    The frozen AE encoder compresses each text chunk into a latent; the CM
    then predicts the next latent; the frozen decoder reconstructs text from
    the predicted latent and the cross-entropy over the reconstruction is the
    primary training signal.  An auxiliary MSE loss penalises the distance
    between predicted and actual latents.

    Args:
        concept_model:   Untrained CM (``ConceptModel`` or ``ConceptModelGPT2``).
        ae_encoder:      Frozen AE encoder (``BottleneckEncoder``).
        ae_decoder:      Frozen AE decoder.
        n_latent_tokens: Number of latent tokens per chunk in the AE.
        lambda_mse:      Weight of the MSE latent loss term.
        train_dataset:   ``ChunkGroupDataset`` for training.
        eval_dataset:    ``ChunkGroupDataset`` for evaluation.
        data_collator:   ``CMCollator`` instance.
        train_config:    ``TrainConfig`` (reused from AE codebase).
    """

    def __init__(
        self,
        concept_model: nn.Module,
        ae_encoder: nn.Module,
        ae_decoder: nn.Module,
        n_latent_tokens: int,
        lambda_mse: float,
        train_dataset,
        eval_dataset,
        data_collator,
        train_config: TrainConfig,
    ):
        self.cfg             = train_config
        self._n_latent       = n_latent_tokens
        self._lambda_mse     = lambda_mse

        self._core = _CMTrainableCore(concept_model)

        # ------------------------------------------------------------------
        # Accelerator
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Optimizer, dataloaders, scheduler
        # ------------------------------------------------------------------
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

        self._steps_per_epoch = math.ceil(
            len(train_dl) / train_config.gradient_accumulation_steps
        )

        (
            self._core,
            self._optimizer,
            self._train_dl,
            self._eval_dl,
            self._scheduler,
        ) = self.accelerator.prepare(self._core, optimizer, train_dl, eval_dl, scheduler)

        # ------------------------------------------------------------------
        # Frozen AE components — on device, excluded from Accelerate/optimizer
        # ------------------------------------------------------------------
        device = self.accelerator.device

        self._ae_encoder = ae_encoder.to(device)
        self._ae_encoder.eval()
        for p in self._ae_encoder.parameters():
            p.requires_grad = False

        self._ae_decoder = ae_decoder.to(device)
        self._ae_decoder.eval()
        for p in self._ae_decoder.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # EMA
        # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode all N chunks per sequence with the frozen AE encoder.

        Args:
            input_ids:      ``(B, N, T)``
            attention_mask: ``(B, N, T)``

        Returns:
            ``(B, N, n, d_ae)`` where ``n = n_latent_tokens``.
        """
        B, N, T = input_ids.shape
        latents = self._ae_encoder(
            input_ids.view(B * N, T),
            attention_mask.view(B * N, T),
        )  # (B*N, n, d_ae)
        return latents.view(B, N, latents.size(1), latents.size(2))

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full CM forward pass: encode → CM → decode → losses.

        CE loss gradient flows: loss → frozen decoder → pred_latents → CM.
        MSE loss gradient flows: loss → pred_latents → CM.
        """
        input_ids         = batch["input_ids"]          # (B, N, T)
        attention_mask    = batch["attention_mask"]     # (B, N, T)
        decoder_input_ids = batch["decoder_input_ids"]  # (B, N, T)
        labels            = batch["labels"]             # (B, N, T)

        B, N, T = input_ids.shape
        n = self._n_latent

        # 1. Encode all chunks with frozen AE encoder (no gradient)
        ae_latents = self._encode_latents(input_ids, attention_mask)  # (B, N, n, d_ae)
        d_ae = ae_latents.size(-1)

        # 2. Flatten and run through CM
        cm_input   = ae_latents.view(B, N * n, d_ae)
        pred_flat  = self._core(cm_input)                 # (B, N*n, d_ae)  ← trainable
        pred_latents = pred_flat.view(B, N, n, d_ae)

        # 3. Shift: output latent i predicts target latent i+1
        N_pred        = N - 1
        pred_shifted  = pred_latents[:, :-1]              # (B, N-1, n, d_ae)
        target_latents = ae_latents[:, 1:].detach()       # (B, N-1, n, d_ae) — fixed target

        # 4. MSE between predicted and actual next latents
        mse_loss = F.mse_loss(pred_shifted, target_latents)

        # 5. CE via frozen AE decoder (teacher-forcing on next-chunk tokens)
        dec_in      = decoder_input_ids[:, 1:].reshape(B * N_pred, T)  # skip first chunk
        lbl         = labels[:, 1:].reshape(B * N_pred, T)
        pred_decode = pred_shifted.reshape(B * N_pred, n, d_ae)

        # Gradients flow through the frozen decoder back to pred_shifted → CM
        logits = self._ae_decoder(
            latent_tokens=pred_decode,
            decoder_input_ids=dec_in,
        )  # (B*N_pred, T, vocab)

        ce_loss = reconstruction_loss(logits, lbl)

        loss = ce_loss + self._lambda_mse * mse_loss
        return {
            "loss":  loss,
            "l_ce":  ce_loss.detach(),
            "l_mse": mse_loss.detach(),
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, resume_from_checkpoint: Optional[str] = None):
        cfg = self.cfg
        self.accelerator.init_trackers("concept_model", config=asdict(cfg))

        global_step            = 0
        start_epoch            = 0
        resume_steps_in_epoch  = 0

        if resume_from_checkpoint is not None:
            if resume_from_checkpoint == "latest":
                resume_from_checkpoint = self._find_latest_checkpoint()
            if resume_from_checkpoint is not None:
                global_step           = self.load_checkpoint(resume_from_checkpoint)
                start_epoch           = global_step // self._steps_per_epoch
                resume_steps_in_epoch = global_step % self._steps_per_epoch

        log_loss_sum = log_ce_sum = log_mse_sum = log_gnorm_sum = 0.0
        log_micro_steps = 0
        last_grad_norm  = 0.0

        for epoch in range(start_epoch, cfg.epochs):
            self._core.train()

            if epoch == start_epoch and resume_steps_in_epoch > 0:
                batches_to_skip  = resume_steps_in_epoch * cfg.gradient_accumulation_steps
                train_dl_epoch   = self.accelerator.skip_first_batches(
                    self._train_dl, batches_to_skip
                )
                steps_this_epoch = self._steps_per_epoch - resume_steps_in_epoch
            else:
                train_dl_epoch   = self._train_dl
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
                    loss    = outputs["loss"]
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self._core.parameters(), cfg.max_grad_norm
                        )
                        last_grad_norm = grad_norm.item()

                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()

                log_loss_sum    += loss.detach().item()
                log_ce_sum      += outputs["l_ce"].item()
                log_mse_sum     += outputs["l_mse"].item()
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
                            global_step, epoch,
                            log_loss_sum, log_ce_sum, log_mse_sum,
                            log_gnorm_sum, log_micro_steps,
                        )
                        log_loss_sum = log_ce_sum = log_mse_sum = log_gnorm_sum = 0.0
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
        ce_sum: float,
        mse_sum: float,
        gnorm_sum: float,
        n: int,
    ):
        n = max(n, 1)
        log_dict = {
            "train/loss":      loss_sum / n,
            "train/l_ce":      ce_sum   / n,
            "train/l_mse":     mse_sum  / n,
            "train/grad_norm": gnorm_sum / n,
            "train/lr":        self._scheduler.get_last_lr()[0],
            "epoch":           epoch,
        }
        self.accelerator.log(log_dict, step=step)

        if self.accelerator.is_main_process:
            tqdm.write(
                f"epoch={epoch + 1}  step={step}"
                f"  loss={loss_sum/n:.4f}  l_ce={ce_sum/n:.4f}  l_mse={mse_sum/n:.4f}"
                f"  gnorm={gnorm_sum/n:.3f}  lr={self._scheduler.get_last_lr()[0]:.2e}"
            )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self, step: Optional[int] = None) -> Dict[str, float]:
        self._core.eval()

        if self._ema is not None:
            self._ema.store()
            self._ema.copy_to()

        device    = self.accelerator.device
        loss_acc  = torch.zeros(1, device=device)
        ce_acc    = torch.zeros(1, device=device)
        mse_acc   = torch.zeros(1, device=device)
        count     = torch.zeros(1, device=device)

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
                ce_acc   += outputs["l_ce"].detach()
                mse_acc  += outputs["l_mse"].detach()
                count    += 1
        finally:
            if self._ema is not None:
                self._ema.restore()

        loss_acc = self.accelerator.reduce(loss_acc, reduction="sum")
        ce_acc   = self.accelerator.reduce(ce_acc,   reduction="sum")
        mse_acc  = self.accelerator.reduce(mse_acc,  reduction="sum")
        count    = self.accelerator.reduce(count,    reduction="sum")

        eval_loss = (loss_acc / count).item()
        eval_ce = (ce_acc / count).item()
        eval_mse = (mse_acc / count).item()
        metrics: Dict[str, float] = {
            "eval/loss":       eval_loss,
            "eval/l_ce":       eval_ce,
            "eval/l_mse":      eval_mse,
            "eval/perplexity": math.exp(min(eval_ce, 20)),
        }

        self.accelerator.log(metrics, step=step)

        if self.accelerator.is_main_process:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            tqdm.write(f"[eval step={step}]  {parts}")

        self._last_eval_metrics = dict(metrics)
        self._last_eval_step = step if step is not None else -1

        return metrics

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    def _build_state_dict(self) -> Dict[str, torch.Tensor]:
        """Collect only the Concept Model weights (no frozen AE components)."""
        core = self.accelerator.unwrap_model(self._core)
        sd: Dict[str, torch.Tensor] = {}
        for k, v in core.concept_model.state_dict().items():
            sd[f"concept_model.{k}"] = v
        return sd

    def save_checkpoint(
        self, output_dir: str, global_step: int = 0, epoch: int = 0, save_ema: bool = True
    ):
        """Save a full training checkpoint (model + optimizer + scheduler + RNG + EMA)."""
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
                save_file(
                    self._build_state_dict(),
                    os.path.join(output_dir, "model.safetensors"),
                )
            finally:
                if save_ema and self._ema is not None:
                    self._ema.restore()

            self._saved_checkpoints.append(output_dir)

    def save_model(self, output_dir: str):
        """Save final CM weights with EMA applied (no optimizer/scheduler state)."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            if self._ema is not None:
                self._ema.store()
                self._ema.copy_to()
            try:
                save_file(
                    self._build_state_dict(),
                    os.path.join(output_dir, "model.safetensors"),
                )
            finally:
                if self._ema is not None:
                    self._ema.restore()

    def save_non_ema_model(self, output_dir: str):
        """Save final raw (non-EMA) CM weights for continued training."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            save_file(
                self._build_state_dict(),
                os.path.join(output_dir, "model.safetensors"),
            )

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load full training state. Returns the global_step at the checkpoint."""
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
                str(d) for d in Path(self.cfg.output_dir).glob("checkpoint-*")
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
                d for d in Path(self.cfg.output_dir).glob("checkpoint-*")
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
