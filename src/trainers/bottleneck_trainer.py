"""Lightweight custom trainer for the Bottleneck autoencoder using Accelerate."""

import math
import os
import shutil
from dataclasses import asdict
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from transformers import get_constant_schedule_with_warmup

from src.losses.reconstruction import reconstruction_loss
from src.losses.semantic import semantic_consistency_loss
from src.utils.config import TrainConfig


# ---------------------------------------------------------------------------
# Trainable module (excludes repr_encoder)
# ---------------------------------------------------------------------------

class _TrainableCore(nn.Module):
    """Training-time module: encoder + decoder + latent_aug + sem_proj.

    repr_encoder is intentionally absent: it is frozen, contributes no
    gradients, and must not be saved to checkpoints.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_aug: nn.Module,
        sem_proj: Optional[nn.Linear],
        lambda_sem: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_aug = latent_aug
        self.sem_proj = sem_proj
        self.lambda_sem = lambda_sem

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        sent_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode → augment → decode → compute loss.

        Args:
            sent_emb: pre-computed sentence embedding from repr_encoder (detached).
                      When None the semantic loss term is skipped.
        """
        z = self.encoder(input_ids, attention_mask)       # (B, L, D)
        z_aug = self.latent_aug(z)                        # no-op during eval

        logits = self.decoder(
            latent_tokens=z_aug,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        l_recon = reconstruction_loss(logits, labels)

        if sent_emb is not None and self.sem_proj is not None:
            z_sem = self.sem_proj(z.mean(dim=1))          # (B, D_s)
            l_sem = semantic_consistency_loss(z_sem, sent_emb)
            loss = l_recon + self.lambda_sem * l_sem
            return {
                "loss": loss,
                "logits": logits,
                "l_recon": l_recon.detach(),
                "l_sem": l_sem.detach(),
            }
        else:
            return {
                "loss": l_recon,
                "logits": logits,
                "l_recon": l_recon.detach(),
            }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BottleneckTrainer:
    """Lightweight Accelerate-based trainer for the Bottleneck autoencoder.

    Takes encoder, decoder, and repr_encoder as separate arguments.
    repr_encoder is used only to produce frozen semantic targets; it is
    never wrapped by Accelerate, never optimized, and never saved.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        repr_encoder: Optional[nn.Module],
        latent_aug: nn.Module,
        lambda_sem: float,
        train_dataset,
        eval_dataset,
        data_collator,
        train_config: TrainConfig,
        compute_metrics: Optional[Callable] = None,
    ):
        self.cfg = train_config
        self.compute_metrics_fn = compute_metrics

        # sem_proj lives here: it knows encoder dim and repr dim, both training-only
        sem_proj: Optional[nn.Linear] = None
        if repr_encoder is not None:
            sem_proj = nn.Linear(encoder.d_model, repr_encoder.sent_dim)

        self._core = _TrainableCore(encoder, decoder, latent_aug, sem_proj, lambda_sem)

        # repr_encoder is kept separate — frozen, on device, excluded from saves
        self._repr_encoder = repr_encoder

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

        # Accelerate wraps core, optimizer, dataloaders, scheduler — NOT repr_encoder
        (
            self._core,
            self._optimizer,
            self._train_dl,
            self._eval_dl,
            self._scheduler,
        ) = self.accelerator.prepare(self._core, optimizer, train_dl, eval_dl, scheduler)

        # Move repr_encoder to the same device but keep it outside Accelerate
        if self._repr_encoder is not None:
            self._repr_encoder = self._repr_encoder.to(self.accelerator.device)
            self._repr_encoder.eval()
            for p in self._repr_encoder.parameters():
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

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _get_sent_emb(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Run the frozen repr_encoder to get sentence embeddings (no_grad)."""
        if self._repr_encoder is None:
            return None
        with torch.no_grad():
            return self._repr_encoder.encode(batch["input_ids"], batch["attention_mask"])

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sent_emb = self._get_sent_emb(batch)
        return self._core(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
            decoder_attention_mask=batch.get("decoder_attention_mask"),
            sent_emb=sent_emb,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        self.accelerator.init_trackers("bottleneck_ae", config=asdict(cfg))

        global_step = 0
        log_loss_sum = log_l_recon_sum = log_l_sem_sum = 0.0
        log_micro_steps = 0

        for epoch in range(cfg.epochs):
            self._core.train()

            for batch in self._train_dl:
                with self.accelerator.accumulate(self._core):
                    outputs = self._forward(batch)
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self._core.parameters(), cfg.max_grad_norm
                        )

                    self._optimizer.step()
                    self._scheduler.step()
                    self._optimizer.zero_grad()

                # Accumulate logging values at micro-step granularity
                log_loss_sum += loss.detach().item()
                log_l_recon_sum += outputs["l_recon"].item()
                if "l_sem" in outputs:
                    log_l_sem_sum += outputs["l_sem"].item()
                log_micro_steps += 1

                if self.accelerator.sync_gradients:
                    if self._ema is not None:
                        self._ema.update()
                    global_step += 1

                    if global_step % cfg.logging_steps == 0:
                        self._log_train(
                            global_step, epoch,
                            log_loss_sum, log_l_recon_sum, log_l_sem_sum, log_micro_steps,
                        )
                        log_loss_sum = log_l_recon_sum = log_l_sem_sum = 0.0
                        log_micro_steps = 0

                    if global_step % cfg.eval_steps == 0:
                        self.evaluate(step=global_step)
                        self._core.train()

                    if global_step % cfg.save_steps == 0:
                        ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        self.save_checkpoint(ckpt_dir)
                        self._rotate_checkpoints()

        self.accelerator.end_training()

    def _log_train(
        self,
        step: int,
        epoch: int,
        loss_sum: float,
        l_recon_sum: float,
        l_sem_sum: float,
        n: int,
    ):
        n = max(n, 1)
        log_dict = {
            "train/loss": loss_sum / n,
            "train/l_recon": l_recon_sum / n,
            "train/lr": self._scheduler.get_last_lr()[0],
            "epoch": epoch,
        }
        if self._repr_encoder is not None:
            log_dict["train/l_sem"] = l_sem_sum / n

        self.accelerator.log(log_dict, step=step)

        if self.accelerator.is_main_process:
            sem_str = f"  l_sem={l_sem_sum/n:.4f}" if self._repr_encoder else ""
            print(
                f"step={step}  loss={loss_sum/n:.4f}"
                f"  l_recon={l_recon_sum/n:.4f}{sem_str}"
                f"  lr={self._scheduler.get_last_lr()[0]:.2e}"
            )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self, step: Optional[int] = None) -> Dict[str, float]:
        self._core.eval()

        if self._ema is not None:
            self._ema.store()
            self._ema.copy_to()

        # Tensor accumulators (allows correct all-reduce across GPUs)
        device = self.accelerator.device
        loss_acc = torch.zeros(1, device=device)
        l_recon_acc = torch.zeros(1, device=device)
        l_sem_acc = torch.zeros(1, device=device)
        count = torch.zeros(1, device=device)

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        try:
            for batch in self._eval_dl:
                with torch.no_grad():
                    outputs = self._forward(batch)

                loss_acc += outputs["loss"].detach()
                l_recon_acc += outputs["l_recon"].detach()
                if "l_sem" in outputs:
                    l_sem_acc += outputs["l_sem"].detach()
                count += 1

                if self.compute_metrics_fn is not None:
                    pred_ids = outputs["logits"].argmax(dim=-1)  # (B, T)
                    pred_ids = self.accelerator.gather_for_metrics(pred_ids)
                    labels = self.accelerator.gather_for_metrics(batch["labels"])
                    if self.accelerator.is_main_process:
                        all_preds.append(pred_ids.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
        finally:
            if self._ema is not None:
                self._ema.restore()

        # Average across all processes
        loss_acc = self.accelerator.reduce(loss_acc, reduction="sum")
        l_recon_acc = self.accelerator.reduce(l_recon_acc, reduction="sum")
        l_sem_acc = self.accelerator.reduce(l_sem_acc, reduction="sum")
        count = self.accelerator.reduce(count, reduction="sum")

        n = count.item()
        eval_loss = (loss_acc / count).item()

        metrics: Dict[str, float] = {
            "eval/loss": eval_loss,
            "eval/l_recon": (l_recon_acc / count).item(),
            "eval/perplexity": math.exp(min(eval_loss, 20)),
        }
        if self._repr_encoder is not None:
            metrics["eval/l_sem"] = (l_sem_acc / count).item()

        if self.compute_metrics_fn is not None and self.accelerator.is_main_process and all_preds:
            predictions = np.concatenate(all_preds, axis=0)
            label_ids = np.concatenate(all_labels, axis=0)
            ep = SimpleNamespace(predictions=predictions, label_ids=label_ids)
            extra = self.compute_metrics_fn(ep)
            metrics.update({f"eval/{k}": v for k, v in extra.items()})

        self.accelerator.log(metrics, step=step)

        if self.accelerator.is_main_process:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[eval step={step}]  {parts}")

        return metrics

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _build_state_dict(self) -> Dict[str, torch.Tensor]:
        """Collect encoder + decoder + sem_proj weights.

        repr_encoder is intentionally excluded.

        The decoder's lm_head.weight is dropped when it is the same
        storage as tok_emb.weight so safetensors does not raise a
        duplicate-pointer error. load_bottleneck_model re-ties the
        weights on load.
        """
        core = self.accelerator.unwrap_model(self._core)
        sd: Dict[str, torch.Tensor] = {}

        for k, v in core.encoder.state_dict().items():
            sd[f"encoder.{k}"] = v

        decoder_sd = core.decoder.state_dict()
        if (
            hasattr(core.decoder, "lm_head")
            and hasattr(core.decoder, "tok_emb")
            and core.decoder.lm_head.weight is core.decoder.tok_emb.weight
        ):
            del decoder_sd["lm_head.weight"]
        for k, v in decoder_sd.items():
            sd[f"decoder.{k}"] = v

        if core.sem_proj is not None:
            for k, v in core.sem_proj.state_dict().items():
                sd[f"sem_proj.{k}"] = v

        return sd

    def save_checkpoint(self, output_dir: str, save_ema: bool = True):
        """Save a checkpoint to output_dir/model.safetensors.

        With save_ema=True (default) the EMA weights are written.
        """
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        os.makedirs(output_dir, exist_ok=True)

        if save_ema and self._ema is not None:
            self._ema.store()
            self._ema.copy_to()
        try:
            save_file(self._build_state_dict(), os.path.join(output_dir, "model.safetensors"))
        finally:
            if save_ema and self._ema is not None:
                self._ema.restore()

        self._saved_checkpoints.append(output_dir)

    def save_model(self, output_dir: str, tokenizer=None):
        """Save final model with EMA weights applied."""
        self.save_checkpoint(output_dir, save_ema=True)
        if tokenizer is not None and self.accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    def save_non_ema_model(self, output_dir: str, tokenizer=None):
        """Save raw (non-EMA) weights for continued training."""
        self.save_checkpoint(output_dir, save_ema=False)
        if tokenizer is not None and self.accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    def _rotate_checkpoints(self):
        """Remove the oldest checkpoint once save_total_limit is exceeded."""
        limit = self.cfg.save_total_limit
        if not limit or limit <= 0:
            return
        while len(self._saved_checkpoints) > limit:
            oldest = self._saved_checkpoints.pop(0)
            if self.accelerator.is_main_process and os.path.isdir(oldest):
                shutil.rmtree(oldest)
