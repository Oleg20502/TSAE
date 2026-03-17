"""Bottleneck AE Trainer: logs l_recon and l_sem for train and validation."""

import math
from typing import Optional

import torch
from torch_ema import ExponentialMovingAverage
from transformers import Trainer


# ---------------------------------------------------------------------------
# For fast handling of logits using gpu
# ---------------------------------------------------------------------------
def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids


class BottleneckTrainer(Trainer):
    """Trainer that logs l_recon and l_sem from the model output.

    Logs train_l_recon / train_l_sem at logging steps. During evaluation,
    accumulates l_recon and l_sem in the same pass as eval_loss and adds
    eval_l_recon / eval_l_sem to the metrics (no second eval pass).

    EMA: validation uses EMA weights (store → copy_to before eval, restore after).
    If validation loss is constant, ensure the model used in the eval loop is the
    same reference we update (unwrapped params at init); with FSDP/sharding, EMA
    may not apply to the eval model.
    """

    def __init__(self, *args, ema_decay: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema: Optional[ExponentialMovingAverage] = None
        if ema_decay is not None and ema_decay > 0.0:
            # Use unwrapped model so EMA tracks the same params after Trainer/Accelerator wrapping
            inner = getattr(self.model, "module", self.model)
            params = [p for p in inner.parameters() if p.requires_grad]
            if params:
                self._ema = ExponentialMovingAverage(params, decay=ema_decay)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            # Cache component losses for the *next* optimizer step.
            # During gradient accumulation, compute_loss is called multiple times
            # while global_step is still the previous optimizer step.
            step_key = int(self.state.global_step) + 1
            if not hasattr(self, "_train_component_loss_cache"):
                self._train_component_loss_cache = {}
            slot = self._train_component_loss_cache.setdefault(
                step_key,
                {
                    "l_recon_sum": 0.0,
                    "l_recon_n": 0,
                    "l_sem_sum": 0.0,
                    "l_sem_n": 0,
                },
            )
            if "l_recon" in outputs:
                slot["l_recon_sum"] += outputs["l_recon"].item()
                slot["l_recon_n"] += 1
            if "l_sem" in outputs:
                slot["l_sem_sum"] += outputs["l_sem"].item()
                slot["l_sem_n"] += 1
        else:
            # Eval: accumulate for mean over validation set
            if not hasattr(self, "_eval_recon_sum"):
                self._eval_recon_sum = 0.0
                self._eval_sem_sum = 0.0
                self._eval_n = 0
            if "l_recon" in outputs:
                self._eval_recon_sum += outputs["l_recon"].item()
            if "l_sem" in outputs:
                self._eval_sem_sum += outputs["l_sem"].item()
            self._eval_n += 1

        if return_outputs:
            return (loss, outputs)
        return loss

    def optimizer_step(
        self,
        *args,
        **kwargs,
    ):
        super().optimizer_step(*args, **kwargs)
        if self._ema is not None:
            self._ema.update()

    def log(self, logs, start_time=None):
        # Merge component losses into the main training log line so everything
        # (loss, grad_norm, lr, train_l_*) is emitted together.
        merged_logs = dict(logs)
        if "loss" in merged_logs and not any(k.startswith("eval_") for k in merged_logs):
            # Trainer accumulates loss as sum over micro-batches; report mean per step.
            merged_logs["loss"] = merged_logs["loss"] / self.args.gradient_accumulation_steps
            cache = getattr(self, "_train_component_loss_cache", None)
            if cache is not None:
                step_key = int(self.state.global_step)
                slot = cache.pop(step_key, None)
                if slot is not None:
                    if slot["l_recon_n"] > 0:
                        merged_logs["train_l_recon"] = slot["l_recon_sum"] / slot["l_recon_n"]
                    if slot["l_sem_n"] > 0:
                        merged_logs["train_l_sem"] = slot["l_sem_sum"] / slot["l_sem_n"]
        return super().log(merged_logs, start_time=start_time)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Reset accumulators before the eval pass
        self._eval_recon_sum = 0.0
        self._eval_sem_sum = 0.0
        self._eval_n = 0

        # Use EMA weights during evaluation if available (same model reference the parent will use)
        if self._ema is not None:
            self._ema.store()
            self._ema.copy_to()
        # Ensure eval mode; parent's loop also sets it, but we need it before parent reads self.model
        if hasattr(self.model, "eval") and callable(self.model.eval):
            self.model.eval()

        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self._ema is not None:
            self._ema.restore()

        # Add component losses and perplexity so they are included when evaluate() logs
        if self._eval_n > 0:
            output.metrics[f"{metric_key_prefix}_l_recon"] = self._eval_recon_sum / self._eval_n
            output.metrics[f"{metric_key_prefix}_l_sem"] = self._eval_sem_sum / self._eval_n
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in output.metrics:
            output.metrics[f"{metric_key_prefix}_perplexity"] = math.exp(output.metrics[loss_key])

        return output

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Save EMA weights if present
        if self._ema is None:
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

        self._ema.store()
        self._ema.copy_to()
        try:
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        finally:
            self._ema.restore()

    def save_non_ema_model(self, output_dir: Optional[str] = None):
        """Save the raw (non-EMA) model weights for continued fine-tuning."""
        return super().save_model(output_dir=output_dir, _internal_call=False)
