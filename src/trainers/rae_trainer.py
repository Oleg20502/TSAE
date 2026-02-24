"""RAE-text Trainer: per-parameter-group learning rates for repr_encoder vs rest."""

from __future__ import annotations

import torch
from transformers import Trainer

from src.models.rae_text import RAEText


class RAETrainer(Trainer):
    """Thin Trainer subclass to set up per-parameter-group learning rates."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            # Cache component losses for the *next* optimizer step so logs are
            # consistent under gradient accumulation.
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

        if return_outputs:
            return (loss, outputs)
        return loss

    def log(self, logs, start_time=None):
        # Emit train_l_recon/train_l_sem together with the Trainer's main log
        # line (loss, grad_norm, lr, epoch).
        merged_logs = dict(logs)
        if "loss" in merged_logs and not any(k.startswith("eval_") for k in merged_logs):
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

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model: RAEText = self.model
        repr_lr = getattr(self, "_repr_lr", 1e-5)
        base_lr = self.args.learning_rate

        # Separate repr encoder params (if any are trainable) from the rest
        repr_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("repr_encoder."):
                repr_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": base_lr},
        ]
        if repr_params:
            param_groups.append({"params": repr_params, "lr": repr_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer
