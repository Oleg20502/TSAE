"""RAE-text Trainer: per-parameter-group learning rates for repr_encoder vs rest."""

from __future__ import annotations

import torch
from transformers import Trainer

from src.models.rae_text import RAEText


class RAETrainer(Trainer):
    """Thin Trainer subclass to set up per-parameter-group learning rates."""

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
