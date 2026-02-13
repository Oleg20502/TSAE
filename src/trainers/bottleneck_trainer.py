"""Bottleneck AE Trainer: logs l_recon and l_sem from model output."""

from __future__ import annotations

from transformers import Trainer


class BottleneckTrainer(Trainer):
    """Trainer that logs l_recon and l_sem from the model output.

    The repr_encoder is always frozen, so no separate parameter group.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Log component losses at logging steps (same keys as TensorBoard)
        if self.state.global_step % self.args.logging_steps == 0:
            logs = {}
            if "l_recon" in outputs:
                logs["train_l_recon"] = outputs["l_recon"].item()
            if "l_sem" in outputs:
                logs["train_l_sem"] = outputs["l_sem"].item()
            if logs:
                self.log(logs)

        if return_outputs:
            return (loss, outputs)
        return loss
