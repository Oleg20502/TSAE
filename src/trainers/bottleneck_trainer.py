"""Bottleneck AE Trainer: logs l_recon and l_sem for train and validation."""

import math

import torch
from transformers import Trainer


# For fast handling of logits using gpu
def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


class BottleneckTrainer(Trainer):
    """Trainer that logs l_recon and l_sem from the model output.

    Logs train_l_recon / train_l_sem at logging steps. During evaluation,
    accumulates l_recon and l_sem in the same pass as eval_loss and adds
    eval_l_recon / eval_l_sem to the metrics (no second eval pass).
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if model.training:
            # Log component losses at logging steps only
            if self.state.global_step % self.args.logging_steps == 0:
                logs = {}
                if "l_recon" in outputs:
                    logs["train_l_recon"] = outputs["l_recon"].item()
                if "l_sem" in outputs:
                    logs["train_l_sem"] = outputs["l_sem"].item()
                if logs:
                    self.log(logs)
        else:
            # Eval: accumulate for mean over validation set (single pass)
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

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Reset accumulators before the single eval pass
        self._eval_recon_sum = 0.0
        self._eval_sem_sum = 0.0
        self._eval_n = 0

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Add component losses from the same pass (no extra forward)
        if self._eval_n > 0:
            metrics[f"{metric_key_prefix}_l_recon"] = self._eval_recon_sum / self._eval_n
            metrics[f"{metric_key_prefix}_l_sem"] = self._eval_sem_sum / self._eval_n

        # Perplexity = exp(eval_loss)
        loss_key = f"{metric_key_prefix}_loss"
        if loss_key in metrics:
            metrics[f"{metric_key_prefix}_perplexity"] = math.exp(metrics[loss_key])

        return metrics
