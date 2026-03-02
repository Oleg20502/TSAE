"""Custom Trainer subclasses for autoencoder training."""

from src.trainers.bottleneck_trainer import BottleneckTrainer, preprocess_logits_for_metrics

__all__ = ["BottleneckTrainer", "preprocess_logits_for_metrics"]
