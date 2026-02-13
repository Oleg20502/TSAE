"""Dataclass-based configuration with YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dacite import from_dict, Config as DaciteConfig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for the data pipeline."""

    dataset_name: str = "wikipedia"
    dataset_config: str = "20220301.en"
    text_column: str = "text"
    max_length: int = 128
    num_train_samples: Optional[int] = None  # None = use full split
    num_val_samples: Optional[int] = 2000
    seed: int = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the RAE-text model."""

    # Backbone
    backbone_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    freeze_repr: bool = True

    # Latent dimensions
    d_sem: int = 256
    d_det: int = 256
    n_detail_tokens: int = 8

    # Detail encoder
    detail_encoder_layers: int = 2
    detail_encoder_heads: int = 4

    # Decoder
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_dim: int = 256
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1
    max_decoder_length: int = 128

    # Loss weights
    lambda_sem: float = 0.2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for Stage-1 autoencoder training."""

    output_dir: str = "outputs/stage1"

    # Optimiser
    lr: float = 1e-4
    repr_lr: float = 1e-5
    weight_decay: float = 0.01

    # Schedule
    epochs: int = 5
    warmup_steps: int = 500
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3

    # Misc
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42
    report_to: str = "tensorboard"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for Stage-1 evaluation."""

    checkpoint_path: str = ""
    batch_size: int = 64
    max_samples: int = 1000
    num_qualitative: int = 20


# ---------------------------------------------------------------------------
# Bottleneck Model
# ---------------------------------------------------------------------------

@dataclass
class BottleneckModelConfig:
    """Configuration for the Bottleneck autoencoder model."""

    # Backbone (used only as semantic loss target)
    backbone_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    freeze_repr: bool = True

    # Encoder
    d_latent: int = 256
    n_latent_tokens: int = 1
    encoder_dim: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_ff_dim: int = 512
    encoder_dropout: float = 0.1
    max_encoder_length: int = 128

    # Decoder
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_dim: int = 256
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1
    max_decoder_length: int = 128

    # Latent augmentation
    noise_std: float = 0.0
    feature_dropout_p: float = 0.0

    # Loss weights
    lambda_sem: float = 0.2


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level config aggregating all sub-configs (RAE-text)."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


@dataclass
class BottleneckExperimentConfig:
    """Top-level config for the Bottleneck autoencoder."""

    data: DataConfig = field(default_factory=DataConfig)
    model: BottleneckModelConfig = field(default_factory=BottleneckModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

_DACITE_CFG = DaciteConfig(strict=False)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    raw = load_yaml(path)
    return from_dict(data_class=ExperimentConfig, data=raw, config=_DACITE_CFG)


def load_bottleneck_config(path: str | Path) -> BottleneckExperimentConfig:
    """Load a BottleneckExperimentConfig from a YAML file."""
    raw = load_yaml(path)
    return from_dict(data_class=BottleneckExperimentConfig, data=raw, config=_DACITE_CFG)


def merge_configs(*paths: str | Path) -> ExperimentConfig:
    """Load and merge multiple YAML files (later files override earlier)."""
    merged: dict = {}
    for p in paths:
        raw = load_yaml(p)
        for section, values in raw.items():
            if section not in merged:
                merged[section] = {}
            if isinstance(values, dict):
                merged[section].update(values)
            else:
                merged[section] = values
    return from_dict(data_class=ExperimentConfig, data=merged, config=_DACITE_CFG)


def merge_bottleneck_configs(*paths: str | Path) -> BottleneckExperimentConfig:
    """Load and merge multiple YAML files for the Bottleneck AE (later files override earlier)."""
    merged: dict = {}
    for p in paths:
        raw = load_yaml(p)
        for section, values in raw.items():
            if section not in merged:
                merged[section] = {}
            if isinstance(values, dict):
                merged[section].update(values)
            else:
                merged[section] = values
    return from_dict(data_class=BottleneckExperimentConfig, data=merged, config=_DACITE_CFG)
