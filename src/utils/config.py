"""Dataclass-based configuration with YAML loading."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, List

import yaml
from dacite import from_dict, Config as DaciteConfig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for the data pipeline."""

    dataset_name: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    text_column: str = "text"
    max_length: int = 128
    num_train_samples: Optional[int] = None  # None = use full split
    num_val_samples: Optional[int] = 2000
    seed: int = 42
    # HuggingFace datasets cache_dir for load_dataset (optional)
    cache_dir: Optional[str] = None
    # If set, load train/validation from this dir (from prepare_dataset script); no HF download or paragraph split at train time
    preprocessed_dir: Optional[str] = None
    # Used only by prepare_dataset: max articles to process before paragraph split (avoids processing full 6M+ wiki)
    max_samples: Optional[int] = 100_000
    # prepare_dataset: num_proc for ds.map (paragraph split); None = 1 (single process)
    prepare_num_proc: Optional[int] = None
    # batch size for preprocessing parallelization
    preprocess_batch_size: Optional[int] = 2000
    # prepare_dataset: preprocess mode — "paragraphs" (split by newlines) or "chunks" (fixed token length)
    preprocess_mode: str = "paragraphs"
    # When preprocess_mode == "chunks": chunk each document into fixed-length segments (GPT-2 tokens)
    chunk_size_tokens: Optional[int] = None
    # When preprocess_mode == "chunks": tokenizer name (e.g. "gpt2")
    gpt2_tokenizer_name: str = "gpt2"
    # When preprocess_mode == "chunks": drop last incomplete chunk per document
    drop_incomplete_chunks: bool = True

    # --- prepare_hybrid_lm_dataset.py (ignored by prepare_dataset.py) ---
    # Bottleneck experiment YAML: backbone tokenizer + model.max_length for AE slices.
    ae_config_path: Optional[str] = None
    max_latent_steps: Optional[int] = None
    prompt_min_len: int = 32
    prompt_max_len: int = 96
    completion_min_len: int = 1
    completion_max_len: int = 4
    tries_per_paragraph: int = 8
    # prepare_hybrid_lm_dataset: paragraphs per Pool chunk (early stop between chunks). None = script default.
    hybrid_build_chunk_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Configuration for autoencoder training."""

    output_dir: str = "outputs/ae"
    init_from_checkpoint: Optional[str] = None

    # Optimiser
    lr: float = 1e-4
    repr_lr: float = 1e-5
    weight_decay: float = 0.01

    # Schedule
    epochs: int = 5
    warmup_steps: int = 500
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3

    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None  # None → infer from metric name

    # Misc
    ema_decay: float = 0.999
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
    """Configuration for autoencoder evaluation."""

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

    d_model: int = 256
    max_length: int = 128
    n_latent_tokens: int = 1
    normalize_latent: bool = False
    
    # Encoder
    encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_ff_dim: int = 512
    encoder_dropout: float = 0.1

    # Decoder
    decoder_type: str = "autoregressive"  # "autoregressive" or "parallel"
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1

    # Latent augmentation
    noise_std: float = 0.0
    sigma_type: str = "abs" # "abs" or "rel" (absolute or relative to the norm of the latent)
    feature_dropout_p: float = 0.0

    # Loss weights
    lambda_sem: float = 0.2


# ---------------------------------------------------------------------------
# Top-level configs
# ---------------------------------------------------------------------------


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


def load_config(path: str | Path) -> BottleneckExperimentConfig:
    """Load a BottleneckExperimentConfig from a YAML file."""
    raw = load_yaml(path)
    return from_dict(data_class=BottleneckExperimentConfig, data=raw, config=_DACITE_CFG)


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


def load_config_from_paths(paths: list[str] | list[Path]) -> BottleneckExperimentConfig:
    """Load Bottleneck config from one or more YAML files.

    - Single path: file must contain full experiment config (model, train, data sections).
    - Multiple paths: files are merged by section (later overrides earlier), same as merge_bottleneck_configs.
    """
    if len(paths) == 1:
        return load_config(paths[0])
    return merge_bottleneck_configs(*paths)


def save_config(cfg: BottleneckExperimentConfig, path: str | Path) -> None:
    """Save experiment config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Concept Model configs
# ---------------------------------------------------------------------------

@dataclass
class ConceptDataConfig:
    """Configuration for the Concept Model data pipeline.

    Each CM training sequence consists of ``n_chunks`` consecutive 16-token
    chunks extracted from the same FineWeb document.  Preprocessing is done
    once by ``scripts/prepare_cm_dataset.py`` and the result is saved to
    ``preprocessed_dir``.  Training then loads directly from disk.
    """

    # ---- Source dataset (used by prepare_cm_dataset.py) ----
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"
    text_column: str = "text"
    cache_dir: Optional[str] = None        # HuggingFace datasets cache root

    # ---- Chunking ----
    n_chunks: int = 64                     # consecutive chunks per CM sequence
    chunk_size_tokens: int = 16            # GPT-2 tokens per chunk
    gpt2_tokenizer_name: str = "gpt2"
    drop_incomplete_chunks: bool = True

    # If set, training uses only the first ``use_n_chunks`` chunks per row (must be
    # ≤ stored length, e.g. 8 on a 32-chunk preprocessed dataset). None = full row.
    use_n_chunks: Optional[int] = None

    # ---- Pre-processed sequences (written by prepare_cm_dataset.py) ----
    preprocessed_dir: Optional[str] = None

    # ---- Dataset limits ----
    max_docs: Optional[int] = None         # cap documents processed (None = all)
    num_val_samples: Optional[int] = None
    seed: int = 42

    # ---- Processing ----
    prepare_num_proc: Optional[int] = None
    preprocess_batch_size: int = 1000


@dataclass
class ConceptModelConfig:
    """Configuration for the Concept Model."""

    # ---- Frozen AE to load ----
    ae_config_path: str = ""
    ae_checkpoint_path: str = ""

    # ---- CM type ----
    # "custom" → scratch transformer; anything else is a HuggingFace causal-LM
    # model name used as the backbone (e.g. "gpt2", "gpt2-medium").
    cm_type: str = "custom"

    # ---- Custom CM architecture ----
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.1

    # ---- Loss ----
    lambda_mse: float = 0.1


@dataclass
class ConceptExperimentConfig:
    """Top-level config for Concept Model training."""

    data: ConceptDataConfig = field(default_factory=ConceptDataConfig)
    model: ConceptModelConfig = field(default_factory=ConceptModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_concept_config(path: str | Path) -> ConceptExperimentConfig:
    """Load a ConceptExperimentConfig from a YAML file."""
    raw = load_yaml(path)
    return from_dict(data_class=ConceptExperimentConfig, data=raw, config=_DACITE_CFG)


def merge_concept_configs(*paths: str | Path) -> ConceptExperimentConfig:
    """Load and merge multiple YAML files for the Concept Model (later overrides earlier)."""
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
    return from_dict(data_class=ConceptExperimentConfig, data=merged, config=_DACITE_CFG)


def load_concept_config_from_paths(paths: List[str] | List[Path]) -> ConceptExperimentConfig:
    """Load ConceptExperimentConfig from one or more YAML files."""
    if len(paths) == 1:
        return load_concept_config(paths[0])
    return merge_concept_configs(*paths)


# ---------------------------------------------------------------------------
# Hybrid latent reasoning (GPT-2 + AE latents)
# ---------------------------------------------------------------------------


@dataclass
class HybridLatentDataConfig:
    dataset_name: str = "booydar/gsm8k"
    dataset_config: str = "default"
    task_column: str = "task"
    cot_column: str = "cot"
    labels_column: str = "labels"
    cache_dir: Optional[str] = None
    # When set, load train/validation from prepare_hybrid_lm_dataset output (GeneralHybridLatentCollator).
    preprocessed_dir: Optional[str] = None
    text_column: str = "text"
    prompt_min_len: int = 32
    prompt_max_len: int = 96
    completion_min_len: int = 1
    completion_max_len: int = 4
    reasoning_trigger: str = "!!!!"
    end_of_thinking_phrase: str = "end of thinking"
    gpt2_tokenizer_name: str = "openai-community/gpt2"
    max_prompt_tokens: int = 256
    max_answer_tokens: int = 128
    max_cot_steps: int = 16
    num_train_samples: Optional[int] = None
    num_val_samples: Optional[int] = None
    seed: int = 42


@dataclass
class HybridLatentModelConfig:
    ae_config_path: str = ""
    ae_checkpoint_path: str = ""
    pretrained_gpt2: str = "openai-community/gpt2"
    max_model_seq_len: Optional[int] = None
    lambda_mse: float = 0.1


@dataclass
class HybridLatentExperimentConfig:
    data: HybridLatentDataConfig = field(default_factory=HybridLatentDataConfig)
    model: HybridLatentModelConfig = field(default_factory=HybridLatentModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_hybrid_latent_config(path: str | Path) -> HybridLatentExperimentConfig:
    raw = load_yaml(path)
    return from_dict(data_class=HybridLatentExperimentConfig, data=raw, config=_DACITE_CFG)


def merge_hybrid_latent_configs(*paths: str | Path) -> HybridLatentExperimentConfig:
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
    return from_dict(data_class=HybridLatentExperimentConfig, data=merged, config=_DACITE_CFG)


def load_hybrid_latent_config_from_paths(paths: List[str] | List[Path]) -> HybridLatentExperimentConfig:
    if len(paths) == 1:
        return load_hybrid_latent_config(paths[0])
    return merge_hybrid_latent_configs(*paths)
