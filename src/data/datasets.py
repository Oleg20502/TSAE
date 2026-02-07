"""Dataset wrappers for text data."""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset, Dataset

from src.utils.config import DataConfig


def load_text_dataset(cfg: DataConfig) -> dict[str, Dataset]:
    """Load and prepare train/validation splits of a text dataset.

    For Wikipedia, each article is split into sentences (simple paragraph
    splitting).  The function returns a dict with ``"train"`` and ``"validation"``
    keys.

    Args:
        cfg: DataConfig with dataset name, column, max samples, etc.

    Returns:
        Dictionary with ``"train"`` and ``"validation"`` HuggingFace Datasets.
    """
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train", trust_remote_code=True)

    # If the text column contains long articles, split into sentences / paragraphs
    if cfg.dataset_name == "wikipedia":
        ds = _split_wiki_paragraphs(ds, text_column=cfg.text_column)

    # Shuffle and subsample
    ds = ds.shuffle(seed=cfg.seed)

    n_train = cfg.num_train_samples or len(ds)
    n_val = cfg.num_val_samples or 2000
    total_needed = n_train + n_val

    if total_needed > len(ds):
        total_needed = len(ds)
        n_val = min(n_val, total_needed // 10)
        n_train = total_needed - n_val

    train_ds = ds.select(range(n_train))
    val_ds = ds.select(range(n_train, n_train + n_val))

    return {"train": train_ds, "validation": val_ds}


def _split_wiki_paragraphs(ds: Dataset, text_column: str = "text") -> Dataset:
    """Split Wikipedia articles into individual paragraphs/sentences."""

    def _split(example):
        text = example[text_column]
        # Split on double newlines (paragraphs) then filter short ones
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
        return {text_column: paragraphs}

    ds = ds.map(_split, batched=False, remove_columns=ds.column_names)
    # Flatten the list column
    # After mapping, text_column contains lists -> use flatten-style select
    # Actually, datasets .map with lists creates nested; let's use a different approach
    all_texts = []
    for example in ds:
        texts = example[text_column]
        if isinstance(texts, list):
            all_texts.extend(texts)
        else:
            all_texts.append(texts)

    return Dataset.from_dict({text_column: all_texts})


def load_simple_text_dataset(texts: list[str], text_column: str = "text") -> Dataset:
    """Create a HuggingFace Dataset from a plain list of strings.

    Useful for testing and quick experiments.
    """
    return Dataset.from_dict({text_column: texts})
