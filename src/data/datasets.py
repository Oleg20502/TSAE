"""Dataset wrappers for text data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset

from src.utils.config import DataConfig


def load_text_dataset(cfg: DataConfig) -> dict[str, Dataset]:
    """Load train/validation splits.

    If cfg.preprocessed_dir is set, loads from disk (output of prepare_dataset script).
    Otherwise loads from HuggingFace, runs paragraph split for Wikipedia, shuffle, and subsample.
    """
    if cfg.preprocessed_dir is not None and (Path(cfg.preprocessed_dir) / "train").exists():
        train_ds = load_from_disk(str(Path(cfg.preprocessed_dir) / "train"))
        val_ds = load_from_disk(str(Path(cfg.preprocessed_dir) / "validation"))
        return {"train": train_ds, "validation": val_ds}

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train", trust_remote_code=True)

    if "wikipedia" in cfg.dataset_name:
        ds = _split_wiki_paragraphs(ds, text_column=cfg.text_column)

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
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
        return {text_column: paragraphs}

    ds = ds.map(_split, batched=False, remove_columns=ds.column_names)
    all_texts = []
    for example in ds:
        texts = example[text_column]
        if isinstance(texts, list):
            all_texts.extend(texts)
        else:
            all_texts.append(texts)
    return Dataset.from_dict({text_column: all_texts})


def _split_wiki_paragraphs_batched(
    ds: Dataset,
    text_column: str = "text",
    batch_size: int = 2000,
    num_proc: Optional[int] = None,
) -> Dataset:
    """Paragraph split in batches. Returns one row per article with list of paragraphs; we flatten to one row per paragraph (works with num_proc)."""
    def _split_batch(examples: dict) -> dict:
        out: list[list[str]] = []
        for text in examples[text_column]:
            if not isinstance(text, str):
                out.append([])
                continue
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
            out.append(paragraphs)
        return {text_column: out}

    map_kwargs: dict = {
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": [c for c in ds.column_names if c != text_column],
        "desc": "Splitting paragraphs",
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    ds = ds.map(_split_batch, **map_kwargs)
    # Flatten: each row is list of paragraphs
    all_texts = [p for row in ds for p in row[text_column]]
    return Dataset.from_dict({text_column: all_texts})


def load_simple_text_dataset(texts: list[str], text_column: str = "text") -> Dataset:
    """Create a HuggingFace Dataset from a plain list of strings.

    Useful for testing and quick experiments.
    """
    return Dataset.from_dict({text_column: texts})
