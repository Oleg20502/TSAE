"""Dataset wrappers for text data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk, Dataset

from src.utils.config import DataConfig

# Lazy GPT-2 tokenizer for FineWeb chunking (shared across calls)
_GPT2_TOKENIZER = None


def _get_gpt2_tokenizer(name: str = "gpt2"):
    global _GPT2_TOKENIZER
    if _GPT2_TOKENIZER is None:
        from transformers import AutoTokenizer
        _GPT2_TOKENIZER = AutoTokenizer.from_pretrained(name)
    return _GPT2_TOKENIZER


def chunk_text_by_gpt2_tokens(
    text: str,
    tokenizer,
    chunk_size: int,
    drop_incomplete: bool = True,
) -> list[str]:
    """Split a single text into chunks of fixed length in GPT-2 tokens.

    Args:
        text: Raw document text.
        tokenizer: GPT-2 tokenizer (or compatible).
        chunk_size: Number of tokens per chunk.
        drop_incomplete: If True, drop the last chunk when it has fewer than chunk_size tokens.

    Returns:
        List of decoded text chunks (each chunk has exactly chunk_size tokens, or fewer only if not drop_incomplete).
    """
    if not text or not text.strip():
        return []
    enc = tokenizer.encode(text, add_special_tokens=False)
    if len(enc) <= chunk_size and (drop_incomplete and len(enc) < chunk_size):
        return []
    chunks = []
    for start in range(0, len(enc), chunk_size):
        end = start + chunk_size
        if end > len(enc):
            if drop_incomplete:
                break
        chunk_ids = enc[start:end]
        if not chunk_ids:
            continue
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


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

    if "fineweb" in cfg.dataset_name.lower() and (cfg.chunk_size_tokens or 0) > 0:
        ds = _chunk_fineweb_batched(
            ds,
            text_column=cfg.text_column,
            chunk_size_tokens=cfg.chunk_size_tokens,
            tokenizer_name=cfg.gpt2_tokenizer_name,
            batch_size=500,
            num_proc=cfg.prepare_num_proc,
            drop_incomplete=True,
        )
    elif "wikipedia" in cfg.dataset_name:
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


def _chunk_fineweb_batched(
    ds: Dataset,
    text_column: str,
    chunk_size_tokens: int,
    tokenizer_name: str = "gpt2",
    batch_size: int = 500,
    num_proc: Optional[int] = None,
    drop_incomplete: bool = True,
) -> Dataset:
    """Split each document into fixed-length (GPT-2 token) chunks; one row per chunk."""
    tokenizer = _get_gpt2_tokenizer(tokenizer_name)

    def _chunk_batch(examples: dict) -> dict:
        out: list[list[str]] = []
        for text in examples[text_column]:
            if not isinstance(text, str):
                out.append([])
                continue
            chunks = chunk_text_by_gpt2_tokens(
                text, tokenizer, chunk_size_tokens, drop_incomplete=drop_incomplete
            )
            out.append(chunks)
        return {text_column: out}

    map_kwargs: dict = {
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": [c for c in ds.column_names if c != text_column],
        "desc": "Chunking by GPT-2 tokens",
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    ds = ds.map(_chunk_batch, **map_kwargs)
    all_texts = [chunk for row in ds for chunk in row[text_column]]
    return Dataset.from_dict({text_column: all_texts})


def load_simple_text_dataset(texts: list[str], text_column: str = "text") -> Dataset:
    """Create a HuggingFace Dataset from a plain list of strings.

    Useful for testing and quick experiments.
    """
    return Dataset.from_dict({text_column: texts})
