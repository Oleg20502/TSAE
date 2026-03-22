"""CM-specific dataset utilities.

Each CM training sequence is a list of ``n_chunks`` consecutive text chunks
that all come from the same FineWeb document.  Chunks are defined by GPT-2
token boundaries (``chunk_size_tokens`` tokens each) so that their length is
consistent with the AE's ``max_length``.

Workflow
--------
1. Run ``scripts/prepare_cm_dataset.py`` once to build and save the sequences.
2. During training, ``load_cm_dataset()`` loads from the saved directory and
   wraps it in ``CMSequenceDataset``.

Each row of the saved HuggingFace Dataset has a single column ``"chunks"``
containing a ``List[str]`` of length ``n_chunks``.  The ``CMCollator`` in
``concept_collators.py`` handles tokenisation.
"""

from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset as HFDataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.datasets import _get_gpt2_tokenizer, chunk_text_by_gpt2_tokens
from src.utils.config import ConceptDataConfig


# ---------------------------------------------------------------------------
# Map-style Dataset wrapper
# ---------------------------------------------------------------------------

class CMSequenceDataset(Dataset):
    """Map-style dataset where each item is a ``List[str]`` of ``n_chunks`` chunks.

    Wraps a HuggingFace ``Dataset`` that has a ``"chunks"`` column produced by
    ``build_cm_sequences()``.
    """

    def __init__(self, hf_dataset: HFDataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> List[str]:
        return self.data[idx]["chunks"]


# ---------------------------------------------------------------------------
# Sequence extraction helpers
# ---------------------------------------------------------------------------

def _extract_sequences_from_batch(
    examples: dict,
    tokenizer,
    n_chunks: int,
    chunk_size: int,
    text_column: str,
    drop_incomplete: bool,
) -> dict:
    """Batched HF map function: documents → sequences of ``n_chunks`` chunks.

    For each document, tokenizes with the GPT-2 tokenizer, cuts into chunks of
    ``chunk_size`` tokens (optionally dropping the final incomplete chunk), then
    produces all non-overlapping windows of ``n_chunks`` consecutive chunks.

    Returns a dict with a ``"chunks"`` key whose value is a flat list of
    sequences — one row per sequence.  Because the output can have more rows
    than the input, HuggingFace handles the size change automatically when
    ``batched=True``.
    """
    all_sequences: List[List[str]] = []
    for text in examples[text_column]:
        if not isinstance(text, str) or not text.strip():
            continue
        chunks = chunk_text_by_gpt2_tokens(
            text, tokenizer, chunk_size, drop_incomplete=drop_incomplete
        )
        # Non-overlapping windows of n_chunks consecutive chunks
        for start in range(0, len(chunks) - n_chunks + 1, n_chunks):
            all_sequences.append(chunks[start : start + n_chunks])
    return {"chunks": all_sequences}


# ---------------------------------------------------------------------------
# Build sequences from FineWeb
# ---------------------------------------------------------------------------

def build_cm_sequences(cfg: ConceptDataConfig) -> Dict[str, HFDataset]:
    """Load FineWeb and extract CM training sequences.

    For every document in FineWeb, chunks it into ``chunk_size_tokens``-token
    pieces (GPT-2 tokens) and groups consecutive chunks into sequences of
    ``n_chunks``.  Returns ``{"train": HFDataset, "validation": HFDataset}``
    where each row has a ``"chunks"`` column (``List[str]``).

    The returned datasets can be saved to disk with ``save_to_disk()`` and
    reloaded with ``load_from_disk()``.
    """
    from datasets import load_dataset

    print(f"Loading {cfg.dataset_name} / {cfg.dataset_config} from HuggingFace …")
    load_kwargs: dict = {"split": "train", "trust_remote_code": True}
    if cfg.cache_dir:
        load_kwargs["cache_dir"] = cfg.cache_dir

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, **load_kwargs)

    if cfg.max_docs is not None:
        n_take = min(cfg.max_docs, len(ds))
        ds = ds.shuffle(seed=cfg.seed).select(range(n_take))
        print(f"Subsampled to {n_take:,} documents.")
    else:
        ds = ds.shuffle(seed=cfg.seed)
        print(f"Loaded {len(ds):,} documents.")

    tokenizer = _get_gpt2_tokenizer(cfg.gpt2_tokenizer_name)
    n_chunks   = cfg.n_chunks
    chunk_size = cfg.chunk_size_tokens

    def extract_fn(examples: dict) -> dict:
        return _extract_sequences_from_batch(
            examples,
            tokenizer=tokenizer,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            text_column=cfg.text_column,
            drop_incomplete=cfg.drop_incomplete_chunks,
        )

    map_kwargs: dict = {
        "batched": True,
        "batch_size": cfg.preprocess_batch_size,
        "remove_columns": ds.column_names,
        "desc": f"Extracting {n_chunks}-chunk sequences",
    }
    if cfg.prepare_num_proc and cfg.prepare_num_proc > 1:
        map_kwargs["num_proc"] = cfg.prepare_num_proc

    processed = ds.map(extract_fn, **map_kwargs)
    n_total = len(processed)
    print(f"Extracted {n_total:,} sequences ({n_chunks} chunks × {chunk_size} GPT-2 tokens each).")

    # Split: last num_val_samples rows go to validation
    n_val = min(cfg.num_val_samples or 10000, n_total // 100)
    n_train = n_total - n_val
    train_ds = processed.select(range(n_train))
    val_ds   = processed.select(range(n_train, n_total))
    print(f"Train: {len(train_ds):,}  Validation: {len(val_ds):,}")

    return {"train": train_ds, "validation": val_ds}


# ---------------------------------------------------------------------------
# Loader used by the training script
# ---------------------------------------------------------------------------

def load_cm_dataset(cfg: ConceptDataConfig) -> Dict[str, CMSequenceDataset]:
    """Load CM training sequences from a pre-processed directory.

    Expects ``preprocessed_dir/train`` and ``preprocessed_dir/validation``
    subdirectories written by ``scripts/prepare_cm_dataset.py``.

    Returns:
        ``{"train": CMSequenceDataset, "validation": CMSequenceDataset}``
    """
    if not cfg.preprocessed_dir:
        raise ValueError(
            "ConceptDataConfig.preprocessed_dir must be set. "
            "Run scripts/prepare_cm_dataset.py first."
        )
    base = Path(cfg.preprocessed_dir)
    train_hf = load_from_disk(str(base / "train"))
    val_hf   = load_from_disk(str(base / "validation"))
    return {
        "train":      CMSequenceDataset(train_hf),
        "validation": CMSequenceDataset(val_hf),
    }
