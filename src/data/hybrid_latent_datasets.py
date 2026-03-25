"""Hybrid latent datasets: GSM8K from HuggingFace or preprocessed LM (Wikipedia-style) from disk."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

from src.utils.config import HybridLatentDataConfig

_COT_STEP_RE = re.compile(r"<<([^>]+)>>")


def parse_cot_steps(cot: str) -> List[str]:
    return [s.strip() for s in _COT_STEP_RE.findall(cot or "") if s.strip()]


class HybridLatentGSM8KDataset(Dataset):
    """Rows from HuggingFace ``booydar/gsm8k``-style tables."""

    def __init__(self, hf_split, cfg: HybridLatentDataConfig):
        self.rows = hf_split
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {
            "task": row[self.cfg.task_column],
            "cot": row[self.cfg.cot_column],
            "labels": row[self.cfg.labels_column],
        }


def load_hybrid_latent_dataset(cfg: HybridLatentDataConfig) -> Dict[str, Union[HybridLatentGSM8KDataset, Any]]:
    if cfg.preprocessed_dir:
        root = Path(cfg.preprocessed_dir)
        train = load_from_disk(str(root / "train"))
        val = load_from_disk(str(root / "validation"))
        if cfg.num_train_samples is not None:
            n = min(cfg.num_train_samples, len(train))
            train = train.shuffle(seed=cfg.seed).select(range(n))
        if cfg.num_val_samples is not None:
            n = min(cfg.num_val_samples, len(val))
            val = val.shuffle(seed=cfg.seed + 1).select(range(n))
        return {"train": train, "validation": val}

    load_kw: Dict = {}
    if cfg.cache_dir:
        load_kw["cache_dir"] = cfg.cache_dir
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, **load_kw)
    train = ds["train"]
    val = ds.get("validation") or ds.get("test") or ds["train"]

    if cfg.num_train_samples is not None:
        n = min(cfg.num_train_samples, len(train))
        train = train.shuffle(seed=cfg.seed).select(range(n))
    if cfg.num_val_samples is not None:
        n = min(cfg.num_val_samples, len(val))
        val = val.shuffle(seed=cfg.seed + 1).select(range(n))

    return {
        "train": HybridLatentGSM8KDataset(train, cfg),
        "validation": HybridLatentGSM8KDataset(val, cfg),
    }
