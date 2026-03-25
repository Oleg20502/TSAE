"""Data utilities for Concept Model training.

``ChunkGroupDataset`` wraps a flat HuggingFace dataset of AE-length text
chunks and returns sequences of ``n_chunks`` consecutive chunks, preserving
local document context within a sequence.

``CMCollator`` tokenises each chunk with the AE tokenizer, builds
teacher-forced decoder inputs, and returns stacked ``(B, N, T)`` tensors
ready for the CM training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ChunkGroupDataset(Dataset):
    """Groups consecutive chunks from a flat HuggingFace dataset.

    ``len(hf_dataset) // n_chunks`` non-overlapping sequences are produced.
    Each ``__getitem__`` returns a list of ``n_chunks`` dataset rows.

    Args:
        hf_dataset: Flat HuggingFace ``Dataset`` of individual text chunks.
        n_chunks:   Number of consecutive chunks to group into one sequence.
        text_column: Name of the text column in ``hf_dataset``.
    """

    def __init__(self, hf_dataset, n_chunks: int, text_column: str = "text"):
        self.hf_dataset  = hf_dataset
        self.n_chunks    = n_chunks
        self.text_column = text_column
        self._len        = len(hf_dataset) // n_chunks

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> List[str]:
        start = idx * self.n_chunks
        return [
            self.hf_dataset[start + i][self.text_column]
            for i in range(self.n_chunks)
        ]


@dataclass
class CMCollator:
    """Tokenise a batch of chunk sequences for Concept Model training.

    Each batch element is a list of ``N`` text chunks (strings).  The
    collator tokenises every chunk independently, builds teacher-forced
    decoder inputs, and stacks everything into ``(B, N, T)`` tensors.

    Returned batch dict keys:
        - ``input_ids``          ``(B, N, T)`` encoder token ids
        - ``attention_mask``     ``(B, N, T)`` encoder mask (1 = real token)
        - ``decoder_input_ids``  ``(B, N, T)`` teacher-forced decoder inputs
        - ``labels``             ``(B, N, T)`` decoder targets (-100 at padding)

    Args:
        tokenizer:   AE tokenizer (same one used during AE training).
        max_length:  Token length of each chunk (``AE.max_length``).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(
        self, batch: List[List[str]]
    ) -> Dict[str, torch.Tensor]:
        B = len(batch)
        N = len(batch[0])

        # Flatten all chunks across the batch into one tokenizer call
        all_texts: List[str] = [chunk for seq in batch for chunk in seq]

        enc = self.tokenizer(
            all_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"]       # (B*N, T)
        attention_mask = enc["attention_mask"]  # (B*N, T)

        bos_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id or 101
        pad_id = self.tokenizer.pad_token_id or 0

        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        decoder_input_ids = torch.full_like(input_ids, pad_id)
        decoder_input_ids[:, 0] = bos_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]

        T = input_ids.size(1)
        return {
            "input_ids":         input_ids.view(B, N, T),
            "attention_mask":    attention_mask.view(B, N, T),
            "decoder_input_ids": decoder_input_ids.view(B, N, T),
            "labels":            labels.view(B, N, T),
        }
