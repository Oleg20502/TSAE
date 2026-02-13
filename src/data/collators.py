"""Data collators that tokenize and prepare batches for the RAE model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class ARDecoderCollator:
    """Collator for the autoencoder model with autregressive decoder.

    Tokenizes raw text and produces encoder inputs + decoder inputs (teacher
    forcing) in a single batch dict.

    The decoder uses the same tokenizer as the encoder.  Decoder inputs are
    the target tokens shifted right (prepended with BOS / [CLS]).
    Labels have padding positions set to -100 so they are ignored by CE loss.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128
    text_column: str = "text"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex[self.text_column] for ex in examples]

        # Tokenize for the encoder
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]            # (B, T)
        attention_mask = enc["attention_mask"]   # (B, T)

        # Decoder targets = same token sequence (autoencoder reconstruction)
        # decoder_input_ids: [BOS] + tokens[:-1]   (teacher forcing)
        # labels:            tokens                  (with pad -> -100)
        bos_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id or 101
        pad_id = self.tokenizer.pad_token_id or 0

        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        # Build decoder_input_ids: shift right, prepend BOS
        decoder_input_ids = torch.full_like(input_ids, pad_id)
        decoder_input_ids[:, 0] = bos_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]

        # Decoder attention mask: same as encoder (all real tokens + BOS are attended)
        decoder_attention_mask = attention_mask.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
