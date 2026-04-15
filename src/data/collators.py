"""Data collators that tokenize and prepare batches for the RAE model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from src.utils.config import BottleneckExperimentConfig


@dataclass
class ARDecoderCollator:
    """Collator for the autoencoder model with autoregressive decoder.

    Tokenizes raw text and produces encoder inputs + decoder inputs (teacher
    forcing) in a single batch dict.

    The decoder uses the same tokenizer as the encoder.  Decoder inputs are
    the target tokens shifted right (prepended with BOS / [CLS]).
    Labels have padding positions set to -100 so they are ignored by CE loss.
    """

    def __init__(
        self,
        ae_tokenizer: PreTrainedTokenizerBase,
        embedder_tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_column: str = "text",
    ):
        self.ae_tokenizer = ae_tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.max_length = max_length
        self.text_column = text_column
        # GPT-style tokenizers often have no explicit PAD token configured.
        if self.ae_tokenizer.pad_token_id is None:
            if self.ae_tokenizer.eos_token is not None:
                self.ae_tokenizer.pad_token = self.ae_tokenizer.eos_token
            elif self.ae_tokenizer.bos_token is not None:
                self.ae_tokenizer.pad_token = self.ae_tokenizer.bos_token

        self._pad_id = (
            self.ae_tokenizer.pad_token_id
            if self.ae_tokenizer.pad_token_id is not None
            else 0
        )
        self._bos_id = next(
            (
                tok_id
                for tok_id in (
                    self.ae_tokenizer.bos_token_id,
                    self.ae_tokenizer.cls_token_id,
                    self.ae_tokenizer.eos_token_id,
                    self.ae_tokenizer.pad_token_id,
                )
                if tok_id is not None
            ),
            0,
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex[self.text_column] for ex in examples]

        # Tokenize for the encoder
        enc = self.ae_tokenizer(
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
        labels = input_ids.clone()
        labels[labels == self._pad_id] = -100

        # Build decoder_input_ids: shift right, prepend BOS
        decoder_input_ids = torch.full_like(input_ids, self._pad_id)
        decoder_input_ids[:, 0] = self._bos_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]

        # Decoder attention mask: same as encoder (all real tokens + BOS are attended)
        decoder_attention_mask = attention_mask.clone()
        decoder_attention_mask[:, 0] = 1

        texts_for_embedder = self.ae_tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)

        embedder_enc = self.embedder_tokenizer(
            texts_for_embedder,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        embedder_input_ids = embedder_enc["input_ids"]            # (B, T)
        embedder_attention_mask = embedder_enc["attention_mask"]   # (B, T)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "embedder_input_ids": embedder_input_ids,
            "embedder_attention_mask": embedder_attention_mask,
        }

@dataclass
class ParallelDecoderCollator:
    """Collator for the autoencoder model with parallel decoder.

    Tokenizes raw text and produces encoder inputs in a single batch dict.
    """

    def __init__(
        self,
        ae_tokenizer: PreTrainedTokenizerBase,
        embedder_tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_column: str = "text",
    ):
        self.ae_tokenizer = ae_tokenizer
        self.embedder_tokenizer = embedder_tokenizer
        self.max_length = max_length
        self.text_column = text_column
        if self.ae_tokenizer.pad_token_id is None:
            if self.ae_tokenizer.eos_token is not None:
                self.ae_tokenizer.pad_token = self.ae_tokenizer.eos_token
            elif self.ae_tokenizer.bos_token is not None:
                self.ae_tokenizer.pad_token = self.ae_tokenizer.bos_token
        self._pad_id = (
            self.ae_tokenizer.pad_token_id
            if self.ae_tokenizer.pad_token_id is not None
            else 0
        )

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex[self.text_column] for ex in examples]

        # Tokenize for the encoder
        enc = self.ae_tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        texts_for_embedder = self.ae_tokenizer.batch_decode(enc["input_ids"], skip_special_tokens=True)

        embedder_enc = self.embedder_tokenizer(
            texts_for_embedder,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]            # (B, T)
        attention_mask = enc["attention_mask"]   # (B, T)
        
        labels = input_ids.clone()
        labels[labels == self._pad_id] = -100

        embedder_input_ids = embedder_enc["input_ids"]            # (B, T)
        embedder_attention_mask = embedder_enc["attention_mask"]   # (B, T)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # Kept for API compatibility with BottleneckTrainer/_core forward.
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
            "embedder_input_ids": embedder_input_ids,
            "embedder_attention_mask": embedder_attention_mask,
            "labels": labels,
        }


def build_collator(
    ae_tokenizer: PreTrainedTokenizerBase,
    embedder_tokenizer: PreTrainedTokenizerBase,
    cfg: BottleneckExperimentConfig,
) -> ARDecoderCollator | ParallelDecoderCollator:
    if cfg.model.decoder_type == "autoregressive":
        return ARDecoderCollator(
            ae_tokenizer,
            embedder_tokenizer,
            cfg.model.max_length,
            cfg.data.text_column,
        )
    elif cfg.model.decoder_type == "parallel":
        return ParallelDecoderCollator(
            ae_tokenizer,
            embedder_tokenizer,
            cfg.model.max_length,
            cfg.data.text_column,
        )
    else:
        raise ValueError(f"Unknown decoder_type: {cfg.model.decoder_type}")