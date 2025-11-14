"""Dynamic iterable dataset for curriculum training with phase-based sampling."""

import torch
from torch.utils.data import IterableDataset
from typing import Dict, Any, Iterator
import random


class CurriculumIterableDataset(IterableDataset):
    """Iterable dataset that dynamically samples based on curriculum phase."""

    def __init__(self, wrapper, tokenizer, max_seq_len: int):
        """
        Initialize dynamic curriculum dataset.

        Args:
            wrapper: CurriculumTrainingWrapper instance
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
        """
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Generate examples dynamically based on current phase."""
        while True:
            # Sample batch based on current phase ratios
            batch = self.wrapper.data_sampler.sample_batch(1)  # Sample 1 at a time

            if not batch:
                continue

            example = batch[0]
            text = example.get("text", "")
            source = example.get("source", "unknown")

            # Check if data is pre-tokenized
            if "input_ids" in example and "attention_mask" in example:
                # Use pre-tokenized data (fast path - RAM lookup only)
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
            else:
                # Tokenize on-the-fly (slow path - backward compatible)
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding=False,
                    return_tensors=None
                )
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

            # Build example dict - let collator handle masking dynamically
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "text": text,
                "source": source
            }

            # Pass through masking_regions if present (for next-utterance format)
            if "masking_regions" in example:
                result["masking_regions"] = example["masking_regions"]

            yield result
