"""
Data collator that uses masking_regions from the messages_to_sft converter.
Handles the formatted text with precise character-based masking regions.
"""

from typing import Dict, List, Any, Optional
import torch
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForMaskingRegions:
    """
    Data collator that applies masking based on masking_regions field.

    Expects data in format from messages_to_sft.py converter:
    {
        "text": "formatted SFT text",
        "masking_regions": [
            {"start": 0, "end": 100, "mask": true, "role": "system"},
            {"start": 100, "end": 200, "mask": false, "role": "assistant"},
            ...
        ]
    }

    The collator:
    1. Tokenizes the text
    2. Maps character positions to token positions
    3. Applies -100 masking to specified regions
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = None
    ignore_index: int = -100

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with masking regions.

        Args:
            features: List of examples with 'text' and 'masking_regions' fields

        Returns:
            Dict with input_ids, attention_mask, and labels tensors
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            # Get text and masking regions
            text = feature.get('text', '')
            masking_regions = feature.get('masking_regions', [])

            # Tokenize the text with offset mapping to track char->token mapping
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True
            )

            input_ids = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']

            # For autoregressive LM: labels are input_ids shifted left by 1
            # Position i predicts token at position i+1
            labels = input_ids[1:] + [self.tokenizer.eos_token_id]

            # Create a mask array for which positions to train on
            # Start with all positions trainable
            train_mask = [True] * len(labels)

            # Apply masking based on regions
            for region in masking_regions:
                if region['mask']:
                    # Find tokens that overlap with this character range
                    start_char = region['start']
                    end_char = region['end']

                    # Find token indices that fall within the character range
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        # Skip special tokens (they have (0, 0) offsets)
                        if token_start == 0 and token_end == 0 and token_idx > 0:
                            continue

                        # Check if this token overlaps with the masking region
                        if token_start < end_char and token_end > start_char:
                            # We don't want to train on predicting this token
                            # Label position for this token is token_idx - 1
                            # (since labels are shifted)
                            if token_idx > 0 and token_idx - 1 < len(train_mask):
                                train_mask[token_idx - 1] = False

            # Apply the mask to labels
            for i, should_mask in enumerate(train_mask):
                if not should_mask:
                    labels[i] = self.ignore_index

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        # Find max length in batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_batch_len = (
                (max_batch_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad all sequences (left padding for Flash Attention compatibility)
        for i in range(len(batch_input_ids)):
            padding_length = max_batch_len - len(batch_input_ids[i])
            if padding_length > 0:
                # Left padding: add padding at the beginning
                batch_input_ids[i] = [self.tokenizer.pad_token_id] * padding_length + batch_input_ids[i]
                batch_labels[i] = [self.ignore_index] * padding_length + batch_labels[i]
                batch_attention_mask[i] = [0] * padding_length + batch_attention_mask[i]

        # Convert to tensors
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long)
        }

    def validate_masking(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate masking on a single example for debugging.

        Args:
            feature: Example with text and masking_regions

        Returns:
            Dict with validation statistics
        """
        text = feature.get('text', '')
        masking_regions = feature.get('masking_regions', [])

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True
        )

        input_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']
        labels = input_ids.copy()

        # Track which roles are masked/unmasked
        masked_tokens_by_role = {}
        unmasked_tokens_by_role = {}

        # Apply masking and track statistics
        for region in masking_regions:
            role = region['role']
            start_char = region['start']
            end_char = region['end']
            should_mask = region['mask']

            # Count tokens in this region
            region_tokens = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end_char and token_end > start_char:
                    region_tokens.append(token_idx)
                    if should_mask:
                        labels[token_idx] = self.ignore_index

            # Track statistics
            if should_mask:
                masked_tokens_by_role[role] = masked_tokens_by_role.get(role, 0) + len(region_tokens)
            else:
                unmasked_tokens_by_role[role] = unmasked_tokens_by_role.get(role, 0) + len(region_tokens)

        # Calculate overall statistics
        total_tokens = len(input_ids)
        total_masked = sum(1 for label in labels if label == self.ignore_index)
        total_unmasked = total_tokens - total_masked

        # Decode unmasked tokens for preview
        unmasked_token_ids = [input_ids[i] for i in range(len(input_ids)) if labels[i] != self.ignore_index]
        unmasked_text = self.tokenizer.decode(unmasked_token_ids, skip_special_tokens=True)

        return {
            'total_tokens': total_tokens,
            'total_masked': total_masked,
            'total_unmasked': total_unmasked,
            'masking_ratio': total_masked / total_tokens if total_tokens > 0 else 0,
            'masked_tokens_by_role': masked_tokens_by_role,
            'unmasked_tokens_by_role': unmasked_tokens_by_role,
            'unmasked_text_preview': unmasked_text[:200] + '...' if len(unmasked_text) > 200 else unmasked_text
        }