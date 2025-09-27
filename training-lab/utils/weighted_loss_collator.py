"""
Weighted loss data collator for CraigslistBargain SFT training.
Implements weighted loss computation: 0.1-0.2x on prompt, 1x on response.
System prompts are completely masked.
"""

import json
from typing import Dict, List, Any, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass


@dataclass
class WeightedLossCollator:
    """
    Data collator that applies weighted loss for different parts of the input.

    Expects data in CraigslistBargain SFT format:
    {
        "id": "conv_id_role_turn",
        "system": "Item: X | Category: Y | Price: Z | Your role: buyer/seller",
        "prompt": "Previous conversation history",
        "response": "Agent's response"
    }

    Loss weights:
    - System: 0.0 (masked completely)
    - Prompt: 0.15x weight (configurable via prompt_weight)
    - Response: 1.0x weight

    The collator returns loss_weights tensor that the custom trainer
    uses to apply different weights to different parts of the sequence.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = 128
    ignore_index: int = -100
    prompt_weight: float = 0.15  # Weight for prompt/context loss

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with weighted loss masking.

        Args:
            features: List of examples with system/prompt/response fields

        Returns:
            Dict with input_ids, attention_mask, labels, and loss_weights tensors
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_loss_weights = []

        for feature in features:
            # Parse if JSON string
            if isinstance(feature, str):
                feature = json.loads(feature)

            # Extract fields
            system = feature.get('system', '')
            prompt = feature.get('prompt', '')
            response = feature.get('response', '')

            # Skip empty responses
            if not response or response.strip() == '':
                continue

            # Format the complete input
            # Use clear delimiters for each section
            formatted_input = ""

            if system:
                formatted_input += f"### System:\n{system}\n\n"

            if prompt:
                formatted_input += f"### Context:\n{prompt}\n\n"

            formatted_input += f"### Response:\n{response}"

            # Tokenize the complete input
            encoding = self.tokenizer(
                formatted_input,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True  # Need this to find section boundaries
            )

            input_ids = encoding['input_ids']
            offsets = encoding['offset_mapping']

            # Find boundaries of each section using the formatted text
            system_end = 0
            prompt_end = 0
            response_start = 0

            # Find where each section ends in the token sequence
            if system:
                system_marker = "### System:"
                system_end_text = system + "\n\n"
                system_end_pos = len(system_marker) + len(system_end_text)
                system_end = self._find_token_position(offsets, system_end_pos)

            if prompt:
                prompt_marker = "### Context:"
                if system:
                    prompt_start_pos = len(f"### System:\n{system}\n\n")
                else:
                    prompt_start_pos = 0
                prompt_end_pos = prompt_start_pos + len(prompt_marker) + len(prompt) + 2  # +2 for \n\n
                prompt_end = self._find_token_position(offsets, prompt_end_pos)

            # Find response start
            response_marker = "### Response:\n"
            response_marker_text = "### Response:"
            response_text_start = formatted_input.find(response_marker_text)
            if response_text_start >= 0:
                response_content_start = response_text_start + len(response_marker)
                response_start = self._find_token_position(offsets, response_content_start)

            # Create labels and loss weights
            labels = []
            loss_weights = []
            for i, token_id in enumerate(input_ids):
                if i < system_end:
                    # System prompt - completely masked
                    labels.append(self.ignore_index)
                    loss_weights.append(0.0)
                elif i < response_start:
                    # Prompt/context - apply reduced weight
                    labels.append(token_id)
                    loss_weights.append(self.prompt_weight)
                else:
                    # Response - full weight
                    labels.append(token_id)
                    loss_weights.append(1.0)

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
            batch_loss_weights.append(loss_weights)

        # Handle empty batch
        if not batch_input_ids:
            return {
                'input_ids': torch.tensor([[]], dtype=torch.long),
                'labels': torch.tensor([[]], dtype=torch.long),
                'attention_mask': torch.tensor([[]], dtype=torch.long),
                'loss_weights': torch.tensor([[]], dtype=torch.float)
            }

        # Find max length in batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)

        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_batch_len = (
                (max_batch_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad all sequences
        for i in range(len(batch_input_ids)):
            padding_length = max_batch_len - len(batch_input_ids[i])
            if padding_length > 0:
                batch_input_ids[i] += [self.tokenizer.pad_token_id] * padding_length
                batch_labels[i] += [self.ignore_index] * padding_length
                batch_attention_mask[i] += [0] * padding_length
                batch_loss_weights[i] += [0.0] * padding_length

        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'loss_weights': torch.tensor(batch_loss_weights, dtype=torch.float)
        }

    def _find_token_position(self, offsets: List[tuple], char_position: int) -> int:
        """
        Find the token index corresponding to a character position.

        Args:
            offsets: List of (start, end) character positions for each token
            char_position: Character position to find

        Returns:
            Token index
        """
        for i, (start, end) in enumerate(offsets):
            if start <= char_position < end:
                return i
        return len(offsets)


class WeightedLossTrainer:
    """
    Custom trainer wrapper that handles weighted loss computation.
    """

    def __init__(self, prompt_weight: float = 0.15):
        """
        Initialize weighted loss trainer.

        Args:
            prompt_weight: Weight to apply to prompt/context tokens (0.1-0.2 recommended)
        """
        self.prompt_weight = prompt_weight

    def compute_weighted_loss(self, logits, labels, ignore_index=-100):
        """
        Compute loss with different weights for prompt and response.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target labels with special encoding for weighted tokens
            ignore_index: Value to ignore in loss computation

        Returns:
            Weighted loss tensor
        """
        import torch.nn.functional as F

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten tensors
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Identify different token types
        # - ignore_index (-100): masked tokens (no loss)
        # - negative values < -10000: weighted prompt tokens
        # - positive values: full weight response tokens

        # Create mask for each type
        masked_tokens = (shift_labels == ignore_index)
        weighted_tokens = (shift_labels < -10000)
        full_weight_tokens = (shift_labels >= 0) & ~masked_tokens

        # Recover actual token IDs for weighted tokens
        actual_labels = shift_labels.clone()
        actual_labels[weighted_tokens] = -(shift_labels[weighted_tokens] + 10000)

        # Compute cross entropy loss without reduction
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        per_token_loss = loss_fct(shift_logits, actual_labels)

        # Apply weights
        weights = torch.ones_like(per_token_loss)
        weights[weighted_tokens] = self.prompt_weight
        weights[masked_tokens] = 0.0

        # Compute weighted mean loss
        weighted_loss = (per_token_loss * weights).sum() / weights.sum().clamp(min=1.0)

        return weighted_loss