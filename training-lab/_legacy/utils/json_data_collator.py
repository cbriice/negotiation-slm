"""
Custom data collator for raw JSON training data.
Handles input/output format directly without text formatting.
"""

import json
from typing import Dict, List, Any, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass


@dataclass
class DataCollatorForJSON:
    """
    Data collator that masks 'input' and trains on 'output' for JSON format.

    Expects data in format:
    {
        "input": {...},  # Context to mask
        "output": "...", # Response to train on
        "meta": {...}    # Optional metadata
    }
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None
    ignore_index: int = -100

    def __post_init__(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of JSON examples.

        Args:
            features: List of examples with 'input' and 'output' fields

        Returns:
            Dict with input_ids, attention_mask, and labels tensors
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            # Extract input and output
            if isinstance(feature, str):
                # If feature is a JSON string, parse it
                feature = json.loads(feature)

            input_data = feature.get('input', {})
            output_text = feature.get('output', '')

            # Format input as a prompt
            input_text = self._format_input(input_data)

            # Tokenize input (context)
            input_tokens = self.tokenizer(
                input_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False
            )['input_ids']

            # Tokenize output (response)
            output_tokens = self.tokenizer(
                output_text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False
            )['input_ids']

            # Add EOS token to output
            output_tokens.append(self.tokenizer.eos_token_id)

            # Combine input and output
            all_tokens = input_tokens + output_tokens

            # Truncate if necessary
            if len(all_tokens) > self.max_length:
                # Prefer truncating input over output
                excess = len(all_tokens) - self.max_length
                if excess < len(input_tokens) - 1:  # Keep at least 1 input token
                    input_tokens = input_tokens[:-excess]
                    all_tokens = input_tokens + output_tokens
                else:
                    # Have to truncate output too
                    all_tokens = all_tokens[:self.max_length]

            # Create labels for autoregressive training (shifted left by 1)
            # labels[i] = all_tokens[i+1] (predict next token)
            labels = all_tokens[1:] + [self.tokenizer.eos_token_id]

            # Mask the input portion (we only train on predicting output)
            # Note: Since labels are shifted, position i predicts token i+1
            # So we mask positions 0 to len(input_tokens)-2 (which predict input tokens)
            for i in range(len(input_tokens) - 1):
                labels[i] = self.ignore_index

            # Create attention mask
            attention_mask = [1] * len(all_tokens)

            # Pad to max length in batch (will be done after collecting all)
            batch_input_ids.append(all_tokens)
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

    def _format_input(self, input_data: Union[Dict, str]) -> str:
        """
        Format input data into a text prompt.

        Args:
            input_data: Input context (dict or string)

        Returns:
            Formatted prompt string
        """
        if isinstance(input_data, str):
            return input_data

        # Build a simple prompt from the input data
        parts = []

        if 'state' in input_data and input_data['state']:
            parts.append(f"State: {json.dumps(input_data['state'])}")

        if 'recent_convo' in input_data and input_data['recent_convo']:
            for turn in input_data['recent_convo']:
                side = turn.get('side', 'unknown')
                text = turn.get('text', turn.get('response', ''))
                if text:
                    parts.append(f"{side}: {text}")

        if 'partner_msg' in input_data and input_data['partner_msg']:
            parts.append(f"Partner: {input_data['partner_msg']}")

        # Join with newlines and add a separator before response
        prompt = '\n'.join(parts)
        if prompt:
            prompt += '\nAgent: '  # Add prompt for agent response

        return prompt