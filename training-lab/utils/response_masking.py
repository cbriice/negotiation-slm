"""
Response masking utilities for proper supervised fine-tuning.
Ensures loss is only computed on model responses, not prompts/context.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM
import training_parameters as params

logger = logging.getLogger(__name__)


class ResponseMaskingConfig:
    """Configuration for response masking behavior."""

    def __init__(
        self,
        response_templates: Optional[List[str]] = None,
        instruction_templates: Optional[List[str]] = None,
        ignore_index: int = -100,
        mask_instructions: bool = True,
        mlm: bool = False,
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Initialize response masking configuration.

        Args:
            response_templates: List of strings that mark start of responses
            instruction_templates: List of strings that mark instructions (optional)
            ignore_index: Token ID to use for masking (default -100 for CrossEntropy)
            mask_instructions: Whether to mask instruction portions
            mlm: Whether using masked language modeling (False for causal LM)
            pad_to_multiple_of: Pad sequences to multiple of this value
        """
        # Use config values if available, otherwise use defaults
        self.response_templates = response_templates or params.RESPONSE_TEMPLATES
        self.instruction_templates = instruction_templates or params.INSTRUCTION_TEMPLATES
        self.ignore_index = ignore_index if ignore_index != -100 else params.MASKING_IGNORE_INDEX
        self.mask_instructions = mask_instructions
        self.mlm = mlm
        self.pad_to_multiple_of = pad_to_multiple_of or params.PAD_TO_MULTIPLE


class ResponseMaskingHandler:
    """
    Handles proper masking of training data to ensure loss is only
    computed on response portions, not prompts/context.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Optional[ResponseMaskingConfig] = None
    ):
        """
        Initialize the response masking handler.

        Args:
            tokenizer: The tokenizer to use
            config: Response masking configuration
        """
        self.tokenizer = tokenizer
        self.config = config or ResponseMaskingConfig()

        # Pre-tokenize response templates for efficient matching
        self._tokenized_templates = {}
        for template in self.config.response_templates:
            tokens = self.tokenizer.encode(template, add_special_tokens=False)
            self._tokenized_templates[template] = tokens

        logger.info(f"Initialized ResponseMaskingHandler with {len(self.config.response_templates)} templates")

    def create_completion_only_collator(
        self,
        response_template: Optional[Union[str, List[int]]] = None,
        instruction_template: Optional[Union[str, List[int]]] = None
    ) -> DataCollatorForCompletionOnlyLM:
        """
        Create a TRL DataCollatorForCompletionOnlyLM for masking non-response tokens.

        Args:
            response_template: Template marking response start (uses first from config if None)
            instruction_template: Template marking instruction (optional)

        Returns:
            Configured DataCollatorForCompletionOnlyLM
        """
        if response_template is None:
            response_template = self.config.response_templates[0]
            logger.info(f"Using default response template: {response_template}")

        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            mlm=self.config.mlm,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            ignore_index=self.config.ignore_index
        )

        return collator

    def mask_labels_manual(
        self,
        input_ids: torch.Tensor,
        text: str,
        response_start: Optional[str] = None
    ) -> torch.Tensor:
        """
        Manually create labels with proper masking.

        Args:
            input_ids: Tokenized input tensor
            text: Original text string
            response_start: String marking where response begins

        Returns:
            Labels tensor with non-response tokens masked
        """
        labels = input_ids.clone()

        # Find response start position
        if response_start is None:
            # Try to find any response template in the text
            response_pos = len(text)
            used_template = None

            for template in self.config.response_templates:
                pos = text.find(template)
                if pos != -1 and pos < response_pos:
                    response_pos = pos
                    used_template = template

            if used_template:
                # Move position to after the template
                response_pos += len(used_template)
            else:
                logger.warning("No response template found in text, masking entire sequence")
                labels[:] = self.config.ignore_index
                return labels
        else:
            response_pos = text.find(response_start)
            if response_pos == -1:
                logger.warning(f"Response start '{response_start}' not found in text")
                labels[:] = self.config.ignore_index
                return labels
            response_pos += len(response_start)

        # Tokenize the portion before response
        prefix_text = text[:response_pos]
        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        prefix_len = len(prefix_tokens)

        # Mask everything before the response
        if prefix_len > 0:
            labels[:prefix_len] = self.config.ignore_index
            logger.debug(f"Masked {prefix_len} prefix tokens out of {len(labels)} total")

        return labels

    def find_response_positions(
        self,
        input_ids: List[int]
    ) -> List[int]:
        """
        Find all positions where responses begin in tokenized input.

        Args:
            input_ids: List of token IDs

        Returns:
            List of positions where responses start
        """
        positions = []

        for template, template_tokens in self._tokenized_templates.items():
            # Search for template tokens in input
            for i in range(len(input_ids) - len(template_tokens) + 1):
                if input_ids[i:i+len(template_tokens)] == template_tokens:
                    # Found template, response starts after it
                    positions.append(i + len(template_tokens))
                    logger.debug(f"Found '{template}' at position {i}")

        return sorted(positions)

    def create_attention_mask_for_responses_only(
        self,
        input_ids: torch.Tensor,
        response_positions: List[int]
    ) -> torch.Tensor:
        """
        Create attention mask that only attends to response portions.

        Args:
            input_ids: Input token IDs
            response_positions: Positions where responses start

        Returns:
            Attention mask tensor
        """
        mask = torch.zeros_like(input_ids)

        if not response_positions:
            # No response found, attend to everything (fallback)
            return torch.ones_like(input_ids)

        # For each response position, unmask from there to the next position or end
        for i, start_pos in enumerate(response_positions):
            if i + 1 < len(response_positions):
                end_pos = response_positions[i + 1]
            else:
                end_pos = len(input_ids)

            mask[start_pos:end_pos] = 1

        return mask

    def validate_masking(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate that masking is working correctly.

        Args:
            input_ids: Input token IDs
            labels: Label token IDs (with masking)
            text: Optional original text for debugging

        Returns:
            Dictionary with validation metrics
        """
        total_tokens = len(input_ids)
        masked_tokens = (labels == self.config.ignore_index).sum().item()
        supervised_tokens = total_tokens - masked_tokens

        metrics = {
            "total_tokens": total_tokens,
            "masked_tokens": masked_tokens,
            "supervised_tokens": supervised_tokens,
            "masking_ratio": masked_tokens / total_tokens if total_tokens > 0 else 0,
            "supervision_ratio": supervised_tokens / total_tokens if total_tokens > 0 else 0
        }

        # Decode supervised portion if possible
        if supervised_tokens > 0:
            supervised_ids = input_ids[labels != self.config.ignore_index]
            if len(supervised_ids) > 0:
                supervised_text = self.tokenizer.decode(supervised_ids, skip_special_tokens=True)
                metrics["supervised_text_preview"] = supervised_text[:100]

        logger.info(f"Masking validation: {masked_tokens}/{total_tokens} masked ({metrics['masking_ratio']:.2%})")

        return metrics


class MultiFormatResponseMasker:
    """
    Handles response masking for multiple training data formats.
    Automatically detects format and applies appropriate masking.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize multi-format masker.

        Args:
            tokenizer: The tokenizer to use
        """
        self.tokenizer = tokenizer

        # Define format-specific templates
        self.format_configs = {
            "instruction": ResponseMaskingConfig(
                response_templates=["### Response:\n", "### Response:"],
                instruction_templates=["### Instruction:\n", "### Instruction:"]
            ),
            "dialogue": ResponseMaskingConfig(
                response_templates=["Assistant:", "Agent:", "You:"],
                instruction_templates=["User:", "Human:", "Partner:"]
            ),
            "context": ResponseMaskingConfig(
                response_templates=["### Response:\n", "You:"],
                instruction_templates=["### Context:\n", "State:", "Partner:"]
            )
        }

        # Create handlers for each format
        self.handlers = {
            fmt: ResponseMaskingHandler(tokenizer, config)
            for fmt, config in self.format_configs.items()
        }

    def detect_format(self, text: str) -> str:
        """
        Detect the format of the training text.

        Args:
            text: Training text

        Returns:
            Format name ("instruction", "dialogue", or "context")
        """
        if "### Instruction:" in text and "### Response:" in text:
            return "instruction"
        elif "### Context:" in text:
            return "context"
        elif any(marker in text for marker in ["User:", "Assistant:", "Human:", "Agent:"]):
            return "dialogue"
        else:
            # Default to instruction format
            return "instruction"

    def get_collator_for_format(self, format_name: str) -> DataCollatorForCompletionOnlyLM:
        """
        Get the appropriate data collator for the detected format.

        Args:
            format_name: Name of the format

        Returns:
            Configured DataCollatorForCompletionOnlyLM
        """
        handler = self.handlers.get(format_name, self.handlers["instruction"])
        return handler.create_completion_only_collator()

    def create_adaptive_collator(self) -> DataCollatorForCompletionOnlyLM:
        """
        Create a collator that works with multiple response templates.

        Returns:
            DataCollatorForCompletionOnlyLM configured for multiple formats
        """
        # Combine all response templates
        all_templates = []
        for config in self.format_configs.values():
            all_templates.extend(config.response_templates)

        # Use the most common template as primary
        primary_template = "### Response:\n"

        # Note: TRL's DataCollatorForCompletionOnlyLM currently supports
        # only one response template at a time. For multiple templates,
        # we'd need to extend it or use multiple collators

        # Use instruction template for proper masking
        instruction_template = "### Context:\n"

        return DataCollatorForCompletionOnlyLM(
            response_template=primary_template,
            instruction_template=instruction_template,
            tokenizer=self.tokenizer,
            mlm=False,
            ignore_index=-100
        )