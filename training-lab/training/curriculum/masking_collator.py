import torch
import re
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer

class CurriculumMaskingCollator:
    """Dynamic masking based on curriculum phase and data source.

    Supports both binary masking (ignore_index=-100) and weighted masking (loss_weights).
    When masking config contains weight values, weighted masking is used.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, current_phase: str, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.current_phase = current_phase
        self.ignore_index = ignore_index
        self.masking_config = {}
        self.use_weighted_masking = False  # Auto-detected from config

    def set_phase(self, phase: str, phase_config: Dict[str, Any]):
        """Update masking strategy for new phase."""
        self.current_phase = phase
        self.masking_config = phase_config.get("masking", {})

        # Auto-detect weighted masking mode
        # If any masking config contains weight dictionaries, use weighted mode
        for source_config in self.masking_config.values():
            if isinstance(source_config, dict) and "user_weight" in source_config:
                self.use_weighted_masking = True
                break

    def compute_labels(self, input_ids: list, text: str, source: str, masking_strategy: str) -> list:
        """Compute labels for a given masking strategy (used for pre-computation)."""
        example = {
            "input_ids": input_ids,
            "text": text,
            "source": source
        }

        if masking_strategy == "none":
            return input_ids.copy()
        elif masking_strategy == "system_only":
            return self._mask_system_only(example).tolist() if torch.is_tensor(self._mask_system_only(example)) else self._mask_system_only(example)
        elif masking_strategy == "alternating":
            return self._mask_alternating(example).tolist() if torch.is_tensor(self._mask_alternating(example)) else self._mask_alternating(example)
        elif masking_strategy == "assistant_only":
            return self._mask_non_assistant(example).tolist() if torch.is_tensor(self._mask_non_assistant(example)) else self._mask_non_assistant(example)
        else:
            return input_ids.copy()

    def mask_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking based on source and current phase."""
        # Skip if labels are already pre-computed
        if "labels" in example:
            return example

        source = example.get("source", "unknown")
        masking_strategy = self.masking_config.get(source, "assistant_only")

        input_ids = example["input_ids"]

        # Use weighted masking if config contains weight dictionaries
        if self.use_weighted_masking and isinstance(masking_strategy, dict):
            labels, loss_weights = self._apply_weighted_masking(example, masking_strategy)
            example["labels"] = labels
            example["loss_weights"] = loss_weights
            return example

        # Otherwise use binary masking (legacy mode)
        if masking_strategy == "none":
            labels = input_ids.clone() if torch.is_tensor(input_ids) else input_ids.copy()

        elif masking_strategy == "system_only":
            labels = self._mask_system_only(example)

        elif masking_strategy == "alternating":
            labels = self._mask_alternating(example)

        elif masking_strategy == "assistant_only":
            labels = self._mask_non_assistant(example)
        else:
            labels = input_ids.clone() if torch.is_tensor(input_ids) else input_ids.copy()

        example["labels"] = labels
        return example

    def _apply_weighted_masking(self, example: Dict[str, Any], weight_config: Dict[str, float]) -> tuple:
        """Apply weighted masking - returns labels and loss_weights tensors."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        # Convert input_ids to tensor if needed
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # All tokens are labels (no masking)
        labels = input_ids.clone()

        # Initialize all weights to 0.0
        loss_weights = torch.zeros_like(input_ids, dtype=torch.float)

        # Get weights from config
        user_weight = weight_config.get("user_weight", 0.0)
        assistant_weight = weight_config.get("assistant_weight", 1.0)
        system_weight = weight_config.get("system_weight", 0.0)

        # Apply system weight (everything before first User:/Assistant:)
        system_pattern = r"^(.*?)(?=(User:|Assistant:))"
        system_match = re.search(system_pattern, text, re.DOTALL)
        if system_match:
            end_pos = self._get_token_position(text, system_match.end())
            loss_weights[:end_pos] = system_weight

        # Apply user weights
        user_pattern = r"User:.*?(?=(Assistant:|User:|$))"
        for match in re.finditer(user_pattern, text, re.DOTALL):
            start_pos = self._get_token_position(text, match.start())
            end_pos = self._get_token_position(text, match.end())
            loss_weights[start_pos:end_pos] = user_weight

        # Apply assistant weights
        assistant_pattern = r"Assistant:.*?(?=(User:|Assistant:|$))"
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_pos = self._get_token_position(text, match.start())
            end_pos = self._get_token_position(text, match.end())
            loss_weights[start_pos:end_pos] = assistant_weight

        return labels, loss_weights

    def _get_token_position(self, text: str, char_position: int) -> int:
        """Convert character position in text to token position."""
        # Tokenize the text up to the character position
        prefix = text[:char_position]
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        return len(prefix_tokens)

    def _mask_system_only(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask only the system prompt section."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        # Convert input_ids to tensor if it's a list
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()

        # Find system section (between "---\nsystem:" and first "User:" or "Assistant:")
        system_pattern = r"---\s*\nsystem:.*?(?=(User:|Assistant:|$))"
        match = re.search(system_pattern, text, re.DOTALL)

        if match:
            # Tokenize the prefix to find position
            prefix = text[:match.end()]
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            labels[:len(prefix_ids)] = self.ignore_index

        return labels

    def _mask_alternating(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask user turns, keep assistant visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        # Convert input_ids to tensor if it's a list
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()

        # Find all User: sections
        user_pattern = r"User:.*?(?=(Assistant:|User:|$))"
        for match in re.finditer(user_pattern, text, re.DOTALL):
            start_text = text[:match.start()]
            end_text = text[:match.end()]

            start_tokens = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_text, add_special_tokens=False))

            labels[start_tokens:end_tokens] = self.ignore_index

        return labels

    def _mask_non_assistant(self, example: Dict[str, Any]) -> torch.Tensor:
        """Standard SFT masking - only assistant responses visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        # Convert input_ids to tensor if it's a list
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = torch.full_like(input_ids, self.ignore_index)

        # Find all Assistant: sections
        assistant_pattern = r"Assistant:.*?(?=(User:|Assistant:|$))"
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_text = text[:match.start()]
            end_text = text[:match.end()]

            start_tokens = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_text, add_special_tokens=False))

            labels[start_tokens:end_tokens] = input_ids[start_tokens:end_tokens]

        return labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch of examples."""
        # Convert lists to tensors first
        for example in examples:
            if "input_ids" in example and not torch.is_tensor(example["input_ids"]):
                example["input_ids"] = torch.tensor(example["input_ids"])
            if "attention_mask" in example and not torch.is_tensor(example["attention_mask"]):
                example["attention_mask"] = torch.tensor(example["attention_mask"])
            if "labels" in example and not torch.is_tensor(example["labels"]):
                example["labels"] = torch.tensor(example["labels"])
            if "loss_weights" in example and not torch.is_tensor(example["loss_weights"]):
                example["loss_weights"] = torch.tensor(example["loss_weights"])

        # Apply masking
        for example in examples:
            self.mask_example(example)

        # Find max length for padding
        max_length = max(len(ex["input_ids"]) for ex in examples)

        # Pad and stack into batch tensors
        batch = {}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for key in ["input_ids", "attention_mask", "labels", "loss_weights"]:
            if key in examples[0]:
                tensors = []
                for ex in examples:
                    tensor = ex[key]
                    if len(tensor) < max_length:
                        # Determine padding value
                        if key == "input_ids":
                            pad_value = pad_token_id
                        elif key == "labels":
                            pad_value = self.ignore_index
                        elif key == "loss_weights":
                            pad_value = 0.0  # Pad loss weights with 0
                        else:  # attention_mask
                            pad_value = 0

                        # Pad tensor
                        padding = torch.full((max_length - len(tensor),),
                                           pad_value,
                                           dtype=tensor.dtype)
                        tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)
                batch[key] = torch.stack(tensors)

        # Don't include text and meta fields in the batch sent to the model
        # They're only needed for masking logic, not for training
        # Keep other non-training fields as lists if needed for debugging
        # but exclude them from the final batch to avoid model errors

        return batch
