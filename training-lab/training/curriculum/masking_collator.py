import torch
import re
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer

class CurriculumMaskingCollator:
    """Dynamic masking based on curriculum phase and data source."""

    def __init__(self, tokenizer: PreTrainedTokenizer, current_phase: str, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.current_phase = current_phase
        self.ignore_index = ignore_index
        self.masking_config = {}

    def set_phase(self, phase: str, phase_config: Dict[str, Any]):
        """Update masking strategy for new phase."""
        self.current_phase = phase
        self.masking_config = phase_config.get("masking", {})

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

        # Apply masking
        for example in examples:
            self.mask_example(example)

        # Find max length for padding
        max_length = max(len(ex["input_ids"]) for ex in examples)

        # Pad and stack into batch tensors
        batch = {}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        for key in ["input_ids", "attention_mask", "labels"]:
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
