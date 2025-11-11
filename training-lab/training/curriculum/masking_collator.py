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

    def mask_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking based on source and current phase."""
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

        labels = input_ids.clone() if torch.is_tensor(input_ids) else torch.tensor(input_ids)

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

        labels = input_ids.clone() if torch.is_tensor(input_ids) else torch.tensor(input_ids)

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

        labels = torch.full_like(input_ids, self.ignore_index) if torch.is_tensor(input_ids) else torch.full((len(input_ids),), self.ignore_index)

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
        for example in examples:
            self.mask_example(example)

        # Stack into batch tensors
        batch = {}
        for key in examples[0].keys():
            if torch.is_tensor(examples[0][key]):
                batch[key] = torch.stack([ex[key] for ex in examples])
            else:
                batch[key] = [ex[key] for ex in examples]

        return batch
