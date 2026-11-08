import bisect
import torch
import re
import logging
from typing import Dict, Any, List, Tuple
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class CurriculumMaskingCollator:
    """Dynamic masking based on curriculum phase and data source.

    Supports both binary masking (ignore_index=-100) and weighted masking (loss_weights).
    When masking config contains weight values, weighted masking is used.

    Character spans (masking_regions or regex matches) are mapped to tokens with the
    tokenizer's offset mapping, re-encoding the text exactly as it was tokenized for
    training (special tokens included). A token belongs to a span when its first
    character falls inside it.
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

    def _token_char_starts(self, text: str, input_ids) -> List[int]:
        """First character index of each token in input_ids (-1 for special tokens).

        Re-encodes text with the same call used to build input_ids and checks the ids
        match, so the mapping can't silently drift (e.g. a missing BOS shifting every
        span by one token).
        """
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Masking requires a fast tokenizer (offset mapping). Load it with use_fast=True.")

        ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
        # The pipeline tokenizes with special tokens; also accept ids built without them
        for add_special_tokens in (True, False):
            enc = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=len(ids),
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            if enc["input_ids"] == ids:
                return [-1 if special else start
                        for (start, _), special in zip(enc["offset_mapping"], enc["special_tokens_mask"])]
        raise RuntimeError(
            "input_ids do not match a fresh tokenization of the example text. "
            "Re-run pre_tokenize_datasets.py with the tokenizer used for training."
        )

    @staticmethod
    def _tokens_in_span(char_starts: List[int], start_char: int, end_char: int) -> torch.Tensor:
        """Boolean mask of tokens whose first character lies in [start_char, end_char)."""
        starts = torch.tensor(char_starts)
        return (starts >= start_char) & (starts < end_char)

    def _apply_weighted_masking(self, example: Dict[str, Any], weight_config: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply weighted masking - returns labels and loss_weights tensors.

        REQUIRES masking_regions in example. Will crash training if not present.
        """
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        # SAFETY CHECK: masking_regions MUST be present
        if "masking_regions" not in example:
            logger.critical("="*80)
            logger.critical("FATAL: masking_regions not found in training example!")
            logger.critical("Legacy regex-based masking is DISABLED (catastrophic for model behavior)")
            logger.critical("Please reformat your dataset using reformat_to_next_utterance.py")
            logger.critical("="*80)
            raise RuntimeError(
                "masking_regions not found in example. "
                "Legacy regex masking is disabled. "
                "Reformat dataset with reformat_to_next_utterance.py"
            )

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # All tokens are labels; loss_weights decide what is learned
        labels = input_ids.clone()
        loss_weights = torch.zeros_like(input_ids, dtype=torch.float)

        user_weight = weight_config.get("user_weight", 0.0)
        assistant_weight = weight_config.get("assistant_weight", 1.0)
        system_weight = weight_config.get("system_weight", 0.0)

        regions = sorted(example["masking_regions"], key=lambda r: r["start"])

        # VALIDATION: regions must not overlap, so every token gets exactly one weight
        for prev, cur in zip(regions, regions[1:]):
            if cur["start"] < prev["end"]:
                logger.critical(f"FATAL: overlapping masking_regions {prev} and {cur}")
                raise RuntimeError(
                    f"Overlapping masking_regions at chars {cur['start']}-{prev['end']}. "
                    "This indicates a bug in masking_regions generation."
                )

        region_starts = [r["start"] for r in regions]
        char_starts = self._token_char_starts(text, input_ids)

        for token_idx, char_pos in enumerate(char_starts):
            if char_pos < 0:
                continue  # special token (BOS/EOS): weight 0
            region_idx = bisect.bisect_right(region_starts, char_pos) - 1
            if region_idx < 0 or char_pos >= regions[region_idx]["end"]:
                continue  # not covered by any region: weight 0
            region = regions[region_idx]
            role = region["role"]

            if role == "system":
                loss_weights[token_idx] = system_weight
            elif role == "user":
                loss_weights[token_idx] = user_weight
            elif role == "assistant":
                # Non-final assistant turns are context (mask=True) and get user_weight;
                # only the final assistant turn (the target) gets assistant_weight
                loss_weights[token_idx] = user_weight if region.get("mask", True) else assistant_weight

        return labels, loss_weights

    def _mask_system_only(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask only the system prompt section."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()

        # Find system section (between "---\nsystem:" and first "User:" or "Assistant:")
        system_pattern = r"---\s*\nsystem:.*?(?=(User:|Assistant:|$))"
        match = re.search(system_pattern, text, re.DOTALL)

        if match:
            char_starts = self._token_char_starts(text, input_ids)
            labels[self._tokens_in_span(char_starts, 0, match.end())] = self.ignore_index

        return labels

    def _mask_alternating(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask user turns, keep assistant visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        char_starts = self._token_char_starts(text, input_ids)

        # Find all User: sections
        user_pattern = r"User:.*?(?=(Assistant:|User:|$))"
        for match in re.finditer(user_pattern, text, re.DOTALL):
            labels[self._tokens_in_span(char_starts, match.start(), match.end())] = self.ignore_index

        return labels

    def _mask_non_assistant(self, example: Dict[str, Any]) -> torch.Tensor:
        """Standard SFT masking - only assistant responses visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = torch.full_like(input_ids, self.ignore_index)
        char_starts = self._token_char_starts(text, input_ids)

        # Find all Assistant: sections
        assistant_pattern = r"Assistant:.*?(?=(User:|Assistant:|$))"
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            span = self._tokens_in_span(char_starts, match.start(), match.end())
            labels[span] = input_ids[span]

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

                        # Pad tensor (respect tokenizer padding_side)
                        padding = torch.full((max_length - len(tensor),),
                                           pad_value,
                                           dtype=tensor.dtype)
                        if self.tokenizer.padding_side == 'left':
                            tensor = torch.cat([padding, tensor])
                        else:
                            tensor = torch.cat([tensor, padding])
                    tensors.append(tensor)
                batch[key] = torch.stack(tensors)

        # Don't include text and meta fields in the batch sent to the model
        # They're only needed for masking logic, not for training
        # Keep other non-training fields as lists if needed for debugging
        # but exclude them from the final batch to avoid model errors

        return batch
