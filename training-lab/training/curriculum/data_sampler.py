import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

class CurriculumDataSampler:
    """Samples from multiple datasets with dynamic ratios."""

    def __init__(self, conversational_path: str, negotiation_path: str, seed: int = 42):
        self.conversational_path = Path(conversational_path)
        self.negotiation_path = Path(negotiation_path)
        self.seed = seed
        random.seed(seed)

        # Load datasets into memory
        self.conversational_data = self._load_jsonl(self.conversational_path)
        self.negotiation_data = self._load_jsonl(self.negotiation_path)

        # Tag with source
        for ex in self.conversational_data:
            ex["source"] = "conversational"
        for ex in self.negotiation_data:
            ex["source"] = "negotiation"

        self.total_examples = len(self.conversational_data) + len(self.negotiation_data)

        # Default ratio
        self.current_ratio = {"conversational": 0.5, "negotiation": 0.5}

        # For iteration
        self.epoch = 0
        self.indices = {"conversational": 0, "negotiation": 0}

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file into memory."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def set_ratio(self, ratio: Dict[str, float]):
        """Update sampling ratio."""
        assert abs(sum(ratio.values()) - 1.0) < 0.01, "Ratios must sum to 1.0"
        self.current_ratio = ratio

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch respecting current ratio."""
        batch = []

        # Calculate samples per dataset
        conv_samples = int(batch_size * self.current_ratio["conversational"])
        neg_samples = batch_size - conv_samples

        # Sample from each dataset
        if conv_samples > 0 and self.conversational_data:
            conv_batch = random.sample(self.conversational_data,
                                      min(conv_samples, len(self.conversational_data)))
            batch.extend(conv_batch)

        if neg_samples > 0 and self.negotiation_data:
            neg_batch = random.sample(self.negotiation_data,
                                    min(neg_samples, len(self.negotiation_data)))
            batch.extend(neg_batch)

        # Shuffle the combined batch
        random.shuffle(batch)

        return batch[:batch_size]  # Ensure exact size

    def get_epoch_iterator(self, batch_size: int, shuffle: bool = True):
        """Get iterator for full epoch with proper ratios."""
        # Prepare indices
        conv_size = int(self.total_examples * self.current_ratio["conversational"])
        neg_size = self.total_examples - conv_size

        # Create repeated indices to match desired ratio
        conv_indices = list(range(len(self.conversational_data))) * (conv_size // len(self.conversational_data) + 1)
        conv_indices = conv_indices[:conv_size]

        neg_indices = list(range(len(self.negotiation_data))) * (neg_size // len(self.negotiation_data) + 1)
        neg_indices = neg_indices[:neg_size]

        # Create combined dataset for epoch
        epoch_data = []
        epoch_data.extend([self.conversational_data[i] for i in conv_indices])
        epoch_data.extend([self.negotiation_data[i] for i in neg_indices])

        if shuffle:
            random.shuffle(epoch_data)

        # Yield batches
        for i in range(0, len(epoch_data), batch_size):
            yield epoch_data[i:i + batch_size]

    def override_ratio(self, ratio: Dict[str, float]):
        """Emergency override for recovery mode."""
        self.set_ratio(ratio)
        print(f"[EMERGENCY] Data ratio overridden to: {ratio}")
