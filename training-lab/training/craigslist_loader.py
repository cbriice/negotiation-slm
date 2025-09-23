"""
CraigslistBargain dataset loader for baseline negotiation training.
Loads and processes the Stanford CraigslistBargains dataset from HuggingFace.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset, Dataset
import re

# Setup logging
logger = logging.getLogger(__name__)


class CraigslistBargainLoader:
    """
    Loads and processes the CraigslistBargain dataset for SFT training.

    Dataset: https://huggingface.co/datasets/stanfordnlp/craigslist_bargains
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CraigslistBargain loader.

        Args:
            cache_dir: Optional directory for caching the dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None

    def load_dataset(self, split: str = "train") -> Dataset:
        """
        Load the CraigslistBargain dataset from HuggingFace.

        Args:
            split: Dataset split to load ('train', 'validation', 'test')

        Returns:
            HuggingFace Dataset object
        """
        import shutil
        logger.info(f"Loading CraigslistBargain dataset (split={split})...")

        # Clear any corrupted cache completely
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            # Remove the entire cache directory for this dataset
            craigslist_cache = cache_path / "stanfordnlp___craigslist_bargains"
            if craigslist_cache.exists():
                logger.info(f"Removing corrupted cache at {craigslist_cache}...")
                shutil.rmtree(craigslist_cache, ignore_errors=True)

            # Also clear downloads cache
            downloads_cache = cache_path / "downloads"
            if downloads_cache.exists():
                logger.info(f"Removing downloads cache at {downloads_cache}...")
                shutil.rmtree(downloads_cache, ignore_errors=True)

        # Load fresh without trust_remote_code
        # Use parquet conversion to avoid deprecated dataset script
        dataset_dict = load_dataset(
            "stanfordnlp/craigslist_bargains",
            cache_dir=self.cache_dir,  # Use the specified cache or None
            revision="refs/convert/parquet"  # Use parquet version to avoid dataset script
        )

        self.dataset = dataset_dict[split]

        logger.info(f"Loaded {len(self.dataset)} conversations from {split} split")
        return self.dataset

    def _extract_dialogue_turns(self, example: Dict) -> List[Tuple[str, str]]:
        """
        Extract dialogue turns from a conversation example.

        Args:
            example: Single example from the dataset

        Returns:
            List of (speaker, utterance) tuples
        """
        turns = []

        # The dataset has 'utterance' and 'dialogue_acts' fields
        if 'utterance' in example:
            utterances = example['utterance']

            # Get agent info if available
            agents = example.get('agent_info', {})

            # Process each utterance
            for i, utt in enumerate(utterances):
                # Skip empty utterances
                if not utt or utt.strip() == '':
                    continue

                # Determine speaker (alternate between buyer/seller based on index)
                # In CraigslistBargains, typically buyer starts
                speaker = "buyer" if i % 2 == 0 else "seller"

                # Clean utterance
                cleaned = utt.strip()
                if cleaned:
                    turns.append((speaker, cleaned))

        return turns

    def _create_system_prompt(self, example: Dict, role: str) -> str:
        """
        Create a system prompt from item metadata.
        Only includes fields that have specific, meaningful values.

        Args:
            example: Dataset example
            role: Either 'buyer' or 'seller'

        Returns:
            System prompt string
        """
        parts = []

        # Extract item information
        items = example.get('items', {})
        if items:
            # Get the first item (usually there's only one)
            item = items.get('item0', {}) if isinstance(items, dict) else {}

            # Only add fields with meaningful values
            title = item.get('Title')
            if title and title not in ['Unknown Item', 'Unknown', '']:
                parts.append(f"Item: {title}")

            category = item.get('Category')
            if category and category not in ['General', 'Unknown', '']:
                parts.append(f"Category: {category}")

            price = item.get('Price')
            if price and price not in ['Negotiable', 'Unknown', '', 'TBD']:
                parts.append(f"Listed Price: {price}")

        # Always add role
        parts.append(f"Your role: {role}")

        # Add agent-specific information if available
        agent_info = example.get('agent_info', {})
        if role in agent_info:
            agent_data = agent_info[role]
            if 'target' in agent_data:
                target = agent_data['target']
                if target:  # Only add if target exists
                    if role == 'buyer':
                        parts.append(f"Target price: {target}")
                    else:
                        parts.append(f"Minimum acceptable: {target}")

        return " | ".join(parts)

    def _create_sft_packets(self, example: Dict) -> List[Dict]:
        """
        Convert a conversation into SFT training packets.
        Creates packets for both buyer and seller perspectives.

        Args:
            example: Single conversation from dataset

        Returns:
            List of SFT packets following the template schema
        """
        packets = []

        # Extract dialogue turns
        turns = self._extract_dialogue_turns(example)

        if len(turns) < 2:
            # Skip conversations that are too short
            return packets

        # Get conversation ID
        conv_id = example.get('dialogue_id', 'unknown')

        # Create packets from both perspectives
        for perspective in ['buyer', 'seller']:
            # Build conversation history for each turn
            for i in range(1, len(turns)):
                # Check if this turn is from the perspective we're training
                speaker, response = turns[i]
                if speaker != perspective:
                    continue

                # Build prompt from conversation history
                history = []
                for j in range(i):
                    hist_speaker, hist_utt = turns[j]
                    role_label = "You" if hist_speaker == perspective else "Partner"
                    history.append(f"{role_label}: {hist_utt}")

                # Create the SFT packet
                packet = {
                    "id": f"{conv_id}_{perspective}_{i}",
                    "system": self._create_system_prompt(example, perspective),
                    "prompt": "\n".join(history),
                    "response": response
                }

                packets.append(packet)

        return packets

    def process_dataset_to_sft(self,
                              dataset: Optional[Dataset] = None,
                              output_file: Optional[str] = None,
                              max_examples: Optional[int] = None) -> str:
        """
        Process entire dataset into SFT format.

        Args:
            dataset: Dataset to process (uses self.dataset if None)
            output_file: Path to save JSONL file
            max_examples: Maximum number of conversations to process

        Returns:
            Path to output file
        """
        if dataset is None:
            dataset = self.dataset

        if dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        logger.info("Processing dataset to SFT format...")

        all_packets = []
        skipped = 0

        # Process each conversation
        num_examples = min(len(dataset), max_examples) if max_examples else len(dataset)

        for i in range(num_examples):
            example = dataset[i]
            packets = self._create_sft_packets(example)

            if packets:
                all_packets.extend(packets)
            else:
                skipped += 1

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{num_examples} conversations, "
                          f"generated {len(all_packets)} packets, skipped {skipped}")

        logger.info(f"Total: {len(all_packets)} SFT packets from {num_examples} conversations")
        logger.info(f"Skipped {skipped} conversations (too short or empty)")

        # Save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                for packet in all_packets:
                    f.write(json.dumps(packet, ensure_ascii=False) + '\n')

            logger.info(f"Saved {len(all_packets)} packets to {output_path}")
            return str(output_path)

        return all_packets

    def load_all_splits(self) -> Dict[str, Dataset]:
        """
        Load all available splits of the dataset.

        Returns:
            Dictionary mapping split names to Dataset objects
        """
        splits = {}
        for split in ['train', 'validation', 'test']:
            try:
                splits[split] = self.load_dataset(split)
                logger.info(f"Loaded {split}: {len(splits[split])} examples")
            except Exception as e:
                logger.warning(f"Could not load {split} split: {e}")

        return splits

    def create_full_training_set(self, output_file: str) -> str:
        """
        Create a comprehensive training set using all available data.
        Uses train + validation for maximum diversity.

        Args:
            output_file: Path to save the combined SFT dataset

        Returns:
            Path to output file
        """
        logger.info("Creating comprehensive training set from all splits...")

        all_packets = []

        # Load and process train split
        train_data = self.load_dataset('train')
        train_packets = self.process_dataset_to_sft(train_data)
        if isinstance(train_packets, list):
            all_packets.extend(train_packets)

        # Load and process validation split (add to training for more diversity)
        try:
            val_data = self.load_dataset('validation')
            val_packets = self.process_dataset_to_sft(val_data)
            if isinstance(val_packets, list):
                all_packets.extend(val_packets)
        except Exception as e:
            logger.warning(f"Could not add validation data: {e}")

        # Save combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
            for packet in all_packets:
                f.write(json.dumps(packet, ensure_ascii=False) + '\n')

        logger.info(f"Created comprehensive training set with {len(all_packets)} packets")
        logger.info(f"Saved to: {output_path}")

        return str(output_path)


def main():
    """CLI interface for testing the loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and process CraigslistBargain dataset")
    parser.add_argument('--split', default='train', choices=['train', 'validation', 'test'],
                       help='Dataset split to load')
    parser.add_argument('--output', type=str, default='data/training/craigslist_sft.jsonl',
                       help='Output file path')
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum number of examples to process')
    parser.add_argument('--all-splits', action='store_true',
                       help='Process all splits into comprehensive training set')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize loader
    loader = CraigslistBargainLoader()

    if args.all_splits:
        # Create comprehensive training set
        loader.create_full_training_set(args.output)
    else:
        # Load single split
        dataset = loader.load_dataset(args.split)

        # Process to SFT
        loader.process_dataset_to_sft(
            dataset=dataset,
            output_file=args.output,
            max_examples=args.max_examples
        )


if __name__ == "__main__":
    main()