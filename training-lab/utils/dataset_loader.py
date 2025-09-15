"""
Unified dataset loader for training modules.
Handles NDJSON, JSON, and HuggingFace dataset loading with consistent preprocessing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


class UnifiedDatasetLoader:
    """Centralized dataset loading and preprocessing for all training modules."""

    @staticmethod
    def load_ndjson(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load data from NDJSON file.

        Args:
            file_path: Path to NDJSON file

        Returns:
            List of dictionaries from the file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = []
        invalid_lines = 0

        logger.info(f"Loading NDJSON from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    invalid_lines += 1

        if not data:
            raise ValueError(f"No valid data found in {file_path}")

        logger.info(f"Loaded {len(data)} items from NDJSON")
        if invalid_lines > 0:
            logger.warning(f"Skipped {invalid_lines} invalid lines")

        return data

    @staticmethod
    def load_dataset_auto(
        file_path: Union[str, Path],
        split: Optional[str] = None
    ) -> Dataset:
        """
        Automatically load dataset from various formats.

        Args:
            file_path: Path to dataset file
            split: Optional split to load (for multi-split datasets)

        Returns:
            HuggingFace Dataset object
        """
        file_path = Path(file_path)

        # Determine file format
        if file_path.suffix in ['.jsonl', '.ndjson']:
            # Load NDJSON
            data = UnifiedDatasetLoader.load_ndjson(file_path)
            return Dataset.from_list(data)

        elif file_path.suffix == '.json':
            # Load regular JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return Dataset.from_list(data)
            else:
                return Dataset.from_dict(data)

        else:
            # Try HuggingFace datasets loader
            dataset = load_dataset('json', data_files=str(file_path))

            # Handle DatasetDict vs Dataset
            if isinstance(dataset, DatasetDict):
                if split:
                    return dataset[split]
                elif 'train' in dataset:
                    return dataset['train']
                else:
                    # Return first available split
                    return list(dataset.values())[0]
            else:
                return dataset

    @staticmethod
    def format_sft_dialogue(example: Dict[str, Any]) -> str:
        """
        Format example for SFT training (dialogue format).

        Args:
            example: Training example

        Returns:
            Formatted text string
        """
        # Handle dialogue format
        if 'dialogue' in example:
            formatted = ""
            for turn in example['dialogue']:
                role = turn.get('role', 'user')
                content = turn.get('content', turn.get('text', ''))

                if role == 'assistant':
                    formatted += f"Assistant: {content}\n"
                else:
                    formatted += f"User: {content}\n"
            return formatted.strip()

        # Handle instruction-response format
        elif 'instruction' in example and 'response' in example:
            instruction = example['instruction']
            response = example['response']
            return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        # Handle input-output format (negotiation specific)
        elif 'input' in example and 'output' in example:
            input_data = example['input']
            output = example['output']

            context = ""
            if isinstance(input_data, dict):
                if 'state' in input_data and input_data['state']:
                    context += f"State: {json.dumps(input_data['state'])}\n"
                if 'recent_convo' in input_data and input_data['recent_convo']:
                    context += f"Recent conversation: {json.dumps(input_data['recent_convo'])}\n"
                if 'partner_msg' in input_data and input_data['partner_msg']:
                    context += f"Partner: {input_data['partner_msg']}\n"
            else:
                context = str(input_data)

            return f"### Context:\n{context}\n### Response:\n{output}"

        # Fallback to plain text
        elif 'text' in example:
            return example['text']

        else:
            raise ValueError(f"Unknown example format with keys: {example.keys()}")

    @staticmethod
    def format_dpo_prompt(context: Dict[str, Any]) -> str:
        """
        Format context into a DPO prompt string.

        Args:
            context: Context dictionary with state, conversation, etc.

        Returns:
            Formatted prompt string
        """
        state = context.get('state', {})
        recent_convo = context.get('recent_convo', [])
        partner_msg = context.get('partner_msg', '')

        prompt_parts = []

        # Add state information
        if state:
            state_str = json.dumps(state, indent=2)
            prompt_parts.append(f"Negotiation State:\n{state_str}")

        # Add conversation history
        if recent_convo:
            prompt_parts.append("\nRecent Conversation:")
            for turn in recent_convo:
                side = turn.get('side', 'unknown')
                text = turn.get('text', '')
                if text:
                    prompt_parts.append(f"{side}: {text}")
                response = turn.get('response', '')
                if response:
                    prompt_parts.append(f"You: {response}")

        # Add current partner message
        if partner_msg:
            prompt_parts.append(f"\nPartner: {partner_msg}")
            prompt_parts.append("You:")

        return "\n".join(prompt_parts)

    @staticmethod
    def preprocess_sft_dataset(
        dataset: Dataset,
        format_func: Optional[Callable] = None
    ) -> Dataset:
        """
        Preprocess dataset for SFT training.

        Args:
            dataset: Raw dataset
            format_func: Optional custom formatting function

        Returns:
            Dataset with 'text' field added
        """
        format_func = format_func or UnifiedDatasetLoader.format_sft_dialogue

        def add_text_field(examples):
            """Add formatted text field to examples."""
            texts = []
            # Get the length of the batch
            batch_size = len(next(iter(examples.values())))

            for i in range(batch_size):
                # Extract single example
                example = {k: v[i] for k, v in examples.items()}

                try:
                    text = format_func(example)
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to format example {i}: {e}")
                    texts.append("")  # Maintain alignment

            return {"text": texts}

        # Apply formatting
        dataset = dataset.map(
            add_text_field,
            batched=True,
            desc="Formatting SFT examples"
        )

        # Filter out empty texts
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: len(x.get('text', '')) > 0)
        filtered = original_size - len(dataset)

        if filtered > 0:
            logger.info(f"Filtered out {filtered} empty examples")

        return dataset

    @staticmethod
    def preprocess_dpo_dataset(
        dataset: Dataset,
        format_func: Optional[Callable] = None
    ) -> Dataset:
        """
        Preprocess dataset for DPO training.

        Args:
            dataset: Raw dataset with context, chosen, rejected fields
            format_func: Optional custom prompt formatting function

        Returns:
            Dataset with prompt, chosen, rejected fields
        """
        format_func = format_func or UnifiedDatasetLoader.format_dpo_prompt

        def format_dpo_examples(examples):
            """Format DPO examples with prompts."""
            prompts = []
            chosen = []
            rejected = []

            batch_size = len(examples.get('context', []))

            for i in range(batch_size):
                try:
                    # Format prompt from context
                    context = examples['context'][i]
                    prompt = format_func(context)
                    prompts.append(prompt)

                    # Keep chosen and rejected as is
                    chosen.append(examples['chosen'][i])
                    rejected.append(examples['rejected'][i])

                except Exception as e:
                    logger.warning(f"Failed to format DPO example {i}: {e}")
                    # Skip this example by not adding it

            return {
                'prompt': prompts,
                'chosen': chosen,
                'rejected': rejected
            }

        # Check if dataset already has correct format
        if 'prompt' in dataset.column_names:
            logger.info("Dataset already has DPO format")
            return dataset

        # Apply formatting if we have context field
        if 'context' in dataset.column_names:
            dataset = dataset.map(
                format_dpo_examples,
                batched=True,
                remove_columns=['context'],
                desc="Formatting DPO examples"
            )

        # Validate required fields
        required_fields = {'prompt', 'chosen', 'rejected'}
        if not required_fields.issubset(set(dataset.column_names)):
            missing = required_fields - set(dataset.column_names)
            raise ValueError(f"DPO dataset missing required fields: {missing}")

        return dataset

    @staticmethod
    def load_sft_dataset(
        file_path: Union[str, Path],
        format_func: Optional[Callable] = None,
        preprocess: bool = True
    ) -> Dataset:
        """
        Load and preprocess dataset for SFT training.

        Args:
            file_path: Path to dataset file
            format_func: Optional custom formatting function
            preprocess: Whether to preprocess the dataset

        Returns:
            Dataset ready for SFT training
        """
        dataset = UnifiedDatasetLoader.load_dataset_auto(file_path)

        if preprocess:
            dataset = UnifiedDatasetLoader.preprocess_sft_dataset(
                dataset, format_func
            )

        logger.info(f"Loaded SFT dataset with {len(dataset)} examples")
        return dataset

    @staticmethod
    def load_dpo_dataset(
        file_path: Union[str, Path],
        format_func: Optional[Callable] = None,
        preprocess: bool = True
    ) -> Dataset:
        """
        Load and preprocess dataset for DPO training.

        Args:
            file_path: Path to dataset file
            format_func: Optional custom prompt formatting function
            preprocess: Whether to preprocess the dataset

        Returns:
            Dataset ready for DPO training
        """
        dataset = UnifiedDatasetLoader.load_dataset_auto(file_path)

        if preprocess:
            dataset = UnifiedDatasetLoader.preprocess_dpo_dataset(
                dataset, format_func
            )

        logger.info(f"Loaded DPO dataset with {len(dataset)} pairs")
        return dataset

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> tuple[Dataset, Dataset]:
        """
        Split dataset into train and eval sets.

        Args:
            dataset: Dataset to split
            train_ratio: Ratio of data for training
            seed: Random seed for splitting

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        split = dataset.train_test_split(
            test_size=1 - train_ratio,
            seed=seed
        )
        return split['train'], split['test']