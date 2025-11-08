"""
Converter for messages-based SFT packets to training format.
Handles conversion from the new negotiation dataset format with messages array
to the expected SFT training format with proper response masking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset

logger = logging.getLogger(__name__)


class MessagesToSFTConverter:
    """
    Converts messages-based SFT packets to the expected training format.

    Input format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "..."},
            ...
        ],
        "system": {...},
        "meta": {...}
    }

    Output format (as per sft_formatting.txt):
    ---
    system:
    assistant_style:
    key_strategy:
    difficulty:
    difficulty_bin:
    ---

    User/Assistant:
    Assistant/User:
    ...
    """

    def __init__(self, mask_non_assistant: bool = True):
        """
        Initialize the converter.

        Args:
            mask_non_assistant: Whether to apply -100 mask to non-assistant responses
        """
        self.mask_non_assistant = mask_non_assistant
        self.ignore_index = -100

    def convert_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single packet from messages format to SFT format.

        Args:
            packet: Input packet with messages, system, and meta fields

        Returns:
            Formatted training example with text field and optional masking info
        """
        messages = packet.get('messages', [])
        system_info = packet.get('system', {})
        meta = packet.get('meta', {})

        # Build the formatted text
        formatted_parts = []

        # Add system header
        formatted_parts.append("---")
        formatted_parts.append("system:")

        # Add system fields if available
        if 'assistant_style' in system_info:
            formatted_parts.append(f"assistant_style: {system_info['assistant_style']}")
        if 'key_strategy' in system_info:
            formatted_parts.append(f"key_strategy: {system_info['key_strategy']}")
        if 'difficulty' in system_info:
            formatted_parts.append(f"difficulty: {system_info['difficulty']}")
        if 'difficulty_bin' in system_info:
            formatted_parts.append(f"difficulty_bin: {system_info['difficulty_bin']}")

        formatted_parts.append("---")
        formatted_parts.append("")

        # Process messages into dialogue format
        dialogue_text = []
        masking_regions = []  # Track which parts should be masked

        # Extract system message if present
        system_message = None
        conversation_messages = []

        for msg in messages:
            # Handle dropout - msg might be missing role or content
            if not isinstance(msg, dict):
                continue

            role = msg.get('role', None)
            content = msg.get('content', '')

            if role == 'system':
                system_message = content if content else None
            elif role:  # Only add if role exists
                conversation_messages.append(msg)

        # Add system message at the top if present
        if system_message:
            formatted_parts.append(f"System: {system_message}")
            formatted_parts.append("")

        # Track masking for system region if masking is enabled
        if self.mask_non_assistant:
            # Calculate the length of the entire system section (header + system message)
            system_section = '\n'.join(formatted_parts)
            system_section_length = len(system_section)

            # Add masking region for the entire system section
            masking_regions.append({
                'start': 0,
                'end': system_section_length,
                'mask': True,
                'role': 'system'
            })

        # Process conversation messages
        for msg in conversation_messages:
            # Handle dropout - safely get role and content
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Skip empty content messages (dropout case)
            if not content:
                continue

            # Map roles to expected format
            if role == 'assistant':
                label = "Assistant"
            elif role == 'user':
                label = "User"
            else:
                label = role.capitalize()

            # Add the message
            message_line = f"{label}: {content}"
            dialogue_text.append(message_line)

            # Track masking info if needed
            if self.mask_non_assistant:
                # Calculate position relative to the full formatted text
                prefix = formatted_parts + dialogue_text[:-1]
                start_pos = len('\n'.join(prefix))
                if prefix:  # If there's content before this message
                    start_pos += 1  # Account for newline

                end_pos = start_pos + len(message_line)

                # Only assistant responses should not be masked
                should_mask = (role != 'assistant')
                masking_regions.append({
                    'start': start_pos,
                    'end': end_pos,
                    'mask': should_mask,
                    'role': role
                })

        # Combine all parts
        formatted_text = '\n'.join(formatted_parts + dialogue_text)

        # Create the output format
        output = {
            'text': formatted_text,
            'original_packet': packet,  # Keep original for reference
        }

        # Add masking info if needed
        if self.mask_non_assistant and masking_regions:
            output['masking_regions'] = masking_regions
            output['response_only_text'] = self._extract_response_only_text(
                dialogue_text, conversation_messages
            )

        # Add metadata
        output['meta'] = {
            **meta,
            'num_turns': len(conversation_messages),
            'has_system': system_message is not None,
            'assistant_turns': sum(1 for msg in conversation_messages if msg['role'] == 'assistant'),
            'user_turns': sum(1 for msg in conversation_messages if msg['role'] == 'user'),
        }

        return output

    def _extract_response_only_text(
        self,
        dialogue_text: List[str],
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Extract only the assistant responses for validation.

        Args:
            dialogue_text: Formatted dialogue lines
            messages: Original messages

        Returns:
            Text containing only assistant responses
        """
        response_lines = []
        dialogue_idx = 0
        for msg in messages:
            # Handle dropout - safely check role and content
            role = msg.get('role', None)
            content = msg.get('content', '')

            # Skip if no role or empty content (dropout)
            if not role or not content:
                continue

            # If this is an assistant message, add it
            if role == 'assistant' and dialogue_idx < len(dialogue_text):
                response_lines.append(dialogue_text[dialogue_idx])

            dialogue_idx += 1

        return '\n'.join(response_lines)

    def convert_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        validate: bool = True,
        shuffle: bool = False,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convert an entire JSONL file from messages format to SFT format.

        Args:
            input_path: Path to input JSONL file with messages format
            output_path: Optional path to save converted JSONL file
            validate: Whether to validate the conversion
            shuffle: Whether to shuffle the converted examples
            seed: Random seed for shuffling

        Returns:
            Tuple of (converted examples, statistics)
        """
        input_path_obj = Path(input_path)
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path_obj}")

        logger.info(f"Converting {input_path_obj} to SFT format...")

        converted_examples = []
        stats = {
            'total_packets': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_messages': 0,
            'total_assistant_turns': 0,
            'total_user_turns': 0,
            'packets_with_system': 0,
            'errors': []
        }

        # Read and convert each packet
        with open(input_path_obj, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                stats['total_packets'] += 1

                try:
                    packet = json.loads(line)
                    converted = self.convert_packet(packet)
                    converted_examples.append(converted)

                    # Update statistics
                    stats['successful_conversions'] += 1
                    stats['total_messages'] += len(packet.get('messages', []))
                    stats['total_assistant_turns'] += converted['meta']['assistant_turns']
                    stats['total_user_turns'] += converted['meta']['user_turns']
                    if converted['meta']['has_system']:
                        stats['packets_with_system'] += 1

                    # Validate if requested
                    if validate and line_num <= 5:  # Validate first 5 examples
                        self._validate_conversion(packet, converted, line_num)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    stats['failed_conversions'] += 1
                    stats['errors'].append(f"Line {line_num}: JSON decode error")
                except Exception as e:
                    logger.error(f"Error converting line {line_num}: {e}")
                    stats['failed_conversions'] += 1
                    stats['errors'].append(f"Line {line_num}: {str(e)}")

        # Shuffle converted examples if requested
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(converted_examples)
            logger.info(f"Shuffled {len(converted_examples)} examples with seed {seed}")

        # Save if output path provided
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path_obj, 'w') as f:
                for example in converted_examples:
                    # Save only essential fields to JSONL
                    output_data = {
                        'text': example['text'],
                        'meta': example['meta']
                    }
                    if 'masking_regions' in example:
                        output_data['masking_regions'] = example['masking_regions']

                    f.write(json.dumps(output_data) + '\n')

            logger.info(f"Saved {len(converted_examples)} converted examples to {output_path_obj}")

        # Log statistics
        self._log_statistics(stats)

        return converted_examples, stats

    def _validate_conversion(
        self,
        original: Dict[str, Any],
        converted: Dict[str, Any],
        line_num: int
    ):
        """
        Validate that conversion preserved essential information.

        Args:
            original: Original packet
            converted: Converted packet
            line_num: Line number for error reporting
        """
        # Check that all messages are present
        original_messages = [
            msg for msg in original.get('messages', [])
            if msg['role'] != 'system'
        ]

        # Count dialogue turns in converted text
        converted_text = converted['text']
        user_count = converted_text.count('User:')
        assistant_count = converted_text.count('Assistant:')

        original_user = sum(1 for msg in original_messages if msg['role'] == 'user')
        original_assistant = sum(1 for msg in original_messages if msg['role'] == 'assistant')

        if user_count != original_user:
            logger.warning(
                f"Line {line_num}: User turn count mismatch - "
                f"original: {original_user}, converted: {user_count}"
            )

        if assistant_count != original_assistant:
            logger.warning(
                f"Line {line_num}: Assistant turn count mismatch - "
                f"original: {original_assistant}, converted: {assistant_count}"
            )

        # Verify system info is preserved
        if 'system' in original and original['system']:
            system_info = original['system']
            for key in ['assistant_style', 'key_strategy', 'difficulty']:
                if key in system_info:
                    if str(system_info[key]) not in converted_text:
                        logger.warning(
                            f"Line {line_num}: System field '{key}' not found in converted text"
                        )

    def _log_statistics(self, stats: Dict[str, Any]):
        """Log conversion statistics."""
        logger.info("=" * 60)
        logger.info("Conversion Statistics:")
        logger.info(f"  Total packets: {stats['total_packets']}")
        logger.info(f"  Successful: {stats['successful_conversions']}")
        logger.info(f"  Failed: {stats['failed_conversions']}")
        logger.info(f"  Total messages: {stats['total_messages']}")
        logger.info(f"  Assistant turns: {stats['total_assistant_turns']}")
        logger.info(f"  User turns: {stats['total_user_turns']}")
        logger.info(f"  Packets with system message: {stats['packets_with_system']}")

        if stats['successful_conversions'] > 0:
            avg_messages = stats['total_messages'] / stats['successful_conversions']
            logger.info(f"  Average messages per packet: {avg_messages:.2f}")

        if stats['errors']:
            logger.warning(f"  Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                logger.warning(f"    - {error}")

        logger.info("=" * 60)

    def convert_to_dataset(
        self,
        input_path: str,
        max_seq_length: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> Dataset:
        """
        Convert JSONL file directly to HuggingFace Dataset.

        Args:
            input_path: Path to input JSONL file
            max_seq_length: Optional maximum sequence length for filtering
            shuffle: Whether to shuffle the dataset (default: True)
            seed: Random seed for shuffling (default: 42)

        Returns:
            HuggingFace Dataset ready for training
        """
        converted_examples, _ = self.convert_file(input_path, validate=True)

        # Extract only the text field for dataset
        dataset_records = []
        for example in converted_examples:
            record = {
                'text': example['text'],
                'assistant_turns': example['meta']['assistant_turns'],
                'user_turns': example['meta']['user_turns'],
                'num_turns': example['meta']['num_turns']
            }

            # Add masking info if available
            if 'response_only_text' in example:
                record['response_only_text'] = example['response_only_text']

            dataset_records.append(record)

        # Create dataset
        dataset = Dataset.from_list(dataset_records)

        # Filter by sequence length if specified
        if max_seq_length:
            original_size = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x['text']) <= max_seq_length,
                desc="Filtering by sequence length"
            )
            filtered = original_size - len(dataset)
            if filtered > 0:
                logger.info(f"Filtered {filtered} examples exceeding {max_seq_length} chars")

        # Shuffle dataset if requested
        if shuffle:
            logger.info(f"Shuffling dataset with seed {seed}...")
            dataset = dataset.shuffle(seed=seed)

        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset


def main():
    """CLI interface for the converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert messages-based SFT packets to training format"
    )
    parser.add_argument(
        'input_file',
        help="Input JSONL file with messages format"
    )
    parser.add_argument(
        '--output',
        '-o',
        help="Output JSONL file (optional)"
    )
    parser.add_argument(
        '--no-mask',
        action='store_true',
        help="Disable masking of non-assistant responses"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help="Validate the conversion (default: True)"
    )
    parser.add_argument(
        '--sample',
        type=int,
        help="Only convert first N packets (for testing)"
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help="Shuffle the dataset after conversion (default: True)"
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help="Disable dataset shuffling"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )

    args = parser.parse_args()

    # Initialize converter
    converter = MessagesToSFTConverter(mask_non_assistant=not args.no_mask)

    # Determine shuffle setting
    should_shuffle = not args.no_shuffle

    # Handle sample mode
    if args.sample:
        # Read only first N packets
        input_path = Path(args.input_file)
        temp_file = Path('/tmp/sample_packets.jsonl')

        with open(input_path, 'r') as inf, open(temp_file, 'w') as outf:
            for i, line in enumerate(inf):
                if i >= args.sample:
                    break
                outf.write(line)

        # Convert the sample
        converted, _ = converter.convert_file(
            str(temp_file),
            args.output,
            validate=args.validate,
            shuffle=should_shuffle,
            seed=args.seed
        )

        # Show sample output
        if converted and not args.output:
            print("\nSample converted output (first example):")
            print("=" * 60)
            print(converted[0]['text'])
            print("=" * 60)
    else:
        # Convert full file
        converted, _ = converter.convert_file(
            args.input_file,
            args.output,
            validate=args.validate,
            shuffle=should_shuffle,
            seed=args.seed
        )

        # Show sample if no output file specified
        if converted and not args.output:
            print("\nSample converted output (first example):")
            print("=" * 60)
            print(converted[0]['text'])
            print("=" * 60)


if __name__ == "__main__":
    main()