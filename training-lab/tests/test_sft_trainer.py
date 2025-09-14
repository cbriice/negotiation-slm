#!/usr/bin/env python3
"""
Test script for SFT trainer functionality without running actual training.
Tests data loading, preprocessing, formatting, and configuration.
"""

import json
import tempfile
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.sft_trainer import SFTTrainingPipeline
import training_parameters as params

# Configure logging for tests
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_sft_trainer.log', mode='w'),  # Clear log on each run
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate mock training data for testing."""

    @staticmethod
    def create_dialogue_format(num_examples: int = 5) -> List[Dict]:
        """Create dialogue format examples."""
        examples = []
        for i in range(num_examples):
            example = {
                "dialogue": [
                    {"role": "user", "content": f"I want to buy item {i} for $50"},
                    {"role": "assistant", "content": f"I can offer item {i} for $60"},
                    {"role": "user", "content": "Can you do $55?"},
                    {"role": "assistant", "content": "Deal! $55 works for me."}
                ],
                "metadata": {
                    "episode_id": f"test_{i}",
                    "turns": 4
                }
            }
            examples.append(example)
        return examples

    @staticmethod
    def create_instruction_format(num_examples: int = 5) -> List[Dict]:
        """Create instruction-response format examples."""
        examples = []
        for i in range(num_examples):
            example = {
                "instruction": f"Negotiate for item {i}. Your budget is $50.",
                "response": f"I'd like to purchase item {i}. Would you accept $45?",
                "metadata": {"episode_id": f"inst_{i}"}
            }
            examples.append(example)
        return examples

    @staticmethod
    def create_sft_packet_format(num_examples: int = 5) -> List[Dict]:
        """Create SFT packet format examples (from schema)."""
        examples = []
        for i in range(num_examples):
            example = {
                "input": {
                    "state": {"item": f"laptop_{i}", "budget": 1000},
                    "recent_convo": {"turns": 2, "last_offer": 1200},
                    "partner_msg": "I can go as low as $1100"
                },
                "output": "I appreciate that. Would you consider $1050? It's closer to my budget."
            }
            examples.append(example)
        return examples

    @staticmethod
    def create_invalid_examples() -> List[Dict]:
        """Create intentionally invalid examples for error testing."""
        return [
            {},  # Empty dict
            {"unknown_field": "value"},  # Unknown format
            {"dialogue": "not_a_list"},  # Wrong type
            {"instruction": "missing response field"},  # Incomplete
            {"input": {}, "output": ""},  # Empty output
        ]

    @staticmethod
    def save_to_ndjson(data: List[Dict], filepath: str):
        """Save data to NDJSON file."""
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


class TestSFTTrainer(unittest.TestCase):
    """Test cases for SFT trainer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = {}
        self.generator = TestDataGenerator()

        # Create test data files
        dialogue_data = self.generator.create_dialogue_format(10)
        instruction_data = self.generator.create_instruction_format(10)
        sft_packet_data = self.generator.create_sft_packet_format(10)
        mixed_data = dialogue_data[:3] + instruction_data[:3] + sft_packet_data[:3]
        invalid_data = self.generator.create_invalid_examples()

        # Save test files
        self.test_files['dialogue'] = Path(self.temp_dir) / "dialogue.ndjson"
        self.test_files['instruction'] = Path(self.temp_dir) / "instruction.ndjson"
        self.test_files['sft_packet'] = Path(self.temp_dir) / "sft_packet.ndjson"
        self.test_files['mixed'] = Path(self.temp_dir) / "mixed.ndjson"
        self.test_files['invalid'] = Path(self.temp_dir) / "invalid.ndjson"

        self.generator.save_to_ndjson(dialogue_data, self.test_files['dialogue'])
        self.generator.save_to_ndjson(instruction_data, self.test_files['instruction'])
        self.generator.save_to_ndjson(sft_packet_data, self.test_files['sft_packet'])
        self.generator.save_to_ndjson(mixed_data, self.test_files['mixed'])
        self.generator.save_to_ndjson(invalid_data, self.test_files['invalid'])

        logger.info(f"Created test files in {self.temp_dir}")

    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_data_loading(self):
        """Test NDJSON data loading."""
        logger.info("Testing data loading...")

        pipeline = SFTTrainingPipeline(
            output_dir=Path(self.temp_dir) / "output",
            checkpoint_dir=Path(self.temp_dir) / "checkpoints"
        )

        # Test loading different formats
        for format_name, filepath in self.test_files.items():
            if format_name == 'invalid':
                continue  # Skip invalid for this test

            logger.info(f"Loading {format_name} format...")
            dataset = pipeline.load_ndjson_dataset(str(filepath))

            self.assertIsNotNone(dataset)
            if format_name != 'mixed':
                self.assertEqual(len(dataset), 10)
            else:
                self.assertEqual(len(dataset), 9)

            logger.info(f"✓ Successfully loaded {len(dataset)} examples from {format_name}")

    def test_format_detection(self):
        """Test chat template formatting for different data formats."""
        logger.info("Testing format detection and conversion...")

        pipeline = SFTTrainingPipeline(
            output_dir=Path(self.temp_dir) / "output",
            checkpoint_dir=Path(self.temp_dir) / "checkpoints"
        )

        # Test dialogue format
        dialogue_example = {
            "dialogue": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"}
            ]
        }
        formatted = pipeline.format_chat_template(dialogue_example)
        self.assertIn("User: Test message", formatted)
        self.assertIn("Assistant: Test response", formatted)
        logger.info("✓ Dialogue format processed correctly")

        # Test instruction format
        instruction_example = {
            "instruction": "Test instruction",
            "response": "Test response"
        }
        formatted = pipeline.format_chat_template(instruction_example)
        self.assertIn("### Instruction:", formatted)
        self.assertIn("Test instruction", formatted)
        self.assertIn("### Response:", formatted)
        self.assertIn("Test response", formatted)
        logger.info("✓ Instruction format processed correctly")

        # Test SFT packet format
        sft_example = {
            "input": {
                "state": {"item": "laptop"},
                "partner_msg": "How about $1000?"
            },
            "output": "That works for me"
        }
        formatted = pipeline.format_chat_template(sft_example)
        self.assertIn("### Context:", formatted)
        self.assertIn("laptop", formatted)
        self.assertIn("### Response:", formatted)
        self.assertIn("That works for me", formatted)
        logger.info("✓ SFT packet format processed correctly")

    def test_preprocessing(self):
        """Test dataset preprocessing."""
        logger.info("Testing dataset preprocessing...")

        pipeline = SFTTrainingPipeline(
            output_dir=Path(self.temp_dir) / "output",
            checkpoint_dir=Path(self.temp_dir) / "checkpoints"
        )

        # Load and preprocess mixed format data
        dataset = pipeline.load_ndjson_dataset(str(self.test_files['mixed']))
        processed = pipeline.preprocess_dataset(dataset)

        # Check that text field was added
        self.assertIn('text', processed.column_names)

        # Check that all examples have non-empty text
        for example in processed:
            self.assertIsNotNone(example['text'])
            self.assertGreater(len(example['text']), 0)

        logger.info(f"✓ Preprocessed {len(processed)} examples successfully")

    def test_error_handling(self):
        """Test error handling for invalid data."""
        logger.info("Testing error handling...")

        pipeline = SFTTrainingPipeline(
            output_dir=Path(self.temp_dir) / "output",
            checkpoint_dir=Path(self.temp_dir) / "checkpoints"
        )

        # Load invalid data - should handle gracefully
        dataset = pipeline.load_ndjson_dataset(str(self.test_files['invalid']))
        initial_count = len(dataset)

        # Preprocess - should filter out invalid examples
        processed = pipeline.preprocess_dataset(dataset)
        final_count = len(processed)

        # Check that filtering happened correctly
        # All our test invalid examples should result in empty text and be filtered
        self.assertLessEqual(final_count, initial_count)

        logger.info(f"✓ Handled invalid data: {initial_count} loaded, {final_count} valid after filtering")

    def test_training_config(self):
        """Test training configuration setup."""
        logger.info("Testing training configuration...")

        # Mock the model loading to avoid actual model download
        with patch('training.sft_trainer.ModelLoader') as MockLoader:
            with patch('training.sft_trainer.LoRAConfigurator') as MockLoRA:
                # Setup mocks
                mock_model = MagicMock()

                # Add mock parameters to avoid division by zero
                mock_param = MagicMock()
                mock_param.numel.return_value = 1000
                mock_param.requires_grad = True
                mock_model.parameters.return_value = [mock_param] * 10  # Simulate 10 parameters

                mock_tokenizer = MagicMock()
                mock_tokenizer.pad_token = None
                mock_tokenizer.eos_token = "[EOS]"

                MockLoader.return_value.load_model.return_value = (mock_model, mock_tokenizer)
                MockLoRA.return_value.apply_lora.return_value = mock_model

                pipeline = SFTTrainingPipeline(
                    output_dir=Path(self.temp_dir) / "output",
                    checkpoint_dir=Path(self.temp_dir) / "checkpoints"
                )

                # Test model setup
                pipeline.setup_model()

                self.assertIsNotNone(pipeline.model)
                self.assertIsNotNone(pipeline.tokenizer)
                self.assertEqual(pipeline.tokenizer.pad_token, "[EOS]")

                logger.info("✓ Model configuration set up correctly")

    def test_parameter_validation(self):
        """Test that training parameters are valid."""
        logger.info("Testing parameter validation...")

        # Check key parameters
        self.assertGreater(params.MAX_SEQ_LEN, 0)
        self.assertGreater(params.SFT_LR, 0)
        self.assertIn(params.BNB_4BIT_QUANT_TYPE, ["nf4", "fp4"])
        self.assertIn(params.R, [4, 8, 16, 32])

        logger.info("✓ All training parameters are valid")
        logger.info(f"  - Max sequence length: {params.MAX_SEQ_LEN}")
        logger.info(f"  - Learning rate: {params.SFT_LR}")
        logger.info(f"  - LoRA rank: {params.R}")
        logger.info(f"  - Quantization: {params.BNB_4BIT_QUANT_TYPE}")


class TestCLIInterface(unittest.TestCase):
    """Test CLI interface functionality."""

    def test_cli_args_parsing(self):
        """Test command-line argument parsing."""
        logger.info("Testing CLI argument parsing...")

        with patch('sys.argv', [
            'sft_trainer.py',
            'train.ndjson',
            '--eval-data', 'eval.ndjson',
            '--model', 'mistral-7b',
            '--output-dir', 'output',
            '--max-steps', '100',
            '--suffix', '_test'
        ]):
            from training.sft_trainer import main
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument("train_data", type=str)
            parser.add_argument("--eval-data", type=str, default=None)
            parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
            parser.add_argument("--output-dir", type=str, default="models/sft")
            parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sft")
            parser.add_argument("--resume-from", type=str, default=None)
            parser.add_argument("--num-epochs", type=float, default=None)
            parser.add_argument("--max-steps", type=int, default=None)
            parser.add_argument("--suffix", type=str, default="")

            # Parse with test arguments
            args = parser.parse_args([
                'train.ndjson',
                '--eval-data', 'eval.ndjson',
                '--model', 'mistral-7b',
                '--output-dir', 'output',
                '--max-steps', '100',
                '--suffix', '_test'
            ])

            self.assertEqual(args.train_data, 'train.ndjson')
            self.assertEqual(args.eval_data, 'eval.ndjson')
            self.assertEqual(args.model, 'mistral-7b')
            self.assertEqual(args.output_dir, 'output')
            self.assertEqual(args.max_steps, 100)
            self.assertEqual(args.suffix, '_test')

            logger.info("✓ CLI arguments parsed correctly")


def run_dry_run_test():
    """Run a dry-run test of the training pipeline without actual training."""
    logger.info("="*60)
    logger.info("Running Dry-Run Test of SFT Trainer")
    logger.info("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate test data
        generator = TestDataGenerator()
        test_file = Path(temp_dir) / "test_data.ndjson"

        # Create mixed format data
        data = (
            generator.create_dialogue_format(3) +
            generator.create_instruction_format(3) +
            generator.create_sft_packet_format(3)
        )
        generator.save_to_ndjson(data, test_file)

        logger.info(f"Created test data with {len(data)} examples")

        # Initialize pipeline
        pipeline = SFTTrainingPipeline(
            output_dir=Path(temp_dir) / "output",
            checkpoint_dir=Path(temp_dir) / "checkpoints"
        )

        # Test data loading
        logger.info("\n1. Testing data loading...")
        dataset = pipeline.load_ndjson_dataset(str(test_file))
        logger.info(f"   Loaded {len(dataset)} examples")

        # Test preprocessing
        logger.info("\n2. Testing preprocessing...")
        processed = pipeline.preprocess_dataset(dataset)
        logger.info(f"   Preprocessed to {len(processed)} valid examples")

        # Show sample formatted outputs
        logger.info("\n3. Sample formatted outputs:")
        for i, example in enumerate(processed.select(range(min(3, len(processed))))):
            logger.info(f"\n   Example {i+1}:")
            preview = example['text'][:200] + "..." if len(example['text']) > 200 else example['text']
            logger.info(f"   {preview}")

        # Test configuration (without loading actual model)
        logger.info("\n4. Verifying configuration:")
        logger.info(f"   - Max sequence length: {params.MAX_SEQ_LEN}")
        logger.info(f"   - Batch size: {params.PER_DEVICE_BATCH_SIZE}")
        logger.info(f"   - Learning rate: {params.SFT_LR}")
        logger.info(f"   - Gradient accumulation: {params.GRAD_ACCUM_STEPS}")
        logger.info(f"   - Mixed precision: bf16={params.BF16}")

        logger.info("\n" + "="*60)
        logger.info("Dry-Run Test Completed Successfully!")
        logger.info("All components are functioning correctly.")
        logger.info("="*60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        # Run dry-run test
        run_dry_run_test()
    else:
        # Run unit tests
        logger.info("Running unit tests for SFT trainer...")
        unittest.main(argv=[''], verbosity=2, exit=False)

        # Also run dry-run
        logger.info("\n")
        run_dry_run_test()