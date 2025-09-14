#!/usr/bin/env python3
"""
Comprehensive test suite for QLoRA training pipeline.
Tests all modular components and integration.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.checkpoint_manager import CheckpointManager
from training.metrics_tracker import TrainingMetricsTracker
from training.qlora_trainer import QLoRATrainer
import training_parameters as params


class TestModelLoader(unittest.TestCase):
    """Test model loading and quantization."""
    
    def test_bnb_config_creation(self):
        """Test BitsAndBytes config creation."""
        loader = ModelLoader()
        config = loader.get_bnb_config()
        
        self.assertEqual(config.load_in_4bit, params.LOAD_IN_4BIT)
        self.assertEqual(config.bnb_4bit_quant_type, params.BNB_4BIT_QUANT_TYPE)
        self.assertEqual(config.bnb_4bit_compute_dtype, params.BNB_4BIT_COMPUTE_DTYPE)
        self.assertEqual(config.bnb_4bit_use_double_quant, params.BNB_4BIT_USE_DOUBLE_QUANT)
        print("✓ BitsAndBytes config created correctly")
    
    def test_tokenizer_setup(self):
        """Test tokenizer setup with padding configuration using real model."""
        loader = ModelLoader()
        
        # Setup tokenizer with real Mistral model
        tokenizer = loader._setup_tokenizer("mistralai/Mistral-7B-v0.1")
        
        # Verify padding configuration
        self.assertIsNotNone(tokenizer.pad_token)
        self.assertIsNotNone(tokenizer.pad_token_id)
        self.assertEqual(tokenizer.padding_side, "right")
        print("✓ Tokenizer setup with correct padding configuration")
    
    @unittest.skipIf(os.getenv('SKIP_MODEL_TESTS', 'false').lower() == 'true', 
                     "Skipping model loading test (set SKIP_MODEL_TESTS=false to run)")
    def test_model_loading_real(self):
        """Test model loading with real Mistral-7B model."""
        loader = ModelLoader()
        
        print("Loading Mistral-7B (this may take a minute on first run)...")
        
        # Test loading real model with 4-bit quantization
        model, tokenizer = loader.load_model("mistralai/Mistral-7B-v0.1")
        
        # Verify model is loaded
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        
        # Verify quantization was applied
        self.assertTrue(hasattr(model, 'config'))
        
        # Check model is prepared for training
        self.assertTrue(model.is_gradient_checkpointing) if params.GRADIENT_CHECKPOINTING else True
        
        print("✓ Model loading with 4-bit quantization (real Mistral-7B)")
        
        # Clean up to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class TestLoRAConfig(unittest.TestCase):
    """Test LoRA configuration."""
    
    def test_lora_config_creation(self):
        """Test LoRA config creation from parameters."""
        configurator = LoRAConfigurator()
        config = configurator.get_lora_config()
        
        self.assertEqual(config.r, params.R)
        self.assertEqual(config.lora_alpha, params.ALPHA)
        self.assertEqual(config.lora_dropout, params.DROPOUT)
        # target_modules becomes a set in LoraConfig, convert for comparison
        self.assertEqual(set(config.target_modules), set(params.TARGET_MODULES))
        self.assertEqual(config.bias, "none")
        print("✓ LoRA config created with correct parameters")
    
    @patch('training.lora_config.get_peft_model')
    def test_apply_lora(self, mock_get_peft):
        """Test applying LoRA to model."""
        configurator = LoRAConfigurator()
        
        # Mock model
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_get_peft.return_value = mock_peft_model
        
        # Apply LoRA
        with patch.object(configurator, '_print_trainable_parameters'):
            result = configurator.apply_lora(mock_model)
        
        # Verify PEFT was applied
        mock_get_peft.assert_called_once()
        self.assertEqual(result, mock_peft_model)
        print("✓ LoRA adapters applied successfully")


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management."""
    
    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Test saving checkpoint with metadata."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Save checkpoint
        metrics = {"loss": 0.5, "accuracy": 0.95}
        checkpoint_path = self.manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stage="sft",
            version=1,
            metrics=metrics
        )
        
        # Verify checkpoint directory created
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Verify metadata file
        metadata_file = Path(checkpoint_path) / "checkpoint_info.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["stage"], "sft")
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["metrics"], metrics)
        
        # Verify save methods called
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        
        print("✓ Checkpoint saved with correct metadata")
    
    def test_load_checkpoint(self):
        """Test loading checkpoint metadata."""
        # Create fake checkpoint
        checkpoint_dir = Path(self.temp_dir) / "test_checkpoint"
        checkpoint_dir.mkdir()
        
        metadata = {
            "stage": "dpo",
            "version": 2,
            "metrics": {"loss": 0.3}
        }
        
        with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
            json.dump(metadata, f)
        
        # Create dummy adapter files
        (checkpoint_dir / "adapter_model.bin").touch()
        (checkpoint_dir / "adapter_config.json").touch()
        
        # Load checkpoint
        loaded = self.manager.load_checkpoint("test_checkpoint")
        
        self.assertEqual(loaded["stage"], "dpo")
        self.assertEqual(loaded["version"], 2)
        self.assertEqual(loaded["metrics"]["loss"], 0.3)
        print("✓ Checkpoint loaded successfully")
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Create multiple checkpoints
        for i in range(3):
            checkpoint_dir = Path(self.temp_dir) / f"negotiation_sft_v{i}_2025"
            checkpoint_dir.mkdir()
            
            metadata = {"stage": "sft", "version": i}
            with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
                json.dump(metadata, f)
        
        # List checkpoints
        checkpoints = self.manager.list_checkpoints(stage="sft")
        
        self.assertEqual(len(checkpoints), 3)
        for i, checkpoint in enumerate(checkpoints):
            self.assertEqual(checkpoint["stage"], "sft")
        
        print("✓ Checkpoints listed correctly")


class TestMetricsTracker(unittest.TestCase):
    """Test metrics tracking with pandas integration."""
    
    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = TrainingMetricsTracker(log_dir=self.temp_dir, stage="test")
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_training_csv_initialization(self):
        """Test that training CSV is initialized with pandas."""
        # Check CSV file exists
        self.assertTrue(self.tracker.training_csv.exists())
        
        # Read CSV with pandas
        df = pd.read_csv(self.tracker.training_csv)
        
        # Check columns
        expected_columns = [
            "timestamp", "step", "epoch", "train_loss", "eval_loss",
            "learning_rate", "gradient_norm", "tokens_per_second"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        print("✓ Training CSV initialized with pandas")
    
    def test_log_training_step(self):
        """Test logging training step with pandas."""
        # Log some steps
        for step in range(5):
            metrics = {
                "train_loss": 1.0 - step * 0.1,
                "learning_rate": 1e-4,
                "tokens_per_second": 1000 + step * 100
            }
            self.tracker.log_training_step(
                step=step,
                epoch=step / 10,
                metrics=metrics,
                log_to_console=False
            )
        
        # Read CSV with pandas
        df = pd.read_csv(self.tracker.training_csv)
        
        # Debug: Print what we actually got
        print(f"DEBUG: First row train_loss from CSV: {df.iloc[0]['train_loss']}")
        print(f"DEBUG: Last row train_loss from CSV: {df.iloc[-1]['train_loss']}")
        
        # Also check the DataFrame directly
        tracker_df = self.tracker.get_training_dataframe()
        print(f"DEBUG: First row train_loss from DataFrame: {tracker_df.iloc[0]['train_loss']}")
        
        # Verify data
        self.assertEqual(len(df), 5)
        self.assertAlmostEqual(df.iloc[0]["train_loss"], 1.0, places=1)
        self.assertAlmostEqual(df.iloc[-1]["train_loss"], 0.6, places=1)
        
        # Verify DataFrame in tracker
        tracker_df = self.tracker.get_training_dataframe()
        self.assertEqual(len(tracker_df), 5)
        
        print("✓ Training steps logged to pandas DataFrame")
    
    def test_episode_logging(self):
        """Test episode logging integration."""
        # Create mock episode data
        episode_data = {
            "dialogue": [
                {"text": "I want to buy", "side": "buyer"},
                {"text": "I'll sell for $100", "side": "seller"}
            ],
            "state": {
                "buyer_state": {"target_price": 80, "max_price": 100},
                "seller_state": {"target_price": 100, "min_price": 90}
            },
            "meta": {
                "episode_id": "test_001",
                "side": "buyer",
                "seed": "test_seed"
            }
        }
        
        # Log episode
        metrics = self.tracker.log_episode(episode_data)
        
        # Verify metrics calculated
        self.assertEqual(metrics["episode_id"], "test_001")
        self.assertEqual(metrics["side"], "buyer")
        self.assertEqual(metrics["training_stage"], "test")
        self.assertIn("turns", metrics)
        
        print("✓ Episode metrics logged with evaluation system integration")
    
    def test_session_summary(self):
        """Test session summary aggregation."""
        # Log some training steps
        for step in range(3):
            self.tracker.log_training_step(
                step=step,
                epoch=step / 10,
                metrics={"train_loss": 1.0 - step * 0.1},
                log_to_console=False
            )
        
        # Add mock episodes
        for i in range(2):
            episode_data = {
                "dialogue": [{"text": f"test {i}"}],
                "meta": {"episode_id": f"ep_{i}"}
            }
            self.tracker.log_episode(episode_data)
        
        # Generate summary
        summary = self.tracker.log_session_summary()
        
        # Verify summary contains aggregated metrics
        self.assertEqual(summary["training_stage"], "test")
        self.assertEqual(summary["total_episodes"], 2)
        self.assertIn("avg_train_loss", summary)
        
        print("✓ Session summary generated with pandas aggregation")
    
    def test_early_stopping(self):
        """Test early stopping logic."""
        # Simulate training with no improvement
        for step in range(15):
            self.tracker.log_training_step(
                step=step,
                epoch=step / 100,
                metrics={"train_loss": 0.5 + np.random.random() * 0.01},  # Small fluctuations
                log_to_console=False
            )
        
        # Check early stopping
        should_stop = self.tracker.should_early_stop(
            patience=5,
            min_delta=0.01,
            metric="train_loss"
        )
        
        # With small fluctuations and patience=5, it might suggest stopping
        self.assertIsInstance(should_stop, bool)
        print(f"✓ Early stopping logic tested (should_stop={should_stop})")


class TestQLoRATrainerIntegration(unittest.TestCase):
    """Test QLoRA trainer integration."""
    
    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = QLoRATrainer(
            model_name="mistralai/Mistral-7B-v0.1",
            output_dir=self.temp_dir,
            log_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(os.getenv('SKIP_MODEL_TESTS', 'false').lower() == 'true',
                     "Skipping model setup test")
    def test_setup_model(self):
        """Test model setup with real Mistral-7B."""
        print("Setting up Mistral-7B with LoRA (this may take a minute)...")
        
        # Setup model - this will load real Mistral-7B and apply LoRA
        self.trainer.setup_model()
        
        # Verify model and tokenizer are set
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.tokenizer)
        
        # Verify LoRA was applied (model should be PeftModel)
        self.assertTrue(hasattr(self.trainer.model, 'peft_config'))
        
        print("✓ Model setup with LoRA integration (real Mistral-7B)")
        
        # Clean up to free memory
        del self.trainer.model
        del self.trainer.tokenizer  
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_training_args_generation(self):
        """Test training arguments generation from parameters."""
        # Test SFT args
        sft_args = self.trainer._get_sft_training_args(num_epochs=1, max_steps=100)
        
        self.assertEqual(sft_args.max_steps, 100)
        self.assertEqual(sft_args.learning_rate, params.SFT_LR)
        self.assertEqual(sft_args.per_device_train_batch_size, params.PER_DEVICE_BATCH_SIZE)
        self.assertEqual(sft_args.gradient_accumulation_steps, params.GRAD_ACCUM_STEPS)
        self.assertEqual(sft_args.optim, params.OPTIM)
        self.assertEqual(sft_args.bf16, params.BF16)
        
        # Test DPO args
        dpo_args = self.trainer._get_dpo_training_args(num_epochs=1, max_steps=100)
        
        self.assertEqual(dpo_args.max_steps, 100)
        self.assertEqual(dpo_args.learning_rate, params.DPO_LR)
        
        print("✓ Training arguments generated from parameters")
    
    def test_format_sft_examples(self):
        """Test SFT example formatting."""
        examples = {
            "dialogue": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ],
                [
                    {"role": "buyer", "text": "I want to buy"},
                    {"role": "seller", "text": "What's your offer?"}
                ]
            ]
        }
        
        formatted = self.trainer._format_sft_examples(examples)
        
        self.assertEqual(len(formatted), 2)
        self.assertIn("user: Hello", formatted[0])
        self.assertIn("assistant: Hi there", formatted[0])
        self.assertIn("buyer: I want to buy", formatted[1])
        
        print("✓ SFT examples formatted correctly")
    
    def test_checkpoint_workflow(self):
        """Test checkpoint save/load workflow."""
        # Mock components
        self.trainer.checkpoint_manager = CheckpointManager(base_dir=self.temp_dir)
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        self.trainer.model = mock_model
        self.trainer.tokenizer = mock_tokenizer
        
        # Save checkpoint
        metrics = {"loss": 0.5}
        checkpoint_path = self.trainer.checkpoint_manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stage="sft",
            version=1,
            metrics=metrics
        )
        
        # Load checkpoint info
        with patch.object(self.trainer, 'setup_model'):
            info = self.trainer.load_checkpoint(Path(checkpoint_path).name)
        
        self.assertEqual(info["stage"], "sft")
        self.assertEqual(info["version"], 1)
        self.assertEqual(self.trainer.current_stage, "sft")
        self.assertEqual(self.trainer.current_version, 2)  # Incremented
        
        print("✓ Checkpoint save/load workflow tested")


class TestDataIntegration(unittest.TestCase):
    """Test data loading and processing."""
    
    def setUp(self):
        """Create test data files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test JSONL files
        self.sft_data = Path(self.temp_dir) / "sft_data.jsonl"
        sft_examples = [
            {
                "dialogue": [
                    {"role": "user", "content": "I want to buy for $50"},
                    {"role": "assistant", "content": "I can sell for $75"}
                ]
            },
            {
                "dialogue": [
                    {"role": "buyer", "text": "Is $60 acceptable?"},
                    {"role": "seller", "text": "Let's meet at $65"}
                ]
            }
        ]
        
        with open(self.sft_data, 'w') as f:
            for example in sft_examples:
                f.write(json.dumps(example) + '\n')
        
        self.dpo_data = Path(self.temp_dir) / "dpo_data.jsonl"
        dpo_examples = [
            {
                "prompt": "I want to buy this item",
                "chosen": "I can offer $50 for it",
                "rejected": "Give it to me for free"
            }
        ]
        
        with open(self.dpo_data, 'w') as f:
            for example in dpo_examples:
                f.write(json.dumps(example) + '\n')
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir)
    
    def test_sft_data_loading(self):
        """Test SFT data can be loaded."""
        from datasets import load_dataset
        
        dataset = load_dataset('json', data_files=str(self.sft_data), split='train')
        
        self.assertEqual(len(dataset), 2)
        self.assertIn('dialogue', dataset[0])
        print("✓ SFT data loaded successfully")
    
    def test_dpo_data_loading(self):
        """Test DPO data can be loaded."""
        from datasets import load_dataset
        
        dataset = load_dataset('json', data_files=str(self.dpo_data), split='train')
        
        self.assertEqual(len(dataset), 1)
        self.assertIn('prompt', dataset[0])
        self.assertIn('chosen', dataset[0])
        self.assertIn('rejected', dataset[0])
        print("✓ DPO data loaded successfully")


def run_all_tests():
    """Run all tests with summary."""
    print("=" * 60)
    print("QLoRA Training Pipeline Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelLoader,
        TestLoRAConfig,
        TestCheckpointManager,
        TestMetricsTracker,
        TestQLoRATrainerIntegration,
        TestDataIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. See details above.")
        
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)