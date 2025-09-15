#!/usr/bin/env python3
"""Test script for CheckpointManager features."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from training.checkpoint_manager import CheckpointManager


def test_checkpoint_manager():
    """Test the CheckpointManager functionality."""

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(base_dir=temp_dir)

        print("Testing CheckpointManager features...")
        print("-" * 50)

        # Test 1: Version management
        print("\n1. Testing version management:")
        next_version = manager.get_next_version("baseline")
        print(f"   Next version for baseline stage: {next_version}")
        assert next_version == 1, "First version should be 1"

        # Test 2: Save checkpoint with dataset stats
        print("\n2. Testing checkpoint save with dataset stats:")

        # Mock model and tokenizer
        mock_model = Mock()
        mock_model.save_pretrained = Mock()
        mock_model.name_or_path = "mistralai/Mistral-7B-v0.1"

        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()

        # Set training start time
        manager.set_training_start_time()

        # Dataset statistics
        dataset_stats = {
            "num_examples": 1500,
            "avg_length": 256,
            "dataset_version": "v1.0",
            "data_distribution": {
                "deals_closed": 0.65,
                "no_deals": 0.35
            }
        }

        # Metrics
        metrics = {
            "loss": 0.45,
            "accuracy": 0.89,
            "deals_closed_rate": 0.65
        }

        # Training args
        training_args = {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "num_epochs": 3
        }

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stage="baseline",
            version=1,
            metrics=metrics,
            training_args=training_args,
            dataset_stats=dataset_stats
        )
        print(f"   Checkpoint saved to: {checkpoint_path}")

        # Test 3: Load checkpoint
        print("\n3. Testing checkpoint load:")
        loaded_info = manager.load_checkpoint(checkpoint_path)
        print(f"   Stage: {loaded_info['stage']}")
        print(f"   Version: {loaded_info['version']}")
        print(f"   Dataset examples: {loaded_info['dataset_stats'].get('num_examples')}")
        print(f"   Training duration: {loaded_info.get('training_duration_seconds')}")

        # Test 4: Get next version (should be 2 now)
        print("\n4. Testing version increment:")
        next_version = manager.get_next_version("baseline")
        print(f"   Next version for baseline stage: {next_version}")
        assert next_version == 2, "Second version should be 2"

        # Test 5: Save another checkpoint for comparison
        print("\n5. Saving second checkpoint for comparison:")
        metrics2 = {
            "loss": 0.35,
            "accuracy": 0.92,
            "deals_closed_rate": 0.72
        }

        checkpoint_path2 = manager.save_checkpoint(
            model=mock_model,
            tokenizer=mock_tokenizer,
            stage="baseline",
            version=2,
            metrics=metrics2,
            training_args=training_args,
            dataset_stats=dataset_stats
        )
        print(f"   Second checkpoint saved")

        # Test 6: Get best checkpoint
        print("\n6. Testing best checkpoint selection:")
        best = manager.get_best_checkpoint("baseline", metric="loss")
        print(f"   Best checkpoint (by loss): {Path(best).name if best else 'None'}")

        # Test 7: Rollback capability
        print("\n7. Testing rollback capability:")
        rollback_info = manager.rollback_to_checkpoint(checkpoint_path)
        print(f"   Rolled back to: {Path(rollback_info['rolled_back_to']).name}")
        print(f"   Rollback history entries: {len(rollback_info['rollback_history'])}")

        # Test 8: Compare checkpoints
        print("\n8. Testing checkpoint comparison:")
        comparison = manager.compare_checkpoints(checkpoint_path, checkpoint_path2)
        print(f"   Metrics differences found: {len(comparison['metrics_diff'])}")
        for key, diff in comparison['metrics_diff'].items():
            print(f"     {key}: {diff['checkpoint1']} -> {diff['checkpoint2']} (change: {diff.get('change')})")

        # Test 9: Get checkpoint history
        print("\n9. Testing checkpoint history:")
        history = manager.get_checkpoint_history("baseline")
        print(f"   Total checkpoints in history: {len(history)}")
        for idx, cp in enumerate(history):
            print(f"     {idx+1}. v{cp['version']} - current: {cp['is_current']}")

        # Test 10: Export summary
        print("\n10. Testing checkpoint summary export:")
        summary_path = manager.export_checkpoint_summary()
        print(f"   Summary exported to: {summary_path}")

        # Test 11: List all checkpoints
        print("\n11. Testing checkpoint listing:")
        all_checkpoints = manager.list_checkpoints()
        print(f"   Total checkpoints: {len(all_checkpoints)}")

        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)


if __name__ == "__main__":
    test_checkpoint_manager()