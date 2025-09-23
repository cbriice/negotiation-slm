#!/usr/bin/env python3
"""
Baseline training script using ParlAI built-in negotiation tasks.
Implements step 2 of the training plan using Deal-or-No-Deal task.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.craigslist_loader import CraigslistBargainLoader
from training.sft_trainer import main as sft_main
import training_parameters as params

# Setup logging - ensure logs directory exists first
Path('logs').mkdir(exist_ok=True)

# Configure root logger to only use file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/baseline_training.log', mode='w')
    ],
    force=True  # Override any existing configuration
)

# Remove any existing console handlers that might have been added
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        root_logger.removeHandler(handler)

logger = logging.getLogger(__name__)


def load_craigslist_dataset(output_dir: str = "data/training/craigslist",
                            use_all_splits: bool = True) -> str:
    """
    Load and process CraigslistBargain dataset to SFT format.

    Args:
        output_dir: Directory to save processed data
        use_all_splits: Whether to use all splits for maximum diversity

    Returns:
        Path to SFT formatted file
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Loading CraigslistBargain dataset from HuggingFace")
    logger.info("=" * 80)

    loader = CraigslistBargainLoader(cache_dir=output_dir)

    if use_all_splits:
        # Use train + validation for maximum diversity
        output_file = f"{output_dir}/craigslist_full_sft.jsonl"
        sft_file = loader.create_full_training_set(output_file)
    else:
        # Use train split only
        dataset = loader.load_dataset('train')
        output_file = f"{output_dir}/craigslist_train_sft.jsonl"
        sft_file = loader.process_dataset_to_sft(
            dataset=dataset,
            output_file=output_file
        )

    logger.info(f"Processed dataset saved to: {sft_file}")
    return sft_file


def validate_sft_format(sft_file: str) -> int:
    """
    Validate and count SFT packets in the file.

    Args:
        sft_file: Path to SFT file

    Returns:
        Number of valid packets
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Validating SFT format")
    logger.info("=" * 80)

    packet_count = 0
    invalid_count = 0

    with open(sft_file, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            try:
                packet = json.loads(line)

                # Validate required fields
                required = ['id', 'system', 'prompt', 'response']
                if all(field in packet for field in required):
                    packet_count += 1

                    # Log sample packets
                    if packet_count <= 3:
                        logger.info(f"Sample packet {packet_count}:")
                        logger.info(f"  ID: {packet['id']}")
                        logger.info(f"  System: {packet['system'][:100]}...")
                        logger.info(f"  Prompt: {packet['prompt'][:100]}...")
                        logger.info(f"  Response: {packet['response'][:100]}...")
                else:
                    invalid_count += 1
                    logger.warning(f"Line {i+1}: Missing fields - has {list(packet.keys())}")

            except json.JSONDecodeError as e:
                invalid_count += 1
                logger.warning(f"Line {i+1}: Invalid JSON - {e}")

    logger.info(f"Validation complete: {packet_count} valid packets, {invalid_count} invalid")
    return packet_count


def calculate_epoch_fraction(sft_file: str,
                           target_epochs_per_task: float = params.SFT_BASELINE_EPOCHS) -> Dict:
    """
    Calculate training parameters for params.SFT_BASELINE_EPOCHS (0.3 as of rn) epochs per task.

    Args:
        sft_file: Path to SFT training file
        target_epochs_per_task: Target epochs per task (default in training_parameters.py)

    Returns:
        Dict with calculated training parameters
    """
    # Count total examples
    total_examples = 0
    role_counts = {'buyer': 0, 'seller': 0}

    with open(sft_file, 'r') as f:
        for line in f:
            if line.strip():
                total_examples += 1
                data = json.loads(line)
                # Extract role from ID (format: convid_role_turn)
                id_parts = data.get('id', '').split('_')
                if len(id_parts) >= 2:
                    role = id_parts[1]
                    if role in role_counts:
                        role_counts[role] += 1

    logger.info(f"Training data statistics:")
    logger.info(f"  Total examples: {total_examples}")
    for role, count in role_counts.items():
        logger.info(f"  {role}: {count} examples")

    # For CraigslistBargain, use smaller epochs since dataset is larger
    # and more diverse than DealNoDeal
    total_epochs = target_epochs_per_task

    # Calculate steps based on batch size
    effective_batch_size = params.PER_DEVICE_BATCH_SIZE * params.GRAD_ACCUM_STEPS
    steps_per_epoch = total_examples // effective_batch_size
    total_steps = int(steps_per_epoch * total_epochs)

    return {
        'num_epochs': total_epochs,
        'max_steps': total_steps,
        'steps_per_epoch': steps_per_epoch,
        'effective_batch_size': effective_batch_size
    }


def run_baseline_sft(sft_file: str,
                    output_dir: str = "models/negotiation_baseline",
                    checkpoint_dir: str = "checkpoints/baseline",
                    num_epochs: float = params.SFT_BASELINE_EPOCHS,
                    max_steps: Optional[int] = None) -> None:
    """
    Run baseline SFT training on negotiation tasks.

    Args:
        sft_file: Path to SFT training file
        output_dir: Output directory for trained model
        checkpoint_dir: Directory for checkpoints
        num_epochs: Number of training epochs
        max_steps: Maximum training steps
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Running baseline SFT training")
    logger.info("=" * 80)
    logger.info(f"Training configuration:")
    logger.info(f"  Data: {sft_file}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Max steps: {max_steps}")
    logger.info(f"  Learning rate: {params.SFT_LR}")
    logger.info(f"  Batch size: {params.PER_DEVICE_BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {params.GRAD_ACCUM_STEPS}")

    # Build command line args for SFT trainer
    sft_args = [
        sft_file,
        '--output-dir', output_dir,
        '--checkpoint-dir', checkpoint_dir,
        '--num-epochs', str(num_epochs),
        '--suffix', 'baseline_v1'
    ]

    if max_steps:
        sft_args.extend(['--max-steps', str(max_steps)])

    # Replace sys.argv and call SFT trainer
    original_argv = sys.argv
    sys.argv = ['sft_trainer.py'] + sft_args

    try:
        logger.info("Starting SFT training...")
        sft_main()
        logger.info("SFT training completed successfully!")

        # Save model info for DPO reference
        model_info = {
            'model_path': output_dir,
            'training_data': sft_file,
            'num_epochs': num_epochs,
            'max_steps': max_steps,
            'description': 'Baseline negotiation model trained on CraigslistBargain dataset'
        }

        info_path = Path(output_dir) / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model info saved to {info_path}")

    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        raise
    finally:
        sys.argv = original_argv


def main():
    """Main entry point for baseline training."""
    parser = argparse.ArgumentParser(
        description="Run baseline training using CraigslistBargain dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script implements step 2 of the negotiation training plan:
- Loads CraigslistBargain dataset from HuggingFace
- Processes multi-turn dialogues into SFT format
- Filters empty utterances and formats system prompts
- Runs training with weighted loss (prompt: 0.1-0.2x, response: 1x)
- Saves the baseline model for future DPO reference

Example usage:
  # Run with default settings (uses all splits for diversity)
  python scripts/run_baseline_training.py

  # Use only training split
  python scripts/run_baseline_training.py --train-only

  # Custom output directories
  python scripts/run_baseline_training.py --output-dir models/my_baseline --checkpoint-dir checkpoints/my_baseline

  # Skip data loading if already processed
  python scripts/run_baseline_training.py --skip-export --sft-file data/training/craigslist/craigslist_full_sft.jsonl
        """
    )

    parser.add_argument('--train-only', action='store_true',
                       help='Use only training split (default: use train+validation)')
    parser.add_argument('--epochs', type=float, default=params.SFT_BASELINE_EPOCHS,
                       help='Number of training epochs (default: in training_parameters.SFT_BASELINE_EPOCHS)')
    parser.add_argument('--output-dir', default='data/models/negotiation_baseline',
                       help='Output directory for trained model')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints/baseline',
                       help='Directory for checkpoints')
    parser.add_argument('--data-dir', default='data/training',
                       help='Directory for training data')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip task export if already done')
    parser.add_argument('--sft-file',
                       help='Path to existing SFT file (use with --skip-export)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Export and convert data but skip training')

    args = parser.parse_args()

    try:
        # Step 1: Load and process CraigslistBargain dataset
        if not args.skip_export:
            sft_file = load_craigslist_dataset(
                output_dir=f"{args.data_dir}/craigslist",
                use_all_splits=not args.train_only
            )

            # Step 2: Validate SFT format
            packet_count = validate_sft_format(sft_file)
        else:
            if not args.sft_file:
                logger.error("--sft-file required when using --skip-export")
                sys.exit(1)
            sft_file = args.sft_file
            logger.info(f"Using existing SFT file: {sft_file}")

        # Calculate training parameters
        training_params = calculate_epoch_fraction(
            sft_file=sft_file,
            target_epochs_per_task=args.epochs
        )

        logger.info("Calculated training parameters:")
        logger.info(f"  Total epochs: {training_params['num_epochs']}")
        logger.info(f"  Max steps: {training_params['max_steps']}")
        logger.info(f"  Steps per epoch: {training_params['steps_per_epoch']}")
        logger.info(f"  Effective batch size: {training_params['effective_batch_size']}")

        # Step 3: Run baseline SFT training
        if not args.dry_run:
            run_baseline_sft(
                sft_file=sft_file,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                num_epochs=training_params['num_epochs'],
                max_steps=int(training_params['max_steps'])
            )

            logger.info("=" * 80)
            logger.info("BASELINE TRAINING COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {args.output_dir}")
            logger.info(f"This model can be used as the reference for DPO training")
            logger.info(f"To use in DPO: python scripts/train.py dpo --reference-model {args.output_dir}")
        else:
            logger.info("Dry run complete - skipping actual training")
            logger.info(f"Would train with {training_params['num_epochs']} epochs, {training_params['max_steps']} steps")

    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()