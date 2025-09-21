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

from training.converters.builtin_export import BuiltinTaskExporter
from training.data_converter import DataConverter
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


def export_builtin_tasks(output_dir: str = "data/training/builtin_tasks",
                         episodes_per_task: int = 1000) -> Dict[str, str]:
    """
    Export Deal-or-No-Deal and Casino tasks to JSONL format.

    Args:
        output_dir: Directory to save exported tasks
        episodes_per_task: Number of episodes to extract per task

    Returns:
        Dict mapping task names to output file paths
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Exporting ParlAI built-in negotiation tasks")
    logger.info("=" * 80)

    exporter = BuiltinTaskExporter(output_dir=output_dir)

    # Export dealnodeal task (casino doesn't exist in ParlAI)
    task_files = exporter.export_all_tasks(
        task_names=['dealnodeal'],
        max_episodes_per_task=episodes_per_task
    )

    logger.info(f"Exported tasks to: {task_files}")
    return task_files


def convert_to_sft_format(task_files: Dict[str, str],
                         output_dir: str = "data/training/sft") -> str:
    """
    Convert exported episodes to SFT training format.

    Args:
        task_files: Dict mapping task names to episode file paths
        output_dir: Directory to save SFT formatted data

    Returns:
        Path to combined SFT training file
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Converting episodes to SFT format")
    logger.info("=" * 80)

    converter = DataConverter(output_dir=output_dir)
    all_sft_files = []

    for task_name, episode_file in task_files.items():
        logger.info(f"Converting {task_name} episodes...")

        # Convert episodes to SFT format
        sft_file = converter.convert_episodes_to_sft(
            episodes_file=episode_file,
            output_file=f"{output_dir}/{task_name}_sft.jsonl"
        )

        all_sft_files.append(sft_file)

        # Count converted packets
        packet_count = 0
        with open(sft_file, 'r') as f:
            for line in f:
                if line.strip():
                    packet_count += 1
        logger.info(f"  Generated {packet_count} SFT packets from {task_name}")

    # Combine all SFT files into one
    output_path = Path(output_dir) / "baseline_negotiation_sft.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_packets = 0
    with open(output_path, 'w') as out_f:
        for sft_file in all_sft_files:
            with open(sft_file, 'r') as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        total_packets += 1

    logger.info(f"Saved {total_packets} total SFT packets to {output_path}")
    return str(output_path)


def calculate_epoch_fraction(sft_file: str,
                           target_epochs_per_task: float = 0.5) -> Dict:
    """
    Calculate training parameters for 0.5 epochs per task.

    Args:
        sft_file: Path to SFT training file
        target_epochs_per_task: Target epochs per task (default 0.5)

    Returns:
        Dict with calculated training parameters
    """
    # Count total examples
    total_examples = 0
    task_counts = {}

    with open(sft_file, 'r') as f:
        for line in f:
            if line.strip():
                total_examples += 1
                data = json.loads(line)
                task = data.get('metadata', {}).get('source_task', 'unknown')
                task_counts[task] = task_counts.get(task, 0) + 1

    logger.info(f"Training data statistics:")
    logger.info(f"  Total examples: {total_examples}")
    for task, count in task_counts.items():
        logger.info(f"  {task}: {count} examples")

    # Calculate epochs - adjust if only one task
    # If we only have dealnodeal, we'll do more epochs on it
    num_tasks = len(task_counts)
    if num_tasks == 1:
        # ----------------------------------------------------------------- num epochs default config ----------------------------------------------------------------------
        total_epochs = 0.3
    else:
        total_epochs = num_tasks * target_epochs_per_task

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
                    num_epochs: float = 1.0,
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
            'description': 'Baseline negotiation model trained on Deal-or-No-Deal and Casino tasks'
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
        description="Run baseline training using ParlAI built-in negotiation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script implements step 2 of the negotiation training plan:
- Exports Deal-or-No-Deal task from ParlAI (Note: Casino task not available in ParlAI)
- Converts it to SFT format
- Runs training epochs as specified
- Saves the baseline model for future DPO reference

Example usage:
  # Run with default settings (recommended)
  python scripts/run_baseline_training.py

  # Custom episode count per task
  python scripts/run_baseline_training.py --episodes-per-task 2000

  # Custom output directories
  python scripts/run_baseline_training.py --output-dir models/my_baseline --checkpoint-dir checkpoints/my_baseline

  # Skip export if already done
  python scripts/run_baseline_training.py --skip-export --sft-file data/training/sft/baseline_negotiation_sft.jsonl
        """
    )

    parser.add_argument('--episodes-per-task', type=int, default=1000,
                       help='Number of episodes to extract per task (default: 1000)')
    parser.add_argument('--epochs-per-task', type=float, default=0.5,
                       help='Training epochs per task (default: 0.5)')
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
        # Step 1: Export built-in tasks
        if not args.skip_export:
            task_files = export_builtin_tasks(
                output_dir=f"{args.data_dir}/builtin_tasks",
                episodes_per_task=args.episodes_per_task
            )

            # Step 2: Convert to SFT format
            sft_file = convert_to_sft_format(
                task_files=task_files,
                output_dir=f"{args.data_dir}/sft"
            )
        else:
            if not args.sft_file:
                logger.error("--sft-file required when using --skip-export")
                sys.exit(1)
            sft_file = args.sft_file
            logger.info(f"Using existing SFT file: {sft_file}")

        # Calculate training parameters
        training_params = calculate_epoch_fraction(
            sft_file=sft_file,
            target_epochs_per_task=args.epochs_per_task
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