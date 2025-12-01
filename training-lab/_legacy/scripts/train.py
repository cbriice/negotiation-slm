#!/usr/bin/env python3
"""
Thin wrapper script for training Mistral-7B negotiation agent.
Provides centralized CLI access to existing training modules without adding logic.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def run_sft(args):
    """Run SFT training using existing module."""
    from training.sft_trainer import main as sft_main

    # Build command line args for SFT trainer
    sft_args = [args.train_data]

    if args.eval_data:
        sft_args.extend(['--eval-data', args.eval_data])
    if args.model:
        sft_args.extend(['--model', args.model])
    if args.output_dir:
        sft_args.extend(['--output-dir', args.output_dir])
    if args.checkpoint_dir:
        sft_args.extend(['--checkpoint-dir', args.checkpoint_dir])
    if args.resume_from:
        sft_args.extend(['--resume-from', args.resume_from])
    if args.num_epochs:
        sft_args.extend(['--num-epochs', str(args.num_epochs)])
    if args.max_steps:
        sft_args.extend(['--max-steps', str(args.max_steps)])
    if args.suffix:
        sft_args.extend(['--suffix', args.suffix])

    # Replace sys.argv and call the main function
    original_argv = sys.argv
    sys.argv = ['sft_trainer.py'] + sft_args
    try:
        sft_main()
    finally:
        sys.argv = original_argv


def run_dpo(args):
    """Run DPO training using existing module."""
    from training.dpo_trainer import main as dpo_main

    # Build command line args for DPO trainer
    dpo_args = []

    if args.train_data:
        dpo_args.extend(['--train-data', args.train_data])
    if args.model:
        dpo_args.extend(['--model', args.model])
    if args.eval_data:
        dpo_args.extend(['--eval-data', args.eval_data])
    if args.reference_model:
        dpo_args.extend(['--reference-model', args.reference_model])
    if args.output_dir:
        dpo_args.extend(['--output-dir', args.output_dir])
    if args.epochs:
        dpo_args.extend(['--epochs', str(args.epochs)])
    if args.resume:
        dpo_args.extend(['--resume', args.resume])
    if args.from_episodes:
        dpo_args.extend(['--from-episodes', args.from_episodes])
    if args.min_score_gap:
        dpo_args.extend(['--min-score-gap', str(args.min_score_gap)])

    # Replace sys.argv and call the main function
    original_argv = sys.argv
    sys.argv = ['dpo_trainer.py'] + dpo_args
    try:
        dpo_main()
    finally:
        sys.argv = original_argv


def run_qlora(args):
    """Run training using QLoRA trainer wrapper."""
    from training.qlora_trainer import QLoRATrainer
    import json

    # Initialize trainer
    trainer = QLoRATrainer(
        model_name=args.model or "mistralai/Mistral-7B-v0.1",
        output_dir=args.output_dir or "models/negotiator"
    )

    # Setup model
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    else:
        trainer.setup_model()

    # Determine training type and run
    if args.mode == 'sft':
        metrics = trainer.train_sft(
            train_data=args.train_data,
            eval_data=args.eval_data,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from
        )
    elif args.mode == 'dpo':
        metrics = trainer.train_dpo(
            train_data=args.train_data,
            reference_model_path=args.reference_model,
            eval_data=args.eval_data,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps
        )
    elif args.mode == 'test':
        if not args.prompt:
            print("Error: --prompt required for test mode")
            sys.exit(1)
        response = trainer.generate(
            prompt=args.prompt,
            max_length=args.max_length or 256,
            temperature=args.temperature or 0.7
        )
        print(f"Response: {response}")
        return
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    print(f"\nTraining Complete!")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Cleanup
    trainer.cleanup()


def manage_checkpoints(args):
    """Manage model checkpoints."""
    from training.checkpoint_manager import CheckpointManager
    import json

    manager = CheckpointManager(base_dir=args.checkpoint_dir or "checkpoints")

    if args.action == 'list':
        checkpoints = manager.list_checkpoints(stage=args.stage)
        if not checkpoints:
            print("No checkpoints found")
            return

        print(f"\nFound {len(checkpoints)} checkpoint(s):")
        print("-" * 80)
        for ckpt in checkpoints:
            print(f"Path: {ckpt['path']}")
            print(f"  Stage: {ckpt.get('stage', 'unknown')}")
            print(f"  Version: {ckpt.get('version', 'unknown')}")
            print(f"  Timestamp: {ckpt.get('timestamp', 'unknown')}")
            metrics = ckpt.get('metrics', {})
            if metrics:
                print(f"  Metrics: loss={metrics.get('train_loss', 'N/A'):.4f}")
            print()

    elif args.action == 'best':
        best = manager.get_best_checkpoint(stage=args.stage, metric=args.metric)
        if best:
            print(f"Best checkpoint for stage '{args.stage}' by {args.metric}:")
            print(f"  {best}")
        else:
            print(f"No checkpoints found for stage '{args.stage}'")

    elif args.action == 'rollback':
        if not args.checkpoint:
            print("Error: --checkpoint required for rollback")
            sys.exit(1)
        result = manager.rollback_to_checkpoint(args.checkpoint)
        print(f"Rolled back to: {result['rolled_back_to']}")
        print(f"Stage: {result['checkpoint_info'].get('stage')}")
        print(f"Version: {result['checkpoint_info'].get('version')}")

    elif args.action == 'compare':
        if not args.checkpoint1 or not args.checkpoint2:
            print("Error: --checkpoint1 and --checkpoint2 required for compare")
            sys.exit(1)
        comparison = manager.compare_checkpoints(args.checkpoint1, args.checkpoint2)
        print("\nCheckpoint Comparison:")
        print("-" * 80)
        print(f"Checkpoint 1: {comparison['checkpoint1']['path']}")
        print(f"  Stage: {comparison['checkpoint1']['stage']}, Version: {comparison['checkpoint1']['version']}")
        print(f"Checkpoint 2: {comparison['checkpoint2']['path']}")
        print(f"  Stage: {comparison['checkpoint2']['stage']}, Version: {comparison['checkpoint2']['version']}")

        if comparison['metrics_diff']:
            print("\nMetric Differences:")
            for key, diff in comparison['metrics_diff'].items():
                print(f"  {key}:")
                print(f"    Checkpoint 1: {diff['checkpoint1']}")
                print(f"    Checkpoint 2: {diff['checkpoint2']}")
                if diff.get('change') is not None:
                    print(f"    Change: {diff['change']:+.4f}")

    elif args.action == 'cleanup':
        manager.cleanup_old_checkpoints(keep_n=args.keep, stage=args.stage)
        print(f"Cleaned up old checkpoints, keeping {args.keep} most recent")

    elif args.action == 'export':
        report_path = manager.export_checkpoint_summary(output_path=args.output)
        print(f"Checkpoint summary exported to: {report_path}")

    elif args.action == 'history':
        history = manager.get_checkpoint_history(stage=args.stage)
        print(f"\nCheckpoint History ({len(history)} total):")
        print("-" * 80)
        for ckpt in history[:10]:  # Show only last 10
            status = "[CURRENT] " if ckpt.get('is_current') else ""
            print(f"{status}{ckpt['path']}")
            print(f"  Stage: {ckpt.get('stage')}, Version: {ckpt.get('version')}")
            print(f"  Timestamp: {ckpt.get('timestamp')}")
            print()


def save_checkpoint(args):
    """Manually save a checkpoint from current model."""
    from training.qlora_trainer import QLoRATrainer
    from training.checkpoint_manager import CheckpointManager

    # Initialize trainer and load model
    trainer = QLoRATrainer(
        model_name=args.model or "mistralai/Mistral-7B-v0.1",
        output_dir="models"
    )

    # Load the model from specified path
    if args.load_from:
        trainer.load_checkpoint(args.load_from)
    else:
        print("Error: --load-from required to specify which model to save")
        sys.exit(1)

    # Initialize checkpoint manager
    manager = CheckpointManager(base_dir=args.checkpoint_dir or "checkpoints")

    # Get version
    version = args.version
    if not version:
        version = manager.get_next_version(args.stage)

    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        stage=args.stage,
        version=version,
        metrics={"manual_save": True, "description": args.description or ""},
        training_args={"source": args.load_from}
    )

    print(f"Checkpoint saved to: {checkpoint_path}")

    # Cleanup
    trainer.cleanup()


def show_info():
    """Display current training parameters."""
    import training_parameters as params

    print("\n" + "="*60)
    print("CURRENT TRAINING PARAMETERS")
    print("="*60)

    print("\n[Model Configuration]")
    print(f"  Base model: mistralai/Mistral-7B-v0.1")
    print(f"  Flash Attention 2: {params.USE_FLASH_ATTENTION_2}")

    print("\n[QLoRA Configuration]")
    print(f"  4-bit quantization: {params.LOAD_IN_4BIT}")
    print(f"  NF4 quantization: {params.NF4}")
    print(f"  LoRA rank (r): {params.R}")
    print(f"  LoRA alpha: {params.ALPHA}")
    print(f"  LoRA dropout: {params.DROPOUT}")
    print(f"  Target modules: {params.TARGET_MODULES}")

    print("\n[Training Configuration]")
    print(f"  Batch size: {params.PER_DEVICE_BATCH_SIZE}")
    print(f"  Gradient accumulation: {params.GRAD_ACCUM_STEPS}")
    print(f"  Max sequence length: {params.MAX_SEQ_LEN}")
    print(f"  Gradient checkpointing: {params.GRADIENT_CHECKPOINTING}")
    print(f"  Mixed precision (bf16): {params.BF16}")
    print(f"  Optimizer: {params.OPTIM}")
    print(f"  Seed: {params.SEED}")

    print("\n[SFT Configuration]")
    print(f"  Learning rate: {params.SFT_LR}")
    print(f"  Max steps: {params.SFT_MAX_STEPS}")
    print(f"  Warmup ratio: {params.SFT_WARMUP_RATIO}")
    print(f"  LR scheduler: {params.SFT_LR_SCHEDULER}")
    print(f"  Packing: {params.PACKING}")

    print("\n[DPO Configuration]")
    print(f"  Learning rate: {params.DPO_LR}")
    print(f"  Beta: {params.DPO_BETA}")
    print(f"  Max steps: {params.DPO_MAX_STEPS}")
    print(f"  Max prompt length: {params.DPO_MAX_PROMPT_LEN}")
    print(f"  Label smoothing: {params.DPO_LABEL_SMOOTH}")

    print("\n[Logging Configuration]")
    print(f"  Logging steps: {params.LOGGING_STEPS}")
    print(f"  Save steps: {params.SAVE_STEPS}")
    print(f"  Save strategy: {params.SAVE_STRATEGY}")
    print(f"  Evaluation strategy: {params.EVAL_STRATEGY}")

    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Centralized training interface for Mistral-7B negotiation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SFT training
  python scripts/train.py sft data/sft_data.jsonl --num-epochs 3

  # Run DPO training with reference model
  python scripts/train.py dpo --train-data data/dpo_pairs.jsonl --reference-model models/sft/final_model

  # Run DPO from episodes
  python scripts/train.py dpo --from-episodes data/episodes.jsonl --min-score-gap 0.5

  # Use QLoRA wrapper for SFT
  python scripts/train.py qlora --mode sft --train-data data/sft.jsonl

  # Test a model
  python scripts/train.py qlora --mode test --checkpoint models/final --prompt "Test prompt"

  # Show current parameters
  python scripts/train.py info

  # Checkpoint management
  python scripts/train.py checkpoint list --stage sft
  python scripts/train.py checkpoint best --stage dpo --metric reward_margin
  python scripts/train.py checkpoint rollback --checkpoint checkpoints/negotiation_sft_v2_20240315

  # Save checkpoint manually
  python scripts/train.py save --load-from models/final --stage sft --description "Production ready"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Training command to run')

    # SFT subcommand - maps to sft_trainer.py
    sft_parser = subparsers.add_parser('sft', help='Run SFT training (delegates to sft_trainer.py)')
    sft_parser.add_argument('train_data', help='Path to training data JSONL')
    sft_parser.add_argument('--eval-data', help='Path to evaluation data')
    sft_parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1', help='Model name or path')
    sft_parser.add_argument('--output-dir', default='models/sft', help='Output directory')
    sft_parser.add_argument('--checkpoint-dir', default='checkpoints/sft', help='Checkpoint directory')
    sft_parser.add_argument('--resume-from', help='Resume from checkpoint')
    sft_parser.add_argument('--num-epochs', type=float, help='Number of epochs')
    sft_parser.add_argument('--max-steps', type=int, help='Maximum steps')
    sft_parser.add_argument('--suffix', default='', help='Output directory suffix')

    # DPO subcommand - maps to dpo_trainer.py
    dpo_parser = subparsers.add_parser('dpo', help='Run DPO training (delegates to dpo_trainer.py)')
    dpo_parser.add_argument('--train-data', help='Path to DPO pairs JSONL')
    dpo_parser.add_argument('--from-episodes', help='Generate pairs from episodes file')
    dpo_parser.add_argument('--eval-data', help='Path to evaluation data')
    dpo_parser.add_argument('--model', default='mistralai/Mistral-7B-v0.1', help='Model name or path')
    dpo_parser.add_argument('--reference-model', help='Reference model path')
    dpo_parser.add_argument('--output-dir', default='checkpoints/dpo', help='Output directory')
    dpo_parser.add_argument('--epochs', type=int, help='Number of epochs')
    dpo_parser.add_argument('--resume', help='Resume from checkpoint')
    dpo_parser.add_argument('--min-score-gap', type=float, default=0.3, help='Min score gap for auto pairs')

    # QLoRA subcommand - uses qlora_trainer.py directly
    qlora_parser = subparsers.add_parser('qlora', help='Run training via QLoRA wrapper')
    qlora_parser.add_argument('--mode', required=True, choices=['sft', 'dpo', 'test'], help='Training mode')
    qlora_parser.add_argument('--train-data', help='Training data path')
    qlora_parser.add_argument('--eval-data', help='Evaluation data path')
    qlora_parser.add_argument('--model', help='Model name or path')
    qlora_parser.add_argument('--reference-model', help='Reference model (DPO only)')
    qlora_parser.add_argument('--output-dir', help='Output directory')
    qlora_parser.add_argument('--checkpoint', help='Load from checkpoint')
    qlora_parser.add_argument('--resume-from', help='Resume training from checkpoint')
    qlora_parser.add_argument('--num-epochs', type=float, help='Number of epochs')
    qlora_parser.add_argument('--max-steps', type=int, help='Maximum steps')
    qlora_parser.add_argument('--prompt', help='Test prompt (test mode only)')
    qlora_parser.add_argument('--max-length', type=int, help='Max generation length (test mode)')
    qlora_parser.add_argument('--temperature', type=float, help='Generation temperature (test mode)')

    # Info subcommand
    info_parser = subparsers.add_parser('info', help='Show current training parameters')

    # Checkpoint management subcommand
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Manage model checkpoints')
    checkpoint_parser.add_argument('action',
                                  choices=['list', 'best', 'rollback', 'compare', 'cleanup', 'export', 'history'],
                                  help='Checkpoint action to perform')
    checkpoint_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    checkpoint_parser.add_argument('--stage', help='Filter by training stage (sft, dpo)')
    checkpoint_parser.add_argument('--metric', default='loss', help='Metric for finding best checkpoint')
    checkpoint_parser.add_argument('--checkpoint', help='Checkpoint path for rollback')
    checkpoint_parser.add_argument('--checkpoint1', help='First checkpoint for comparison')
    checkpoint_parser.add_argument('--checkpoint2', help='Second checkpoint for comparison')
    checkpoint_parser.add_argument('--keep', type=int, default=5, help='Number of checkpoints to keep for cleanup')
    checkpoint_parser.add_argument('--output', help='Output path for export')

    # Save checkpoint subcommand
    save_parser = subparsers.add_parser('save', help='Manually save a checkpoint')
    save_parser.add_argument('--load-from', required=True, help='Model path to create checkpoint from')
    save_parser.add_argument('--stage', required=True, help='Training stage label')
    save_parser.add_argument('--version', type=int, help='Version number (auto-increments if not specified)')
    save_parser.add_argument('--description', help='Description for this checkpoint')
    save_parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    save_parser.add_argument('--model', help='Model name or path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Route to appropriate handler
    if args.command == 'sft':
        run_sft(args)
    elif args.command == 'dpo':
        if not args.train_data and not args.from_episodes:
            print("Error: Either --train-data or --from-episodes required")
            sys.exit(1)
        run_dpo(args)
    elif args.command == 'qlora':
        run_qlora(args)
    elif args.command == 'info':
        show_info()
    elif args.command == 'checkpoint':
        manage_checkpoints(args)
    elif args.command == 'save':
        save_checkpoint(args)


if __name__ == "__main__":
    main()