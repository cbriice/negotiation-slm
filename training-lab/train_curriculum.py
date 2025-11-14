#!/usr/bin/env python3
"""
Curriculum training script for preventing catastrophic forgetting.
Uses phased training with dynamic masking and automatic recovery.
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time
from datetime import datetime
import platform
import torch
import torch.nn as nn
from transformers import TrainingArguments
from trl import SFTTrainer

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.curriculum.wrapper import CurriculumTrainingWrapper, CurriculumCallback, MetricsLoggingCallback
import training_parameters as params
import curriculum_parameters as cp

# Global logger (will be configured in setup_logging)
logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "logs/curriculum/training_info.log"):
    """
    Configure comprehensive logging to file only (console has HF training output).

    Args:
        log_file: Path to log file
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    root_logger.handlers.clear()

    # File handler with rotation (max 50MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_file


def log_system_info():
    """Log system and environment information."""
    logger.info("=" * 80)
    logger.info("CURRICULUM TRAINING SYSTEM INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    logger.info("=" * 80)


class WeightedLossSFTTrainer(SFTTrainer):
    """SFTTrainer with weighted loss support for curriculum learning."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with per-token weighting if loss_weights present."""
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if loss_weights is None:
            # Standard loss
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
            if loss is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        # Weighted loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())
        weighted_loss = per_token_loss * shift_weights
        # Normalize by sum of weights, not total token count
        # This ensures proper loss scaling when most tokens have weight=0
        loss = weighted_loss.sum() / shift_weights.sum()

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Curriculum training for negotiation model")
    parser.add_argument("--conversational-data", type=str,
                      default="data/training_data/conversational-dataset_formatted.jsonl")
    parser.add_argument("--negotiation-data", type=str,
                      default="data/training_data/negotiation-dataset_formatted.jsonl")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output-dir", type=str, default="models/curriculum")
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--disable-recovery", action="store_true", help="Disable automatic recovery")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with minimal steps")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Resume from checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs/curriculum/metrics", help="Directory for metrics logging")
    parser.add_argument("--training-log", type=str, default="logs/curriculum/training_info.log",
                      help="Path to detailed training log file")

    args = parser.parse_args()

    # Setup comprehensive logging
    log_file = setup_logging(args.training_log)
    logger.info("Logging initialized to: %s", log_file)

    # Log system information
    log_system_info()

    # Log configuration
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Conversational data: {args.conversational_data}")
    logger.info(f"Negotiation data: {args.negotiation_data}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Base learning rate: {args.base_lr}")
    logger.info(f"Recovery enabled: {not args.disable_recovery}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Resume from checkpoint: {args.resume_from_checkpoint or 'None'}")
    logger.info(f"Metrics log directory: {args.log_dir}")
    logger.info("=" * 80)

    # Initialize curriculum wrapper
    logger.info("=" * 80)
    logger.info("INITIALIZING CURRICULUM WRAPPER")
    logger.info("=" * 80)
    start_time = time.time()
    wrapper = CurriculumTrainingWrapper(
        conversational_path=args.conversational_data,
        negotiation_path=args.negotiation_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        recovery_enabled=not args.disable_recovery,
        base_lr=args.base_lr
    )
    wrapper_time = time.time() - start_time
    logger.info(f"Curriculum wrapper initialized in {wrapper_time:.2f}s")
    logger.info(f"Total training examples: {wrapper.data_sampler.total_examples}")
    logger.info(f"Conversational examples: {len(wrapper.data_sampler.conversational_data)}")
    logger.info(f"Negotiation examples: {len(wrapper.data_sampler.negotiation_data)}")

    # Load model with LoRA
    logger.info("=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)
    logger.info(f"Model name: {args.model_name}")
    start_time = time.time()
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model(model_name=args.model_name)
    model_load_time = time.time() - start_time
    logger.info(f"Model loaded in {model_load_time:.2f}s")

    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Apply LoRA
    logger.info("=" * 80)
    logger.info("APPLYING LORA CONFIGURATION")
    logger.info("=" * 80)
    from peft import LoraConfig, TaskType
    lora_config = LoraConfig(
        r=64,  # Increased capacity
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(f"LoRA rank (r): {lora_config.r}")
    logger.info(f"LoRA alpha: {lora_config.lora_alpha}")
    logger.info(f"LoRA dropout: {lora_config.lora_dropout}")
    logger.info(f"Target modules: {lora_config.target_modules}")

    lora_configurator = LoRAConfigurator()
    model = lora_configurator.apply_lora(model, lora_config)

    # Log LoRA parameter counts
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA trainable parameters: {lora_params:,}")
    logger.info(f"LoRA percentage of total: {100 * lora_params / total_params:.2f}%")

    # Set model reference in wrapper for checkpoint saving
    wrapper.set_model(model)
    logger.info("Model reference set in wrapper for checkpoint management")

    # Use max_steps from curriculum_parameters unless overridden
    max_steps = args.max_steps if args.max_steps != 8000 else cp.MAX_STEPS
    if args.test_mode:
        max_steps = 10
        logger.info("TEST MODE ENABLED: Limiting to 10 steps")

    # Training arguments
    logger.info("=" * 80)
    logger.info("TRAINING ARGUMENTS")
    logger.info("=" * 80)
    effective_batch_size = args.batch_size * args.gradient_accumulation
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Per-device batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Learning rate: {args.base_lr}")
    logger.info(f"LR scheduler: cosine")
    logger.info(f"Warmup ratio: 0.1")
    logger.info(f"BF16: {params.BF16}")
    logger.info(f"TF32: {params.TF32}")
    logger.info(f"Gradient checkpointing: {params.GRADIENT_CHECKPOINTING}")
    logger.info(f"Optimizer: {params.OPTIM}")
    logger.info(f"Max sequence length: {params.MAX_SEQ_LEN}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.base_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Extended warmup
        logging_steps=10,
        save_steps=500,
        save_strategy="steps",
        evaluation_strategy="no",  # We handle validation ourselves
        bf16=params.BF16,
        tf32=params.TF32,
        gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if params.GRADIENT_CHECKPOINTING else None,
        optim=params.OPTIM,
        report_to=["tensorboard"] if not args.test_mode else [],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,  # Single-threaded (multiprocessing issues with tokenizer)
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        ignore_data_skip=True,  # For IterableDataset resume - don't hang trying to skip samples
    )

    # Create dynamic iterable dataset that samples based on current phase
    logger.info("=" * 80)
    logger.info("CREATING CURRICULUM DATASET")
    logger.info("=" * 80)
    from training.curriculum.dynamic_dataset import CurriculumIterableDataset

    train_dataset = CurriculumIterableDataset(
        wrapper=wrapper,
        tokenizer=tokenizer,
        max_seq_len=params.MAX_SEQ_LEN
    )
    logger.info("Dynamic curriculum dataset created")

    # Create trainer with curriculum callback (using weighted loss trainer)
    logger.info("=" * 80)
    logger.info("CREATING TRAINER")
    logger.info("=" * 80)
    logger.info("Trainer type: WeightedLossSFTTrainer (supports per-token loss weighting)")
    logger.info("Callbacks: CurriculumCallback, MetricsLoggingCallback")

    trainer = WeightedLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=wrapper.get_data_collator(),
        callbacks=[
            CurriculumCallback(wrapper),
            MetricsLoggingCallback(log_dir=args.log_dir, stage="curriculum")
        ],
        max_seq_length=params.MAX_SEQ_LEN,
    )
    logger.info("Trainer initialized successfully")

    # Run training
    logger.info("=" * 80)
    logger.info("CURRICULUM TRAINING PHASES")
    logger.info("=" * 80)
    logger.info(f"Total training steps: {max_steps}")
    logger.info(f"Number of phases: {len(cp.PHASES)}")
    logger.info("")

    for i, phase in enumerate(cp.PHASES, 1):
        conv_ratio = phase['data_ratio']['conversational']
        neg_ratio = phase['data_ratio']['negotiation']
        user_w = phase['masking'].get('negotiation', {}).get('user_weight', 0.0)
        phase_steps = phase['step_end'] - phase['step_start']
        logger.info(f"Phase {i}: {phase['name']} (steps {phase['step_start']}-{phase['step_end']}, {phase_steps} steps)")
        logger.info(f"  Data ratio: Conv {conv_ratio:.0%} / Neg {neg_ratio:.0%}")
        logger.info(f"  Loss weights: User {user_w:.3f} / Assistant 1.0")
        logger.info(f"  LR multiplier: {phase.get('lr_multiplier', 1.0)}")
        logger.info(f"  Validation frequency: {phase.get('validation_frequency', 100)} steps")

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    training_start_time = time.time()

    try:
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()

        training_duration = time.time() - training_start_time
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total training time: {training_duration:.2f}s ({training_duration/3600:.2f} hours)")
        logger.info(f"Average time per step: {training_duration/max_steps:.2f}s")

    except Exception as e:
        training_duration = time.time() - training_start_time
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Training time before failure: {training_duration:.2f}s")
        logger.error(f"Error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

    # Save final model as phase checkpoint
    logger.info("=" * 80)
    logger.info("SAVING FINAL MODEL")
    logger.info("=" * 80)

    final_model_path = str(Path(args.output_dir) / "final")
    logger.info(f"Saving final model to: {final_model_path}")
    start_time = time.time()
    trainer.save_model(final_model_path)
    save_time = time.time() - start_time
    logger.info(f"Final model saved in {save_time:.2f}s")

    # Also save as final phase checkpoint
    logger.info("Saving final phase checkpoint...")
    from training.curriculum.checkpoint_utils import save_phase_checkpoint
    final_phase_num = len(cp.PHASES)
    final_phase_name = cp.PHASES[-1]['name']
    checkpoint_dir = save_phase_checkpoint(
        model=model,
        phase_num=final_phase_num,
        phase_name=final_phase_name,
        step=max_steps,
        total_phases=len(cp.PHASES),
        output_dir=Path(args.output_dir)
    )
    logger.info(f"Final phase checkpoint saved to: {checkpoint_dir}")

    logger.info("=" * 80)
    logger.info("CURRICULUM TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Final model: {final_model_path}")
    logger.info(f"Final checkpoint: {checkpoint_dir}")
    logger.info(f"Metrics logs: {args.log_dir}")
    logger.info(f"Training log: {log_file}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
