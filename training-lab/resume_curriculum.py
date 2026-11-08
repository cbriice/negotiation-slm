#!/usr/bin/env python3
"""
Resume curriculum training from a specific phase checkpoint.
Load LoRA adapters and continue training with modified parameters.
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import TrainingArguments

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.weighted_trainer import WeightedLossSFTTrainer
from training.curriculum.wrapper import CurriculumTrainingWrapper, CurriculumCallback, MetricsLoggingCallback
from training.curriculum.checkpoint_utils import (
    load_phase_checkpoint,
    adjust_phase_ranges,
    print_resume_banner,
    validate_phase_compatibility,
    save_phase_checkpoint
)
import training_parameters as params
import curriculum_parameters as cp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Resume curriculum training from checkpoint")
    parser.add_argument("--start-from-phase", type=int, required=True,
                      help="Phase number to resume from (1-indexed)")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                      help="Path to phase checkpoint directory")
    parser.add_argument("--conversational-data", type=str,
                      default="data/training_data/conversational-dataset_formatted.jsonl")
    parser.add_argument("--negotiation-data", type=str,
                      default="data/training_data/negotiation-dataset_formatted.jsonl")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output-dir", type=str, default="models/curriculum_resumed")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--disable-recovery", action="store_true", help="Disable automatic recovery")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with minimal steps")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for metrics logging")

    args = parser.parse_args()

    # Load checkpoint and metadata
    logger.info(f"Loading checkpoint from {args.checkpoint_dir}")
    checkpoint_path, metadata = load_phase_checkpoint(args.checkpoint_dir)

    # Validate phase compatibility
    validate_phase_compatibility(metadata, cp.PHASES, args.start_from_phase)

    # Adjust phase ranges to start from 0
    adjusted_phases = adjust_phase_ranges(cp.PHASES, args.start_from_phase)

    # Print resume banner
    print_resume_banner(checkpoint_path, metadata, adjusted_phases)

    # Create temporary phase config module for adjusted phases
    # We'll monkey-patch curriculum_parameters.PHASES for this run
    original_phases = cp.PHASES
    cp.PHASES = adjusted_phases

    # Calculate adjusted max_steps
    adjusted_max_steps = adjusted_phases[-1]["step_end"]
    if args.test_mode:
        adjusted_max_steps = 10

    # Initialize curriculum wrapper with adjusted phases
    logger.info("Initializing curriculum training wrapper...")
    wrapper = CurriculumTrainingWrapper(
        conversational_path=args.conversational_data,
        negotiation_path=args.negotiation_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        recovery_enabled=not args.disable_recovery,
        base_lr=args.base_lr
    )

    # Load base model
    logger.info("Loading base model...")
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model(model_name=args.model_name)

    # Apply LoRA (EXACT same as train_curriculum.py)
    logger.info("Applying LoRA configuration...")
    from peft import LoraConfig, TaskType
    lora_config = LoraConfig(
        r=64,  # Increased capacity
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_configurator = LoRAConfigurator()
    model = lora_configurator.apply_lora(model, lora_config)

    # Now load checkpoint weights into the LoRA adapters
    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    adapters_weights = load_file(Path(checkpoint_path) / "adapter_model.safetensors")
    set_peft_model_state_dict(model, adapters_weights)

    # Set model reference in wrapper for checkpoint saving
    wrapper.set_model(model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=adjusted_max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.base_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        save_strategy="steps",
        evaluation_strategy="no",
        bf16=params.BF16,
        tf32=params.TF32,
        gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if params.GRADIENT_CHECKPOINTING else None,
        optim=params.OPTIM,
        report_to=["tensorboard"] if not args.test_mode else [],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        ignore_data_skip=True,
    )

    # Create dynamic iterable dataset
    logger.info("Creating dynamic curriculum dataset...")
    from training.curriculum.dynamic_dataset import CurriculumIterableDataset

    train_dataset = CurriculumIterableDataset(
        wrapper=wrapper,
        tokenizer=tokenizer,
        max_seq_len=params.MAX_SEQ_LEN
    )

    # Get data collator and ensure its tokenizer has correct padding side
    data_collator = wrapper.get_data_collator()
    if hasattr(data_collator, 'tokenizer') and data_collator.tokenizer is not None:
        data_collator.tokenizer.padding_side = 'left'

    tokenizer.padding_side = 'left'
    # Create trainer with curriculum callback
    trainer = WeightedLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            CurriculumCallback(wrapper),
            MetricsLoggingCallback(log_dir=args.log_dir, stage="curriculum_resumed")
        ],
        max_seq_length=params.MAX_SEQ_LEN,
    )


    # Run training
    logger.info("Starting resumed curriculum training...")
    logger.info(f"Total steps: {adjusted_max_steps}")
    logger.info("")
    for i, phase in enumerate(adjusted_phases, args.start_from_phase):
        conv_ratio = phase['data_ratio']['conversational']
        neg_ratio = phase['data_ratio']['negotiation']
        user_w = phase['masking'].get('negotiation', {}).get('user_weight', 0.0)
        logger.info(f"Phase {i}: {phase['name']} ({phase['step_start']}-{phase['step_end']})")
        logger.info(f"  Data: Conv {conv_ratio:.0%} / Neg {neg_ratio:.0%}")
        logger.info(f"  User weight: {user_w:.3f} / Assistant weight: 1.0")

    if args.test_mode:
        logger.info("TEST MODE: Running only 10 steps")

    # Start training (adapters already loaded, no resume_from_checkpoint needed)
    trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(Path(args.output_dir) / "final"))

    # Also save as final phase checkpoint
    final_phase_num = len(original_phases)  # Use original phase count
    final_phase_name = adjusted_phases[-1]['name']
    save_phase_checkpoint(
        model=model,
        phase_num=final_phase_num,
        phase_name=final_phase_name,
        step=adjusted_max_steps,
        total_phases=len(original_phases),
        output_dir=Path(args.output_dir)
    )

    logger.info("Resumed curriculum training complete!")

    # Restore original phases
    cp.PHASES = original_phases


if __name__ == "__main__":
    main()
