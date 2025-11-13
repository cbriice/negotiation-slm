#!/usr/bin/env python3
"""
Curriculum training script for preventing catastrophic forgetting.
Uses phased training with dynamic masking and automatic recovery.
"""

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import TrainingArguments
from trl import SFTTrainer

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.curriculum.wrapper import CurriculumTrainingWrapper, CurriculumCallback
import training_parameters as params
import curriculum_parameters as cp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        loss = weighted_loss.sum() / shift_weights.numel()

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

    args = parser.parse_args()

    # Initialize curriculum wrapper
    logger.info("Initializing curriculum training wrapper...")
    wrapper = CurriculumTrainingWrapper(
        conversational_path=args.conversational_data,
        negotiation_path=args.negotiation_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        recovery_enabled=not args.disable_recovery,
        base_lr=args.base_lr
    )

    # Load model with LoRA
    logger.info("Loading model with LoRA configuration...")
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model(model_name=args.model_name)

    # Apply LoRA
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

    # Set model reference in wrapper for checkpoint saving
    wrapper.set_model(model)

    # Use max_steps from curriculum_parameters unless overridden
    max_steps = args.max_steps if args.max_steps != 8000 else cp.MAX_STEPS
    if args.test_mode:
        max_steps = 10

    # Training arguments
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
        optim=params.OPTIM,
        report_to=["tensorboard"] if not args.test_mode else [],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,  # Single-threaded (multiprocessing issues with tokenizer)
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
        ignore_data_skip=True,  # For IterableDataset resume - don't hang trying to skip samples
    )

    # Create dynamic iterable dataset that samples based on current phase
    logger.info("Creating dynamic curriculum dataset...")
    from training.curriculum.dynamic_dataset import CurriculumIterableDataset

    train_dataset = CurriculumIterableDataset(
        wrapper=wrapper,
        tokenizer=tokenizer,
        max_seq_len=params.MAX_SEQ_LEN
    )

    # Create trainer with curriculum callback (using weighted loss trainer)
    trainer = WeightedLossSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=wrapper.get_data_collator(),
        callbacks=[CurriculumCallback(wrapper)],
        max_seq_length=params.MAX_SEQ_LEN,
    )

    # Run training
    logger.info("Starting curriculum training with WEIGHTED MASKING...")
    logger.info(f"Total steps: {max_steps}")
    logger.info("")
    for i, phase in enumerate(cp.PHASES, 1):
        conv_ratio = phase['data_ratio']['conversational']
        neg_ratio = phase['data_ratio']['negotiation']
        user_w = phase['masking'].get('negotiation', {}).get('user_weight', 0.0)
        logger.info(f"Phase {i}: {phase['name']} ({phase['step_start']}-{phase['step_end']})")
        logger.info(f"  Data: Conv {conv_ratio:.0%} / Neg {neg_ratio:.0%}")
        logger.info(f"  User weight: {user_w:.3f} / Assistant weight: 1.0")

    if args.test_mode:
        logger.info("TEST MODE: Running only 10 steps")

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model as phase checkpoint
    logger.info("Saving final model...")
    trainer.save_model(str(Path(args.output_dir) / "final"))

    # Also save as final phase checkpoint
    from training.curriculum.checkpoint_utils import save_phase_checkpoint
    final_phase_num = len(cp.PHASES)
    final_phase_name = cp.PHASES[-1]['name']
    save_phase_checkpoint(
        model=model,
        phase_num=final_phase_num,
        phase_name=final_phase_name,
        step=max_steps,
        total_phases=len(cp.PHASES),
        output_dir=Path(args.output_dir)
    )

    logger.info("Curriculum training complete!")

if __name__ == "__main__":
    main()
