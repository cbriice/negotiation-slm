"""
Standalone SFT trainer module for QLoRA fine-tuning with CLI interface.
Loads NDJSON datasets and performs supervised fine-tuning using TRL's SFTTrainer.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from trl import SFTTrainer

# Import existing QLoRA components
from .model_loader import ModelLoader
from .lora_config import LoRAConfigurator
from .checkpoint_manager import CheckpointManager
from .metrics_tracker import TrainingMetricsTracker
import training_parameters as params

# Configure logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sft_trainer.log', mode='w')  # 'w' mode clears log on each run
    ]
)
logger = logging.getLogger(__name__)


class SFTTrainingPipeline:
    """Handles SFT training pipeline for QLoRA fine-tuning."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "models/sft",
        checkpoint_dir: str = "checkpoints/sft"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing SFTTrainingPipeline")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")

        # Initialize components
        self.model_loader = ModelLoader()
        self.lora_configurator = LoRAConfigurator()
        self.checkpoint_manager = CheckpointManager(base_dir=self.checkpoint_dir)

        # Model and tokenizer (initialized during setup)
        self.model = None
        self.tokenizer = None

        # Initialize metrics tracker
        self.metrics_tracker = None

    def load_ndjson_dataset(self, file_path: str) -> Dataset:
        """
        Load and parse NDJSON file into HuggingFace Dataset.

        Args:
            file_path: Path to NDJSON file

        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset from: {file_path}")

        data = []
        invalid_lines = 0
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    invalid_lines += 1
                    continue

        if not data:
            logger.error(f"No valid data found in {file_path}")
            raise ValueError(f"No valid data found in {file_path}")

        logger.info(f"Successfully loaded {len(data)} examples from NDJSON")
        if invalid_lines > 0:
            logger.warning(f"Skipped {invalid_lines} invalid lines")

        # Convert to Dataset
        dataset = Dataset.from_list(data)
        return dataset

    def format_chat_template(self, example: Dict[str, Any]) -> str:
        """
        Format training example using chat template.
        Supports both dialogue format and instruction-response format.

        Args:
            example: Single training example

        Returns:
            Formatted text for training
        """
        # Check if it's dialogue format (list of turns)
        if 'dialogue' in example:
            formatted = ""
            for turn in example['dialogue']:
                role = turn.get('role', 'user')
                content = turn.get('content', turn.get('text', ''))

                # Use simple format for now, can be enhanced with proper chat templates
                if role == 'assistant':
                    formatted += f"Assistant: {content}\n"
                else:
                    formatted += f"User: {content}\n"

            return formatted.strip()

        # Check if it's instruction-response format
        elif 'instruction' in example and 'response' in example:
            instruction = example['instruction']
            response = example['response']

            # Format as instruction-following
            formatted = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            return formatted

        # Check if it's input-output format (from SFT packet template)
        elif 'input' in example and 'output' in example:
            input_data = example['input']
            output = example['output']

            # Build context from input components
            context = ""
            if isinstance(input_data, dict):
                if 'state' in input_data and input_data['state']:
                    context += f"State: {json.dumps(input_data['state'])}\n"
                if 'recent_convo' in input_data and input_data['recent_convo']:
                    context += f"Recent conversation: {json.dumps(input_data['recent_convo'])}\n"
                if 'partner_msg' in input_data and input_data['partner_msg']:
                    context += f"Partner: {input_data['partner_msg']}\n"
            else:
                context = str(input_data)

            formatted = f"### Context:\n{context}\n### Response:\n{output}"
            return formatted

        # Fallback: treat as plain text
        elif 'text' in example:
            return example['text']

        else:
            raise ValueError(f"Unknown example format: {example.keys()}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for SFT training.

        Args:
            dataset: Raw dataset

        Returns:
            Preprocessed dataset with 'text' field
        """
        def add_text_field(examples):
            """Add formatted text field to examples."""
            texts = []
            for i in range(len(next(iter(examples.values())))):
                # Extract single example
                example = {k: v[i] for k, v in examples.items()}

                # Format using chat template
                try:
                    text = self.format_chat_template(example)
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to format example {i}: {e}")
                    texts.append("")  # Add empty string to maintain alignment

            return {"text": texts}

        # Apply formatting
        dataset = dataset.map(
            add_text_field,
            batched=True,
            desc="Formatting examples"
        )

        # Filter out empty texts
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: len(x['text']) > 0)
        filtered = original_size - len(dataset)

        logger.info(f"Preprocessed {len(dataset)} examples")
        if filtered > 0:
            logger.warning(f"Filtered out {filtered} empty examples")
        return dataset

    def setup_model(self, resume_from: Optional[str] = None):
        """
        Initialize model with QLoRA configuration.

        Args:
            resume_from: Optional checkpoint path to resume from
        """
        logger.info("Setting up model with QLoRA...")
        logger.info(f"  4-bit quantization: {params.LOAD_IN_4BIT}")
        logger.info(f"  LoRA rank: {params.R}")
        logger.info(f"  LoRA alpha: {params.ALPHA}")
        logger.info(f"  LoRA dropout: {params.DROPOUT}")

        # Load base model with 4-bit quantization
        start_time = time.time()
        self.model, self.tokenizer = self.model_loader.load_model(self.model_name)
        load_time = time.time() - start_time
        logger.info(f"Base model loaded in {load_time:.2f}s")

        if resume_from:
            logger.info(f"Loading checkpoint from: {resume_from}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resume_from)
            logger.info("Checkpoint loaded successfully")
        else:
            # Apply LoRA configuration
            self.model = self.lora_configurator.apply_lora(self.model)
            logger.info("LoRA configuration applied")

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")

        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model setup complete")
        if total_params > 0:
            logger.info(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            logger.info(f"  Total params: {total_params:,}")
        else:
            logger.info(f"  Model has no parameters (likely a mock for testing)")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: Optional[float] = None,
        max_steps: Optional[int] = None,
        output_suffix: str = ""
    ) -> Dict[str, Any]:
        """
        Execute SFT training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            num_epochs: Number of epochs (overrides params if provided)
            max_steps: Maximum steps (overrides params if provided)
            output_suffix: Suffix for output directory

        Returns:
            Training metrics
        """
        # Determine output directory
        output_dir = self.output_dir / f"sft_run{output_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*60)
        logger.info("Starting SFT Training Run")
        logger.info("="*60)
        logger.info(f"Output directory: {output_dir}")

        # Initialize metrics tracker for this training run
        self.metrics_tracker = TrainingMetricsTracker(
            log_dir=str(output_dir / "logs"),
            stage="sft"
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs if num_epochs else params.SFT_MAX_STEPS / len(train_dataset),
            max_steps=max_steps if max_steps else params.SFT_MAX_STEPS,
            per_device_train_batch_size=params.PER_DEVICE_BATCH_SIZE,
            per_device_eval_batch_size=params.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
            learning_rate=params.SFT_LR,
            warmup_ratio=params.SFT_WARMUP_RATIO,
            lr_scheduler_type=params.SFT_LR_SCHEDULER,
            optim=params.OPTIM,
            bf16=params.BF16,
            fp16=not params.BF16,  # Use fp16 if bf16 is not available
            gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
            logging_steps=params.LOGGING_STEPS,
            save_steps=params.SAVE_STEPS,
            save_strategy=params.SAVE_STRATEGY,
            evaluation_strategy=params.EVAL_STRATEGY if eval_dataset else "no",
            eval_steps=params.SAVE_STEPS if eval_dataset else None,
            seed=params.SEED,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            logging_dir=str(output_dir / "logs"),
        )

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",  # Use preprocessed text field
            max_seq_length=params.MAX_SEQ_LEN,
            packing=params.PACKING,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Optimize for tensor cores
            ),
        )

        logger.info("Training Configuration:")
        logger.info(f"  - Train examples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"  - Eval examples: {len(eval_dataset)}")
        logger.info(f"  - Max steps: {training_args.max_steps}")
        logger.info(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")
        logger.info(f"  - Warmup ratio: {training_args.warmup_ratio}")
        logger.info(f"  - LR scheduler: {training_args.lr_scheduler_type}")
        logger.info(f"  - Optimizer: {training_args.optim}")
        logger.info(f"  - Max sequence length: {params.MAX_SEQ_LEN}")
        logger.info(f"  - Packing: {params.PACKING}")
        logger.info(f"  - Mixed precision: bf16={params.BF16}, fp16={not params.BF16}")
        logger.info(f"  - Gradient checkpointing: {params.GRADIENT_CHECKPOINTING}")

        # Add callback for logging
        class LoggingCallback(TrainerCallback):
            def __init__(self, metrics_tracker):
                self.metrics_tracker = metrics_tracker

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step = state.global_step
                    # Log to our metrics tracker
                    if self.metrics_tracker:
                        self.metrics_tracker.log_training_step(
                            step=step,
                            epoch=state.epoch or 0,
                            metrics=logs
                        )
                    # Log key metrics to logger
                    if "loss" in logs:
                        logger.info(f"Step {step}: loss={logs['loss']:.4f}")

        trainer.add_callback(LoggingCallback(self.metrics_tracker))

        # Train
        logger.info("Starting training...")
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(str(output_dir / "final_model"))
        logger.info(f"Model saved to: {output_dir / 'final_model'}")

        # Save training metrics
        metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "total_training_time": training_time,
            "total_steps": train_result.global_step,
        }

        # Save metrics to file
        with open(output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save training summary from metrics tracker
        if self.metrics_tracker:
            summary = self.metrics_tracker.save_training_summary()
            logger.info("Training metrics saved to CSV")

        logger.info("="*60)
        logger.info("Training Completed Successfully!")
        logger.info("="*60)
        logger.info(f"Final loss: {metrics['train_loss']:.4f}")
        logger.info(f"Total time: {training_time/60:.1f} minutes")
        logger.info(f"Training speed: {metrics['train_samples_per_second']:.2f} samples/sec")
        logger.info(f"Model saved to: {output_dir / 'final_model'}")

        return metrics


def main():
    """CLI entry point for SFT training."""
    parser = argparse.ArgumentParser(
        description="SFT trainer for QLoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "train_data",
        type=str,
        help="Path to training data NDJSON file"
    )

    # Optional arguments
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data NDJSON file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name or path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/sft",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/sft",
        help="Directory for saving checkpoints"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--num-epochs",
        type=float,
        default=None,
        help="Number of training epochs (overrides config)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output directory naming"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("SFT Training Pipeline - CLI")
    logger.info("="*60)

    # Log arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Validate paths
    train_path = Path(args.train_data)
    if not train_path.exists():
        logger.error(f"Training data file not found: {train_path}")
        sys.exit(1)

    if args.eval_data:
        eval_path = Path(args.eval_data)
        if not eval_path.exists():
            logger.error(f"Evaluation data file not found: {eval_path}")
            sys.exit(1)

    try:
        # Initialize pipeline
        logger.info("Initializing training pipeline...")
        pipeline = SFTTrainingPipeline(
            model_name=args.model,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir
        )

        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = pipeline.load_ndjson_dataset(args.train_data)
        eval_dataset = None
        if args.eval_data:
            eval_dataset = pipeline.load_ndjson_dataset(args.eval_data)

        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = pipeline.preprocess_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = pipeline.preprocess_dataset(eval_dataset)

        # Setup model
        logger.info("Setting up model...")
        pipeline.setup_model(resume_from=args.resume_from)

        # Run training
        metrics = pipeline.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            output_suffix=args.suffix
        )

        logger.info("\nFinal Training Summary:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()