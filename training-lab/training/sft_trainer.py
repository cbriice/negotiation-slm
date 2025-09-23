"""
Standalone SFT trainer module for QLoRA fine-tuning with CLI interface.
Loads NDJSON datasets and performs supervised fine-tuning using TRL's SFTTrainer.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Import existing QLoRA components
from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.checkpoint_manager import CheckpointManager
from training.metrics_tracker import TrainingMetricsTracker
from utils.dataset_loader import UnifiedDatasetLoader
from utils.json_data_collator import DataCollatorForJSON
from utils.weighted_loss_collator import WeightedLossCollator, WeightedLossTrainer
import training_parameters as params

# Import error handling modules
from utils.error_handler import ErrorHandler
from utils.training_state import TrainingStateManager, TrainingState
from utils.batch_size_optimizer import BatchSizeOptimizer, BatchSizeConfig
from utils.gradient_monitor import GradientMonitor, GradientHealthConfig
from utils.loss_validator import LossValidator, LossValidationConfig
from utils.precision_handler import PrecisionHandler, PrecisionConfig
from utils.tensorboard_writer import TensorBoardWriter

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
        checkpoint_dir: str = "checkpoints/sft",
        enable_error_handling: bool = True
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.enable_error_handling = enable_error_handling

        logger.info(f"Initializing SFTTrainingPipeline")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Error handling: {enable_error_handling}")

        # Initialize components
        self.model_loader = ModelLoader()
        self.lora_configurator = LoRAConfigurator()
        self.checkpoint_manager = CheckpointManager(base_dir=str(self.checkpoint_dir))
        self.dataset_loader = UnifiedDatasetLoader()

        # Initialize error handling modules if enabled
        if self.enable_error_handling:
            self.error_handler = ErrorHandler(
                log_dir=str(self.output_dir / "error_logs"),
                enable_auto_recovery=True,
                max_recovery_attempts=3
            )
            self.state_manager = TrainingStateManager(
                checkpoint_dir=str(self.checkpoint_dir),
                state_dir=str(self.output_dir / "training_states"),
                auto_save_interval=100  # Save every 100 steps
            )
            self.batch_optimizer = BatchSizeOptimizer(
                config=BatchSizeConfig(
                    initial_batch_size=params.PER_DEVICE_BATCH_SIZE,
                    min_batch_size=1,
                    max_batch_size=32,
                    gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
                    memory_fraction_target=0.85,  # Target 85% GPU memory usage
                    enable_auto_tuning=True
                )
            )
            self.gradient_monitor = GradientMonitor(
                config=GradientHealthConfig(
                    max_gradient_norm=10.0,
                    explosion_threshold=100.0,
                    vanishing_threshold=1e-7,
                    enable_clipping=True,
                    adaptive_clipping=True
                )
            )
            self.loss_validator = LossValidator(
                config=LossValidationConfig(
                    max_loss=100.0,
                    min_loss=0.0,
                    nan_action="skip",
                    inf_action="skip",
                    plateau_patience=100,
                    enable_anomaly_detection=True
                )
            )
            self.precision_handler = PrecisionHandler(
                config=PrecisionConfig(
                    use_bf16=params.BF16,
                    use_fp16=not params.BF16,
                    use_gradient_scaling=not params.BF16,  # Use gradient scaling for fp16
                    use_amp=True,
                    overflow_fallback="reduce_scale"
                )
            )
            logger.info("Error handling modules initialized")
        else:
            self.error_handler = None
            self.state_manager = None
            self.batch_optimizer = None
            self.gradient_monitor = None
            self.loss_validator = None
            self.precision_handler = None

        # Model and tokenizer (initialized during setup)
        self.model = None
        self.tokenizer = None

        # Initialize metrics tracker and TensorBoard writer
        self.metrics_tracker = None
        self.tb_writer = None

        # Data collator will be initialized after tokenizer is loaded
        self.json_collator = None
        self.weighted_collator = None
        self.use_weighted_loss = False  # Will be set based on data format

    def load_ndjson_dataset(self, file_path: str) -> Dataset:
        """
        Load and parse NDJSON file into HuggingFace Dataset.
        Now delegates to UnifiedDatasetLoader.

        Args:
            file_path: Path to NDJSON file

        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset from: {file_path}")
        return self.dataset_loader.load_dataset_auto(file_path)

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess dataset for SFT training.
        Detects format and sets appropriate collator.

        Args:
            dataset: Raw dataset

        Returns:
            Dataset ready for training
        """
        logger.info(f"Loaded {len(dataset)} raw examples")

        if len(dataset) > 0:
            sample = dataset[0]

            # Check for CraigslistBargain format (system, prompt, response)
            if all(field in sample for field in ['system', 'prompt', 'response']):
                logger.info("Detected CraigslistBargain format - using weighted loss")
                self.use_weighted_loss = True

                # Log sample
                logger.debug(f"Sample system: {sample.get('system', '')[:100]}...")
                logger.debug(f"Sample prompt: {sample.get('prompt', '')[:100]}...")
                logger.debug(f"Sample response: {sample.get('response', '')[:100]}...")

            # Check for standard format (input, output)
            elif all(field in sample for field in ['input', 'output']):
                logger.info("Detected standard JSON format - using standard masking")
                self.use_weighted_loss = False

                # Log sample
                logger.debug(f"Sample input: {str(sample.get('input', ''))[:200]}...")
                logger.debug(f"Sample output: {sample.get('output', '')[:100]}...")

            else:
                available = list(sample.keys()) if isinstance(sample, dict) else []
                raise ValueError(f"Unknown dataset format. Have fields: {available}")

        return dataset

    def _validate_masking_on_sample(self, dataset: Dataset, data_collator) -> None:
        """
        Validate that JSON masking is working correctly on sample data.

        Args:
            dataset: Training dataset with input/output JSON format
            data_collator: JSON data collator
        """
        if len(dataset) == 0:
            logger.warning("Cannot validate masking on empty dataset")
            return

        if not self.tokenizer or not self.json_collator:
            logger.warning("Cannot validate masking - tokenizer or collator not initialized")
            return

        logger.info("="*60)
        logger.info("Validating JSON Input/Output Masking")
        logger.info("="*60)

        # Take a few samples for validation
        num_samples = min(3, len(dataset))
        samples = dataset.select(range(num_samples))

        total_masked = 0
        total_supervised = 0
        total_tokens = 0

        for i in range(len(samples)):
            example = samples[i]

            # Log the raw JSON data
            logger.info(f"\nSample {i+1}:")
            if 'input' in example:
                input_str = str(example['input'])[:200]
                logger.info(f"  Input: {input_str}...")
            if 'output' in example:
                logger.info(f"  Output: {example['output'][:100]}...")

            # Apply collator to get tokenized and masked data
            batch = data_collator([example])
            input_ids = batch["input_ids"][0]
            labels = batch["labels"][0]

            # Calculate masking statistics
            masked_count = (labels == params.MASKING_IGNORE_INDEX).sum().item()
            supervised_count = (labels != params.MASKING_IGNORE_INDEX).sum().item()
            token_count = len(labels)

            total_masked += masked_count
            total_supervised += supervised_count
            total_tokens += token_count

            masking_ratio = masked_count / token_count if token_count > 0 else 0

            # Log statistics
            logger.info(f"  Total tokens: {token_count}")
            logger.info(f"  Masked tokens (input): {masked_count} ({masking_ratio:.1%})")
            logger.info(f"  Supervised tokens (output): {supervised_count} ({(1-masking_ratio):.1%})")

            # Show preview of what's being supervised
            if supervised_count > 0:
                supervised_ids = input_ids[labels != params.MASKING_IGNORE_INDEX]
                if len(supervised_ids) > 0:
                    supervised_text = self.tokenizer.decode(supervised_ids, skip_special_tokens=True)
                    preview_len = min(100, len(supervised_text))
                    logger.info(f"  Training on: '{supervised_text[:preview_len]}...'")

        # Calculate overall statistics
        if total_tokens > 0:
            overall_masking_ratio = total_masked / total_tokens
            overall_supervision_ratio = total_supervised / total_tokens

            logger.info("\n" + "="*60)
            logger.info("Overall Masking Statistics:")
            logger.info(f"  Total tokens: {total_tokens}")
            logger.info(f"  Total masked: {total_masked} ({overall_masking_ratio:.1%})")
            logger.info(f"  Total supervised: {total_supervised} ({overall_supervision_ratio:.1%})")

            # Warnings for unusual masking ratios
            if overall_supervision_ratio < 0.1:
                logger.warning(f"⚠️  Low supervision ratio ({overall_supervision_ratio:.1%}) - check that outputs are being parsed correctly")
            elif overall_supervision_ratio > 0.9:
                logger.warning(f"⚠️  High supervision ratio ({overall_supervision_ratio:.1%}) - may be training on input context")
            else:
                logger.info(f"✓ Masking ratio looks reasonable ({overall_masking_ratio:.1%} input masked, {overall_supervision_ratio:.1%} output supervised)")

        logger.info("="*60 + "\n")

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
        logger.info(f"  Flash Attention 2: {params.USE_FLASH_ATTENTION_2}")

        # Load base model with 4-bit quantization and Flash Attention 2
        start_time = time.time()
        self.model, self.tokenizer = self.model_loader.load_model(
            self.model_name,
            use_flash_attention_2=params.USE_FLASH_ATTENTION_2
        )
        load_time = time.time() - start_time
        logger.info(f"Base model loaded in {load_time:.2f}s with Flash Attention 2: {params.USE_FLASH_ATTENTION_2}")

        if resume_from:
            logger.info(f"Loading checkpoint from: {resume_from}")
            from peft.peft_model import PeftModel
            self.model = PeftModel.from_pretrained(self.model, resume_from)
            logger.info("Checkpoint loaded successfully")
        else:
            # Apply LoRA configuration
            self.model = self.lora_configurator.apply_lora(self.model)
            logger.info("LoRA configuration applied")

        # Ensure tokenizer has padding token and proper configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")

        # Ensure tokenizer has proper padding settings for SFTTrainer
        self.tokenizer.padding_side = "right"  # SFTTrainer expects right padding
        self.tokenizer.truncation_side = "right"

        # Set model's pad_token_id to match tokenizer
        if hasattr(self.model, 'config'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Initialize appropriate data collator
        if hasattr(self, 'use_weighted_loss') and self.use_weighted_loss:
            logger.info("Initializing weighted loss collator for CraigslistBargain...")
            self.weighted_collator = WeightedLossCollator(
                tokenizer=self.tokenizer,
                max_length=params.MAX_SEQ_LEN,
                pad_to_multiple_of=params.PAD_TO_MULTIPLE,
                ignore_index=params.MASKING_IGNORE_INDEX,
                prompt_weight=0.15  # 0.15x weight for prompts
            )
            logger.info("Weighted loss collator initialized (system: 0x, prompt: 0.15x, response: 1x)")
        else:
            logger.info("Initializing JSON data collator...")
            self.json_collator = DataCollatorForJSON(
                tokenizer=self.tokenizer,
                max_length=params.MAX_SEQ_LEN,
                pad_to_multiple_of=params.PAD_TO_MULTIPLE,
                ignore_index=params.MASKING_IGNORE_INDEX
            )
            logger.info("JSON data collator initialized for input/output training")

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

        # Initialize TensorBoard writer
        self.tb_writer = TensorBoardWriter(
            log_dir=str(output_dir.parent / "runs"),
            experiment_name=f"sft{output_suffix}",
            comment=f"lr_{params.SFT_LR}_bs_{params.PER_DEVICE_BATCH_SIZE}"
        )

        # Setup training arguments using SFTConfig for compatibility
        training_args = SFTConfig(
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
            remove_unused_columns=False,  # Keep all columns for custom collator
            report_to=["tensorboard"],
            logging_dir=str(output_dir / "logs"),
            # SFT specific configs
            max_seq_length=params.MAX_SEQ_LEN,
            packing=False,  # Disable packing when using custom collator
            # Skip prepare dataset since our collator handles everything
            dataset_kwargs={
                "skip_prepare_dataset": True
            }
        )

        # Select appropriate collator
        if self.use_weighted_loss and self.weighted_collator:
            logger.info("Using weighted loss collator for CraigslistBargain data...")
            data_collator = self.weighted_collator
        else:
            logger.info("Using JSON data collator for input/output masking...")
            data_collator = self.json_collator

        # Initialize SFTTrainer with JSON data collator
        # Keep dataset raw with {"input": ..., "output": ...} format
        # The DataCollatorForJSON will handle all tokenization and masking

        # IMPORTANT: We pass the raw dataset and let the collator handle everything
        # No formatting_func, no dataset_text_field - just the collator
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # Pass tokenizer for model saving
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=data_collator,  # JSON collator handles everything
            # No formatting_func or dataset_text_field
            # The collator receives raw data and produces input_ids/labels/attention_mask
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
        if self.use_weighted_loss:
            logger.info(f"  - Loss weighting: ENABLED (system: 0x, prompt: 0.15x, response: 1x)")
        else:
            logger.info(f"  - Response masking: ENABLED (loss computed on responses only)")

        # Validate masking on sample batch
        self._validate_masking_on_sample(train_dataset, data_collator)

        # Add callback for logging and error handling
        class ErrorHandlingCallback(TrainerCallback):
            def __init__(self, metrics_tracker, error_handling_enabled, parent):
                self.metrics_tracker = metrics_tracker
                self.error_handling_enabled = error_handling_enabled
                self.parent = parent  # Reference to SFTTrainingPipeline
                self.last_loss = None

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

                    # Log to TensorBoard
                    if self.parent.tb_writer:
                        # Log loss
                        if "loss" in logs:
                            self.parent.tb_writer.log_loss_metrics(
                                loss=logs['loss'],
                                step=step,
                                loss_type="train"
                            )

                        # Log eval loss if present
                        if "eval_loss" in logs:
                            self.parent.tb_writer.log_loss_metrics(
                                loss=logs['eval_loss'],
                                step=step,
                                loss_type="eval"
                            )

                        # Log learning rate
                        if "learning_rate" in logs:
                            self.parent.tb_writer.log_learning_rate(
                                logs['learning_rate'],
                                step
                            )

                        # Log performance metrics
                        if "train_samples_per_second" in logs:
                            self.parent.tb_writer.log_performance_metrics(
                                samples_per_second=logs.get("train_samples_per_second"),
                                step=step
                            )

                        # Log all error handlers (pass model to gradient monitor)
                        if self.parent.gradient_monitor:
                            # Set model for gradient monitoring if not already set
                            if not hasattr(self.parent.gradient_monitor, 'model') or self.parent.gradient_monitor.model is None:
                                self.parent.gradient_monitor.model = kwargs.get('model', self.parent.model)

                        self.parent.tb_writer.log_all_error_handlers(
                            batch_optimizer=self.parent.batch_optimizer,
                            gradient_monitor=self.parent.gradient_monitor,
                            loss_validator=self.parent.loss_validator,
                            precision_handler=self.parent.precision_handler,
                            step=step
                        )

                    # Log key metrics to logger
                    if "loss" in logs:
                        logger.info(f"Step {step}: loss={logs['loss']:.4f}")
                        self.last_loss = logs['loss']

                        # Validate loss if error handling is enabled
                        if self.error_handling_enabled and self.parent.loss_validator:
                            is_valid, action, info = self.parent.loss_validator.validate_loss(
                                logs['loss'], step
                            )
                            if not is_valid:
                                logger.warning(f"Loss validation failed at step {step}: {info}")
                                if action == "skip_update":
                                    control.should_training_stop = False
                                    control.should_save = False
                                elif action == "restore_checkpoint":
                                    control.should_training_stop = True
                                    logger.error("Loss diverged - stopping training for checkpoint restore")

                                # Log error to TensorBoard
                                if self.parent.tb_writer:
                                    self.parent.tb_writer.log_error_event(
                                        error_type="loss_validation",
                                        error_message=str(info),
                                        step=step,
                                        severity="error" if action == "restore_checkpoint" else "warning"
                                    )

            def on_before_optimizer_step(self, args, state, control, **kwargs):
                """Check gradients before optimizer step when they're available."""
                if self.error_handling_enabled and self.parent.gradient_monitor:
                    model = kwargs.get('model', self.parent.model)
                    if model:
                        is_healthy, stats = self.parent.gradient_monitor.check_gradients(model)
                        if not is_healthy:
                            logger.warning(f"Unhealthy gradients detected at step {state.global_step}")
                            # Apply adaptive gradient clipping
                            self.parent.gradient_monitor.clip_gradients(model)

            def on_step_end(self, args, state, control, **kwargs):
                """Handle post-step operations."""
                # Gradients are now checked in on_before_optimizer_step

                # Save state periodically for recovery
                if self.error_handling_enabled and self.parent.state_manager:
                    if state.global_step % self.parent.state_manager.auto_save_interval == 0:
                        training_state = TrainingState(
                            stage="sft",
                            epoch=state.epoch or 0.0,
                            global_step=state.global_step,
                            best_metric=self.last_loss or 0.0,
                            best_metric_name="loss",
                            total_epochs=args.num_train_epochs,
                            total_steps=args.max_steps or 0,
                            samples_seen=state.global_step * args.per_device_train_batch_size,
                            tokens_processed=0,  # Would need to track this
                            model_name=self.parent.model_name,
                            checkpoint_path=str(output_dir / f"checkpoint-{state.global_step}"),
                            adapter_config={},
                            learning_rate=args.learning_rate,
                            optimizer_state_dict=None,  # Too large to save every time
                            scheduler_state_dict=None,
                            dataset_path="",  # Would need to be passed in
                            dataset_index=0,
                            batch_size=args.per_device_train_batch_size,
                            gradient_accumulation_steps=args.gradient_accumulation_steps,
                            train_loss=self.last_loss or 0.0,
                            eval_loss=None,
                            metrics_history={},
                            timestamp=datetime.now().isoformat(),
                            recovery_count=0,
                            last_error=None,
                            cuda_memory_allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                            cuda_memory_reserved=torch.cuda.memory_reserved() if torch.cuda.is_available() else None
                        )
                        self.parent.state_manager.save_state(training_state)

        trainer.add_callback(ErrorHandlingCallback(
            self.metrics_tracker,
            self.enable_error_handling,
            self
        ))

        # Wrap training with error handler if enabled
        if self.enable_error_handling and self.error_handler:
            logger.info("Starting training with error handling enabled...")

            # Suppress TRL warnings during training
            warnings.filterwarnings("ignore", message="Could not find response key")
            warnings.filterwarnings("ignore", category=UserWarning, module="trl")

            start_time = time.time()

            # Create training context for error recovery
            training_context = {
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "max_seq_length": params.MAX_SEQ_LEN,
                "gradient_checkpointing": params.GRADIENT_CHECKPOINTING,
                "stage": "sft"
            }

            # Use precision handler context if available
            if self.precision_handler:
                with self.precision_handler.autocast_context():
                    train_result = self.error_handler.safe_execute(
                        trainer.train,
                        context=training_context
                    )
            else:
                train_result = self.error_handler.safe_execute(
                    trainer.train,
                    context=training_context
                )

            # Handle OOM or other errors
            if train_result is None:
                logger.error("Training failed after maximum recovery attempts")
                # Try to recover with reduced batch size
                if self.batch_optimizer:
                    new_batch_size, new_grad_accum = self.batch_optimizer.reduce_batch_size_on_oom()
                    logger.info(f"Retrying with batch_size={new_batch_size}, grad_accum={new_grad_accum}")
                    training_args.per_device_train_batch_size = new_batch_size
                    training_args.gradient_accumulation_steps = new_grad_accum
                    # Recreate trainer with new batch size
                    trainer = SFTTrainer(
                        model=self.model,
                        tokenizer=self.tokenizer,  # Pass tokenizer for model saving
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        args=training_args,
                        data_collator=data_collator,  # Use the JSON collator
                        # No formatting_func or dataset_text_field
                    )
                    train_result = trainer.train()
                else:
                    raise RuntimeError("Training failed and recovery was not possible")

            training_time = time.time() - start_time
        else:
            # Train without error handling
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

        # Log final hyperparameters and metrics to TensorBoard
        if self.tb_writer:
            hparams = {
                "learning_rate": params.SFT_LR,
                "batch_size": params.PER_DEVICE_BATCH_SIZE,
                "grad_accum_steps": params.GRAD_ACCUM_STEPS,
                "max_seq_len": params.MAX_SEQ_LEN,
                "lora_rank": params.R,
                "lora_alpha": params.ALPHA,
                "lora_dropout": params.DROPOUT,
                "optimizer": params.OPTIM,
                "scheduler": params.SFT_LR_SCHEDULER,
                "warmup_ratio": params.SFT_WARMUP_RATIO,
            }
            final_metrics = {
                "final_loss": metrics['train_loss'],
                "total_steps": metrics['total_steps'],
                "train_runtime": metrics['train_runtime'],
                "samples_per_second": metrics['train_samples_per_second'],
            }
            self.tb_writer.log_hyperparameters(hparams, final_metrics)
            self.tb_writer.close()

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