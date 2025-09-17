"""
DPO (Direct Preference Optimization) training module using TRL's DPOTrainer.
Handles QLoRA DPO training with reward margin tracking and early stopping.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from datasets import Dataset as HFDataset
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
    TrainerCallback
)
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig

from peft.mapping import get_peft_model

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.checkpoint_manager import CheckpointManager
from training.metrics_tracker import TrainingMetricsTracker
from utils.dataset_loader import UnifiedDatasetLoader
import training_parameters as params

# Import error handling modules
from utils.error_handler import ErrorHandler
from utils.training_state import TrainingStateManager, TrainingState
from utils.batch_size_optimizer import BatchSizeOptimizer, BatchSizeConfig
from utils.gradient_monitor import GradientMonitor, GradientHealthConfig
from utils.loss_validator import LossValidator, LossValidationConfig
from utils.precision_handler import PrecisionHandler, PrecisionConfig
from utils.tensorboard_writer import TensorBoardWriter
from datetime import datetime
import gc

logger = logging.getLogger(__name__)


# DPODataset class removed - functionality moved to UnifiedDatasetLoader


class DPOErrorHandlingCallback(TrainerCallback):
    """
    Custom callback for error handling during DPO training.
    Monitors loss, gradients, and DPO-specific metrics like reward margins.
    """

    def __init__(self,
                 gradient_monitor: Optional[GradientMonitor] = None,
                 loss_validator: Optional[LossValidator] = None,
                 reward_validator: Optional[LossValidator] = None,
                 state_manager: Optional[TrainingStateManager] = None,
                 model: Optional[torch.nn.Module] = None):
        """
        Initialize DPO error handling callback.

        Args:
            gradient_monitor: Monitor for gradient health
            loss_validator: Validator for loss values
            reward_validator: Validator for reward margins (DPO-specific)
            state_manager: Manager for training state persistence
            model: Model for gradient checking
        """
        self.gradient_monitor = gradient_monitor
        self.loss_validator = loss_validator
        self.reward_validator = reward_validator
        self.state_manager = state_manager
        self.model = model
        self.steps_since_save = 0
        self.reward_margin_history = []

    def __init__(self,
                 gradient_monitor: Optional[GradientMonitor] = None,
                 loss_validator: Optional[LossValidator] = None,
                 reward_validator: Optional[LossValidator] = None,
                 state_manager: Optional[TrainingStateManager] = None,
                 model: Optional[torch.nn.Module] = None,
                 tb_writer: Optional[TensorBoardWriter] = None):
        """Initialize with TensorBoard writer."""
        super().__init__(gradient_monitor, loss_validator, reward_validator, state_manager, model)
        self.tb_writer = tb_writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Monitor metrics during logging.

        Args:
            args: Training arguments (unused but required by interface)
            state: Current training state
            control: Training control flow
            logs: Dictionary of logged metrics
            **kwargs: Additional arguments (unused but required by interface)
        """
        if not logs:
            return control

        step = state.global_step

        # Log to TensorBoard if available
        if self.tb_writer and logs:
            # Log DPO specific metrics
            if "reward_margin" in logs:
                self.tb_writer.log_dpo_metrics(
                    chosen_rewards=logs.get("chosen_reward", 0),
                    rejected_rewards=logs.get("rejected_reward", 0),
                    reward_margin=logs["reward_margin"],
                    step=step
                )

            # Log loss
            if "loss" in logs:
                self.tb_writer.log_loss_metrics(
                    loss=logs["loss"],
                    step=step,
                    loss_type="train"
                )

            # Log eval loss
            if "eval_loss" in logs:
                self.tb_writer.log_loss_metrics(
                    loss=logs["eval_loss"],
                    step=step,
                    loss_type="eval"
                )

            # Log learning rate
            if "learning_rate" in logs:
                self.tb_writer.log_learning_rate(
                    logs["learning_rate"],
                    step
                )

        # Check standard loss
        if self.loss_validator and "loss" in logs:
            loss = logs["loss"]
            is_valid, action, info = self.loss_validator.validate_loss(
                torch.tensor(loss), step=state.global_step
            )

            if not is_valid:
                logger.warning(f"Loss validation failed at step {state.global_step}: {action}")
                if action == "skip_update":
                    control.should_training_stop = False
                    control.should_log = False
                elif action == "restore_checkpoint":
                    control.should_training_stop = True
                    logger.error("Critical loss issue - stopping training")

        # Check DPO-specific reward margins
        if self.reward_validator:
            if "reward_margin" in logs:
                margin = logs["reward_margin"]
                self.reward_margin_history.append(margin)

                is_valid, action, info = self.reward_validator.validate_loss(
                    torch.tensor(margin), step=state.global_step
                )

                if not is_valid:
                    logger.warning(f"Reward margin issue at step {state.global_step}: {info}")

                    # Check for reward collapse
                    if "plateau" in str(info).lower() or margin < 0.01:
                        logger.error("Reward collapse detected - may need to adjust DPO beta")
                        control.should_training_stop = True

            # Monitor chosen vs rejected rewards separately
            if "chosen_reward" in logs and "rejected_reward" in logs:
                chosen = logs["chosen_reward"]
                rejected = logs["rejected_reward"]

                # Check for reward inversion (rejected > chosen)
                if rejected > chosen:
                    logger.warning(f"Reward inversion at step {state.global_step}: "
                                   f"rejected ({rejected:.4f}) > chosen ({chosen:.4f})")

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Check gradients and save state after each step.

        Args:
            args: Training arguments (unused but required by interface)
            state: Current training state
            control: Training control flow (unused but required by interface)
            **kwargs: Additional arguments (unused but required by interface)
        """
        # Log error handler metrics to TensorBoard
        if self.tb_writer:
            self.tb_writer.log_all_error_handlers(
                batch_optimizer=None,  # Not available in this context
                gradient_monitor=self.gradient_monitor,
                loss_validator=self.loss_validator,
                precision_handler=None,  # Not available in this context
                step=state.global_step
            )

        # Check gradient health
        if self.gradient_monitor and self.model:
            is_healthy, stats = self.gradient_monitor.check_gradients(self.model)

            if not is_healthy and stats:
                issues = stats.get("issues", [])
                logger.warning(f"Gradient issues at step {state.global_step}: {issues}")

                # Apply gradient clipping if needed
                if any("explosion" in str(issue).lower() for issue in issues):
                    self.gradient_monitor.clip_gradients(self.model)
                    logger.info("Applied adaptive gradient clipping")

                # Log issues to TensorBoard
                if self.tb_writer:
                    for issue in issues:
                        self.tb_writer.log_error_event(
                            error_type="gradient_issue",
                            error_message=issue,
                            step=state.global_step,
                            severity="warning"
                        )

        # Save state periodically
        self.steps_since_save += 1
        if self.state_manager and self.steps_since_save >= 100:
            # Create a simple state dictionary for DPO training
            state_dict = {
                "step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric or 0.0,
                "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
                "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
                "reward_margins": self.reward_margin_history[-10:] if self.reward_margin_history else [],
                "timestamp": datetime.now().isoformat()
            }
            # Save as JSON directly since we're tracking DPO-specific metrics
            import json
            state_path = self.state_manager.checkpoint_dir / f"dpo_state_step_{state.global_step}.json"
            with open(state_path, 'w') as f:
                json.dump(state_dict, f, indent=2)
            self.steps_since_save = 0


class DPOTrainingModule:
    """
    DPO training module using TRL's DPOTrainer with QLoRA.
    Integrates with existing data pipeline from EpisodeLogger/DataConverter.
    """

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "checkpoints/dpo",
                 reference_model_path: Optional[str] = None,
                 enable_error_handling: bool = True):
        """
        Initialize DPO training module.

        Args:
            model_name: Base model name or path
            output_dir: Directory for saving checkpoints
            reference_model_path: Path to reference model (if None, uses base model)
            enable_error_handling: Whether to enable comprehensive error handling
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_model_path = reference_model_path or model_name
        self.enable_error_handling = enable_error_handling

        # Initialize components
        self.checkpoint_manager = CheckpointManager(str(self.output_dir))
        self.metrics_tracker = TrainingMetricsTracker(log_dir=str(self.output_dir / "metrics"), stage="dpo")
        self.dataset_loader = UnifiedDatasetLoader()
        self.tb_writer = None  # Initialize during training

        # Track training state
        self.reward_margins = []
        self.best_reward_margin = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = 5

        # Initialize error handlers if enabled
        if self.enable_error_handling:
            # Central error handler
            self.error_handler = ErrorHandler()

            # Training state manager for crash recovery
            self.state_manager = TrainingStateManager(
                checkpoint_dir=str(self.output_dir / "training_states")
            )

            # Batch size optimizer for OOM recovery
            self.batch_optimizer = BatchSizeOptimizer(
                config=BatchSizeConfig(
                    initial_batch_size=params.PER_DEVICE_BATCH_SIZE,
                    min_batch_size=1,
                    max_batch_size=params.PER_DEVICE_BATCH_SIZE * 2,
                    gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
                    enable_auto_tuning=True,
                    memory_fraction_target=0.9,
                    reduction_factor=0.5
                ),
                log_dir=str(self.output_dir)
            )

            # Gradient monitor for health checking
            self.gradient_monitor = GradientMonitor(
                config=GradientHealthConfig(
                    max_gradient_norm=1.0,
                    explosion_threshold=100.0,
                    vanishing_threshold=1e-7,
                    enable_clipping=True,
                    clip_value=1.0,
                    adaptive_clipping=True,
                    log_frequency=params.LOGGING_STEPS
                )
            )

            # Loss validator for detecting training issues
            self.loss_validator = LossValidator(
                config=LossValidationConfig(
                    max_loss=100.0,
                    min_loss=0.0,
                    nan_action="skip",
                    inf_action="skip",
                    stability_window=50,
                    enable_anomaly_detection=True,
                    divergence_threshold=1000.0,
                    divergence_patience=5
                )
            )

            # Precision handler for mixed precision training
            self.precision_handler = PrecisionHandler(
                config=PrecisionConfig(
                    use_bf16=params.BF16,
                    use_fp16=not params.BF16,
                    use_fp32_fallback=True,
                    use_amp=True,
                    use_gradient_scaling=not params.BF16,
                    auto_detect_hardware=True
                )
            )

            logger.info("Error handling initialized with all protection modules")

            # DPO-specific: Initialize reward margin validator
            self.reward_validator = LossValidator(
                config=LossValidationConfig(
                    max_loss=10.0,  # Max acceptable reward margin
                    min_loss=-10.0,  # Min acceptable reward margin
                    nan_action="skip",
                    inf_action="skip",
                    stability_window=20,
                    plateau_threshold=0.001,  # Detect reward collapse
                    plateau_patience=10
                )
            )

    def validate_preference_pair(self, chosen: str, rejected: str) -> Tuple[bool, str]:
        """
        Validate a DPO preference pair for quality issues.

        Args:
            chosen: The preferred response
            rejected: The rejected response

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for empty or identical pairs
        if not chosen or not rejected:
            return False, "Empty response in pair"

        if chosen.strip() == rejected.strip():
            return False, "Identical chosen and rejected responses"

        # Check for extreme length differences (might indicate data issues)
        len_ratio = len(chosen) / max(len(rejected), 1)
        if len_ratio > 10 or len_ratio < 0.1:
            return False, f"Extreme length ratio: {len_ratio:.2f}"

        # Check for suspicious patterns
        if chosen.count("[") != chosen.count("]") or rejected.count("[") != rejected.count("]"):
            return False, "Unmatched brackets (potential truncation)"

        return True, "Valid pair"

    def prepare_dataset(self, data_path: str, tokenizer: AutoTokenizer) -> HFDataset:
        """
        Prepare dataset for DPO training using UnifiedDatasetLoader.

        Args:
            data_path: Path to NDJSON file with DPO pairs
            tokenizer: Tokenizer for processing (kept for compatibility)

        Returns:
            HuggingFace Dataset ready for DPOTrainer
        """
        # Load and preprocess using unified loader
        hf_dataset = self.dataset_loader.load_dpo_dataset(data_path)

        logger.info(f"Prepared {len(hf_dataset)} DPO pairs for training")
        return hf_dataset

    def setup_training_args(self, num_train_epochs: Optional[int] = None) -> DPOConfig:
        """
        Setup training arguments for DPO.

        Args:
            num_train_epochs: Number of training epochs (overrides max_steps if set)

        Returns:
            DPOConfig configured for DPO training
        """
        # When load_best_model_at_end is True, eval_strategy must match save_strategy
        # For DPO, we want to save and evaluate at the same intervals
        save_strategy = params.SAVE_STRATEGY
        eval_strategy = "steps" if save_strategy == "steps" else "epoch"
        eval_steps = params.SAVE_STEPS if save_strategy == "steps" else None

        args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs if num_train_epochs else -1,
            max_steps=params.DPO_MAX_STEPS if not num_train_epochs else -1,
            per_device_train_batch_size=params.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
            learning_rate=params.DPO_LR,
            optim=params.OPTIM,
            save_steps=params.SAVE_STEPS,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            logging_steps=params.LOGGING_STEPS,
            warmup_ratio=0.1,  # Fixed warmup for DPO
            lr_scheduler_type="linear",  # Linear decay for DPO
            bf16=params.BF16,
            gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
            remove_unused_columns=False,
            # DPO-specific parameters
            beta=params.DPO_BETA,
            label_smoothing=params.DPO_LABEL_SMOOTH,
            max_length=params.MAX_SEQ_LEN,
            max_prompt_length=params.DPO_MAX_PROMPT_LEN,
            generate_during_eval=False,
            load_best_model_at_end=True,
            metric_for_best_model="reward_margin",
            greater_is_better=True,
            report_to=["tensorboard"],
            logging_dir=str(self.output_dir / "logs"),
            seed=params.SEED,
        )

        return args

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics for DPO training.

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        # Extract rewards from predictions
        predictions = eval_pred.predictions

        if len(predictions) >= 2:
            # Convert to numpy arrays if they're lists
            chosen_rewards = np.array(predictions[0])
            rejected_rewards = np.array(predictions[1])

            # Compute reward margin (higher is better)
            reward_margins = chosen_rewards - rejected_rewards
            avg_margin = float(np.mean(reward_margins))

            # Track for early stopping
            self.reward_margins.append(avg_margin)

            # Check for improvement
            if avg_margin > self.best_reward_margin:
                self.best_reward_margin = avg_margin
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            metrics = {
                'reward_margin': avg_margin,
                'chosen_reward': float(np.mean(chosen_rewards)),
                'rejected_reward': float(np.mean(rejected_rewards)),
                'patience': self.patience_counter
            }

            # Log to metrics tracker
            self.metrics_tracker.log_training_step(
                step=len(self.reward_margins),
                epoch=0.0,  # Will be updated by trainer
                metrics=metrics
            )

            return metrics

        return {'reward_margin': 0.0}

    def should_early_stop(self) -> bool:
        """
        Check if training should stop early based on reward divergence.

        Returns:
            True if should stop early
        """
        # Check patience
        if self.patience_counter >= self.early_stopping_patience:
            logger.warning(f"Early stopping triggered: no improvement for {self.patience_counter} evaluations")
            return True

        # Check for reward collapse (margins getting too small)
        if len(self.reward_margins) >= 5:
            recent_margins = self.reward_margins[-5:]
            avg_recent = sum(recent_margins) / len(recent_margins)
            if avg_recent < 0.01:  # Threshold for collapse
                logger.warning(f"Early stopping triggered: reward margin collapsed to {avg_recent}")
                return True

        return False

    def train(self,
              train_data_path: str,
              eval_data_path: Optional[str] = None,
              num_train_epochs: Optional[int] = None,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run DPO training.

        Args:
            train_data_path: Path to training DPO pairs (NDJSON)
            eval_data_path: Optional path to eval DPO pairs
            num_train_epochs: Number of epochs (overrides max_steps)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting DPO training...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Reference model: {self.reference_model_path}")
        logger.info(f"Training data: {train_data_path}")

        # Initialize TensorBoard writer
        run_name = f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tb_writer = TensorBoardWriter(
            log_dir=str(self.output_dir.parent / "runs"),
            experiment_name=run_name,
            comment=f"beta_{params.DPO_BETA}_lr_{params.DPO_LR}"
        )

        # Load model and tokenizer with Flash Attention 2
        logger.info("Loading model with QLoRA and Flash Attention 2...")
        model, tokenizer = ModelLoader.load_model(
            self.model_name,
            use_flash_attention_2=params.USE_FLASH_ATTENTION_2
        )

        # Load reference model (frozen in 4-bit) with Flash Attention 2
        logger.info("Loading reference model with Flash Attention 2...")
        ref_model = ModelLoader.load_reference_model(
            self.reference_model_path,
            use_flash_attention_2=params.USE_FLASH_ATTENTION_2
        )

        # Apply LoRA configuration
        lora_config = LoRAConfigurator.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data_path, tokenizer)
        eval_dataset = None
        if eval_data_path:
            eval_dataset = self.prepare_dataset(eval_data_path, tokenizer)

        # Setup training arguments
        training_args = self.setup_training_args(num_train_epochs)

        # Initialize DPO trainer
        trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )

        # Add callbacks
        if eval_dataset:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=0.001
            )
            trainer.add_callback(early_stopping)

        # Add error handling callback if enabled
        if self.enable_error_handling:
            error_callback = DPOErrorHandlingCallback(
                gradient_monitor=self.gradient_monitor,
                loss_validator=self.loss_validator,
                reward_validator=self.reward_validator if hasattr(self, 'reward_validator') else None,
                state_manager=self.state_manager,
                model=model,
                tb_writer=self.tb_writer
            )
            trainer.add_callback(error_callback)
            logger.info("Error handling callback added to trainer")

        # Training context for error handler
        training_context = {
            "model": model,
            "ref_model": ref_model,
            "tokenizer": tokenizer,
            "training_args": training_args,
            "output_dir": str(self.output_dir),
            "resume_checkpoint": resume_from_checkpoint
        }

        # Start training with error handling
        logger.info("Starting DPO training loop...")

        if self.enable_error_handling:
            # Handle dual model memory for DPO
            if torch.cuda.is_available():
                # Clear cache before training
                torch.cuda.empty_cache()
                gc.collect()

                # Log memory usage for both models
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU memory before training: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

            # Wrap training with precision handling and error recovery
            with self.precision_handler.autocast_context():
                train_result = self.error_handler.safe_execute(
                    trainer.train,
                    context=training_context,
                    resume_from_checkpoint=resume_from_checkpoint
                )

            # Handle OOM errors with batch size reduction
            if train_result is None:
                logger.warning("Training failed, attempting recovery with reduced batch size")

                # Get new batch size from optimizer
                new_batch_size, new_grad_accum = self.batch_optimizer.reduce_batch_size_on_oom()

                # Update training args
                training_args.per_device_train_batch_size = new_batch_size
                training_args.gradient_accumulation_steps = new_grad_accum

                logger.info(f"Retrying with batch_size={new_batch_size}, grad_accum={new_grad_accum}")

                # Recreate trainer with new batch size
                trainer = TRLDPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=self.compute_metrics if eval_dataset else None,
                )

                # Re-add error callback
                error_callback = DPOErrorHandlingCallback(
                    gradient_monitor=self.gradient_monitor,
                    loss_validator=self.loss_validator,
                    reward_validator=self.reward_validator if hasattr(self, 'reward_validator') else None,
                    state_manager=self.state_manager,
                    model=model,
                    tb_writer=self.tb_writer
                )
                trainer.add_callback(error_callback)

                # Re-add early stopping if we have eval data
                if eval_dataset:
                    early_stopping_retry = EarlyStoppingCallback(
                        early_stopping_patience=self.early_stopping_patience,
                        early_stopping_threshold=0.001
                    )
                    trainer.add_callback(early_stopping_retry)

                # Retry training
                with self.precision_handler.autocast_context():
                    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            # Standard training without error handling
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "reward_margins": self.reward_margins,
            "best_reward_margin": self.best_reward_margin,
            "total_steps": train_result.global_step,
        }

        # Save metrics to file
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log final statistics
        logger.info("=" * 50)
        logger.info("DPO Training Complete!")
        logger.info(f"Final loss: {metrics['train_loss']:.4f}")
        logger.info(f"Best reward margin: {self.best_reward_margin:.4f}")
        logger.info(f"Total steps: {metrics['total_steps']}")
        logger.info(f"Model saved to: {self.output_dir / 'final_model'}")
        logger.info("=" * 50)

        # Log final hyperparameters and metrics to TensorBoard
        if self.tb_writer:
            hparams = {
                "beta": params.DPO_BETA,
                "learning_rate": params.DPO_LR,
                "batch_size": params.PER_DEVICE_BATCH_SIZE,
                "grad_accum_steps": params.GRAD_ACCUM_STEPS,
                "max_prompt_length": params.DPO_MAX_PROMPT_LEN,
                "label_smoothing": params.DPO_LABEL_SMOOTH,
                "lora_rank": params.R,
                "lora_alpha": params.ALPHA,
            }
            final_metrics = {
                "final_loss": metrics["train_loss"],
                "best_reward_margin": metrics["best_reward_margin"],
                "total_steps": metrics["total_steps"],
            }
            self.tb_writer.log_hyperparameters(hparams, final_metrics)
            self.tb_writer.close()

        return metrics

    def train_from_episodes(self,
                            episodes_path: str,
                            min_score_gap: float = 0.3,
                            num_train_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train DPO directly from episode files by auto-generating pairs.

        Args:
            episodes_path: Path to episodes NDJSON file
            min_score_gap: Minimum protocol score gap for pair selection
            num_train_epochs: Number of training epochs

        Returns:
            Training results
        """
        from training.data_converter import DataConverter

        logger.info(f"Auto-generating DPO pairs from episodes: {episodes_path}")

        # Use DataConverter to generate DPO pairs
        converter = DataConverter(output_dir=str(self.output_dir / "data"))
        dpo_file = converter.auto_generate_dpo(
            episodes_path,
            output_file=str(self.output_dir / "data" / "auto_dpo_pairs.jsonl"),
            score_threshold=min_score_gap
        )

        # Train on generated pairs
        return self.train(
            train_data_path=dpo_file,
            num_train_epochs=num_train_epochs
        )

    def merge_and_save(self, checkpoint_path: str, output_path: str):
        """
        Merge LoRA weights with base model and save.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            output_path: Path to save merged model
        """
        from peft.peft_model import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Merging LoRA weights from {checkpoint_path}...")

        # Load base model WITHOUT quantization for merging
        # Cannot merge into a quantized model
        logger.info("Loading base model without quantization for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if params.BF16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA weights
        logger.info("Loading LoRA adapter weights...")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

        # Verify we have a PEFT model with merge_and_unload method
        if not hasattr(peft_model, 'merge_and_unload'):
            raise AttributeError(f"Model does not have merge_and_unload method. Got type: {type(peft_model)}")

        # Merge and unload
        logger.info("Merging LoRA weights into base model...")
        # Type ignore for Pylance - merge_and_unload returns a model, not a Tensor
        merged_model = peft_model.merge_and_unload()  # type: ignore

        # Save merged model
        logger.info(f"Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")


def main():
    """Command-line interface for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO training for negotiation agents")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                       help="Base model name or path")
    parser.add_argument("--train-data", required=True,
                       help="Path to training DPO pairs (NDJSON)")
    parser.add_argument("--eval-data", help="Path to eval DPO pairs")
    parser.add_argument("--reference-model", help="Path to reference model")
    parser.add_argument("--output-dir", default="checkpoints/dpo",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--from-episodes", help="Generate DPO pairs from episodes file")
    parser.add_argument("--min-score-gap", type=float, default=0.3,
                       help="Minimum score gap for auto-generated pairs")

    args = parser.parse_args()

    # Initialize trainer
    trainer = DPOTrainingModule(
        model_name=args.model,
        output_dir=args.output_dir,
        reference_model_path=args.reference_model
    )

    # Train
    if args.from_episodes:
        results = trainer.train_from_episodes(
            args.from_episodes,
            min_score_gap=args.min_score_gap,
            num_train_epochs=args.epochs
        )
    else:
        results = trainer.train(
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            num_train_epochs=args.epochs,
            resume_from_checkpoint=args.resume
        )

    print(f"\nTraining complete! Results saved to {args.output_dir}")
    print(f"Final metrics: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()