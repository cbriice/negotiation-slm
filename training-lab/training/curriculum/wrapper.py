import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import torch
from transformers import TrainerCallback, TrainingArguments, AutoTokenizer

from training.curriculum.phase_config import CurriculumPhase, PhaseConfig
from training.curriculum.masking_collator import CurriculumMaskingCollator
from training.curriculum.data_sampler import CurriculumDataSampler
from training.curriculum.monitor import CurriculumMonitor
from training.curriculum.checkpoint_utils import save_phase_checkpoint
from training.metrics_tracker import TrainingMetricsTracker
import curriculum_parameters as cp

logger = logging.getLogger(__name__)

class CurriculumTrainingWrapper:
    """Main wrapper orchestrating curriculum training."""

    def __init__(
        self,
        conversational_path: str,
        negotiation_path: str,
        model_name: str,
        output_dir: str = "models/curriculum",
        recovery_enabled: bool = True,
        base_lr: float = 1e-5
    ):
        self.conversational_path = conversational_path
        self.negotiation_path = negotiation_path
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_enabled = recovery_enabled
        self.base_lr = base_lr

        # Initialize components
        self.phase_config = PhaseConfig()
        self.data_sampler = CurriculumDataSampler(conversational_path, negotiation_path)
        self.monitor = CurriculumMonitor()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get the first phase from configuration (supports any phase naming)
        first_phase_name = self.phase_config.get_phase_at_step(0)

        # Initialize masking collator
        self.masking_collator = CurriculumMaskingCollator(
            self.tokenizer,
            first_phase_name
        )

        # Training state
        self.current_step = 0
        self.current_phase = first_phase_name
        self.recovery_mode = False
        self.recovery_end_step = 0
        self.model = None  # Set via set_model() after initialization

        # Apply initial phase configuration
        self._apply_phase_config(self.current_phase)

        logger.info(f"Curriculum wrapper initialized with {self.data_sampler.total_examples} total examples")

    def _apply_phase_config(self, phase: str):
        """Apply configuration for a specific phase."""
        config = self.phase_config.phases[phase]

        # Update data sampling ratio
        self.data_sampler.set_ratio(config["data_ratio"])

        # Update masking strategy
        self.masking_collator.set_phase(phase, config)

        logger.info(f"Applied phase configuration: {phase}")
        logger.info(f"  Data ratio: {config['data_ratio']}")
        logger.info(f"  Masking strategy: {config['masking']}")
        logger.info(f"  LR multiplier: {config['lr_multiplier']}")

    def update_step(self, step: int):
        """Update current step and check for phase transition."""
        self.current_step = step

        # Check if we should transition phase
        new_phase = self.phase_config.get_phase_at_step(step)
        if new_phase != self.current_phase:
            self._transition_phase(new_phase)

    def _transition_phase(self, new_phase: str):
        """Transition to a new training phase."""
        logger.info(f"Transitioning from {self.current_phase} to {new_phase} at step {self.current_step}")

        self.current_phase = new_phase
        self._apply_phase_config(new_phase)

        # Save checkpoint at phase transition
        self._save_phase_checkpoint()

    def set_model(self, model):
        """Set model reference for checkpoint saving."""
        self.model = model

    def _get_phase_number(self, phase_name: str) -> int:
        """Get 1-indexed phase number from phase name."""
        phase_names = list(self.phase_config.phases.keys())
        try:
            return phase_names.index(phase_name) + 1
        except ValueError:
            logger.warning(f"Unknown phase name: {phase_name}, using step-based fallback")
            # Fallback: count which phase we're in based on cp.PHASES
            for i, phase in enumerate(cp.PHASES, 1):
                if phase.get('name') == phase_name:
                    return i
            return 1  # Default fallback

    def _save_phase_checkpoint(self):
        """Save checkpoint at phase boundary."""
        if self.model is None:
            logger.warning("Model not set, skipping checkpoint save")
            return

        phase_num = self._get_phase_number(self.current_phase)
        checkpoint_dir = save_phase_checkpoint(
            model=self.model,
            phase_num=phase_num,
            phase_name=self.current_phase,
            step=self.current_step,
            total_phases=len(cp.PHASES),
            output_dir=self.output_dir
        )
        logger.info(f"Phase {phase_num} checkpoint saved to {checkpoint_dir}")

    def get_current_phase(self) -> str:
        """Get the current training phase."""
        return self.current_phase

    def get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for current phase."""
        if self.recovery_mode:
            return 0.1  # Emergency reduction
        return self.phase_config.phases[self.current_phase]["lr_multiplier"]

    def should_validate(self) -> bool:
        """Check if validation should run at current step."""
        if self.current_step == 0:
            return False  # Skip validation at start (model not trained yet)

        validation_freq = self.phase_config.phases[self.current_phase]["validation_frequency"]
        return self.current_step % validation_freq == 0

    def run_health_check(self, model) -> Tuple[str, Dict[str, Any]]:
        """Run health check on current model."""
        return self.monitor.checkpoint_health_check(model, self.tokenizer, self.current_step)

    def execute_recovery_protocol(self, model, optimizer, metrics: Dict[str, Any]):
        """Execute emergency recovery when collapse detected."""
        logger.critical(f"EXECUTING RECOVERY PROTOCOL at step {self.current_step}")

        # 1. Reduce learning rate to 10%
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.base_lr * 0.1

        # 2. Override data mixture to 70% conversational
        self.data_sampler.override_ratio({"conversational": 0.7, "negotiation": 0.3})

        # 3. Disable all masking temporarily
        self.masking_collator.set_phase("recovery", {"masking": {
            "conversational": "none",
            "negotiation": "none"
        }})

        # 4. Set recovery mode
        self.recovery_mode = True
        self.recovery_end_step = self.current_step + 200

        # 5. Save emergency checkpoint
        self.monitor.save_emergency_checkpoint(
            model, optimizer, self.current_step, metrics, str(self.output_dir)
        )

        logger.info("Recovery protocol activated:")
        logger.info("  - Learning rate reduced to 10%")
        logger.info("  - Data ratio: 70% conversational, 30% negotiation")
        logger.info("  - All masking disabled")
        logger.info(f"  - Recovery mode until step {self.recovery_end_step}")

    def check_recovery_complete(self) -> bool:
        """Check if recovery mode should end."""
        if self.recovery_mode and self.current_step >= self.recovery_end_step:
            logger.info("Recovery mode complete, resuming normal training")
            self.recovery_mode = False
            # Reapply current phase configuration
            self._apply_phase_config(self.current_phase)
            return True
        return False

    def get_data_collator(self):
        """Get the masking collator for training."""
        return self.masking_collator

    def get_dataset_iterator(self, batch_size: int, shuffle: bool = True):
        """Get dataset iterator with proper mixing."""
        return self.data_sampler.get_epoch_iterator(batch_size, shuffle)


class CurriculumCallback(TrainerCallback):
    """Trainer callback for curriculum training integration."""

    def __init__(self, wrapper: CurriculumTrainingWrapper):
        self.wrapper = wrapper

    def on_step_begin(self, args, state, control, **kwargs):
        """Update wrapper at step beginning."""
        self.wrapper.update_step(state.global_step)

        # Check if recovery mode should end
        self.wrapper.check_recovery_complete()

        # Adjust learning rate based on phase
        lr_multiplier = self.wrapper.get_lr_multiplier()
        for param_group in kwargs['optimizer'].param_groups:
            param_group['lr'] = self.wrapper.base_lr * lr_multiplier

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run validation and health checks."""
        if self.wrapper.should_validate() and model is not None:
            status, metrics = self.wrapper.run_health_check(model)

            if status == "CRITICAL" and self.wrapper.recovery_enabled:
                self.wrapper.execute_recovery_protocol(
                    model, kwargs.get('optimizer'), metrics
                )
            elif status == "WARNING":
                logger.warning(f"Health check warning at step {state.global_step}")
                # Reduce LR by 50%
                for param_group in kwargs.get('optimizer').param_groups:
                    param_group['lr'] *= 0.5


class MetricsLoggingCallback(TrainerCallback):
    """Trainer callback for logging metrics to CSV using TrainingMetricsTracker."""

    def __init__(self, log_dir: str = "logs", stage: str = "curriculum"):
        self.metrics_tracker = TrainingMetricsTracker(log_dir=log_dir, stage=stage)
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize timing."""
        import time
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics when trainer logs."""
        if logs is None:
            return

        # Calculate epoch
        epoch = state.epoch if state.epoch is not None else 0.0

        # Extract relevant metrics from logs
        metrics = {}
        for key in ["loss", "learning_rate", "grad_norm"]:
            if key in logs:
                if key == "loss":
                    metrics["train_loss"] = logs[key]
                elif key == "learning_rate":
                    metrics["learning_rate"] = logs[key]
                elif key == "grad_norm":
                    metrics["gradient_norm"] = logs[key]

        # Calculate tokens per second if we have the info
        if self.start_time is not None and state.global_step > 0:
            import time
            elapsed = time.time() - self.start_time
            if "num_input_tokens_seen" in state.__dict__:
                tokens_per_second = state.num_input_tokens_seen / elapsed
                metrics["tokens_per_second"] = tokens_per_second

        # Log to CSV
        self.metrics_tracker.log_training_step(
            step=state.global_step,
            epoch=epoch,
            metrics=metrics,
            log_to_console=False  # Trainer already logs to console
        )

    def on_train_end(self, args, state, control, **kwargs):
        """Save summary at end of training."""
        logger.info("Saving metrics summary...")
        summary = self.metrics_tracker.save_training_summary()
        logger.info(f"Metrics saved to: {self.metrics_tracker.training_csv}")
