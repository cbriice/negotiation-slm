"""Training state persistence and recovery for crash resilience."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import torch
import logging
from dataclasses import dataclass, asdict
import hashlib
import shutil


@dataclass
class TrainingState:
    """Complete training state for recovery."""

    # Basic info
    stage: str  # "sft" or "dpo"
    epoch: float
    global_step: int
    best_metric: float
    best_metric_name: str

    # Training progress
    total_epochs: float
    total_steps: int
    samples_seen: int
    tokens_processed: int

    # Model info
    model_name: str
    checkpoint_path: Optional[str]
    adapter_config: Dict[str, Any]

    # Optimizer state
    learning_rate: float
    optimizer_state_dict: Optional[Dict[str, Any]]
    scheduler_state_dict: Optional[Dict[str, Any]]

    # Data state
    dataset_path: str
    dataset_index: int
    batch_size: int
    gradient_accumulation_steps: int

    # Metrics
    train_loss: float
    eval_loss: Optional[float]
    metrics_history: Dict[str, list]

    # Recovery info
    timestamp: str
    recovery_count: int
    last_error: Optional[str]

    # Hardware state
    cuda_memory_allocated: Optional[int]
    cuda_memory_reserved: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        state_dict = asdict(self)
        # Remove optimizer state for JSON serialization (too large)
        state_dict.pop('optimizer_state_dict', None)
        state_dict.pop('scheduler_state_dict', None)
        return state_dict


class TrainingStateManager:
    """Manages training state persistence and recovery."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        state_dir: str = "training_states",
        auto_save_interval: int = 100,  # steps
        keep_last_n: int = 5
    ):
        """
        Initialize state manager.

        Args:
            checkpoint_dir: Directory for model checkpoints
            state_dir: Directory for training states
            auto_save_interval: Auto-save every N steps
            keep_last_n: Keep last N state files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save_interval = auto_save_interval
        self.keep_last_n = keep_last_n

        self.logger = logging.getLogger("TrainingStateManager")
        self.current_state: Optional[TrainingState] = None
        self.state_history: list = []

    def save_state(
        self,
        state: TrainingState,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        force: bool = False
    ) -> Path:
        """
        Save training state to disk.

        Args:
            state: Current training state
            optimizer: Optimizer to save
            scheduler: LR scheduler to save
            force: Force save regardless of interval

        Returns:
            Path to saved state file
        """
        # Check if we should save
        if not force and state.global_step % self.auto_save_interval != 0:
            return None

        # Update state with optimizer/scheduler info
        if optimizer:
            state.optimizer_state_dict = optimizer.state_dict()
        if scheduler:
            state.scheduler_state_dict = scheduler.state_dict()

        # Update hardware state
        if torch.cuda.is_available():
            state.cuda_memory_allocated = torch.cuda.memory_allocated()
            state.cuda_memory_reserved = torch.cuda.memory_reserved()

        # Create state filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = self.state_dir / f"state_{state.stage}_{state.global_step}_{timestamp}.pkl"

        # Save state
        try:
            # Save full state with pickle (includes optimizer)
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)

            # Save lightweight JSON version for inspection
            json_file = state_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2, default=str)

            self.logger.info(f"Saved training state to {state_file}")

            # Add to history
            self.state_history.append(state_file)

            # Clean up old states
            self._cleanup_old_states()

            # Update current state
            self.current_state = state

            # Create symlink to latest state
            latest_link = self.state_dir / "latest_state.pkl"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(state_file.name)

            return state_file

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

            # Try emergency save to /tmp
            emergency_file = Path("/tmp") / f"emergency_state_{timestamp}.pkl"
            try:
                with open(emergency_file, 'wb') as f:
                    pickle.dump(state, f)
                self.logger.warning(f"Emergency state saved to {emergency_file}")
                return emergency_file
            except:
                return None

    def load_state(
        self,
        state_file: Optional[Union[str, Path]] = None,
        load_optimizer: bool = True
    ) -> Optional[TrainingState]:
        """
        Load training state from disk.

        Args:
            state_file: Specific state file to load (or latest if None)
            load_optimizer: Whether to load optimizer state

        Returns:
            Loaded training state or None if failed
        """
        # Find state file
        if state_file is None:
            # Load latest state
            latest_link = self.state_dir / "latest_state.pkl"
            if latest_link.exists():
                state_file = latest_link.resolve()
            else:
                # Find most recent state file
                state_files = sorted(self.state_dir.glob("state_*.pkl"))
                if not state_files:
                    self.logger.warning("No state files found")
                    return None
                state_file = state_files[-1]
        else:
            state_file = Path(state_file)

        if not state_file.exists():
            self.logger.error(f"State file not found: {state_file}")
            return None

        # Load state
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)

            if not load_optimizer:
                state.optimizer_state_dict = None
                state.scheduler_state_dict = None

            self.logger.info(f"Loaded training state from {state_file}")
            self.logger.info(f"Resuming from step {state.global_step}, epoch {state.epoch}")

            # Increment recovery count
            state.recovery_count += 1
            state.timestamp = datetime.now().isoformat()

            self.current_state = state
            return state

        except Exception as e:
            self.logger.error(f"Failed to load state from {state_file}: {e}")

            # Try loading JSON version as fallback
            json_file = state_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        state_dict = json.load(f)
                    # Create TrainingState from dict (without optimizer)
                    state = TrainingState(**{
                        k: v for k, v in state_dict.items()
                        if k in TrainingState.__annotations__
                    })
                    self.logger.warning("Loaded state from JSON (no optimizer state)")
                    return state
                except Exception as e2:
                    self.logger.error(f"Failed to load JSON state: {e2}")

            return None

    def get_resume_info(self) -> Dict[str, Any]:
        """
        Get information needed to resume training.

        Returns:
            Resume information dictionary
        """
        if not self.current_state:
            self.load_state()  # Try loading latest

        if not self.current_state:
            return {
                "can_resume": False,
                "reason": "No saved state found"
            }

        state = self.current_state
        resume_info = {
            "can_resume": True,
            "stage": state.stage,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "samples_seen": state.samples_seen,
            "checkpoint_path": state.checkpoint_path,
            "dataset_index": state.dataset_index,
            "recovery_count": state.recovery_count,
            "last_loss": state.train_loss,
            "timestamp": state.timestamp
        }

        # Check if checkpoint exists
        if state.checkpoint_path:
            checkpoint_path = Path(state.checkpoint_path)
            if not checkpoint_path.exists():
                resume_info["can_resume"] = False
                resume_info["reason"] = f"Checkpoint not found: {checkpoint_path}"

        return resume_info

    def create_recovery_checkpoint(
        self,
        model: torch.nn.Module,
        error: Exception,
        context: Dict[str, Any]
    ) -> Path:
        """
        Create emergency checkpoint after error.

        Args:
            model: Current model
            error: Exception that occurred
            context: Training context

        Returns:
            Path to recovery checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recovery_dir = self.state_dir / "recovery" / timestamp
        recovery_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_file = recovery_dir / "model_emergency.pt"
        torch.save(model.state_dict(), model_file)

        # Save error info
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": timestamp,
            "context": context
        }

        with open(recovery_dir / "error_info.json", 'w') as f:
            json.dump(error_info, f, indent=2, default=str)

        # Save current state if available
        if self.current_state:
            self.current_state.last_error = str(error)
            self.save_state(self.current_state, force=True)

        self.logger.info(f"Created recovery checkpoint at {recovery_dir}")
        return recovery_dir

    def _cleanup_old_states(self):
        """Remove old state files keeping only last N."""
        state_files = sorted(self.state_dir.glob("state_*.pkl"))

        if len(state_files) > self.keep_last_n:
            to_remove = state_files[:-self.keep_last_n]
            for file in to_remove:
                try:
                    file.unlink()
                    # Also remove JSON version
                    json_file = file.with_suffix('.json')
                    if json_file.exists():
                        json_file.unlink()
                    self.logger.debug(f"Removed old state file: {file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file}: {e}")

    def validate_state(self, state: TrainingState) -> bool:
        """
        Validate training state for consistency.

        Args:
            state: State to validate

        Returns:
            Whether state is valid
        """
        issues = []

        # Check basic fields
        if state.global_step < 0:
            issues.append("Invalid global_step")
        if state.epoch < 0:
            issues.append("Invalid epoch")
        if state.batch_size <= 0:
            issues.append("Invalid batch_size")

        # Check progress consistency
        if state.global_step > state.total_steps:
            issues.append("global_step exceeds total_steps")
        if state.epoch > state.total_epochs:
            issues.append("epoch exceeds total_epochs")

        # Check loss values
        if torch.isnan(torch.tensor(state.train_loss)):
            issues.append("NaN train_loss")
        if state.train_loss < 0:
            issues.append("Negative train_loss")

        if issues:
            self.logger.warning(f"State validation issues: {issues}")
            return False

        return True

    def get_state_checksum(self, state: TrainingState) -> str:
        """
        Get checksum of state for integrity checking.

        Args:
            state: State to checksum

        Returns:
            Hex checksum string
        """
        # Create checksum from key fields
        key_data = f"{state.stage}_{state.global_step}_{state.epoch}_{state.samples_seen}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def export_training_history(self, output_file: str):
        """
        Export complete training history to file.

        Args:
            output_file: Path to output file
        """
        history = []

        for state_file in sorted(self.state_dir.glob("state_*.json")):
            try:
                with open(state_file, 'r') as f:
                    state_dict = json.load(f)
                    history.append({
                        "file": str(state_file),
                        "step": state_dict.get("global_step"),
                        "epoch": state_dict.get("epoch"),
                        "loss": state_dict.get("train_loss"),
                        "timestamp": state_dict.get("timestamp")
                    })
            except Exception as e:
                self.logger.warning(f"Failed to read {state_file}: {e}")

        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Exported training history to {output_file}")


class StateAutoSaver:
    """Context manager for automatic state saving."""

    def __init__(
        self,
        state_manager: TrainingStateManager,
        initial_state: TrainingState,
        save_on_exception: bool = True
    ):
        """
        Initialize auto-saver.

        Args:
            state_manager: State manager instance
            initial_state: Initial training state
            save_on_exception: Save state on exception
        """
        self.state_manager = state_manager
        self.state = initial_state
        self.save_on_exception = save_on_exception

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, save state if needed."""
        if exc_type is not None and self.save_on_exception:
            # Save state on exception
            self.state.last_error = str(exc_val)
            self.state_manager.save_state(self.state, force=True)
            self.state_manager.logger.warning(f"Saved state due to exception: {exc_val}")

    def update(self, **kwargs):
        """Update state fields."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)