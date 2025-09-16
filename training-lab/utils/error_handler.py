"""Comprehensive error handling and recovery for QLoRA training pipeline."""

import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from pathlib import Path
import torch
import gc
import os
import signal
import sys
from datetime import datetime
import json

T = TypeVar('T')

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass

class OOMError(TrainingError):
    """Out of memory error during training."""
    pass

class GradientError(TrainingError):
    """Gradient-related error (explosion, NaN, etc.)."""
    pass

class CheckpointError(TrainingError):
    """Error related to checkpoint saving/loading."""
    pass

class DataError(TrainingError):
    """Error related to data loading or processing."""
    pass


class ErrorHandler:
    """Centralized error handling for training pipeline."""

    def __init__(
        self,
        log_dir: str = "logs/errors",
        enable_auto_recovery: bool = True,
        max_recovery_attempts: int = 3
    ):
        """
        Initialize error handler.

        Args:
            log_dir: Directory for error logs
            enable_auto_recovery: Whether to attempt automatic recovery
            max_recovery_attempts: Maximum recovery attempts per error type
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_auto_recovery = enable_auto_recovery
        self.max_recovery_attempts = max_recovery_attempts

        # Track recovery attempts
        self.recovery_attempts: Dict[str, int] = {}

        # Track error counts by type
        self.error_counts: Dict[str, int] = {}

        # Setup logging
        self.logger = self._setup_logger()

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

        # Recovery callbacks
        self.recovery_callbacks: Dict[type, Callable] = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup error logger with file and console handlers."""
        logger = logging.getLogger("ErrorHandler")
        logger.setLevel(logging.DEBUG)

        # File handler for all errors
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            self.log_dir / f"errors_{timestamp}.log"
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler for warnings and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}")
            self.handle_interrupt()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def register_recovery_callback(
        self,
        error_type: type,
        callback: Callable[[Exception], Any]
    ):
        """
        Register a recovery callback for a specific error type.

        Args:
            error_type: Type of error to handle
            callback: Recovery function to call
        """
        self.recovery_callbacks[error_type] = callback
        self.logger.info(f"Registered recovery callback for {error_type.__name__}")

    def handle_oom_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle out-of-memory errors with automatic recovery.

        Args:
            error: The OOM error
            context: Training context (batch_size, model, etc.)

        Returns:
            Recovery actions to take
        """
        self.logger.error(f"OOM Error detected: {error}")

        recovery_actions = {
            "clear_cache": True,
            "reduce_batch_size": False,
            "enable_gradient_checkpointing": False,
            "reduce_sequence_length": False,
            "offload_to_cpu": False
        }

        # Clear CUDA cache immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Cleared CUDA cache")

        # Check current batch size
        current_batch_size = context.get("batch_size", 1)
        if current_batch_size > 1:
            recovery_actions["reduce_batch_size"] = True
            recovery_actions["new_batch_size"] = max(1, current_batch_size // 2)
            self.logger.info(f"Recommending batch size reduction: {current_batch_size} -> {recovery_actions['new_batch_size']}")

        # Check gradient checkpointing
        if not context.get("gradient_checkpointing", False):
            recovery_actions["enable_gradient_checkpointing"] = True
            self.logger.info("Recommending gradient checkpointing")

        # Check sequence length
        current_seq_len = context.get("max_seq_length", 4096)
        if current_seq_len > 2048:
            recovery_actions["reduce_sequence_length"] = True
            recovery_actions["new_seq_length"] = max(2048, current_seq_len // 2)
            self.logger.info(f"Recommending sequence length reduction: {current_seq_len} -> {recovery_actions['new_seq_length']}")

        # Log memory stats
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.logger.info(f"GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        return recovery_actions

    def handle_gradient_error(
        self,
        error: Exception,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Handle gradient-related errors (explosion, NaN).

        Args:
            error: The gradient error
            gradients: Current gradients if available

        Returns:
            Recovery actions to take
        """
        self.logger.error(f"Gradient Error detected: {error}")

        recovery_actions = {
            "clip_gradients": False,
            "reduce_learning_rate": False,
            "skip_batch": False,
            "reset_optimizer": False
        }

        # Analyze gradients if provided
        if gradients:
            max_grad_norm = 0
            has_nan = False
            has_inf = False

            for name, grad in gradients.items():
                if grad is not None:
                    if torch.isnan(grad).any():
                        has_nan = True
                        self.logger.warning(f"NaN gradient in {name}")
                    if torch.isinf(grad).any():
                        has_inf = True
                        self.logger.warning(f"Inf gradient in {name}")

                    grad_norm = grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)

            self.logger.info(f"Max gradient norm: {max_grad_norm}")

            if has_nan or has_inf:
                recovery_actions["skip_batch"] = True
                recovery_actions["reset_optimizer"] = True
                self.logger.info("Recommending batch skip and optimizer reset due to NaN/Inf")
            elif max_grad_norm > 100:
                recovery_actions["clip_gradients"] = True
                recovery_actions["max_grad_norm"] = 1.0
                self.logger.info(f"Recommending gradient clipping to {recovery_actions['max_grad_norm']}")
            elif max_grad_norm > 10:
                recovery_actions["reduce_learning_rate"] = True
                recovery_actions["lr_scale"] = 0.5
                self.logger.info("Recommending learning rate reduction")

        return recovery_actions

    def handle_loss_error(self, loss_value: float, step: int) -> Dict[str, Any]:
        """
        Handle abnormal loss values (NaN, explosion).

        Args:
            loss_value: Current loss value
            step: Current training step

        Returns:
            Recovery actions to take
        """
        recovery_actions = {
            "skip_update": False,
            "restore_checkpoint": False,
            "reduce_learning_rate": False,
            "reset_optimizer": False
        }

        if torch.isnan(torch.tensor(loss_value)):
            self.logger.error(f"NaN loss detected at step {step}")
            recovery_actions["skip_update"] = True
            recovery_actions["reset_optimizer"] = True
        elif torch.isinf(torch.tensor(loss_value)):
            self.logger.error(f"Inf loss detected at step {step}")
            recovery_actions["skip_update"] = True
            recovery_actions["reset_optimizer"] = True
        elif loss_value > 1000:
            self.logger.warning(f"Abnormally high loss {loss_value} at step {step}")
            recovery_actions["reduce_learning_rate"] = True
            recovery_actions["lr_scale"] = 0.1

        return recovery_actions

    def handle_checkpoint_error(self, error: Exception, checkpoint_path: str) -> bool:
        """
        Handle checkpoint saving/loading errors.

        Args:
            error: The checkpoint error
            checkpoint_path: Path to checkpoint

        Returns:
            Whether recovery was successful
        """
        self.logger.error(f"Checkpoint Error at {checkpoint_path}: {error}")

        # Try backup locations
        backup_paths = [
            Path(checkpoint_path).parent / "backup",
            Path("/tmp") / "emergency_checkpoint",
            Path.home() / ".cache" / "training_checkpoints"
        ]

        for backup_path in backup_paths:
            try:
                backup_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Attempting backup to {backup_path}")
                return True
            except Exception as e:
                self.logger.warning(f"Backup to {backup_path} failed: {e}")

        return False

    def handle_interrupt(self):
        """Handle keyboard interrupt or termination signal."""
        self.logger.warning("Training interrupted - attempting graceful shutdown")

        # Save emergency checkpoint
        emergency_path = Path("/tmp/emergency_checkpoint")
        emergency_path.mkdir(parents=True, exist_ok=True)

        # Log interrupt metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "reason": "interrupt",
            "recovery_possible": True
        }

        with open(emergency_path / "interrupt_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Emergency checkpoint metadata saved to {emergency_path}")

    def safe_execute(
        self,
        func: Callable[..., T],
        *args,
        error_type: Optional[type] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[T, None]:
        """
        Safely execute a function with error handling and recovery.

        Args:
            func: Function to execute
            error_type: Expected error type for specialized handling
            context: Additional context for error recovery
            *args, **kwargs: Arguments for func

        Returns:
            Function result or None if failed
        """
        error_key = f"{func.__name__}_{error_type.__name__ if error_type else 'general'}"

        # Check recovery attempts
        if error_key not in self.recovery_attempts:
            self.recovery_attempts[error_key] = 0

        if self.recovery_attempts[error_key] >= self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts reached for {error_key}")
            return None

        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {e}")
            self.logger.debug(traceback.format_exc())

            # Track error counts by exception type
            error_type_name = type(e).__name__
            if error_type_name not in self.error_counts:
                self.error_counts[error_type_name] = 0
            self.error_counts[error_type_name] += 1

            # Increment recovery attempts
            self.recovery_attempts[error_key] += 1

            # Try recovery if enabled
            if self.enable_auto_recovery:
                # Check for registered recovery callback
                for error_class, callback in self.recovery_callbacks.items():
                    if isinstance(e, error_class):
                        self.logger.info(f"Attempting recovery with {callback.__name__}")
                        try:
                            return callback(e)
                        except Exception as recovery_error:
                            self.logger.error(f"Recovery failed: {recovery_error}")

                # Default recovery based on error type
                if "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                    recovery_actions = self.handle_oom_error(e, context or {})
                    if recovery_actions.get("clear_cache"):
                        # Retry after clearing cache
                        return self.safe_execute(func, *args, error_type=error_type, context=context, **kwargs)

            return None

    def log_error_summary(self):
        """Log summary of all errors encountered."""
        self.logger.info("=" * 50)
        self.logger.info("Error Summary")
        self.logger.info("=" * 50)

        for error_key, attempts in self.recovery_attempts.items():
            self.logger.info(f"{error_key}: {attempts} recovery attempts")

        self.logger.info("=" * 50)


def with_error_handling(
    error_handler: ErrorHandler,
    error_type: Optional[type] = None,
    context_func: Optional[Callable[..., Dict[str, Any]]] = None
):
    """
    Decorator for adding error handling to functions.

    Args:
        error_handler: ErrorHandler instance
        error_type: Expected error type
        context_func: Function to get context for error recovery
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            context = context_func(*args, **kwargs) if context_func else {}
            return error_handler.safe_execute(
                func,
                *args,
                error_type=error_type,
                context=context,
                **kwargs
            )
        return wrapper
    return decorator