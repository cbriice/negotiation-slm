"""TensorBoard logging with error handling integration."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class TensorBoardWriter:
    """Centralized TensorBoard writer with error handling integration."""

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        comment: str = "",
        flush_secs: int = 120
    ):
        """
        Initialize TensorBoard writer.

        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment
            comment: Additional comment for run naming
            flush_secs: How often to flush to disk
        """
        self.logger = logging.getLogger("TensorBoardWriter")

        # Create unique run directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if comment:
            run_name = f"{experiment_name}_{comment}"
        else:
            run_name = experiment_name

        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize writer
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs
        )

        # Track what we've logged
        self.logged_metrics = set()
        self.step_count = 0

        # Custom layout for better organization
        self._setup_custom_layout()

        self.logger.info(f"TensorBoard logging to: {self.log_dir}")

    def _setup_custom_layout(self):
        """Setup custom scalar layouts for better visualization."""
        layout = {
            "Training": {
                "Loss": ["Multiline", ["loss/train", "loss/eval", "loss/ema"]],
                "Learning Rate": ["Multiline", ["training/learning_rate"]],
                "Throughput": ["Multiline", ["performance/tokens_per_second",
                                             "performance/samples_per_second"]]
            },
            "Error Handling": {
                "Batch Size": ["Multiline", ["batch_optimizer/batch_size",
                                            "batch_optimizer/effective_batch_size"]],
                "Gradients": ["Multiline", ["gradients/norm", "gradients/anomaly_score"]],
                "Loss Health": ["Multiline", ["loss_validator/divergence_count",
                                             "loss_validator/plateau_count"]],
                "Memory": ["Multiline", ["memory/allocated_gb", "memory/reserved_gb"]]
            },
            "DPO Metrics": {
                "Rewards": ["Multiline", ["dpo/reward_margin", "dpo/chosen_reward",
                                         "dpo/rejected_reward"]],
                "Policy": ["Multiline", ["dpo/kl_divergence"]]
            }
        }

        self.writer.add_custom_scalars(layout)

    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "training"
    ):
        """
        Log standard training metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Global training step
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            if value is not None and not (np.isnan(value) or np.isinf(value)):
                self.writer.add_scalar(f"{prefix}/{name}", value, step)

    def log_loss_metrics(
        self,
        loss: float,
        step: int,
        loss_type: str = "train",
        additional_losses: Optional[Dict[str, float]] = None
    ):
        """
        Log loss metrics.

        Args:
            loss: Main loss value
            step: Training step
            loss_type: Type of loss (train/eval/test)
            additional_losses: Additional loss components
        """
        self.writer.add_scalar(f"loss/{loss_type}", loss, step)

        if additional_losses:
            for name, value in additional_losses.items():
                if value is not None:
                    self.writer.add_scalar(f"loss/{name}", value, step)

    def log_dpo_metrics(
        self,
        chosen_rewards: float,
        rejected_rewards: float,
        reward_margin: float,
        step: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log DPO-specific metrics.

        Args:
            chosen_rewards: Mean reward for chosen responses
            rejected_rewards: Mean reward for rejected responses
            reward_margin: Margin between chosen and rejected
            step: Training step
            additional_metrics: Additional DPO metrics
        """
        self.writer.add_scalar("dpo/chosen_reward", chosen_rewards, step)
        self.writer.add_scalar("dpo/rejected_reward", rejected_rewards, step)
        self.writer.add_scalar("dpo/reward_margin", reward_margin, step)

        if additional_metrics:
            for name, value in additional_metrics.items():
                if value is not None:
                    self.writer.add_scalar(f"dpo/{name}", value, step)

    def log_batch_optimizer_metrics(
        self,
        batch_optimizer,
        step: int
    ):
        """
        Log metrics from BatchSizeOptimizer.

        Args:
            batch_optimizer: BatchSizeOptimizer instance
            step: Training step
        """
        if batch_optimizer is None:
            return

        summary = batch_optimizer.get_summary()

        self.writer.add_scalar(
            "batch_optimizer/batch_size",
            summary["current_batch_size"],
            step
        )
        self.writer.add_scalar(
            "batch_optimizer/grad_accum_steps",
            summary["gradient_accumulation_steps"],
            step
        )
        self.writer.add_scalar(
            "batch_optimizer/effective_batch_size",
            summary["effective_batch_size"],
            step
        )

        # Log best throughput if available
        if summary.get("best_throughput"):
            self.writer.add_scalar(
                "batch_optimizer/best_throughput",
                summary["best_throughput"],
                step
            )

        # Log OOM events
        if summary.get("oom_batch_sizes"):
            for oom_size in summary["oom_batch_sizes"]:
                self.writer.add_scalar(
                    "batch_optimizer/oom_events",
                    oom_size,
                    step
                )

    def log_gradient_monitor_metrics(
        self,
        gradient_monitor,
        step: int
    ):
        """
        Log metrics from GradientMonitor.

        Args:
            gradient_monitor: GradientMonitor instance
            step: Training step
        """
        if gradient_monitor is None:
            return

        # Check gradients and get stats
        is_healthy, stats = gradient_monitor.check_gradients(return_stats=True)

        if stats and stats.get("global"):
            global_stats = stats["global"]

            # Log gradient norm
            self.writer.add_scalar(
                "gradients/norm",
                global_stats["norm"],
                step
            )

            # Log gradient statistics
            self.writer.add_scalar(
                "gradients/mean",
                global_stats["mean"],
                step
            )
            self.writer.add_scalar(
                "gradients/std",
                global_stats["std"],
                step
            )

            # Log gradient health
            self.writer.add_scalar(
                "gradients/is_healthy",
                float(is_healthy),
                step
            )

            # Log sparsity (dead neurons indicator)
            self.writer.add_scalar(
                "gradients/sparsity",
                global_stats["sparsity"],
                step
            )

            # Log issues as events
            if not is_healthy and stats.get("issues"):
                for issue in stats["issues"]:
                    self.writer.add_text(
                        "gradient_issues",
                        f"Step {step}: {issue}",
                        step
                    )

        # Log gradient flow for important layers
        gradient_flow = gradient_monitor.get_gradient_flow()
        for layer_name, history in gradient_flow.items():
            if "lora" in layer_name.lower() and len(history) > 0:
                # Log latest gradient norm for LoRA layers
                self.writer.add_scalar(
                    f"gradients/layers/{layer_name}",
                    history[-1],
                    step
                )

    def log_loss_validator_metrics(
        self,
        loss_validator,
        step: int
    ):
        """
        Log metrics from LossValidator.

        Args:
            loss_validator: LossValidator instance
            step: Training step
        """
        if loss_validator is None:
            return

        summary = loss_validator.get_summary()

        if summary.get("status") != "no_data":
            # Log loss statistics
            if summary.get("ema_loss") is not None:
                self.writer.add_scalar(
                    "loss/ema",
                    summary["ema_loss"],
                    step
                )

            if summary.get("std_loss") is not None:
                self.writer.add_scalar(
                    "loss_validator/std",
                    summary["std_loss"],
                    step
                )

            # Log loss trend
            if summary.get("trend") is not None:
                self.writer.add_scalar(
                    "loss_validator/trend",
                    summary["trend"],
                    step
                )

            # Log stability indicators
            self.writer.add_scalar(
                "loss_validator/is_stable",
                float(summary.get("is_stable", False)),
                step
            )
            self.writer.add_scalar(
                "loss_validator/is_plateaued",
                float(summary.get("is_plateaued", False)),
                step
            )

            # Log issue counts
            issues = summary.get("issues", {})
            for issue_type, count in issues.items():
                if count > 0:
                    self.writer.add_scalar(
                        f"loss_validator/{issue_type}",
                        count,
                        step
                    )

    def log_precision_handler_metrics(
        self,
        precision_handler,
        step: int
    ):
        """
        Log metrics from PrecisionHandler.

        Args:
            precision_handler: PrecisionHandler instance
            step: Training step
        """
        if precision_handler is None:
            return

        summary = precision_handler.get_summary()

        # Log current precision
        dtype_map = {
            "torch.float32": 32,
            "torch.float16": 16,
            "torch.bfloat16": 16
        }
        precision_bits = dtype_map.get(summary["current_dtype"], 32)
        self.writer.add_scalar(
            "precision/bits",
            precision_bits,
            step
        )

        # Log overflow events
        self.writer.add_scalar(
            "precision/overflow_count",
            summary["overflow_count"],
            step
        )

        # Log precision switches
        if summary["precision_switches"] > 0:
            self.writer.add_scalar(
                "precision/switches",
                summary["precision_switches"],
                step
            )

        # Log memory usage
        memory_stats = summary.get("memory_gb", {})
        for mem_type, value in memory_stats.items():
            self.writer.add_scalar(
                f"memory/{mem_type}_gb",
                value,
                step
            )

        # Log gradient scale if using FP16
        if summary.get("scale") is not None:
            self.writer.add_scalar(
                "precision/gradient_scale",
                summary["scale"],
                step
            )

    def log_performance_metrics(
        self,
        tokens_per_second: Optional[float] = None,
        samples_per_second: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        step: Optional[int] = None
    ):
        """
        Log performance metrics.

        Args:
            tokens_per_second: Training throughput in tokens/sec
            samples_per_second: Training throughput in samples/sec
            gpu_utilization: GPU utilization percentage
            step: Training step
        """
        step = step or self.step_count

        if tokens_per_second is not None:
            self.writer.add_scalar(
                "performance/tokens_per_second",
                tokens_per_second,
                step
            )

        if samples_per_second is not None:
            self.writer.add_scalar(
                "performance/samples_per_second",
                samples_per_second,
                step
            )

        if gpu_utilization is not None:
            self.writer.add_scalar(
                "performance/gpu_utilization",
                gpu_utilization,
                step
            )

    def log_histograms(
        self,
        model: torch.nn.Module,
        step: int,
        log_gradients: bool = True,
        log_weights: bool = True
    ):
        """
        Log weight and gradient histograms.

        Args:
            model: Model to log histograms for
            step: Training step
            log_gradients: Whether to log gradient histograms
            log_weights: Whether to log weight histograms
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weights
                if log_weights and param.data is not None:
                    self.writer.add_histogram(
                        f"weights/{name}",
                        param.data.cpu().numpy(),
                        step
                    )

                # Log gradients
                if log_gradients and param.grad is not None:
                    self.writer.add_histogram(
                        f"gradients/{name}",
                        param.grad.data.cpu().numpy(),
                        step
                    )

    def log_learning_rate(
        self,
        learning_rate: float,
        step: int
    ):
        """
        Log learning rate.

        Args:
            learning_rate: Current learning rate
            step: Training step
        """
        self.writer.add_scalar("training/learning_rate", learning_rate, step)

    def log_error_event(
        self,
        error_type: str,
        error_message: str,
        step: int,
        severity: str = "warning"
    ):
        """
        Log an error or warning event.

        Args:
            error_type: Type of error (e.g., "OOM", "divergence")
            error_message: Detailed error message
            step: Training step
            severity: Severity level (warning/error/critical)
        """
        # Log as text event
        self.writer.add_text(
            f"errors/{error_type}",
            f"[{severity.upper()}] Step {step}: {error_message}",
            step
        )

        # Also log as scalar for tracking frequency
        self.writer.add_scalar(
            f"errors/{error_type}_count",
            1,
            step
        )

    def log_checkpoint_saved(
        self,
        checkpoint_path: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log checkpoint save event.

        Args:
            checkpoint_path: Path to saved checkpoint
            step: Training step
            metrics: Metrics at checkpoint time
        """
        message = f"Checkpoint saved: {checkpoint_path}"
        if metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            message += f"\nMetrics: {metrics_str}"

        self.writer.add_text(
            "checkpoints",
            message,
            step
        )

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters and optionally final metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional final metrics
        """
        # Convert non-scalar hparams to strings
        scalar_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                scalar_hparams[key] = value
            else:
                scalar_hparams[key] = str(value)

        if metrics:
            self.writer.add_hparams(scalar_hparams, metrics)
        else:
            # Just log hyperparameters without metrics
            for key, value in scalar_hparams.items():
                self.writer.add_text(
                    "hyperparameters",
                    f"{key}: {value}",
                    0
                )

    def log_all_error_handlers(
        self,
        batch_optimizer=None,
        gradient_monitor=None,
        loss_validator=None,
        precision_handler=None,
        step: Optional[int] = None
    ):
        """
        Convenience method to log all error handler metrics at once.

        Args:
            batch_optimizer: BatchSizeOptimizer instance
            gradient_monitor: GradientMonitor instance
            loss_validator: LossValidator instance
            precision_handler: PrecisionHandler instance
            step: Training step
        """
        step = step or self.step_count

        self.log_batch_optimizer_metrics(batch_optimizer, step)
        self.log_gradient_monitor_metrics(gradient_monitor, step)
        self.log_loss_validator_metrics(loss_validator, step)
        self.log_precision_handler_metrics(precision_handler, step)

        self.step_count = step

    def flush(self):
        """Flush all pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        self.logger.info("TensorBoard writer closed")