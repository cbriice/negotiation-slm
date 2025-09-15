"""Loss validation and anomaly detection utilities."""

import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings
import math


@dataclass
class LossValidationConfig:
    """Configuration for loss validation."""

    # Thresholds
    max_loss: float = 100.0
    min_loss: float = 0.0
    nan_action: str = "skip"  # "skip", "stop", "reset"
    inf_action: str = "skip"

    # Stability checks
    stability_window: int = 50
    max_variance: float = 10.0
    oscillation_threshold: float = 0.5
    plateau_threshold: float = 1e-4
    plateau_patience: int = 100

    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_zscore: float = 3.0
    anomaly_patience: int = 3

    # Smoothing
    enable_smoothing: bool = True
    smoothing_factor: float = 0.9  # Exponential moving average

    # Divergence detection
    divergence_threshold: float = 1000.0
    divergence_patience: int = 5


@dataclass
class LossStatistics:
    """Statistics for loss monitoring."""

    current: float
    mean: float
    std: float
    min: float
    max: float
    ema: float  # Exponential moving average
    trend: float  # Linear trend coefficient
    is_stable: bool
    is_plateaued: bool
    anomaly_score: float


class LossValidator:
    """Validates loss values and detects training issues."""

    def __init__(self, config: LossValidationConfig):
        """
        Initialize loss validator.

        Args:
            config: Loss validation configuration
        """
        self.config = config
        self.logger = logging.getLogger("LossValidator")

        # History tracking
        self.loss_history = deque(maxlen=config.stability_window * 2)
        self.ema_loss = None
        self.step_count = 0

        # Issue tracking
        self.nan_count = 0
        self.inf_count = 0
        self.anomaly_count = 0
        self.divergence_count = 0
        self.plateau_count = 0

        # Statistics
        self.current_stats: Optional[LossStatistics] = None
        self.best_loss = float('inf')
        self.worst_loss = float('-inf')

    def validate_loss(
        self,
        loss: torch.Tensor,
        step: Optional[int] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate a loss value.

        Args:
            loss: Loss tensor
            step: Current training step

        Returns:
            Tuple of (is_valid, action, info_dict)
        """
        self.step_count = step if step is not None else self.step_count + 1

        # Convert to scalar
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        # Check for NaN
        if math.isnan(loss_value):
            return self._handle_nan_loss()

        # Check for Inf
        if math.isinf(loss_value):
            return self._handle_inf_loss(loss_value)

        # Check bounds
        if loss_value > self.config.max_loss:
            return self._handle_excessive_loss(loss_value)

        if loss_value < self.config.min_loss:
            return self._handle_negative_loss(loss_value)

        # Update history
        self.loss_history.append(loss_value)
        self._update_ema(loss_value)

        # Calculate statistics
        self.current_stats = self._calculate_statistics(loss_value)

        # Check for anomalies
        is_anomaly, anomaly_info = self._check_anomaly(loss_value)
        if is_anomaly:
            return self._handle_anomaly(loss_value, anomaly_info)

        # Check stability
        is_stable, stability_info = self._check_stability()
        if not is_stable:
            return self._handle_instability(stability_info)

        # Check for divergence
        is_diverging, divergence_info = self._check_divergence()
        if is_diverging:
            return self._handle_divergence(divergence_info)

        # Check for plateau
        is_plateaued, plateau_info = self._check_plateau()
        if is_plateaued:
            self._handle_plateau(plateau_info)

        # Update best/worst
        self.best_loss = min(self.best_loss, loss_value)
        self.worst_loss = max(self.worst_loss, loss_value)

        # Reset issue counters on valid loss
        self.nan_count = 0
        self.inf_count = 0
        self.anomaly_count = 0
        self.divergence_count = 0

        return True, "continue", {"loss": loss_value, "stats": self.current_stats}

    def _handle_nan_loss(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle NaN loss."""
        self.nan_count += 1
        self.logger.error(f"NaN loss detected at step {self.step_count} (count: {self.nan_count})")

        action = self.config.nan_action
        info = {
            "issue": "nan_loss",
            "count": self.nan_count,
            "step": self.step_count
        }

        if action == "stop":
            return False, "stop", info
        elif action == "reset":
            return False, "reset_optimizer", info
        else:  # skip
            return False, "skip_update", info

    def _handle_inf_loss(self, loss_value: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle infinite loss."""
        self.inf_count += 1
        sign = "positive" if loss_value > 0 else "negative"
        self.logger.error(
            f"{sign} Inf loss detected at step {self.step_count} (count: {self.inf_count})"
        )

        action = self.config.inf_action
        info = {
            "issue": "inf_loss",
            "count": self.inf_count,
            "step": self.step_count,
            "sign": sign
        }

        if action == "stop":
            return False, "stop", info
        elif action == "reset":
            return False, "reset_optimizer", info
        else:  # skip
            return False, "skip_update", info

    def _handle_excessive_loss(self, loss_value: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle excessively high loss."""
        self.logger.warning(
            f"Excessive loss {loss_value:.2f} at step {self.step_count} "
            f"(max: {self.config.max_loss})"
        )

        info = {
            "issue": "excessive_loss",
            "loss": loss_value,
            "max_allowed": self.config.max_loss,
            "step": self.step_count
        }

        return False, "reduce_learning_rate", info

    def _handle_negative_loss(self, loss_value: float) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle negative loss."""
        self.logger.warning(
            f"Negative loss {loss_value:.4f} at step {self.step_count}"
        )

        info = {
            "issue": "negative_loss",
            "loss": loss_value,
            "step": self.step_count
        }

        # Negative loss usually indicates a bug
        return False, "stop", info

    def _update_ema(self, loss_value: float):
        """Update exponential moving average."""
        if self.config.enable_smoothing:
            if self.ema_loss is None:
                self.ema_loss = loss_value
            else:
                alpha = self.config.smoothing_factor
                self.ema_loss = alpha * self.ema_loss + (1 - alpha) * loss_value

    def _calculate_statistics(self, current_loss: float) -> LossStatistics:
        """Calculate loss statistics."""
        if len(self.loss_history) < 2:
            return LossStatistics(
                current=current_loss,
                mean=current_loss,
                std=0.0,
                min=current_loss,
                max=current_loss,
                ema=self.ema_loss or current_loss,
                trend=0.0,
                is_stable=True,
                is_plateaued=False,
                anomaly_score=0.0
            )

        history_array = np.array(list(self.loss_history))

        # Basic statistics
        mean = np.mean(history_array)
        std = np.std(history_array)
        min_loss = np.min(history_array)
        max_loss = np.max(history_array)

        # Calculate trend
        if len(history_array) >= 5:
            x = np.arange(len(history_array))
            trend = np.polyfit(x, history_array, 1)[0]
        else:
            trend = 0.0

        # Check stability (coefficient of variation)
        cv = std / (mean + 1e-8)
        is_stable = cv < 0.1  # Less than 10% variation

        # Check plateau
        recent_window = min(self.config.plateau_patience, len(history_array))
        recent_losses = history_array[-recent_window:]
        is_plateaued = np.std(recent_losses) < self.config.plateau_threshold

        # Anomaly score (z-score)
        anomaly_score = abs((current_loss - mean) / (std + 1e-8))

        return LossStatistics(
            current=current_loss,
            mean=mean,
            std=std,
            min=min_loss,
            max=max_loss,
            ema=self.ema_loss or current_loss,
            trend=trend,
            is_stable=is_stable,
            is_plateaued=is_plateaued,
            anomaly_score=anomaly_score
        )

    def _check_anomaly(self, loss_value: float) -> Tuple[bool, Dict[str, Any]]:
        """Check if loss is anomalous."""
        if not self.config.enable_anomaly_detection:
            return False, {}

        if len(self.loss_history) < 10:
            return False, {}

        stats = self.current_stats
        if stats and stats.anomaly_score > self.config.anomaly_zscore:
            info = {
                "z_score": stats.anomaly_score,
                "loss": loss_value,
                "expected_range": (
                    stats.mean - self.config.anomaly_zscore * stats.std,
                    stats.mean + self.config.anomaly_zscore * stats.std
                )
            }
            return True, info

        return False, {}

    def _handle_anomaly(
        self,
        loss_value: float,
        anomaly_info: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle anomalous loss."""
        self.anomaly_count += 1

        self.logger.warning(
            f"Anomalous loss detected at step {self.step_count}: "
            f"loss={loss_value:.4f}, z-score={anomaly_info['z_score']:.2f}"
        )

        if self.anomaly_count >= self.config.anomaly_patience:
            return False, "reduce_learning_rate", {
                "issue": "repeated_anomalies",
                "count": self.anomaly_count,
                **anomaly_info
            }

        # Single anomaly - skip update but continue
        return False, "skip_update", {
            "issue": "anomaly",
            **anomaly_info
        }

    def _check_stability(self) -> Tuple[bool, Dict[str, Any]]:
        """Check loss stability."""
        if len(self.loss_history) < self.config.stability_window:
            return True, {}

        history_array = np.array(list(self.loss_history)[-self.config.stability_window:])

        # Check variance
        variance = np.var(history_array)
        if variance > self.config.max_variance:
            return False, {
                "issue": "high_variance",
                "variance": variance,
                "threshold": self.config.max_variance
            }

        # Check oscillation
        differences = np.diff(history_array)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        oscillation_rate = sign_changes / len(differences)

        if oscillation_rate > self.config.oscillation_threshold:
            return False, {
                "issue": "oscillation",
                "rate": oscillation_rate,
                "threshold": self.config.oscillation_threshold
            }

        return True, {}

    def _handle_instability(self, stability_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle loss instability."""
        issue = stability_info.get("issue", "unknown")

        if issue == "high_variance":
            self.logger.warning(
                f"High loss variance detected: {stability_info['variance']:.4f}"
            )
            return True, "reduce_learning_rate", stability_info

        elif issue == "oscillation":
            self.logger.warning(
                f"Loss oscillation detected: rate={stability_info['rate']:.2f}"
            )
            return True, "reduce_learning_rate", stability_info

        return True, "continue", stability_info

    def _check_divergence(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if loss is diverging."""
        if len(self.loss_history) < 10:
            return False, {}

        stats = self.current_stats
        if not stats:
            return False, {}

        # Check if loss is consistently increasing
        if stats.trend > 0.1 and stats.current > self.config.divergence_threshold:
            self.divergence_count += 1
            if self.divergence_count >= self.config.divergence_patience:
                return True, {
                    "trend": stats.trend,
                    "current_loss": stats.current,
                    "count": self.divergence_count
                }

        return False, {}

    def _handle_divergence(self, divergence_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Handle loss divergence."""
        self.logger.error(
            f"Loss divergence detected at step {self.step_count}: "
            f"trend={divergence_info['trend']:.4f}, loss={divergence_info['current_loss']:.2f}"
        )

        return False, "restore_checkpoint", {
            "issue": "divergence",
            **divergence_info
        }

    def _check_plateau(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if loss has plateaued."""
        if len(self.loss_history) < self.config.plateau_patience:
            return False, {}

        stats = self.current_stats
        if stats and stats.is_plateaued:
            self.plateau_count += 1
            if self.plateau_count >= self.config.plateau_patience:
                return True, {
                    "std": stats.std,
                    "mean": stats.mean,
                    "duration": self.plateau_count
                }
        else:
            self.plateau_count = 0

        return False, {}

    def _handle_plateau(self, plateau_info: Dict[str, Any]):
        """Handle loss plateau (logging only)."""
        self.logger.info(
            f"Loss plateau detected at step {self.step_count}: "
            f"mean={plateau_info['mean']:.4f}, duration={plateau_info['duration']} steps"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.current_stats:
            return {"status": "no_data"}

        return {
            "step": self.step_count,
            "current_loss": self.current_stats.current,
            "ema_loss": self.current_stats.ema,
            "mean_loss": self.current_stats.mean,
            "std_loss": self.current_stats.std,
            "trend": self.current_stats.trend,
            "is_stable": self.current_stats.is_stable,
            "is_plateaued": self.current_stats.is_plateaued,
            "best_loss": self.best_loss if self.best_loss != float('inf') else None,
            "worst_loss": self.worst_loss if self.worst_loss != float('-inf') else None,
            "issues": {
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "anomaly_count": self.anomaly_count,
                "divergence_count": self.divergence_count
            }
        }

    def should_early_stop(self, patience: int = 50) -> bool:
        """
        Check if training should stop early.

        Args:
            patience: Number of steps without improvement

        Returns:
            Whether to stop training
        """
        if len(self.loss_history) < patience:
            return False

        # Check if loss hasn't improved in patience steps
        recent_losses = list(self.loss_history)[-patience:]
        best_recent = min(recent_losses)

        # Compare with historical best (excluding recent)
        if len(self.loss_history) > patience:
            historical = list(self.loss_history)[:-patience]
            best_historical = min(historical)

            if best_recent >= best_historical:
                self.logger.info(
                    f"No improvement in {patience} steps. "
                    f"Best recent: {best_recent:.4f}, Best historical: {best_historical:.4f}"
                )
                return True

        return False

    def reset(self):
        """Reset validator state."""
        self.loss_history.clear()
        self.ema_loss = None
        self.step_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.anomaly_count = 0
        self.divergence_count = 0
        self.plateau_count = 0
        self.current_stats = None
        self.best_loss = float('inf')
        self.worst_loss = float('-inf')