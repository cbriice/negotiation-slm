"""Gradient health monitoring and explosion detection."""

import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import math


@dataclass
class GradientStatistics:
    """Statistics for gradient monitoring."""

    mean: float
    std: float
    min: float
    max: float
    norm: float
    has_nan: bool
    has_inf: bool
    sparsity: float  # Percentage of near-zero gradients
    variance_ratio: float  # Ratio of variance to mean squared


@dataclass
class GradientHealthConfig:
    """Configuration for gradient health monitoring."""

    # Thresholds
    max_gradient_norm: float = 10.0
    explosion_threshold: float = 100.0
    vanishing_threshold: float = 1e-7
    nan_tolerance: int = 0  # Number of NaN gradients to tolerate

    # Clipping
    enable_clipping: bool = True
    clip_value: float = 1.0
    adaptive_clipping: bool = True
    clip_percentile: float = 95.0

    # History tracking
    history_size: int = 100
    anomaly_detection: bool = True
    anomaly_factor: float = 3.0  # Number of std devs for anomaly

    # Reporting
    log_frequency: int = 10  # Log every N steps
    verbose: bool = False


class GradientMonitor:
    """Monitors gradient health during training."""

    def __init__(
        self,
        config: GradientHealthConfig,
        model: Optional[torch.nn.Module] = None
    ):
        """
        Initialize gradient monitor.

        Args:
            config: Gradient monitoring configuration
            model: Model to monitor
        """
        self.config = config
        self.model = model
        self.logger = logging.getLogger("GradientMonitor")

        # History tracking
        self.gradient_norm_history = deque(maxlen=config.history_size)
        self.layer_gradient_history: Dict[str, deque] = {}
        self.anomaly_count = 0
        self.step_count = 0

        # Statistics
        self.global_stats: Optional[GradientStatistics] = None
        self.layer_stats: Dict[str, GradientStatistics] = {}

        # Adaptive clipping
        self.adaptive_clip_value = config.clip_value

    def check_gradients(
        self,
        model: Optional[torch.nn.Module] = None,
        return_stats: bool = True
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check gradient health.

        Args:
            model: Model to check (uses self.model if None)
            return_stats: Whether to return detailed statistics

        Returns:
            Tuple of (is_healthy, statistics_dict)
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided for gradient checking")

        self.step_count += 1

        # Collect gradients
        gradients = self._collect_gradients(model)
        if not gradients:
            self.logger.warning("No gradients found")
            return False, None

        # Calculate statistics
        self.global_stats = self._calculate_global_stats(gradients)
        self.layer_stats = self._calculate_layer_stats(gradients)

        # Check health
        is_healthy, issues = self._check_health()

        # Update history
        self._update_history()

        # Log if necessary
        if self.step_count % self.config.log_frequency == 0:
            self._log_status(is_healthy, issues)

        # Prepare return statistics
        stats = None
        if return_stats:
            stats = {
                "global": self._stats_to_dict(self.global_stats),
                "layers": {
                    name: self._stats_to_dict(stat)
                    for name, stat in self.layer_stats.items()
                },
                "is_healthy": is_healthy,
                "issues": issues,
                "step": self.step_count
            }

        return is_healthy, stats

    def _collect_gradients(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Collect all gradients from model."""
        gradients = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach()

        return gradients

    def _calculate_global_stats(self, gradients: Dict[str, torch.Tensor]) -> GradientStatistics:
        """Calculate global gradient statistics."""
        all_grads = []
        total_norm = 0

        for grad in gradients.values():
            all_grads.append(grad.flatten())
            total_norm += grad.norm().item() ** 2

        total_norm = math.sqrt(total_norm)
        all_grads = torch.cat(all_grads)

        # Calculate statistics
        has_nan = torch.isnan(all_grads).any().item()
        has_inf = torch.isinf(all_grads).any().item()

        if has_nan or has_inf:
            return GradientStatistics(
                mean=float('nan'),
                std=float('nan'),
                min=float('nan'),
                max=float('nan'),
                norm=total_norm,
                has_nan=has_nan,
                has_inf=has_inf,
                sparsity=0,
                variance_ratio=0
            )

        mean = all_grads.mean().item()
        std = all_grads.std().item()
        min_val = all_grads.min().item()
        max_val = all_grads.max().item()

        # Calculate sparsity (percentage of near-zero gradients)
        near_zero = (all_grads.abs() < 1e-8).float().mean().item()

        # Calculate variance ratio
        variance_ratio = std ** 2 / (mean ** 2 + 1e-8)

        return GradientStatistics(
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            norm=total_norm,
            has_nan=has_nan,
            has_inf=has_inf,
            sparsity=near_zero,
            variance_ratio=variance_ratio
        )

    def _calculate_layer_stats(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, GradientStatistics]:
        """Calculate per-layer gradient statistics."""
        layer_stats = {}

        for name, grad in gradients.items():
            flat_grad = grad.flatten()

            has_nan = torch.isnan(flat_grad).any().item()
            has_inf = torch.isinf(flat_grad).any().item()

            if has_nan or has_inf:
                layer_stats[name] = GradientStatistics(
                    mean=float('nan'),
                    std=float('nan'),
                    min=float('nan'),
                    max=float('nan'),
                    norm=grad.norm().item(),
                    has_nan=has_nan,
                    has_inf=has_inf,
                    sparsity=0,
                    variance_ratio=0
                )
                continue

            mean = flat_grad.mean().item()
            std = flat_grad.std().item()
            min_val = flat_grad.min().item()
            max_val = flat_grad.max().item()
            norm = grad.norm().item()

            near_zero = (flat_grad.abs() < 1e-8).float().mean().item()
            variance_ratio = std ** 2 / (mean ** 2 + 1e-8)

            layer_stats[name] = GradientStatistics(
                mean=mean,
                std=std,
                min=min_val,
                max=max_val,
                norm=norm,
                has_nan=has_nan,
                has_inf=has_inf,
                sparsity=near_zero,
                variance_ratio=variance_ratio
            )

            # Update layer history
            if name not in self.layer_gradient_history:
                self.layer_gradient_history[name] = deque(maxlen=self.config.history_size)
            self.layer_gradient_history[name].append(norm)

        return layer_stats

    def _check_health(self) -> Tuple[bool, List[str]]:
        """Check gradient health and return issues."""
        issues = []

        if self.global_stats is None:
            return False, ["No gradient statistics available"]

        # Check for NaN/Inf
        if self.global_stats.has_nan:
            issues.append("NaN gradients detected")
        if self.global_stats.has_inf:
            issues.append("Inf gradients detected")

        # Check gradient explosion
        if self.global_stats.norm > self.config.explosion_threshold:
            issues.append(f"Gradient explosion: norm={self.global_stats.norm:.2f}")

        # Check gradient vanishing
        if self.global_stats.norm < self.config.vanishing_threshold:
            issues.append(f"Gradient vanishing: norm={self.global_stats.norm:.2e}")

        # Check high sparsity (dead neurons)
        if self.global_stats.sparsity > 0.9:
            issues.append(f"High gradient sparsity: {self.global_stats.sparsity:.1%}")

        # Anomaly detection
        if self.config.anomaly_detection and len(self.gradient_norm_history) > 10:
            self._check_anomalies(issues)

        # Per-layer checks
        for name, stats in self.layer_stats.items():
            if stats.has_nan or stats.has_inf:
                issues.append(f"Layer {name}: NaN/Inf detected")
            elif stats.norm > self.config.explosion_threshold:
                issues.append(f"Layer {name}: explosion (norm={stats.norm:.2f})")
            elif stats.norm < self.config.vanishing_threshold:
                issues.append(f"Layer {name}: vanishing (norm={stats.norm:.2e})")

        is_healthy = len(issues) == 0
        return is_healthy, issues

    def _check_anomalies(self, issues: List[str]):
        """Check for anomalies in gradient history."""
        if len(self.gradient_norm_history) < 2:
            return

        history_array = np.array(list(self.gradient_norm_history))
        mean = np.mean(history_array[:-1])  # Exclude current
        std = np.std(history_array[:-1])

        if std > 0:
            z_score = abs((self.global_stats.norm - mean) / std)
            if z_score > self.config.anomaly_factor:
                issues.append(
                    f"Gradient anomaly detected: z-score={z_score:.2f}, "
                    f"current={self.global_stats.norm:.2f}, mean={mean:.2f}"
                )
                self.anomaly_count += 1

    def _update_history(self):
        """Update gradient history."""
        if self.global_stats:
            self.gradient_norm_history.append(self.global_stats.norm)

    def _log_status(self, is_healthy: bool, issues: List[str]):
        """Log gradient status."""
        if not is_healthy:
            self.logger.warning(f"Gradient health issues at step {self.step_count}: {issues}")
        elif self.config.verbose:
            self.logger.info(
                f"Step {self.step_count}: Gradient norm={self.global_stats.norm:.4f}, "
                f"mean={self.global_stats.mean:.2e}, std={self.global_stats.std:.2e}"
            )

    def _stats_to_dict(self, stats: GradientStatistics) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "norm": stats.norm,
            "has_nan": stats.has_nan,
            "has_inf": stats.has_inf,
            "sparsity": stats.sparsity,
            "variance_ratio": stats.variance_ratio
        }

    def clip_gradients(
        self,
        model: Optional[torch.nn.Module] = None,
        max_norm: Optional[float] = None
    ) -> float:
        """
        Clip gradients by norm.

        Args:
            model: Model to clip (uses self.model if None)
            max_norm: Maximum norm (uses config value if None)

        Returns:
            Total gradient norm before clipping
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided for gradient clipping")

        max_norm = max_norm or self.adaptive_clip_value

        # Update adaptive clipping value if enabled
        if self.config.adaptive_clipping and len(self.gradient_norm_history) > 10:
            self._update_adaptive_clipping()
            max_norm = self.adaptive_clip_value

        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm
        )

        if self.config.verbose and total_norm > max_norm:
            self.logger.info(
                f"Clipped gradients: {total_norm:.2f} -> {max_norm:.2f}"
            )

        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

    def _update_adaptive_clipping(self):
        """Update adaptive clipping value based on history."""
        if len(self.gradient_norm_history) < 10:
            return

        history_array = np.array(list(self.gradient_norm_history))

        # Use percentile-based clipping
        clip_value = np.percentile(history_array, self.config.clip_percentile)

        # Smooth the update
        self.adaptive_clip_value = 0.9 * self.adaptive_clip_value + 0.1 * clip_value

        # Apply bounds
        self.adaptive_clip_value = max(
            0.1,  # Minimum clip value
            min(self.adaptive_clip_value, self.config.explosion_threshold)
        )

    def reset_anomaly_detection(self):
        """Reset anomaly detection state."""
        self.anomaly_count = 0
        self.gradient_norm_history.clear()
        self.layer_gradient_history.clear()

    def get_gradient_flow(self) -> Dict[str, List[float]]:
        """
        Get gradient flow through layers.

        Returns:
            Dictionary mapping layer names to gradient norm history
        """
        return {
            name: list(history)
            for name, history in self.layer_gradient_history.items()
        }

    def diagnose_gradient_issues(self) -> Dict[str, Any]:
        """
        Comprehensive gradient diagnosis.

        Returns:
            Diagnostic information
        """
        diagnosis = {
            "step": self.step_count,
            "anomaly_count": self.anomaly_count,
            "adaptive_clip_value": self.adaptive_clip_value,
            "recommendations": []
        }

        if self.global_stats:
            # Check for specific issues
            if self.global_stats.has_nan or self.global_stats.has_inf:
                diagnosis["recommendations"].append(
                    "Critical: NaN/Inf gradients. Check loss function and data."
                )

            if self.global_stats.norm > self.config.explosion_threshold:
                diagnosis["recommendations"].append(
                    f"Gradient explosion detected. Reduce learning rate or enable clipping."
                )

            if self.global_stats.norm < self.config.vanishing_threshold:
                diagnosis["recommendations"].append(
                    f"Gradient vanishing detected. Check model initialization or use residual connections."
                )

            if self.global_stats.sparsity > 0.8:
                diagnosis["recommendations"].append(
                    f"High sparsity ({self.global_stats.sparsity:.1%}). "
                    f"Many neurons may be dead. Check activation functions."
                )

            # Analyze layer-specific issues
            problematic_layers = []
            for name, stats in self.layer_stats.items():
                if stats.norm > self.config.explosion_threshold or \
                   stats.norm < self.config.vanishing_threshold:
                    problematic_layers.append(name)

            if problematic_layers:
                diagnosis["problematic_layers"] = problematic_layers
                diagnosis["recommendations"].append(
                    f"Issues in layers: {', '.join(problematic_layers[:5])}"
                )

        # Check gradient flow
        if len(self.gradient_norm_history) > 10:
            history_array = np.array(list(self.gradient_norm_history))
            trend = np.polyfit(range(len(history_array)), history_array, 1)[0]

            if abs(trend) > 0.1:
                direction = "increasing" if trend > 0 else "decreasing"
                diagnosis["gradient_trend"] = direction
                diagnosis["recommendations"].append(
                    f"Gradient norms are {direction}. Monitor for instability."
                )

        return diagnosis