"""Automatic batch size optimization for OOM recovery."""

import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import gc
import time
import math


@dataclass
class BatchSizeConfig:
    """Configuration for batch size optimization."""

    initial_batch_size: int
    min_batch_size: int = 1
    max_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Auto-tuning parameters
    enable_auto_tuning: bool = True
    memory_fraction_target: float = 0.9  # Target GPU memory usage
    performance_weight: float = 0.7  # Balance between speed and memory

    # Search parameters
    search_mode: str = "binary"  # "binary", "linear", "exponential"
    max_search_iterations: int = 10

    # Recovery parameters
    reduction_factor: float = 0.5  # Factor to reduce batch size on OOM
    increase_factor: float = 1.1  # Factor to increase batch size if stable


class BatchSizeOptimizer:
    """Optimizes batch size for maximum throughput without OOM."""

    def __init__(
        self,
        config: BatchSizeConfig,
        model: Optional[torch.nn.Module] = None,
        log_dir: str = "logs"
    ):
        """
        Initialize batch size optimizer.

        Args:
            config: Batch size configuration
            model: Model to optimize for
            log_dir: Directory for logs
        """
        self.config = config
        self.model = model
        self.logger = logging.getLogger("BatchSizeOptimizer")

        # Current state
        self.current_batch_size = config.initial_batch_size
        self.current_grad_accum = config.gradient_accumulation_steps
        self.effective_batch_size = self.current_batch_size * self.current_grad_accum

        # Performance tracking
        self.throughput_history: Dict[int, float] = {}
        self.memory_history: Dict[int, float] = {}
        self.oom_batch_sizes: set = set()
        self.stable_batch_sizes: set = set()

        # Search state
        self.search_iteration = 0
        self.best_batch_size = config.initial_batch_size
        self.best_throughput = 0.0

    def find_optimal_batch_size(
        self,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...],
        max_sequence_length: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Find optimal batch size through auto-tuning.

        Args:
            forward_func: Function to run forward pass
            input_shape: Shape of single input sample
            max_sequence_length: Maximum sequence length for transformers

        Returns:
            Tuple of (optimal_batch_size, gradient_accumulation_steps)
        """
        self.logger.info("Starting batch size optimization...")

        if self.config.search_mode == "binary":
            optimal_size = self._binary_search(forward_func, input_shape)
        elif self.config.search_mode == "exponential":
            optimal_size = self._exponential_search(forward_func, input_shape)
        else:
            optimal_size = self._linear_search(forward_func, input_shape)

        # Optimize gradient accumulation
        grad_accum = self._optimize_gradient_accumulation(optimal_size)

        self.logger.info(
            f"Optimal configuration: batch_size={optimal_size}, "
            f"grad_accum={grad_accum}, effective_batch={optimal_size * grad_accum}"
        )

        return optimal_size, grad_accum

    def _binary_search(
        self,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...]
    ) -> int:
        """
        Binary search for optimal batch size.

        Args:
            forward_func: Forward pass function
            input_shape: Input shape

        Returns:
            Optimal batch size
        """
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        best_batch_size = low
        best_throughput = 0

        while low <= high and self.search_iteration < self.config.max_search_iterations:
            mid = (low + high) // 2

            # Skip if known to OOM
            if mid in self.oom_batch_sizes:
                high = mid - 1
                continue

            success, throughput, memory = self._test_batch_size(
                mid, forward_func, input_shape
            )

            if success:
                self.stable_batch_sizes.add(mid)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = mid

                # Try larger batch size
                if memory < self.config.memory_fraction_target:
                    low = mid + 1
                else:
                    # Close to memory limit, be conservative
                    high = mid - 1
            else:
                self.oom_batch_sizes.add(mid)
                high = mid - 1

            self.search_iteration += 1

        return best_batch_size

    def _exponential_search(
        self,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...]
    ) -> int:
        """
        Exponential search for optimal batch size.

        Args:
            forward_func: Forward pass function
            input_shape: Input shape

        Returns:
            Optimal batch size
        """
        batch_size = self.config.min_batch_size
        best_batch_size = batch_size
        best_throughput = 0

        while batch_size <= self.config.max_batch_size and \
              self.search_iteration < self.config.max_search_iterations:

            success, throughput, memory = self._test_batch_size(
                batch_size, forward_func, input_shape
            )

            if success:
                self.stable_batch_sizes.add(batch_size)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size

                # Exponentially increase
                batch_size = min(batch_size * 2, self.config.max_batch_size)
            else:
                self.oom_batch_sizes.add(batch_size)
                # Binary search between last good and current
                if best_batch_size < batch_size - 1:
                    return self._binary_search_range(
                        best_batch_size, batch_size - 1,
                        forward_func, input_shape
                    )
                break

            self.search_iteration += 1

        return best_batch_size

    def _linear_search(
        self,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...]
    ) -> int:
        """
        Linear search for optimal batch size.

        Args:
            forward_func: Forward pass function
            input_shape: Input shape

        Returns:
            Optimal batch size
        """
        best_batch_size = self.config.min_batch_size
        best_throughput = 0

        for batch_size in range(
            self.config.min_batch_size,
            min(self.config.max_batch_size + 1, self.config.min_batch_size + self.config.max_search_iterations)
        ):
            if batch_size in self.oom_batch_sizes:
                break

            success, throughput, memory = self._test_batch_size(
                batch_size, forward_func, input_shape
            )

            if success:
                self.stable_batch_sizes.add(batch_size)
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
            else:
                self.oom_batch_sizes.add(batch_size)
                break

        return best_batch_size

    def _binary_search_range(
        self,
        low: int,
        high: int,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...]
    ) -> int:
        """Binary search within a specific range."""
        best_batch_size = low
        best_throughput = 0

        while low <= high:
            mid = (low + high) // 2

            success, throughput, memory = self._test_batch_size(
                mid, forward_func, input_shape
            )

            if success:
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1

        return best_batch_size

    def _test_batch_size(
        self,
        batch_size: int,
        forward_func: Callable[[int], torch.Tensor],
        input_shape: Tuple[int, ...]
    ) -> Tuple[bool, float, float]:
        """
        Test a specific batch size.

        Args:
            batch_size: Batch size to test
            forward_func: Forward pass function
            input_shape: Input shape

        Returns:
            Tuple of (success, throughput, memory_fraction)
        """
        self.logger.debug(f"Testing batch size: {batch_size}")

        # Clear cache before test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        try:
            # Measure initial memory
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            # Run forward pass
            start_time = time.time()
            loss = forward_func(batch_size)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN/Inf loss with batch size {batch_size}")
                return False, 0, 0

            # Backward pass
            loss.backward()

            # Synchronize for accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_time = time.time() - start_time

            # Calculate metrics
            throughput = batch_size / elapsed_time
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_fraction = peak_memory / total_memory
            else:
                memory_fraction = 0

            # Clear gradients
            if self.model:
                self.model.zero_grad()

            # Store history
            self.throughput_history[batch_size] = throughput
            self.memory_history[batch_size] = memory_fraction

            self.logger.debug(
                f"Batch size {batch_size}: throughput={throughput:.2f} samples/s, "
                f"memory={memory_fraction:.2%}"
            )

            return True, throughput, memory_fraction

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.debug(f"OOM with batch size {batch_size}")
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                return False, 0, 1.0
            else:
                raise

    def _optimize_gradient_accumulation(self, batch_size: int) -> int:
        """
        Optimize gradient accumulation steps for effective batch size.

        Args:
            batch_size: Per-device batch size

        Returns:
            Optimal gradient accumulation steps
        """
        # Target effective batch size (common values)
        target_effective_sizes = [16, 32, 64, 128, 256]

        best_grad_accum = 1
        min_remainder = float('inf')

        for target in target_effective_sizes:
            if target >= batch_size:
                grad_accum = target // batch_size
                remainder = target % batch_size

                # Prefer exact divisors
                if remainder < min_remainder:
                    min_remainder = remainder
                    best_grad_accum = grad_accum

                if remainder == 0 and grad_accum <= 16:  # Reasonable limit
                    return grad_accum

        return min(best_grad_accum, 16)

    def reduce_batch_size_on_oom(self) -> Tuple[int, int]:
        """
        Reduce batch size after OOM error.

        Returns:
            New (batch_size, gradient_accumulation_steps)
        """
        self.oom_batch_sizes.add(self.current_batch_size)

        # Try reducing batch size
        new_batch_size = max(
            self.config.min_batch_size,
            int(self.current_batch_size * self.config.reduction_factor)
        )

        # If we can't reduce batch size, increase gradient accumulation
        if new_batch_size == self.current_batch_size and new_batch_size == self.config.min_batch_size:
            new_grad_accum = min(self.current_grad_accum * 2, 32)
            self.logger.warning(
                f"Cannot reduce batch size further. "
                f"Increasing gradient accumulation to {new_grad_accum}"
            )
            self.current_grad_accum = new_grad_accum
        else:
            self.current_batch_size = new_batch_size
            self.logger.info(f"Reduced batch size to {new_batch_size} after OOM")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return self.current_batch_size, self.current_grad_accum

    def try_increase_batch_size(
        self,
        current_memory_usage: float,
        recent_loss_stable: bool
    ) -> Tuple[int, int]:
        """
        Try to increase batch size if memory and training are stable.

        Args:
            current_memory_usage: Current GPU memory usage fraction
            recent_loss_stable: Whether recent loss is stable

        Returns:
            New (batch_size, gradient_accumulation_steps)
        """
        if not recent_loss_stable:
            return self.current_batch_size, self.current_grad_accum

        # Only increase if we have memory headroom
        if current_memory_usage < 0.7:  # Conservative threshold
            new_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * self.config.increase_factor)
            )

            # Check if new size was previously OOM
            if new_batch_size not in self.oom_batch_sizes:
                self.logger.info(
                    f"Memory usage low ({current_memory_usage:.1%}), "
                    f"trying batch size {new_batch_size}"
                )
                self.current_batch_size = new_batch_size

                # Potentially reduce gradient accumulation
                if self.current_grad_accum > 1:
                    target_effective = self.current_batch_size * self.current_grad_accum
                    new_grad_accum = max(1, target_effective // new_batch_size)
                    self.current_grad_accum = new_grad_accum

        return self.current_batch_size, self.current_grad_accum

    def get_memory_estimate(
        self,
        batch_size: int,
        sequence_length: int,
        model_params: int,
        dtype_bytes: int = 2  # bf16
    ) -> float:
        """
        Estimate memory usage for given batch size.

        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            model_params: Number of model parameters
            dtype_bytes: Bytes per parameter

        Returns:
            Estimated memory in GB
        """
        # Model weights
        model_memory = model_params * dtype_bytes

        # Activations (rough estimate for transformers)
        # Each layer stores: attention scores, values, FFN intermediates
        num_layers = 32  # Typical for 7B model
        hidden_size = 4096  # Typical for 7B model
        attention_heads = 32

        activation_memory = (
            batch_size * sequence_length * hidden_size * dtype_bytes * num_layers * 4
        )

        # Gradients (same size as weights)
        gradient_memory = model_memory

        # Optimizer states (Adam: 2x parameters)
        optimizer_memory = model_params * dtype_bytes * 2

        total_memory_bytes = (
            model_memory + activation_memory + gradient_memory + optimizer_memory
        )

        return total_memory_bytes / (1024 ** 3)  # Convert to GB

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            "current_batch_size": self.current_batch_size,
            "gradient_accumulation_steps": self.current_grad_accum,
            "effective_batch_size": self.effective_batch_size,
            "best_batch_size": self.best_batch_size,
            "best_throughput": self.best_throughput,
            "oom_batch_sizes": list(self.oom_batch_sizes),
            "stable_batch_sizes": list(self.stable_batch_sizes),
            "throughput_history": self.throughput_history,
            "memory_history": self.memory_history
        }