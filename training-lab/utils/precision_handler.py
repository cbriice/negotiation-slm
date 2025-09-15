"""Mixed precision training handler with automatic fallback."""

import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
from contextlib import contextmanager
import gc


@dataclass
class PrecisionConfig:
    """Configuration for precision handling."""

    # Preferred precision settings
    use_bf16: bool = True
    use_fp16: bool = False
    use_fp32_fallback: bool = True

    # Automatic mixed precision (AMP)
    use_amp: bool = True
    amp_dtype: Optional[torch.dtype] = None  # Auto-detect

    # Gradient scaling for fp16
    use_gradient_scaling: bool = True
    initial_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # Overflow handling
    max_consecutive_overflows: int = 3
    overflow_fallback: str = "reduce_scale"  # "reduce_scale", "skip_step", "switch_precision"

    # Hardware detection
    auto_detect_hardware: bool = True
    force_cpu_fp32: bool = False


class PrecisionHandler:
    """Handles mixed precision training with automatic fallback."""

    def __init__(self, config: PrecisionConfig):
        """
        Initialize precision handler.

        Args:
            config: Precision configuration
        """
        self.config = config
        self.logger = logging.getLogger("PrecisionHandler")

        # Detect hardware capabilities
        self.hardware_info = self._detect_hardware()
        self.supported_dtypes = self._get_supported_dtypes()

        # Select optimal precision
        self.current_dtype = self._select_precision()
        self.original_dtype = self.current_dtype

        # Gradient scaler for fp16
        self.scaler = None
        if self.current_dtype == torch.float16 and self.config.use_gradient_scaling:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.initial_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval
            )

        # Tracking
        self.overflow_count = 0
        self.consecutive_overflows = 0
        self.step_count = 0
        self.precision_switches = 0

        self.logger.info(f"Initialized with dtype: {self.current_dtype}")

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware capabilities."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": 0,
            "gpu_name": None,
            "compute_capability": None,
            "supports_bf16": False,
            "supports_fp16": False,
            "supports_tf32": False,
            "tensor_cores": False
        }

        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)

            # Check compute capability
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = (major, minor)

            # BF16 support (Ampere and newer, compute capability >= 8.0)
            info["supports_bf16"] = major >= 8

            # FP16 support (Pascal and newer, compute capability >= 6.0)
            info["supports_fp16"] = major >= 6

            # TF32 support (Ampere and newer)
            info["supports_tf32"] = major >= 8

            # Tensor cores (Volta and newer, compute capability >= 7.0)
            info["tensor_cores"] = major >= 7

            self.logger.info(
                f"GPU: {info['gpu_name']}, "
                f"Compute capability: {major}.{minor}, "
                f"BF16: {info['supports_bf16']}, "
                f"FP16: {info['supports_fp16']}, "
                f"Tensor cores: {info['tensor_cores']}"
            )

        return info

    def _get_supported_dtypes(self) -> list:
        """Get list of supported data types based on hardware."""
        dtypes = [torch.float32]  # Always supported

        if self.hardware_info["cuda_available"]:
            if self.hardware_info["supports_fp16"]:
                dtypes.append(torch.float16)
            if self.hardware_info["supports_bf16"]:
                dtypes.append(torch.bfloat16)

        return dtypes

    def _select_precision(self) -> torch.dtype:
        """Select optimal precision based on config and hardware."""
        # Force CPU FP32 if configured
        if self.config.force_cpu_fp32 or not self.hardware_info["cuda_available"]:
            self.logger.info("Using FP32 (CPU or forced)")
            return torch.float32

        # Try BF16 first (preferred for stability)
        if self.config.use_bf16 and torch.bfloat16 in self.supported_dtypes:
            self.logger.info("Using BF16 precision")
            return torch.bfloat16

        # Try FP16
        if self.config.use_fp16 and torch.float16 in self.supported_dtypes:
            self.logger.info("Using FP16 precision (gradient scaling enabled)")
            return torch.float16

        # Fallback to FP32
        if self.config.use_fp32_fallback:
            self.logger.info("Using FP32 precision (fallback)")
            return torch.float32

        raise ValueError("No suitable precision available")

    @contextmanager
    def autocast_context(
        self,
        device_type: str = "cuda",
        custom_dtype: Optional[torch.dtype] = None
    ):
        """
        Context manager for automatic mixed precision.

        Args:
            device_type: Device type for autocast
            custom_dtype: Override dtype for this context

        Yields:
            Autocast context
        """
        dtype = custom_dtype or self.current_dtype

        if dtype == torch.float32:
            # No autocast needed for FP32
            yield
        else:
            # Use autocast for mixed precision
            with torch.autocast(device_type=device_type, dtype=dtype):
                yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss
        """
        if self.scaler is not None and self.current_dtype == torch.float16:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        grad_clip_fn: Optional[Callable] = None
    ) -> bool:
        """
        Step optimizer with mixed precision handling.

        Args:
            optimizer: Optimizer to step
            grad_clip_fn: Optional gradient clipping function

        Returns:
            Whether step was successful (not skipped due to overflow)
        """
        self.step_count += 1

        if self.scaler is not None and self.current_dtype == torch.float16:
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)

            # Apply gradient clipping if provided
            if grad_clip_fn:
                grad_clip_fn()

            # Check for infs/nans
            optimizer_step_success = self._check_gradients(optimizer)

            if optimizer_step_success:
                # Step optimizer
                self.scaler.step(optimizer)
                self.scaler.update()
                self.consecutive_overflows = 0
                return True
            else:
                # Skip step due to overflow
                self.scaler.update()
                return self._handle_overflow()
        else:
            # Regular optimizer step
            if grad_clip_fn:
                grad_clip_fn()

            # Check for numerical issues even in bf16/fp32
            if not self._check_gradients(optimizer):
                return self._handle_overflow()

            optimizer.step()
            return True

    def _check_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Check gradients for numerical issues.

        Args:
            optimizer: Optimizer with gradients

        Returns:
            Whether gradients are valid
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        return False
        return True

    def _handle_overflow(self) -> bool:
        """
        Handle gradient overflow.

        Returns:
            Whether to continue training
        """
        self.overflow_count += 1
        self.consecutive_overflows += 1

        self.logger.warning(
            f"Gradient overflow detected (count: {self.overflow_count}, "
            f"consecutive: {self.consecutive_overflows})"
        )

        # Check if we need to take action
        if self.consecutive_overflows >= self.config.max_consecutive_overflows:
            action = self.config.overflow_fallback

            if action == "reduce_scale" and self.scaler:
                # Reduce gradient scale
                current_scale = self.scaler.get_scale()
                new_scale = current_scale * self.config.backoff_factor
                self.scaler.update(new_scale)
                self.logger.info(f"Reduced gradient scale: {current_scale} -> {new_scale}")

            elif action == "switch_precision":
                # Switch to more stable precision
                success = self.fallback_precision()
                if not success:
                    self.logger.error("Failed to switch precision")
                    return False

            elif action == "skip_step":
                # Just skip this step
                self.logger.info("Skipping optimizer step due to overflow")

        return False  # Step was skipped

    def fallback_precision(self) -> bool:
        """
        Fallback to more stable precision.

        Returns:
            Whether fallback was successful
        """
        self.precision_switches += 1

        if self.current_dtype == torch.float16:
            # Try BF16 if available
            if torch.bfloat16 in self.supported_dtypes:
                self.logger.info("Falling back from FP16 to BF16")
                self.current_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need gradient scaling
                return True
            # Otherwise fall back to FP32
            elif self.config.use_fp32_fallback:
                self.logger.info("Falling back from FP16 to FP32")
                self.current_dtype = torch.float32
                self.scaler = None
                return True

        elif self.current_dtype == torch.bfloat16:
            # Fall back to FP32
            if self.config.use_fp32_fallback:
                self.logger.info("Falling back from BF16 to FP32")
                self.current_dtype = torch.float32
                return True

        self.logger.error(f"No fallback available from {self.current_dtype}")
        return False

    def convert_model(
        self,
        model: torch.nn.Module,
        dtype: Optional[torch.dtype] = None
    ) -> torch.nn.Module:
        """
        Convert model to specified precision.

        Args:
            model: Model to convert
            dtype: Target dtype (uses current if None)

        Returns:
            Converted model
        """
        target_dtype = dtype or self.current_dtype

        if target_dtype == torch.float32:
            # No conversion needed for FP32
            return model

        # Convert model parameters
        model = model.to(dtype=target_dtype)

        # Log conversion
        param_count = sum(p.numel() for p in model.parameters())
        memory_bytes = param_count * self._dtype_size(target_dtype)
        memory_gb = memory_bytes / (1024**3)

        self.logger.info(
            f"Converted model to {target_dtype}: "
            f"{param_count:,} parameters, ~{memory_gb:.2f} GB"
        )

        return model

    def _dtype_size(self, dtype: torch.dtype) -> int:
        """Get size in bytes for dtype."""
        if dtype == torch.float32:
            return 4
        elif dtype in [torch.float16, torch.bfloat16]:
            return 2
        elif dtype == torch.int8:
            return 1
        else:
            return 4  # Default

    def optimize_for_hardware(self) -> Dict[str, Any]:
        """
        Optimize settings for detected hardware.

        Returns:
            Optimization recommendations
        """
        recommendations = {
            "dtype": self.current_dtype,
            "use_tensor_cores": False,
            "use_channels_last": False,
            "use_tf32": False,
            "cudnn_benchmark": False
        }

        if not self.hardware_info["cuda_available"]:
            return recommendations

        # Enable tensor cores if available
        if self.hardware_info["tensor_cores"]:
            recommendations["use_tensor_cores"] = True
            # Ensure dimensions are multiples of 8 for tensor core efficiency
            self.logger.info("Tensor cores available - ensure dimensions are multiples of 8")

        # Enable TF32 for Ampere GPUs
        if self.hardware_info["supports_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            recommendations["use_tf32"] = True
            self.logger.info("Enabled TF32 for matrix operations")

        # Enable cudNN autotuner for consistent input sizes
        torch.backends.cudnn.benchmark = True
        recommendations["cudnn_benchmark"] = True

        # Channels-last format for vision models (not applicable for transformers)
        # recommendations["use_channels_last"] = False

        return recommendations

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Memory usage statistics in GB
        """
        stats = {
            "allocated": 0,
            "reserved": 0,
            "free": 0
        }

        if torch.cuda.is_available():
            stats["allocated"] = torch.cuda.memory_allocated() / (1024**3)
            stats["reserved"] = torch.cuda.memory_reserved() / (1024**3)
            stats["free"] = (
                torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_reserved()
            ) / (1024**3)

        return stats

    def estimate_memory_requirement(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: int,
        dtype: Optional[torch.dtype] = None
    ) -> float:
        """
        Estimate memory requirement for training.

        Args:
            model_params: Number of model parameters
            batch_size: Batch size
            sequence_length: Sequence length
            dtype: Data type (uses current if None)

        Returns:
            Estimated memory in GB
        """
        dtype = dtype or self.current_dtype
        dtype_size = self._dtype_size(dtype)

        # Model parameters
        model_memory = model_params * dtype_size

        # Gradients (same size as parameters)
        gradient_memory = model_memory

        # Optimizer states (Adam uses 2x parameters for momentum)
        optimizer_memory = model_params * 4 * 2  # FP32 optimizer states

        # Activations (rough estimate for transformers)
        # Assumes 32 layers, hidden size 4096
        num_layers = 32
        hidden_size = 4096
        activation_memory = (
            batch_size * sequence_length * hidden_size * dtype_size * num_layers * 4
        )

        total_bytes = model_memory + gradient_memory + optimizer_memory + activation_memory
        total_gb = total_bytes / (1024**3)

        self.logger.info(
            f"Estimated memory requirement: {total_gb:.2f} GB "
            f"(dtype: {dtype}, batch_size: {batch_size}, seq_len: {sequence_length})"
        )

        return total_gb

    def reset_overflow_tracking(self):
        """Reset overflow tracking counters."""
        self.overflow_count = 0
        self.consecutive_overflows = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get precision handler summary."""
        memory_stats = self.get_memory_usage()

        return {
            "current_dtype": str(self.current_dtype),
            "original_dtype": str(self.original_dtype),
            "hardware": self.hardware_info,
            "supported_dtypes": [str(d) for d in self.supported_dtypes],
            "overflow_count": self.overflow_count,
            "precision_switches": self.precision_switches,
            "step_count": self.step_count,
            "memory_gb": memory_stats,
            "using_gradient_scaling": self.scaler is not None,
            "scale": self.scaler.get_scale() if self.scaler else None
        }