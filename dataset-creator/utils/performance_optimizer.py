"""
Performance optimization utilities for LLM inference.
Provides safe GPU optimizations without risking OOM.
"""

import torch
import gc
from typing import Optional, Dict, Any
import logging
from transformers import BitsAndBytesConfig
import os

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Manages performance optimizations for LLM inference."""

    def __init__(self):
        self.original_settings = {}
        self.optimizations_applied = []

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage stats."""
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_reserved()) / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }

    def enable_tf32(self):
        """Enable TF32 for matrix multiplications (10-15% speedup on Ampere/Ada GPUs)."""
        if torch.cuda.is_available():
            # Store original settings
            self.original_settings["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            self.original_settings["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32

            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.optimizations_applied.append("TF32")
            logger.info("✓ TF32 enabled for matrix operations")
            return True
        return False

    def get_optimized_quantization_config(self, use_bfloat16: bool = True) -> BitsAndBytesConfig:
        """
        Get optimized quantization config with better compute dtype.

        Args:
            use_bfloat16: Whether to use bfloat16 (better for RTX 40 series)
        """
        compute_dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_bf16_supported() else torch.float16

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Reduces memory by ~0.4 bits/param
        )

        dtype_name = "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
        logger.info(f"✓ Quantization config optimized with {dtype_name} compute")
        self.optimizations_applied.append(f"4-bit-{dtype_name}")

        return config

    def optimize_device_map(self, model_size_gb: float = 7.0) -> str:
        """
        Get optimized device map based on available VRAM.

        Args:
            model_size_gb: Estimated model size in GB (quantized)

        Returns:
            Device map strategy ('auto', 'balanced', 'sequential')
        """
        if not torch.cuda.is_available():
            return "cpu"

        mem_info = self.get_gpu_memory_info()
        free_vram = mem_info.get("free_gb", 0)

        # Conservative: Keep 2GB buffer for activations/KV cache
        safety_buffer_gb = 2.0

        if free_vram > model_size_gb + safety_buffer_gb:
            # Enough VRAM for full model
            logger.info(f"✓ VRAM sufficient ({free_vram:.1f}GB free) for full GPU loading")
            return "cuda:0"
        elif free_vram > model_size_gb * 0.7:
            # Can fit most of model on GPU
            logger.info(f"✓ Using balanced device mapping ({free_vram:.1f}GB free)")
            return "balanced"
        else:
            # Need significant CPU offloading
            logger.info(f"⚠ Limited VRAM ({free_vram:.1f}GB free), using auto device mapping")
            return "auto"

    def enable_flash_attention(self, model_name: str) -> Dict[str, Any]:
        """
        Get model kwargs for Flash Attention 2 if supported.

        Args:
            model_name: Name of the model being loaded
        """
        kwargs = {}

        # Check if model supports optimized attention
        supports_flash_attn = any(x in model_name.lower() for x in ["qwen2", "llama", "mistral"])

        if supports_flash_attn and torch.cuda.is_available():
            # Use SDPA (Scaled Dot Product Attention) - built into PyTorch 2.0+
            # Nearly as fast as Flash Attention 2 but without compatibility issues
            kwargs["attn_implementation"] = "sdpa"
            logger.info("✓ Using SDPA attention (optimized for RTX 50 series)")
            self.optimizations_applied.append("SDPA")

        return kwargs

    def optimize_generation_config(self) -> Dict[str, Any]:
        """Get optimized generation configuration."""
        return {
            "use_cache": True,  # Enable KV cache
            "do_sample": True,  # Required for temperature/top_p
            "pad_token_id": None,  # Will be set from tokenizer
            "eos_token_id": None,  # Will be set from tokenizer
        }

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("✓ GPU cache cleared")

    def get_summary(self) -> str:
        """Get summary of applied optimizations."""
        if not self.optimizations_applied:
            return "No optimizations applied"

        mem_info = self.get_gpu_memory_info()
        summary = f"Optimizations applied: {', '.join(self.optimizations_applied)}\n"

        if mem_info:
            summary += f"GPU Memory: {mem_info['allocated_gb']:.1f}GB used / {mem_info['total_gb']:.1f}GB total"

        return summary

    def reset(self):
        """Reset all optimizations to original settings."""
        if "matmul_allow_tf32" in self.original_settings:
            torch.backends.cuda.matmul.allow_tf32 = self.original_settings["matmul_allow_tf32"]
        if "cudnn_allow_tf32" in self.original_settings:
            torch.backends.cudnn.allow_tf32 = self.original_settings["cudnn_allow_tf32"]

        self.original_settings.clear()
        self.optimizations_applied.clear()
        logger.info("All optimizations reset")


def benchmark_generation(model, tokenizer, prompt: str, iterations: int = 5) -> Dict[str, float]:
    """
    Benchmark generation performance.

    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        prompt: Test prompt
        iterations: Number of iterations to average

    Returns:
        Dictionary with performance metrics
    """
    import time

    device = next(model.parameters()).device

    # Warmup
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    times = []
    tokens_generated = []

    for _ in range(iterations):
        start = time.perf_counter()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        end = time.perf_counter()
        times.append(end - start)
        tokens_generated.append(outputs.shape[1] - inputs.input_ids.shape[1])

    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    return {
        "avg_time_seconds": avg_time,
        "avg_tokens_generated": avg_tokens,
        "tokens_per_second": avg_tokens / avg_time,
        "iterations": iterations
    }