#!/usr/bin/env python3
"""
Performance benchmarking script for testing GPU optimizations.
Compares baseline vs optimized configurations.
"""

import sys
import os
import time
import torch
import gc
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_interface import LLMInterface, LLMConfig
from utils.performance_optimizer import PerformanceOptimizer, benchmark_generation
from utils.parallel_runner import ParallelNegotiationRunner
from utils.logger_config import get_logger

class PerformanceBenchmark:
    """Benchmark suite for testing performance optimizations."""

    def __init__(self):
        self.logger = get_logger()
        self.results = {}

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "bf16_supported": torch.cuda.is_bf16_supported()
        }

    def benchmark_baseline(self) -> Dict[str, float]:
        """Benchmark with baseline configuration (no optimizations)."""
        print("\n" + "="*60)
        print("BASELINE BENCHMARK (No Optimizations)")
        print("="*60)

        # Create baseline config
        config = LLMConfig(
            load_in_4bit=True,
            torch_dtype="float16",  # Standard float16
            device_map="auto"  # Default device mapping
        )

        # Disable TF32 for baseline
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        try:
            # Load model
            print("Loading model with baseline config...")
            start = time.time()
            llm = LLMInterface(config)
            load_time = time.time() - start
            print(f"✓ Model loaded in {load_time:.2f}s")

            # Test prompts
            test_prompts = [
                "I'm interested in buying this item. What's your best price?",
                "The list price seems high. Can you offer a discount?",
                "I need this for my business. What's the bulk pricing?",
                "Is this your final offer? I'm comparing with other sellers.",
                "Can you include free shipping at this price?"
            ]

            # Benchmark single generation
            print("\nTesting single generation...")
            single_times = []
            for prompt in test_prompts[:3]:
                start = time.time()
                response = llm.generate(prompt, temperature=0.7, sim_id="baseline_single")
                elapsed = time.time() - start
                single_times.append(elapsed)
                print(f"  Generated in {elapsed:.3f}s: {len(response.split())} words")

            # Benchmark sequential batch
            print("\nTesting sequential batch (5 prompts)...")
            start = time.time()
            responses = []
            for prompt in test_prompts:
                response = llm.generate(prompt, temperature=0.7, sim_id="baseline_seq")
                responses.append(response)
            seq_batch_time = time.time() - start
            print(f"✓ Sequential batch completed in {seq_batch_time:.2f}s")

            # Get memory usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                memory_allocated = 0
                memory_reserved = 0

            results = {
                "load_time": load_time,
                "avg_single_gen": sum(single_times) / len(single_times),
                "seq_batch_time": seq_batch_time,
                "seq_batch_per_prompt": seq_batch_time / len(test_prompts),
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved
            }

            # Cleanup
            llm.cleanup()
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"✗ Baseline benchmark failed: {e}")
            return {}

    def benchmark_optimized(self) -> Dict[str, float]:
        """Benchmark with all optimizations enabled."""
        print("\n" + "="*60)
        print("OPTIMIZED BENCHMARK (All Optimizations)")
        print("="*60)

        # Create optimized config
        config = LLMConfig(
            load_in_4bit=True,
            torch_dtype="auto",  # Will use bfloat16 on RTX 50 series
            device_map="auto"  # Optimized device mapping
        )

        try:
            # Load model with optimizations
            print("Loading model with optimizations...")
            start = time.time()
            llm = LLMInterface(config)  # Optimizer is initialized in __init__ now
            load_time = time.time() - start
            print(f"✓ Model loaded in {load_time:.2f}s")
            print(f"✓ Optimizations: {llm.perf_optimizer.get_summary()}")

            # Test prompts
            test_prompts = [
                "I'm interested in buying this item. What's your best price?",
                "The list price seems high. Can you offer a discount?",
                "I need this for my business. What's the bulk pricing?",
                "Is this your final offer? I'm comparing with other sellers.",
                "Can you include free shipping at this price?"
            ]

            # Benchmark single generation
            print("\nTesting single generation...")
            single_times = []
            for prompt in test_prompts[:3]:
                start = time.time()
                response = llm.generate(prompt, temperature=0.7, sim_id="optimized_single")
                elapsed = time.time() - start
                single_times.append(elapsed)
                print(f"  Generated in {elapsed:.3f}s: {len(response.split())} words")

            # Benchmark true batch generation
            print("\nTesting batched generation (5 prompts)...")
            start = time.time()
            responses = llm.batch_generate(test_prompts, temperature=0.7, sim_id="optimized_batch")
            batch_time = time.time() - start
            print(f"✓ Batch generation completed in {batch_time:.2f}s")
            print(f"  Per-prompt time: {batch_time/len(test_prompts):.3f}s")

            # Test parallel processing with smaller batch
            print("\nTesting parallel processing simulation...")
            parallel_runner = ParallelNegotiationRunner(
                llm_interface=llm,
                max_concurrent_negotiations=2,  # Conservative for testing
                batch_size=4
            )

            # Simulate parallel prompts
            parallel_prompts = test_prompts * 2  # 10 total prompts
            start = time.time()

            # Process in parallel batches
            from utils.parallel_runner import GenerationRequest
            for i, prompt in enumerate(parallel_prompts):
                req = GenerationRequest(
                    request_id=f"req_{i}",
                    prompt=prompt,
                    temperature=0.7,
                    negotiation_id=f"test_{i}",
                    turn_number=1
                )
                parallel_runner.generation_queue.put(req)

            # Wait for processing
            time.sleep(2)  # Allow batch processor to work
            parallel_time = time.time() - start
            print(f"✓ Parallel simulation completed in {parallel_time:.2f}s")

            # Get memory usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                memory_allocated = 0
                memory_reserved = 0

            results = {
                "load_time": load_time,
                "avg_single_gen": sum(single_times) / len(single_times),
                "batch_time": batch_time,
                "batch_per_prompt": batch_time / len(test_prompts),
                "parallel_time": parallel_time,
                "parallel_per_prompt": parallel_time / len(parallel_prompts),
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved
            }

            # Cleanup
            parallel_runner.shutdown()
            llm.cleanup()
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"✗ Optimized benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def compare_results(self, baseline: Dict, optimized: Dict):
        """Compare and display benchmark results."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)

        if not baseline or not optimized:
            print("✗ Insufficient data for comparison")
            return

        # Calculate speedups
        metrics = [
            ("Model Load Time", "load_time", "s"),
            ("Single Generation", "avg_single_gen", "s"),
            ("Batch Processing (5 prompts)", "batch_time", "s", "seq_batch_time"),
            ("Per-Prompt (batched)", "batch_per_prompt", "s", "seq_batch_per_prompt"),
            ("Memory Usage", "memory_allocated_gb", "GB"),
            ("Memory Reserved", "memory_reserved_gb", "GB")
        ]

        print(f"\n{'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Speedup':>10}")
        print("-" * 64)

        for metric_info in metrics:
            name = metric_info[0]
            key_opt = metric_info[1]
            unit = metric_info[2]
            key_base = metric_info[3] if len(metric_info) > 3 else key_opt

            base_val = baseline.get(key_base, 0)
            opt_val = optimized.get(key_opt, 0)

            if base_val > 0 and opt_val > 0:
                if "memory" in key_opt.lower():
                    # For memory, lower is better
                    speedup = base_val / opt_val
                    speedup_str = f"{speedup:.2f}x less"
                else:
                    # For time, lower is better
                    speedup = base_val / opt_val
                    speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{name:<30} {base_val:>11.3f}{unit} {opt_val:>11.3f}{unit} {speedup_str:>10}")

        # Overall assessment
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)

        # Calculate overall speedup for generation tasks
        if baseline.get("seq_batch_time") and optimized.get("batch_time"):
            overall_speedup = baseline["seq_batch_time"] / optimized["batch_time"]
            print(f"\n✓ Overall batch processing speedup: {overall_speedup:.2f}x faster")

        if baseline.get("memory_allocated_gb") and optimized.get("memory_allocated_gb"):
            memory_reduction = (baseline["memory_allocated_gb"] - optimized["memory_allocated_gb"])
            if memory_reduction > 0:
                print(f"✓ Memory usage reduced by {memory_reduction:.1f}GB")

        # GPU utilization estimate
        if optimized.get("batch_per_prompt") and baseline.get("seq_batch_per_prompt"):
            efficiency_gain = (baseline["seq_batch_per_prompt"] - optimized["batch_per_prompt"]) / baseline["seq_batch_per_prompt"] * 100
            print(f"✓ Per-prompt efficiency improved by {efficiency_gain:.1f}%")

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("\n" + "="*60)
        print("GPU PERFORMANCE BENCHMARK SUITE")
        print("="*60)

        # Display system info
        gpu_info = self.get_gpu_info()
        print("\nSystem Information:")
        print(f"  GPU: {gpu_info.get('gpu_name', 'N/A')}")
        print(f"  VRAM: {gpu_info.get('gpu_memory_gb', 0):.1f}GB")
        print(f"  CUDA: {gpu_info.get('cuda_version', 'N/A')}")
        print(f"  PyTorch: {gpu_info.get('pytorch_version', 'N/A')}")
        print(f"  BF16 Support: {gpu_info.get('bf16_supported', False)}")

        # Run benchmarks
        baseline_results = self.benchmark_baseline()
        optimized_results = self.benchmark_optimized()

        # Compare results
        self.compare_results(baseline_results, optimized_results)

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": gpu_info,
            "baseline": baseline_results,
            "optimized": optimized_results
        }

        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU performance optimizations")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark only")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline benchmark")
    parser.add_argument("--optimized-only", action="store_true", help="Run only optimized benchmark")
    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    if args.baseline_only:
        benchmark.benchmark_baseline()
    elif args.optimized_only:
        benchmark.benchmark_optimized()
    else:
        benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()