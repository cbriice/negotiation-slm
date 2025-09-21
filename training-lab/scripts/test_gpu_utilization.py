#!/usr/bin/env python3
"""
Test script to verify GPU utilization improvements with parallel batch inference.
Compares single episode vs parallel episode processing.
"""

import time
import torch
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from parlai.core.params import ParlaiParser
from worlds.selfchat_world import SelfChatWorld


def monitor_gpu_utilization():
    """Monitor and return current GPU utilization stats."""
    if torch.cuda.is_available():
        # Get GPU memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB

        # Get GPU utilization (requires nvidia-ml-py)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory
            pynvml.nvmlShutdown()
        except:
            gpu_util = -1
            mem_util = -1

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'gpu_utilization': gpu_util,
            'memory_utilization': mem_util
        }
    return None


def test_single_episode_mode(num_episodes=10):
    """Test traditional single episode processing."""
    print("\n" + "="*60)
    print("TESTING: Single Episode Mode (Sequential)")
    print("="*60)

    parser = ParlaiParser()
    parser = SelfChatWorld.add_cmdline_args(parser)

    opt = parser.parse_args([
        '--num-episodes', str(num_episodes),
        '--parallel-episodes', '1',  # Single episode at a time
        '--use-batch-inference', 'False',
        '--output-dir', 'data/test_single/'
    ])

    world = SelfChatWorld(opt)

    # Monitor GPU during execution
    start_time = time.time()
    start_stats = monitor_gpu_utilization()
    print(f"Initial GPU stats: {start_stats}")

    max_gpu_util = 0
    samples_count = 0
    total_gpu_util = 0

    # Run episodes
    for i in range(num_episodes):
        world.parley()

        # Sample GPU utilization
        stats = monitor_gpu_utilization()
        if stats and stats['gpu_utilization'] > 0:
            max_gpu_util = max(max_gpu_util, stats['gpu_utilization'])
            total_gpu_util += stats['gpu_utilization']
            samples_count += 1

        if (i + 1) % 5 == 0:
            print(f"Completed {i+1}/{num_episodes} episodes")
            if stats:
                print(f"  Current GPU util: {stats['gpu_utilization']}%")

    duration = time.time() - start_time

    # Calculate metrics
    avg_gpu_util = total_gpu_util / samples_count if samples_count > 0 else 0
    episodes_per_second = num_episodes / duration

    print(f"\nResults for Single Episode Mode:")
    print(f"  Total time: {duration:.2f}s")
    print(f"  Episodes/second: {episodes_per_second:.2f}")
    print(f"  Max GPU utilization: {max_gpu_util}%")
    print(f"  Avg GPU utilization: {avg_gpu_util:.2f}%")

    world.shutdown()
    return {
        'mode': 'single',
        'duration': duration,
        'episodes_per_second': episodes_per_second,
        'max_gpu_util': max_gpu_util,
        'avg_gpu_util': avg_gpu_util
    }


def test_parallel_episode_mode(num_episodes=10, parallel_batch=4):
    """Test parallel episode processing with batch inference."""
    print("\n" + "="*60)
    print(f"TESTING: Parallel Episode Mode (Batch={parallel_batch})")
    print("="*60)

    parser = ParlaiParser()
    parser = SelfChatWorld.add_cmdline_args(parser)

    opt = parser.parse_args([
        '--num-episodes', str(num_episodes),
        '--parallel-episodes', str(parallel_batch),
        '--use-batch-inference', 'True',
        '--output-dir', 'data/test_parallel/'
    ])

    world = SelfChatWorld(opt)

    # Monitor GPU during execution
    start_time = time.time()
    start_stats = monitor_gpu_utilization()
    print(f"Initial GPU stats: {start_stats}")

    max_gpu_util = 0
    samples_count = 0
    total_gpu_util = 0

    # Run episodes (will be processed in batches)
    episodes_run = 0
    while episodes_run < num_episodes:
        world.parley()
        episodes_run = world.episode_count

        # Sample GPU utilization
        stats = monitor_gpu_utilization()
        if stats and stats['gpu_utilization'] > 0:
            max_gpu_util = max(max_gpu_util, stats['gpu_utilization'])
            total_gpu_util += stats['gpu_utilization']
            samples_count += 1

        if episodes_run > 0 and episodes_run % 5 == 0:
            print(f"Completed {episodes_run}/{num_episodes} episodes")
            if stats:
                print(f"  Current GPU util: {stats['gpu_utilization']}%")

    duration = time.time() - start_time

    # Calculate metrics
    avg_gpu_util = total_gpu_util / samples_count if samples_count > 0 else 0
    episodes_per_second = num_episodes / duration

    print(f"\nResults for Parallel Episode Mode:")
    print(f"  Total time: {duration:.2f}s")
    print(f"  Episodes/second: {episodes_per_second:.2f}")
    print(f"  Max GPU utilization: {max_gpu_util}%")
    print(f"  Avg GPU utilization: {avg_gpu_util:.2f}%")

    world.shutdown()
    return {
        'mode': f'parallel_{parallel_batch}',
        'duration': duration,
        'episodes_per_second': episodes_per_second,
        'max_gpu_util': max_gpu_util,
        'avg_gpu_util': avg_gpu_util
    }


def main():
    """Run GPU utilization tests."""
    parser = argparse.ArgumentParser(description='Test GPU utilization improvements')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8],
                       help='Batch sizes to test')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("GPU UTILIZATION TEST")
    print("="*60)
    print(f"Testing with {args.episodes} episodes")
    print(f"Batch sizes: {args.batch_sizes}")

    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Tests will run on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    results = []

    # Test each batch size
    for batch_size in args.batch_sizes:
        if batch_size == 1:
            result = test_single_episode_mode(args.episodes)
        else:
            result = test_parallel_episode_mode(args.episodes, batch_size)
        results.append(result)

        # Clear GPU cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)  # Cool down between tests

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    baseline = results[0] if results else None

    for result in results:
        speedup = result['episodes_per_second'] / baseline['episodes_per_second'] if baseline else 1.0
        gpu_improvement = result['avg_gpu_util'] - baseline['avg_gpu_util'] if baseline else 0

        print(f"\n{result['mode'].upper()} Mode:")
        print(f"  Episodes/sec: {result['episodes_per_second']:.2f} ({speedup:.2f}x speedup)")
        print(f"  Avg GPU util: {result['avg_gpu_util']:.1f}% ({gpu_improvement:+.1f}% change)")
        print(f"  Max GPU util: {result['max_gpu_util']:.0f}%")
        print(f"  Total time: {result['duration']:.2f}s")

    # Calculate overall improvement
    if len(results) > 1:
        best = max(results, key=lambda x: x['avg_gpu_util'])
        worst = min(results, key=lambda x: x['avg_gpu_util'])

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Best GPU utilization: {best['mode']} with {best['avg_gpu_util']:.1f}%")
        print(f"Worst GPU utilization: {worst['mode']} with {worst['avg_gpu_util']:.1f}%")
        print(f"GPU utilization improvement: {best['avg_gpu_util'] - worst['avg_gpu_util']:.1f}%")
        print(f"Speed improvement: {best['episodes_per_second'] / worst['episodes_per_second']:.2f}x")


if __name__ == "__main__":
    main()