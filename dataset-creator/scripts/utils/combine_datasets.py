#!/usr/bin/env python3
"""
Utility script to combine multiple simulation runs into a single dataset.

This script allows you to merge JSONL datasets from multiple run directories
into a single combined dataset file for training.
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def combine_runs(run_directories: list, output_file: str = None):
    """
    Combine multiple simulation runs into a single dataset file.

    Args:
        run_directories: List of run directory paths to combine
        output_file: Output file path (default: data/combined_dataset_<timestamp>.jsonl)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_file = os.path.join(project_root, "data", f"combined_dataset_{timestamp}.jsonl")

    combined_entries = []
    total_negotiations = 0

    for run_dir in run_directories:
        # Find the dataset JSONL file in the run directory
        jsonl_files = [f for f in os.listdir(run_dir) if f.startswith("dataset_") and f.endswith(".jsonl")]

        if not jsonl_files:
            print(f"Warning: No dataset file found in {run_dir}")
            continue

        dataset_file = os.path.join(run_dir, jsonl_files[0])

        # Read and combine the data
        with open(dataset_file, 'r') as f:
            for line in f:
                combined_entries.append(line.strip())
                total_negotiations += 1

    # Save combined dataset
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        for entry in combined_entries:
            f.write(entry + '\n')

    print(f"Combined {total_negotiations} negotiations from {len(run_directories)} runs")
    print(f"Saved to: {output_file}")

    return output_file


def find_run_directories(base_path: str = "data/runs/", pattern: str = "run_*"):
    """
    Find all run directories matching a pattern.

    Args:
        base_path: Base directory to search in
        pattern: Directory pattern to match

    Returns:
        List of matching directory paths
    """
    search_path = os.path.join(base_path, pattern)
    return sorted(glob(search_path))


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple negotiation simulation runs into a single dataset"
    )

    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Specific run directories to combine (if not provided, combines all runs)"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/runs/",
        help="Base directory containing run folders (default: data/runs/)"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="run_*",
        help="Pattern to match run directories (default: run_*)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for combined dataset (default: data/combined_dataset_<timestamp>.jsonl)"
    )

    parser.add_argument(
        "--latest",
        type=int,
        help="Combine only the N most recent runs"
    )

    args = parser.parse_args()

    # Determine which directories to combine
    if args.run_dirs:
        # Use specific directories provided
        run_directories = args.run_dirs
    else:
        # Find directories based on pattern
        run_directories = find_run_directories(args.base_dir, args.pattern)

        if args.latest:
            # Take only the most recent N runs
            run_directories = run_directories[-args.latest:]

    if not run_directories:
        print(f"No run directories found in {args.base_dir} matching pattern '{args.pattern}'")
        sys.exit(1)

    print(f"Found {len(run_directories)} run directories to combine:")
    for dir_path in run_directories:
        print(f"  - {dir_path}")

    print("\nCombining datasets...")

    # Combine the runs
    output_file = combine_runs(run_directories, args.output)

    print(f"\nCombination complete!")
    print(f"Combined dataset saved to: {output_file}")


if __name__ == "__main__":
    main()