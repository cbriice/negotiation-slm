#!/usr/bin/env python3
"""
Script to add difficulty_bin field to SFT training packets based on difficulty value.
Difficulty bins (output as "diff_N"):
- diff_1: 0.0-0.1
- diff_2: 0.1-0.2
- ...
- diff_10: 0.9-1.0
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any

def calculate_difficulty_bin(difficulty: float) -> int:
    """
    Calculate the difficulty bin based on the difficulty value.

    Args:
        difficulty: Float value between 0.0 and 1.0

    Returns:
        Difficulty bin (1-10)
    """
    # Handle edge case where difficulty is exactly 1.0
    if difficulty == 1.0:
        return 10

    # Calculate bin: floor(difficulty * 10) + 1
    # This gives us:
    # 0.0-0.099... -> bin 1
    # 0.1-0.199... -> bin 2
    # etc.
    bin_value = int(difficulty * 10) + 1

    # Ensure bin is within valid range (1-10)
    return min(10, max(1, bin_value))

def process_jsonl_file(input_path: str, output_path: str = None) -> int:
    """
    Process a JSONL file and add difficulty_bin to each packet's system field.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (optional, defaults to adding _with_bins suffix)

    Returns:
        Number of packets processed
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_with_bins{input_path.suffix}"
    else:
        output_path = Path(output_path)

    packets_processed = 0
    packets_skipped = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON packet
                packet = json.loads(line.strip())

                # Check if packet has required structure
                if 'system' not in packet:
                    print(f"Warning: Line {line_num} missing 'system' field, skipping")
                    packets_skipped += 1
                    outfile.write(line)  # Write unchanged
                    continue

                if 'difficulty' not in packet['system']:
                    print(f"Warning: Line {line_num} missing 'difficulty' field in system, skipping")
                    packets_skipped += 1
                    outfile.write(line)  # Write unchanged
                    continue

                # Calculate and add difficulty_bin
                difficulty = packet['system']['difficulty']
                difficulty_bin = calculate_difficulty_bin(difficulty)
                packet['system']['difficulty_bin'] = f"diff_{difficulty_bin}"

                # Write updated packet
                outfile.write(json.dumps(packet) + '\n')
                packets_processed += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                packets_skipped += 1
                outfile.write(line)  # Write unchanged
            except Exception as e:
                print(f"Unexpected error at line {line_num}: {e}")
                packets_skipped += 1
                outfile.write(line)  # Write unchanged

    print(f"Processing complete:")
    print(f"  - Packets processed: {packets_processed}")
    print(f"  - Packets skipped: {packets_skipped}")
    print(f"  - Output written to: {output_path}")

    return packets_processed

def process_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Process all JSONL files in a directory.

    Args:
        input_dir: Directory containing JSONL files
        output_dir: Output directory (optional, defaults to same as input)
    """
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to process")

    total_processed = 0
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")
        output_path = output_dir / f"{jsonl_file.stem}_with_bins{jsonl_file.suffix}"
        processed = process_jsonl_file(jsonl_file, output_path)
        total_processed += processed

    print(f"\n{'='*50}")
    print(f"Total packets processed across all files: {total_processed}")

def main():
    parser = argparse.ArgumentParser(
        description="Add difficulty_bin field to SFT training packets based on difficulty value"
    )
    parser.add_argument(
        "input",
        nargs='?',  # Make input optional when using --test
        help="Input JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory (defaults to input with _with_bins suffix)",
        default=None
    )
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Run test to verify bin calculation logic"
    )

    args = parser.parse_args()

    if args.test:
        # Run tests
        test_cases = [
            (0.0, 1),
            (0.05, 1),
            (0.099, 1),
            (0.1, 2),
            (0.15, 2),
            (0.2, 3),
            (0.21, 3),
            (0.3, 4),
            (0.38, 4),
            (0.4, 5),
            (0.5, 6),
            (0.6, 7),
            (0.7, 8),
            (0.8, 9),
            (0.9, 10),
            (0.92, 10),
            (0.99, 10),
            (1.0, 10),
        ]

        print("Testing difficulty bin calculation:")
        print("-" * 40)
        for difficulty, expected_bin in test_cases:
            calculated_bin = calculate_difficulty_bin(difficulty)
            status = "✓" if calculated_bin == expected_bin else "✗"
            print(f"Difficulty {difficulty:4.2f} -> diff_{calculated_bin} {status} (expected: diff_{expected_bin})")

        return

    if not args.input:
        parser.error("Input file or directory is required when not in test mode")

    input_path = Path(args.input)

    if input_path.is_dir():
        process_directory(args.input, args.output)
    elif input_path.is_file():
        process_jsonl_file(args.input, args.output)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        exit(1)

if __name__ == "__main__":
    main()