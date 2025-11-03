#!/usr/bin/env python3
"""
Script to track personality pairs for each seed across all JSONL dataset files.
Flags any seeds that have duplicate personality pairs and reports which files they're in.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set


def load_jsonl_file(filepath: Path) -> List[Dict]:
    """Load a JSONL file and return list of negotiation records."""
    negotiations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                negotiations.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {filepath}: {e}")
    return negotiations


def analyze_seed_personalities(datasets_dir: Path) -> None:
    """
    Analyze all JSONL files in the datasets directory for duplicate personality pairs per seed.

    Args:
        datasets_dir: Path to the directory containing JSONL dataset files
    """
    # Structure: seed_id -> {(buyer_personality, seller_personality): [(filename, line_number)]}
    seed_personality_map: Dict[int, Dict[Tuple[str, str], List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))

    # Track all unique personality pairs per seed
    seed_unique_pairs: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)

    # Get all JSONL files in the directory
    jsonl_files = sorted(datasets_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {datasets_dir}")
        return

    print(f"Analyzing {len(jsonl_files)} JSONL files in {datasets_dir}\n")

    # Process each file
    total_negotiations = 0
    for filepath in jsonl_files:
        filename = filepath.name
        print(f"Processing: {filename}")

        negotiations = load_jsonl_file(filepath)
        total_negotiations += len(negotiations)

        for line_num, negotiation in enumerate(negotiations, 1):
            seed_id = negotiation.get('seed_id')
            buyer_personality = negotiation.get('buyer_personality')
            seller_personality = negotiation.get('seller_personality')

            if seed_id is None or buyer_personality is None or seller_personality is None:
                print(f"  Warning: Missing data in {filename} line {line_num}")
                continue

            personality_pair = (buyer_personality, seller_personality)
            seed_personality_map[seed_id][personality_pair].append((filename, line_num))
            seed_unique_pairs[seed_id].add(personality_pair)

    print(f"\nTotal negotiations analyzed: {total_negotiations}")
    print(f"Unique seeds found: {len(seed_personality_map)}")

    # Find and report duplicates
    duplicates_found = False
    duplicate_count = 0

    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS")
    print("=" * 80)

    for seed_id in sorted(seed_personality_map.keys()):
        personality_pairs = seed_personality_map[seed_id]

        for personality_pair, occurrences in personality_pairs.items():
            if len(occurrences) > 1:
                duplicates_found = True
                duplicate_count += 1

                buyer_personality, seller_personality = personality_pair
                print(f"\n⚠️  DUPLICATE FOUND for Seed {seed_id}:")
                print(f"   Personality Pair: Buyer='{buyer_personality}', Seller='{seller_personality}'")
                print(f"   Found {len(occurrences)} occurrences:")

                for filename, line_num in occurrences:
                    print(f"     - {filename}, line {line_num}")

    if not duplicates_found:
        print("\n✅ No duplicate personality pairs found for any seed!")
    else:
        print(f"\n⚠️  Total duplicate personality pairs found: {duplicate_count}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Seeds with most unique personality pairs
    seeds_by_pair_count = sorted(
        [(seed_id, len(pairs)) for seed_id, pairs in seed_unique_pairs.items()],
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nTop 10 seeds by number of unique personality pairs:")
    for i, (seed_id, pair_count) in enumerate(seeds_by_pair_count[:10], 1):
        print(f"  {i}. Seed {seed_id}: {pair_count} unique personality pairs")

    # Distribution of personality pair counts per seed
    pair_count_distribution = defaultdict(int)
    for seed_id, pairs in seed_unique_pairs.items():
        pair_count_distribution[len(pairs)] += 1

    print(f"\nDistribution of personality pair counts per seed:")
    for pair_count in sorted(pair_count_distribution.keys()):
        seed_count = pair_count_distribution[pair_count]
        print(f"  {pair_count} personality pair(s): {seed_count} seed(s)")

    # Most common personality pairs across all seeds
    personality_pair_counts = defaultdict(int)
    for seed_id, pairs in seed_personality_map.items():
        for pair in pairs.keys():
            personality_pair_counts[pair] += 1

    most_common_pairs = sorted(
        personality_pair_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\nTop 10 most common personality pairs across all seeds:")
    for i, ((buyer, seller), count) in enumerate(most_common_pairs, 1):
        print(f"  {i}. Buyer='{buyer}', Seller='{seller}': {count} seeds")


def main():
    """Main function to run the analysis."""
    # Set the path to the datasets directory
    datasets_dir = Path("data/runs/datasets_raw")

    if not datasets_dir.exists():
        print(f"Error: Directory {datasets_dir} does not exist!")
        return

    if not datasets_dir.is_dir():
        print(f"Error: {datasets_dir} is not a directory!")
        return

    analyze_seed_personalities(datasets_dir)


if __name__ == "__main__":
    main()