#!/usr/bin/env python3
"""Analyze labeled SFT dataset for strategy statistics and personality associations."""

import json
from collections import defaultdict
from pathlib import Path
import sys

def analyze_dataset(filepath):
    """Analyze SFT dataset for strategy statistics."""

    # Initialize counters
    strategy_stats = defaultdict(lambda: {
        'count': 0,
        'deals': 0,
        'no_deals': 0,
        'personalities': defaultdict(int)
    })

    total_entries = 0

    # Read and process dataset
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                total_entries += 1
                data = json.loads(line)

                # Extract relevant fields
                strategy = data['system']['key_strategy']
                personality = data['system']['assistant_style']
                deal_closed = data['meta']['deal_closed'] == 'true'

                # Update statistics
                stats = strategy_stats[strategy]
                stats['count'] += 1

                if deal_closed:
                    stats['deals'] += 1
                else:
                    stats['no_deals'] += 1

                stats['personalities'][personality] += 1

    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Calculate and display results
    print(f"\n{'='*70}")
    print(f"SFT Dataset Analysis - Total Entries: {total_entries}")
    print(f"{'='*70}\n")

    # Sort strategies by count
    sorted_strategies = sorted(strategy_stats.items(), key=lambda x: x[1]['count'], reverse=True)

    for strategy, stats in sorted_strategies:
        count = stats['count']
        deals = stats['deals']
        no_deals = stats['no_deals']
        deal_rate = (deals / count * 100) if count > 0 else 0
        dataset_percentage = (count / total_entries * 100) if total_entries > 0 else 0

        print(f"Strategy: {strategy.upper()}")
        print(f"  Count: {count} ({dataset_percentage:.1f}% of dataset)")
        print(f"  Deals: {deals} | No Deals: {no_deals}")
        print(f"  Deal Rate: {deal_rate:.1f}%")

        # Show top personalities for this strategy
        top_personalities = sorted(stats['personalities'].items(),
                                 key=lambda x: x[1], reverse=True)[:3]

        print(f"  Top Personalities:")
        for personality, p_count in top_personalities:
            p_percentage = (p_count / count * 100) if count > 0 else 0
            print(f"    - {personality}: {p_count} ({p_percentage:.1f}%)")
        print()

    # Summary statistics
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_deals = sum(s['deals'] for s in strategy_stats.values())
    overall_deal_rate = (total_deals / total_entries * 100) if total_entries > 0 else 0

    print(f"Total Strategies: {len(strategy_stats)}")
    print(f"Overall Deal Rate: {overall_deal_rate:.1f}%")
    print(f"Total Deals: {total_deals}")
    print(f"Total No Deals: {total_entries - total_deals}")

if __name__ == "__main__":
    # Default file path
    default_file = "data/runs/bulk_data/training_data/dataset_20251026_114958_sft_labeled.jsonl"

    # Use command line argument if provided, otherwise use default
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_file

    if not Path(filepath).exists():
        print(f"Error: File not found - {filepath}")
        sys.exit(1)

    analyze_dataset(filepath)