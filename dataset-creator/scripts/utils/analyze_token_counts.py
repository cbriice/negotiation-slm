#!/usr/bin/env python
"""
analyze_token_counts.py

Analyzes token counts in JSONL training data files, focusing on
the "messages" and "system" fields that will be used for training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import statistics
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use tiktoken for accurate token counting
try:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    USE_TIKTOKEN = True
except ImportError:
    print("Warning: tiktoken not installed. Using word-based approximation (1.3 tokens per word)")
    USE_TIKTOKEN = False

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or word approximation."""
    if USE_TIKTOKEN:
        return len(encoder.encode(text))
    else:
        # Approximate: average 1.3 tokens per word
        return int(len(text.split()) * 1.3)

def analyze_record(record: dict) -> Dict[str, int]:
    """
    Analyze token counts for a single record.

    Returns dict with:
    - messages_tokens: total tokens in messages array
    - system_tokens: tokens in system field (assistant_style, opponent_style)
    - total_training_tokens: sum of above (what gets trained on)
    """
    stats = {
        'messages_tokens': 0,
        'system_tokens': 0,
        'total_training_tokens': 0,
        'message_count': 0,
        'turn_count': 0
    }

    # Count tokens in messages
    if 'messages' in record:
        messages = record['messages']
        stats['message_count'] = len(messages)
        stats['turn_count'] = sum(1 for m in messages if m.get('role') in ['assistant', 'user'])

        for message in messages:
            # Convert message to training format
            role = message.get('role', '')
            content = message.get('content', '')

            # Format as it would appear in training
            if role == 'system':
                formatted = f"System: {content}"
            elif role == 'assistant':
                formatted = f"Assistant: {content}"
            elif role == 'user':
                formatted = f"User: {content}"
            else:
                formatted = content

            stats['messages_tokens'] += count_tokens(formatted)

    # Count tokens in system field (personality styles)
    if 'system' in record:
        system = record['system']
        assistant_style = system.get('assistant_style', '')
        opponent_style = system.get('opponent_style', '')

        # These would typically be used as system prompts or tags
        system_text = f"Assistant style: {assistant_style}, Opponent style: {opponent_style}"
        stats['system_tokens'] = count_tokens(system_text)

    stats['total_training_tokens'] = stats['messages_tokens'] + stats['system_tokens']

    return stats

def analyze_file(filepath: Path) -> Tuple[List[Dict], Dict]:
    """
    Analyze all records in a JSONL file.

    Returns:
    - List of per-record stats
    - Aggregated statistics
    """
    record_stats = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                stats = analyze_record(record)
                record_stats.append(stats)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at {filepath}:{line_num} - {e}")
            except Exception as e:
                print(f"Unexpected error at {filepath}:{line_num} - {e}")

    # Calculate aggregate stats
    if record_stats:
        aggregate = {
            'file': filepath.name,
            'record_count': len(record_stats),
            'avg_messages_tokens': statistics.mean([r['messages_tokens'] for r in record_stats]),
            'avg_system_tokens': statistics.mean([r['system_tokens'] for r in record_stats]),
            'avg_total_tokens': statistics.mean([r['total_training_tokens'] for r in record_stats]),
            'avg_message_count': statistics.mean([r['message_count'] for r in record_stats]),
            'avg_turn_count': statistics.mean([r['turn_count'] for r in record_stats]),
            'min_tokens': min([r['total_training_tokens'] for r in record_stats]),
            'max_tokens': max([r['total_training_tokens'] for r in record_stats]),
            'median_tokens': statistics.median([r['total_training_tokens'] for r in record_stats]),
            'stdev_tokens': statistics.stdev([r['total_training_tokens'] for r in record_stats]) if len(record_stats) > 1 else 0
        }
    else:
        aggregate = {'file': filepath.name, 'record_count': 0}

    return record_stats, aggregate

def main():
    """Main function to analyze token counts across all training data."""

    # Define data directory
    data_dir = Path("data/runs/bulk_data/real_training_data")
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    # Get all JSONL files
    jsonl_files = list(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    print(f"Analyzing {len(jsonl_files)} JSONL files...")
    print(f"Token counting method: {'tiktoken (accurate)' if USE_TIKTOKEN else 'word approximation'}")
    print("-" * 80)

    all_records = []
    file_stats = []

    # Analyze each file
    for filepath in jsonl_files:
        print(f"Processing {filepath.name}...")
        records, aggregate = analyze_file(filepath)
        all_records.extend(records)
        file_stats.append(aggregate)

    # Calculate overall statistics
    if all_records:
        print("\n" + "=" * 80)
        print("OVERALL DATASET STATISTICS")
        print("=" * 80)

        total_packets = len(all_records)
        all_total_tokens = [r['total_training_tokens'] for r in all_records]
        all_messages_tokens = [r['messages_tokens'] for r in all_records]
        all_system_tokens = [r['system_tokens'] for r in all_records]
        all_turn_counts = [r['turn_count'] for r in all_records]

        print(f"\nTotal dataset packets: {total_packets:,}")
        print(f"Total files analyzed: {len(file_stats)}")

        print(f"\n--- TOKEN COUNTS (Training Data Only) ---")
        print(f"Average total tokens per packet: {statistics.mean(all_total_tokens):.1f}")
        print(f"  - Messages tokens: {statistics.mean(all_messages_tokens):.1f}")
        print(f"  - System tokens: {statistics.mean(all_system_tokens):.1f}")

        print(f"\nMedian total tokens: {statistics.median(all_total_tokens):.1f}")
        print(f"Min tokens: {min(all_total_tokens):,}")
        print(f"Max tokens: {max(all_total_tokens):,}")
        print(f"Std deviation: {statistics.stdev(all_total_tokens):.1f}")

        # Distribution analysis
        print(f"\n--- TOKEN DISTRIBUTION ---")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        sorted_tokens = sorted(all_total_tokens)
        for p in percentiles:
            idx = int(len(sorted_tokens) * p / 100)
            print(f"{p}th percentile: {sorted_tokens[idx]:,} tokens")

        print(f"\n--- CONVERSATION STATISTICS ---")
        print(f"Average turns per conversation: {statistics.mean(all_turn_counts):.1f}")
        print(f"Min turns: {min(all_turn_counts)}")
        print(f"Max turns: {max(all_turn_counts)}")

        # Token buckets
        print(f"\n--- TOKEN BUCKETS ---")
        buckets = [
            (0, 500),
            (500, 1000),
            (1000, 1500),
            (1500, 2000),
            (2000, 3000),
            (3000, 5000),
            (5000, float('inf'))
        ]

        for low, high in buckets:
            if high == float('inf'):
                count = sum(1 for t in all_total_tokens if t >= low)
                pct = (count / total_packets) * 100
                print(f"{low}+ tokens: {count:,} packets ({pct:.1f}%)")
            else:
                count = sum(1 for t in all_total_tokens if low <= t < high)
                pct = (count / total_packets) * 100
                print(f"{low}-{high} tokens: {count:,} packets ({pct:.1f}%)")

        # File-level summary
        print(f"\n--- PER-FILE AVERAGES ---")
        for stat in sorted(file_stats, key=lambda x: x['avg_total_tokens'], reverse=True)[:10]:
            print(f"{stat['file']}: {stat['avg_total_tokens']:.1f} tokens avg ({stat['record_count']} records)")

        # Total dataset size estimate
        total_dataset_tokens = sum(all_total_tokens)
        print(f"\n--- TOTAL DATASET SIZE ---")
        print(f"Total training tokens: {total_dataset_tokens:,}")
        print(f"Estimated size: {total_dataset_tokens / 1_000_000:.2f}M tokens")

        # Save detailed stats to file
        output_file = Path("data/token_analysis_report.json")
        report = {
            'summary': {
                'total_packets': total_packets,
                'total_files': len(file_stats),
                'total_tokens': total_dataset_tokens,
                'avg_tokens_per_packet': statistics.mean(all_total_tokens),
                'median_tokens': statistics.median(all_total_tokens),
                'min_tokens': min(all_total_tokens),
                'max_tokens': max(all_total_tokens),
                'stdev_tokens': statistics.stdev(all_total_tokens)
            },
            'file_stats': file_stats,
            'token_counting_method': 'tiktoken' if USE_TIKTOKEN else 'word_approximation'
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {output_file}")

if __name__ == "__main__":
    main()