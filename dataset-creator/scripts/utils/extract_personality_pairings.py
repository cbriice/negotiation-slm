#!/usr/bin/env python
"""
extract_personality_pairings.py

Analyzes JSONL files in data/runs/bulk_data/real_training_data to extract
all existing personality pairings for each seed ID.

Outputs a JSON file mapping seed IDs to lists of existing personality pairings.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prompts.personality_deck import (
    StraightToPoint, Desperate, Skeptic, Bluffer,
    Confrontational, Haggler, WeakEmotional, Rationalizer,
    StoneWall, Opportunist, HighHorse, Stubborn
)

def get_tag_to_personality_map() -> Dict[str, str]:
    """Create mapping from personality tags to personality class names."""
    personality_classes = [
        StraightToPoint(), Desperate(), Skeptic(), Bluffer(),
        Confrontational(), Haggler(), WeakEmotional(), Rationalizer(),
        StoneWall(), Opportunist(), HighHorse(), Stubborn()
    ]

    tag_to_name = {}
    for personality in personality_classes:
        tag = personality.tag
        class_name = personality.__class__.__name__
        tag_to_name[tag] = class_name

    return tag_to_name

def extract_seed_id(meta_data: dict) -> str:
    """Extract seed ID from metadata."""
    if 'seed_id' in meta_data:
        return str(meta_data['seed_id'])
    elif 'item' in meta_data:
        return meta_data['item']
    else:
        return "unknown"

def extract_personality_pairing(record: dict, tag_to_name: Dict[str, str]) -> Tuple[str, str, str]:
    """
    Extract personality pairing and seed ID from a JSONL record.

    Returns:
        Tuple of (seed_id, assistant_personality, opponent_personality)
    """
    try:
        assistant_tag = record.get('system', {}).get('assistant_style', 'unknown')
        opponent_tag = record.get('meta', {}).get('opponent_style', 'unknown')
        seed_id = extract_seed_id(record.get('meta', {}))

        assistant_personality = tag_to_name.get(assistant_tag, assistant_tag)
        opponent_personality = tag_to_name.get(opponent_tag, opponent_tag)

        return seed_id, assistant_personality, opponent_personality
    except Exception as e:
        print(f"Error extracting pairing: {e}")
        return "unknown", "unknown", "unknown"

def process_jsonl_file(filepath: Path, tag_to_name: Dict[str, str]) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Process a single JSONL file and extract all personality pairings per seed.

    Returns:
        Dictionary mapping seed IDs to sets of (assistant, opponent) personality tuples
    """
    seed_pairings = defaultdict(set)

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                seed_id, assistant_personality, opponent_personality = extract_personality_pairing(record, tag_to_name)

                if seed_id != "unknown":
                    seed_pairings[seed_id].add((assistant_personality, opponent_personality))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at {filepath}:{line_num} - {e}")
            except Exception as e:
                print(f"Unexpected error at {filepath}:{line_num} - {e}")

    return seed_pairings

def main():
    """Main function to extract all personality pairings from training data."""

    # Get tag to personality name mapping
    tag_to_name = get_tag_to_personality_map()
    print(f"Loaded {len(tag_to_name)} personality mappings:")
    for tag, name in tag_to_name.items():
        print(f"  {tag} -> {name}")
    print()

    # Define data directory
    data_dir = Path("data/runs/bulk_data/real_training_data")
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    # Process all JSONL files
    all_seed_pairings = defaultdict(set)
    jsonl_files = list(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {data_dir}")
        return

    print(f"Processing {len(jsonl_files)} JSONL files...")

    for filepath in jsonl_files:
        print(f"  Processing {filepath.name}...")
        file_pairings = process_jsonl_file(filepath, tag_to_name)

        # Merge pairings
        for seed_id, pairings in file_pairings.items():
            all_seed_pairings[seed_id].update(pairings)

    # Convert sets to sorted lists for JSON serialization
    output_data = {}
    for seed_id, pairings in all_seed_pairings.items():
        output_data[seed_id] = sorted([
            {"assistant": assistant, "opponent": opponent}
            for assistant, opponent in pairings
        ], key=lambda x: (x["assistant"], x["opponent"]))

    # Save to output file
    output_file = Path("data/existing_personality_pairings.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    print(f"\nExtracted personality pairings saved to {output_file}")

    # Print summary statistics
    total_seeds = len(output_data)
    total_pairings = sum(len(pairings) for pairings in output_data.values())
    avg_pairings_per_seed = total_pairings / total_seeds if total_seeds > 0 else 0

    print(f"\nSummary:")
    print(f"  Total seeds: {total_seeds}")
    print(f"  Total unique pairings: {total_pairings}")
    print(f"  Average pairings per seed: {avg_pairings_per_seed:.2f}")

    # Show sample of pairings for first few seeds
    print(f"\nSample pairings (first 3 seeds):")
    for i, (seed_id, pairings) in enumerate(sorted(output_data.items())[:3]):
        print(f"\n  Seed: {seed_id}")
        for pairing in pairings[:5]:  # Show max 5 pairings per seed
            print(f"    {pairing['assistant']} vs {pairing['opponent']}")
        if len(pairings) > 5:
            print(f"    ... and {len(pairings) - 5} more")

if __name__ == "__main__":
    main()