#!/usr/bin/env python
"""
Test script to verify duplicate personality pairing prevention.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulation_runner import SimulationRunner

def test_duplicate_prevention():
    """Test that duplicate personality pairings are properly avoided."""

    # Create simulation runner with a specific seed file
    print("\n" + "="*60)
    print("TESTING DUPLICATE PERSONALITY PAIRING PREVENTION")
    print("="*60 + "\n")

    # Initialize runner
    runner = SimulationRunner(
        seed_directory="seeds/",
        model_name="Qwen/Qwen2.5-14B-Instruct",
        max_turns=5,  # Short for testing
        specific_seed_file="seeds/tech_seeds.json",
        same_personality_count=0  # No same personality
    )

    # Check what existing pairings were loaded
    print("\nExisting pairings loaded:")
    if runner.existing_pairings:
        for seed_id, pairs in list(runner.existing_pairings.items())[:3]:  # Show first 3
            unique_pairs = len(pairs) // 2  # Divide by 2 since we store both directions
            print(f"  Seed {seed_id}: {unique_pairs} existing pairs")
            # Show a sample
            shown_pairs = set()
            for pair in pairs:
                if pair not in shown_pairs and (pair[1], pair[0]) not in shown_pairs:
                    print(f"    - {pair[0]} vs {pair[1]}")
                    shown_pairs.add(pair)
                    if len(shown_pairs) >= 2:  # Show only 2 pairs per seed
                        break
    else:
        print("  No existing pairings file found")

    print("\n" + "-"*60)
    print("Testing personality selection for specific seeds...")
    print("-"*60 + "\n")

    # Test with a few seeds
    test_seeds = ["1001", "1002", "1003"]  # These should have existing pairings

    for seed_id in test_seeds:
        print(f"\n--- Testing seed {seed_id} ---")

        # Get seed with this ID
        seed = None
        for s in runner.seed_loader.all_seeds:
            if str(s.seed_id) == seed_id:
                seed = s
                break

        if not seed:
            print(f"  Seed {seed_id} not found in loaded seeds")
            continue

        # Check existing pairs
        existing = runner.existing_pairings.get(seed_id, set())
        print(f"  Existing pairs for this seed: {len(existing)//2}")

        # Try to select a new personality pair
        print("  Attempting to select new personality pair...")

        # Simulate the selection logic
        from prompts.personality_deck import PersonalityDeck
        deck = PersonalityDeck()

        attempts = 0
        max_attempts = 20
        found_new = False

        while attempts < max_attempts:
            buyer, seller = deck.select_two_personalities(avoid_same=True)
            pair_id = (buyer.__class__.__name__, seller.__class__.__name__)

            if pair_id not in existing:
                print(f"  ✓ Found NEW pair: {pair_id[0]} vs {pair_id[1]}")
                found_new = True
                break
            else:
                print(f"  ✗ Duplicate detected: {pair_id[0]} vs {pair_id[1]} (attempt {attempts+1})")

            attempts += 1

        if not found_new:
            print(f"  ⚠ Could not find new pair after {max_attempts} attempts")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_duplicate_prevention()