#!/usr/bin/env python3
"""
Test that converted eval seeds can be loaded by SelfPlayTeacher.
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.selfplay_teacher import SelfPlayTeacher


def test_load_eval_seeds():
    """Test loading converted eval seeds."""

    # Create minimal opt with required ParlAI fields
    opt = {
        'datatype': 'train',
        'task': 'selfplay',
        'negotiation_data_dir': 'data/negotiations/selfchat',
        'selfplay_seed_file': 'seeds/eval_seeds_converted.jsonl',
        'regression_seed_ratio': 0.1,
        'seed_mutation_rate': 0.0,
        'opening_prompt_style': 'minimal',
        'balanced_sides': True
    }

    # Initialize teacher
    print("Initializing SelfPlayTeacher with eval seeds...")
    teacher = SelfPlayTeacher(opt, shared=None)

    # Check loaded seeds
    print(f"Loaded {len(teacher.state_seeds)} state seeds")
    print(f"Loaded {len(teacher.regression_seeds)} regression seeds")

    if teacher.state_seeds:
        print("\nSample loaded seed:")
        sample = teacher.state_seeds[0]
        print(json.dumps(sample, indent=2))

        # Validate structure
        required_keys = ['category', 'item', 'buyer_state', 'seller_state']
        for key in required_keys:
            if key not in sample:
                print(f"ERROR: Missing required key: {key}")
                return False

        # Validate buyer_state
        buyer_required = ['role', 'max_price', 'target_price', 'logistics']
        for key in buyer_required:
            if key not in sample['buyer_state']:
                print(f"ERROR: Missing buyer_state key: {key}")
                return False

        # Validate seller_state
        seller_required = ['role', 'min_price', 'logistics']
        for key in seller_required:
            if key not in sample['seller_state']:
                print(f"ERROR: Missing seller_state key: {key}")
                return False

        print("\n✓ Seed structure validation passed!")
        return True
    else:
        print("ERROR: No seeds loaded!")
        return False


if __name__ == '__main__':
    success = test_load_eval_seeds()
    sys.exit(0 if success else 1)
