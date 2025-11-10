#!/usr/bin/env python3
"""
Convert eval_seeds.json to selfchatworld-compatible format.
Transforms flat seed format into nested buyer_state/seller_state structure.
"""

import json
import random
from pathlib import Path


def convert_seed(old_seed):
    """Convert old seed format to selfchatworld format."""

    # Parse item name to extract details
    item_name = old_seed['item']

    # Build structured item dict
    item = {
        'name': item_name,
        'condition': old_seed.get('condition', 'used'),
        'list_price': old_seed.get('seller_list', 0)
    }

    # Calculate target price (typically ~15% below max)
    buyer_max = old_seed.get('buyer_max', 0)
    buyer_target = int(buyer_max * 0.85)

    # Build buyer state
    buyer_state = {
        'role': 'buyer',
        'max_price': buyer_max,
        'target_price': buyer_target,
        'preferences': {},
        'logistics': {
            'region': old_seed.get('locality', 'urban'),
            'meet_ok': random.choice([True, False]),
            'ship_ok': True,
            'return_days': random.choice([0, 7, 14, 30])
        }
    }

    # Build seller state
    seller_min = old_seed.get('seller_min', 0)

    seller_state = {
        'role': 'seller',
        'min_price': seller_min,
        'target_price': old_seed.get('seller_list', seller_min),
        'item_issues': [],
        'logistics': {
            'region': old_seed.get('locality', 'urban'),
            'meet_ok': random.choice([True, False]),
            'ship_ok': True,
            'return_days': random.choice([0, 7, 14])
        }
    }

    # Add extras as potential add-ons
    if old_seed.get('extras_available'):
        seller_state['extras_available'] = old_seed['extras_available']

    # Determine difficulty based on price gap
    price_gap = seller_min - buyer_max
    list_price = old_seed.get('seller_list', seller_min)
    gap_ratio = abs(price_gap) / list_price if list_price > 0 else 0

    if gap_ratio < 0.05:
        difficulty = 'easy'
    elif gap_ratio < 0.15:
        difficulty = 'medium'
    else:
        difficulty = 'hard'

    # Map context to metadata
    context = old_seed.get('context', {})
    timeline = context.get('timeline', 'neutral')

    # Build new seed
    new_seed = {
        'category': old_seed.get('category', 'general'),
        'item': item,
        'buyer_state': buyer_state,
        'seller_state': seller_state,
        'seed_metadata': {
            'seed_id': old_seed.get('seed_id'),
            'difficulty': difficulty,
            'time_pressure': timeline,
            'market_type': context.get('market_type', 'marketplace'),
            'communication_style': context.get('communication_style', 'regular'),
            'notes': old_seed.get('notes', '')
        }
    }

    # Add context fields if available
    if 'scarcity' in context:
        new_seed['seed_metadata']['scarcity'] = context['scarcity']
    if 'info_advantage' in context:
        new_seed['seed_metadata']['info_advantage'] = context['info_advantage']
    if 'competing_offer' in context:
        new_seed['seed_metadata']['competing_offer'] = context['competing_offer']

    return new_seed


def main():
    # Paths
    input_path = Path('data/negotiations/selfchat/seeds/eval_seeds.json')
    output_path = Path('data/negotiations/selfchat/seeds/eval_seeds_converted.jsonl')

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Load old seeds
    print(f"Loading seeds from {input_path}")
    with open(input_path, 'r') as f:
        old_seeds = json.load(f)

    print(f"Found {len(old_seeds)} seeds to convert")

    # Convert seeds
    converted_seeds = []
    for old_seed in old_seeds:
        try:
            new_seed = convert_seed(old_seed)
            converted_seeds.append(new_seed)
        except Exception as e:
            print(f"Warning: Failed to convert seed {old_seed.get('seed_id')}: {e}")

    print(f"Successfully converted {len(converted_seeds)} seeds")

    # Save as JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for seed in converted_seeds:
            f.write(json.dumps(seed) + '\n')

    print(f"Saved converted seeds to {output_path}")

    # Show sample
    if converted_seeds:
        print("\nSample converted seed:")
        print(json.dumps(converted_seeds[0], indent=2))


if __name__ == '__main__':
    main()
