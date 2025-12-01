"""
Load and manage evaluation seeds for negotiation testing.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional


class EvalSeedLoader:
    """Load and manage evaluation seeds."""

    def __init__(self, seed_file: Optional[str] = None):
        """
        Initialize seed loader.

        Args:
            seed_file: Path to seed file (JSON or JSONL)
        """
        # Default to converted seeds if available
        if seed_file is None:
            default_paths = [
                "data/negotiations/selfchat/seeds/eval_seeds_converted.jsonl",
                "data/negotiations/selfchat/seeds/eval_seeds.json",
            ]
            for path in default_paths:
                if Path(path).exists():
                    seed_file = path
                    break

        if seed_file is None:
            raise ValueError("No seed file found. Please provide a path.")

        self.seed_file = Path(seed_file)
        self.seeds = self._load_seeds()

    def _load_seeds(self) -> List[Dict[str, Any]]:
        """Load seeds from file."""
        seeds = []

        if self.seed_file.suffix == '.jsonl':
            # Load JSONL format
            with open(self.seed_file, 'r') as f:
                for line in f:
                    seed = json.loads(line.strip())
                    seeds.append(seed)
        else:
            # Load JSON format
            with open(self.seed_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    seeds = data
                else:
                    seeds = [data]

        # Convert old format to new if needed
        converted_seeds = []
        for seed in seeds:
            if 'buyer_state' in seed and 'seller_state' in seed:
                # Already in new format
                converted_seeds.append(seed)
            else:
                # Convert old format
                converted_seeds.append(self._convert_old_format(seed))

        return converted_seeds

    def _convert_old_format(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old seed format to new format."""
        converted = {
            'seed_id': seed.get('seed_id', random.randint(1, 10000)),
            'category': seed.get('category', 'general'),
            'item': {
                'name': seed.get('item', 'Unknown Item'),
                'condition': seed.get('condition', 'used'),
                'list_price': seed.get('seller_list', 1000),
            },
            'buyer_state': {
                'role': 'buyer',
                'max_price': seed.get('buyer_max', 900),
                'target_price': seed.get('buyer_max', 900) * 0.85,  # 85% of max as target
                'preferences': {},
                'logistics': {
                    'region': seed.get('locality', 'urban'),
                    'meet_ok': True,
                    'ship_ok': True,
                    'return_days': 14,
                }
            },
            'seller_state': {
                'role': 'seller',
                'min_price': seed.get('seller_min', 800),
                'target_price': seed.get('seller_list', 1000),
                'item_issues': [],
                'logistics': {
                    'region': seed.get('locality', 'urban'),
                    'meet_ok': True,
                    'ship_ok': True,
                    'return_days': 7,
                }
            }
        }

        # Add extras if available
        if 'extras_available' in seed:
            converted['seller_state']['extras'] = seed['extras_available']

        # Add context notes
        if 'notes' in seed:
            converted['notes'] = seed['notes']

        return converted

    def get_seed(self, seed_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a specific seed or random seed.

        Args:
            seed_id: Specific seed ID to retrieve

        Returns:
            Seed data
        """
        if seed_id is not None:
            # Find specific seed
            for seed in self.seeds:
                if seed.get('seed_id') == seed_id:
                    return seed
            # If not found, return random
            print(f"Seed {seed_id} not found, returning random seed")

        # Return random seed
        return random.choice(self.seeds)

    def get_seeds(self,
                  n: Optional[int] = None,
                  category: Optional[str] = None,
                  shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        Get multiple seeds.

        Args:
            n: Number of seeds to return (None for all)
            category: Filter by category
            shuffle: Whether to shuffle seeds

        Returns:
            List of seed data
        """
        seeds = self.seeds.copy()

        # Filter by category if specified
        if category:
            seeds = [s for s in seeds if s.get('category') == category]

        # Shuffle if requested
        if shuffle:
            random.shuffle(seeds)

        # Return requested number
        if n is not None:
            seeds = seeds[:n]

        return seeds

    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        categories = set()
        for seed in self.seeds:
            if 'category' in seed:
                categories.add(seed['category'])
        return sorted(list(categories))

    def stats(self) -> Dict[str, Any]:
        """Get statistics about loaded seeds."""
        stats = {
            'total_seeds': len(self.seeds),
            'categories': {}
        }

        # Count by category
        for seed in self.seeds:
            cat = seed.get('category', 'unknown')
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

        # Calculate price ranges
        buyer_max_prices = [s['buyer_state']['max_price'] for s in self.seeds
                           if 'buyer_state' in s and 'max_price' in s['buyer_state']]
        seller_min_prices = [s['seller_state']['min_price'] for s in self.seeds
                            if 'seller_state' in s and 'min_price' in s['seller_state']]

        if buyer_max_prices:
            stats['buyer_max_range'] = (min(buyer_max_prices), max(buyer_max_prices))
        if seller_min_prices:
            stats['seller_min_range'] = (min(seller_min_prices), max(seller_min_prices))

        return stats