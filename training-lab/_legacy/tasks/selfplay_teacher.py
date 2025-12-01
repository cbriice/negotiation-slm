"""
Teacher for self-play negotiation scenarios.
Provides state seeds and scenarios optimized for self-play training.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional
from pathlib import Path

from tasks.negotiation_teacher import NegotiationJSONTeacher

# Set up logging
logger = logging.getLogger(__name__)


class SelfPlayTeacher(NegotiationJSONTeacher):
    """
    Specialized teacher for self-play scenarios.
    
    Focuses on:
    - Balanced state seeds for both sides
    - Diverse negotiation scenarios
    - Regression control with maintained seed subsets
    - Minimal initial prompts to encourage exploration
    """
    
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        """Add command-line arguments for self-play teacher."""
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Self-Play Teacher Arguments')
        agent.add_argument(
            '--selfplay-seed-file',
            type=str,
            default='state_seeds.jsonl',
            help='File containing state seeds for self-play'
        )
        agent.add_argument(
            '--regression-seed-ratio',
            type=float,
            default=0.1,
            help='Ratio of seeds to maintain for regression testing'
        )
        agent.add_argument(
            '--seed-mutation-rate',
            type=float,
            default=0.0,
            help='Rate at which to apply mutations to seeds'
        )
        agent.add_argument(
            '--opening-prompt-style',
            type=str,
            default='minimal',
            choices=['minimal', 'guided', 'structured'],
            help='Style of opening prompts for self-play'
        )
        agent.add_argument(
            '--balanced-sides',
            type=bool,
            default=True,
            help='Ensure balanced representation of buyer/seller perspectives'
        )
        return parser
    
    def __init__(self, opt, shared=None):
        """Initialize the self-play teacher."""
        # Set default data directory for self-play
        if 'negotiation_data_dir' not in opt:
            opt['negotiation_data_dir'] = 'data/selfplay'
        if 'negotiation_data_file' not in opt:
            opt['negotiation_data_file'] = 'selfplay_episodes.jsonl'
        
        # Initialize parent
        super().__init__(opt, shared)
        
        # Self-play specific configuration
        self.seed_file = opt.get('selfplay_seed_file', 'state_seeds.jsonl')
        self.regression_ratio = opt.get('regression_seed_ratio', 0.1)
        self.mutation_rate = opt.get('seed_mutation_rate', 0.0)
        self.opening_style = opt.get('opening_prompt_style', 'minimal')
        self.balanced_sides = opt.get('balanced_sides', True)
        
        # Load or initialize seeds
        if shared:
            self.state_seeds = shared.get('state_seeds', [])
            self.regression_seeds = shared.get('regression_seeds', [])
        else:
            self._load_state_seeds()
            self._select_regression_seeds()
    
    def share(self):
        """Share data between teacher instances."""
        shared = super().share()
        shared['state_seeds'] = self.state_seeds
        shared['regression_seeds'] = self.regression_seeds
        return shared
    
    def _load_state_seeds(self):
        """Load state seeds from file."""
        seed_path = self.data_dir / self.seed_file
        self.state_seeds = []
        
        if seed_path.exists():
            with open(seed_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            seed = json.loads(line)
                            self.state_seeds.append(seed)
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Loaded {len(self.state_seeds)} state seeds")
        else:
            # Generate default seeds if file doesn't exist
            self.state_seeds = self._generate_default_seeds()
            self._save_state_seeds(seed_path)
    
    def _generate_default_seeds(self) -> List[Dict[str, Any]]:
        """Generate default state seeds for self-play."""
        seeds = []
        
        # Categories and items for diverse scenarios
        scenarios = [
            {
                'category': 'electronics',
                'item': {'name': 'Laptop', 'brand': 'Dell', 'model': 'XPS 13', 'year': '2023', 
                        'condition': 'like new', 'list_price': 1200},
                'buyer': {'max_price': 1000, 'target_price': 850},
                'seller': {'min_price': 900}
            },
            {
                'category': 'vehicles',
                'item': {'name': 'Mountain Bike', 'brand': 'Trek', 'model': 'Marlin 7', 
                        'year': '2022', 'condition': 'good', 'list_price': 800},
                'buyer': {'max_price': 650, 'target_price': 550},
                'seller': {'min_price': 600}
            },
            {
                'category': 'furniture',
                'item': {'name': 'Office Chair', 'brand': 'Herman Miller', 'model': 'Aeron',
                        'condition': 'used', 'list_price': 500},
                'buyer': {'max_price': 400, 'target_price': 350},
                'seller': {'min_price': 375}
            },
            {
                'category': 'collectibles',
                'item': {'name': 'Vintage Watch', 'brand': 'Omega', 'model': 'Speedmaster',
                        'year': '1969', 'condition': 'excellent', 'list_price': 5000},
                'buyer': {'max_price': 4500, 'target_price': 4000},
                'seller': {'min_price': 4200}
            },
            {
                'category': 'appliances',
                'item': {'name': 'Refrigerator', 'brand': 'Samsung', 'model': 'French Door',
                        'condition': 'good', 'list_price': 1500},
                'buyer': {'max_price': 1200, 'target_price': 1000},
                'seller': {'min_price': 1100}
            }
        ]
        
        # Generate seeds with variations
        for scenario in scenarios:
            # Create base seed
            seed = {
                'category': scenario['category'],
                'item': scenario['item'],
                'buyer_state': {
                    'role': 'buyer',
                    'max_price': scenario['buyer']['max_price'],
                    'target_price': scenario['buyer']['target_price'],
                    'preferences': {},
                    'logistics': {
                        'region': random.choice(['north', 'south', 'east', 'west']),
                        'meet_ok': random.choice([True, False]),
                        'ship_ok': True,
                        'return_days': random.choice([0, 7, 14, 30])
                    }
                },
                'seller_state': {
                    'role': 'seller',
                    'min_price': scenario['seller']['min_price'],
                    'item_issues': [],
                    'logistics': {
                        'region': random.choice(['north', 'south', 'east', 'west']),
                        'meet_ok': random.choice([True, False]),
                        'ship_ok': True,
                        'return_days': random.choice([0, 7, 14])
                    }
                },
                'seed_metadata': {
                    'difficulty': random.choice(['easy', 'medium', 'hard']),
                    'time_pressure': random.choice(['none', 'low', 'high'])
                }
            }
            
            seeds.append(seed)
            
            # Create variations with different price gaps
            for gap_multiplier in [0.8, 1.2]:
                variant = json.loads(json.dumps(seed))  # Deep copy
                variant['buyer_state']['max_price'] = int(scenario['buyer']['max_price'] * gap_multiplier)
                variant['buyer_state']['target_price'] = int(scenario['buyer']['target_price'] * gap_multiplier)
                variant['seed_metadata']['difficulty'] = 'hard' if gap_multiplier < 1 else 'easy'
                seeds.append(variant)
        
        return seeds
    
    def _save_state_seeds(self, filepath: Path):
        """Save state seeds to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for seed in self.state_seeds:
                f.write(json.dumps(seed) + '\n')
        logger.info(f"Saved {len(self.state_seeds)} state seeds to {filepath}")
    
    def _select_regression_seeds(self):
        """Select subset of seeds for regression testing."""
        if not self.state_seeds:
            self.regression_seeds = []
            return
        
        num_regression = max(1, int(len(self.state_seeds) * self.regression_ratio))
        self.regression_seeds = random.sample(self.state_seeds, num_regression)
        logger.info(f"Selected {len(self.regression_seeds)} seeds for regression testing")
    
    def _mutate_seed(self, seed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutations to a seed."""
        mutated = json.loads(json.dumps(seed))  # Deep copy
        
        mutations = []
        
        # Price jitter (±10%)
        if random.random() < 0.3:
            jitter = random.uniform(0.9, 1.1)
            if 'buyer_state' in mutated:
                mutated['buyer_state']['max_price'] = int(mutated['buyer_state']['max_price'] * jitter)
                mutated['buyer_state']['target_price'] = int(mutated['buyer_state']['target_price'] * jitter)
            if 'seller_state' in mutated:
                mutated['seller_state']['min_price'] = int(mutated['seller_state']['min_price'] * jitter)
            mutations.append('price_jitter')
        
        # Add random item issues
        if random.random() < 0.2 and 'seller_state' in mutated:
            issues = ['minor scratches', 'needs cleaning', 'missing manual', 'cosmetic wear']
            mutated['seller_state']['item_issues'] = random.sample(issues, random.randint(1, 2))
            mutations.append('item_issues')
        
        # Change logistics preferences
        if random.random() < 0.25:
            for state_key in ['buyer_state', 'seller_state']:
                if state_key in mutated and 'logistics' in mutated[state_key]:
                    mutated[state_key]['logistics']['meet_ok'] = random.choice([True, False])
                    mutated[state_key]['logistics']['ship_ok'] = random.choice([True, False])
            mutations.append('logistics_change')
        
        # Add mutation metadata
        if 'seed_metadata' not in mutated:
            mutated['seed_metadata'] = {}
        mutated['seed_metadata']['mutations'] = mutations
        
        return mutated
    
    def _generate_opening_prompt(self, state: Dict[str, Any], side: str) -> str:
        """Generate opening prompt based on style."""
        if self.opening_style == 'minimal':
            # Very minimal prompt to encourage exploration
            if side == 'buyer':
                return f"Hi, I'm interested in the {state['item']['name']}."
            else:
                return f"Hello! Thanks for your interest in the {state['item']['name']}."
        
        elif self.opening_style == 'guided':
            # Slightly more structured opening
            if side == 'buyer':
                price = state['item'].get('list_price', 'your asking price')
                return f"Hi, I saw your listing for the {state['item']['name']} at ${price}. I'm interested but was hoping we could discuss the price."
            else:
                condition = state['item'].get('condition', 'great condition')
                return f"Hello! Yes, the {state['item']['name']} is available. It's in {condition}. What would you like to know?"
        
        elif self.opening_style == 'structured':
            # More structured opening with clear intent
            if side == 'buyer':
                target = state['buyer_state'].get('target_price')
                return f"[PROPOSE] Hi, I'm interested in your {state['item']['name']}. Based on the condition and market prices, would you consider ${target}?"
            else:
                min_price = state['seller_state'].get('min_price')
                return f"[PROPOSE] Thanks for your interest! The {state['item']['name']} is in {state['item'].get('condition', 'good')} condition. I'm asking ${min_price} for it."
        
        return ""
    
    def setup_data(self, datafile=None):
        """Set up self-play training data."""
        # If we have existing episodes, use them
        if self.episodes:
            yield from super().setup_data(datafile)
            return
        
        # Otherwise, generate episodes from seeds
        for seed_idx, seed in enumerate(self.state_seeds):
            # Apply mutation if configured
            if self.mutation_rate > 0 and random.random() < self.mutation_rate:
                seed = self._mutate_seed(seed)
            
            # Generate episodes for both sides if balanced
            sides = ['buyer', 'seller'] if self.balanced_sides else [random.choice(['buyer', 'seller'])]
            
            for side in sides:
                # Create episode structure
                episode = {
                    'state': seed,
                    'dialogue': [],
                    'meta': {
                        'episode_id': f"selfplay_seed{seed_idx}_{side}",
                        'dialogue_id': f"selfplay_d{seed_idx}_{side}",
                        'side': side,
                        'seed_idx': seed_idx,
                        'is_regression': seed in self.regression_seeds,
                        'opening_style': self.opening_style
                    }
                }
                
                # Generate opening prompt
                opening = self._generate_opening_prompt(seed, side)
                
                # Format as observation
                obs = {
                    'text': self._format_state_context(seed, side),
                    'labels': [opening] if opening else None,
                    'episode_id': episode['meta']['episode_id'],
                    'dialogue_id': episode['meta']['dialogue_id'],
                    'side': side,
                    'episode_done': True  # Single turn for seeding
                }
                
                if obs['labels']:
                    yield obs, True
    
    def get_regression_seeds(self) -> List[Dict[str, Any]]:
        """Get seeds selected for regression testing."""
        return self.regression_seeds
    
    def log_selfplay_episode(self, dialogue: List[Dict[str, Any]], 
                            state: Dict[str, Any], 
                            metrics: Dict[str, Any]):
        """Log a completed self-play episode."""
        episode = {
            'state': state,
            'dialogue': dialogue,
            'meta': {
                'episode_id': f"selfplay_{len(self.episodes)}",
                'dialogue_id': f"selfplay_d{len(self.episodes)}",
                'timestamp': metrics.get('timestamp'),
                'model_version': metrics.get('model_version')
            },
            'metrics': metrics
        }
        
        self.episodes.append(episode)
        
        # Update episode metrics
        self.update_episode_metrics(episode['meta']['episode_id'], metrics)
        
        # Periodically save episodes
        if len(self.episodes) % 100 == 0:
            self.save_episodes()