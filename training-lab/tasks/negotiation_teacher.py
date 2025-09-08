"""
Base teacher class for negotiation training using ParlAI.
Loads JSONL episodes and injects state seeds for training.
"""

import json
import logging
import os
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from parlai.core.teachers import FixedDialogTeacher
from parlai.core.message import Message

# Set up logging
logger = logging.getLogger(__name__)


class NegotiationJSONTeacher(FixedDialogTeacher):
    """
    Base teacher class for negotiation training.
    
    Loads JSONL episodes from data directory, parses state seeds,
    formats observations with negotiation context, and tracks metrics.
    """
    
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        """Add command-line arguments specific to negotiation teachers."""
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Negotiation Teacher Arguments')
        agent.add_argument(
            '--negotiation-data-file',
            type=str,
            default='episodes.jsonl',
            help='Name of the JSONL file containing negotiation episodes'
        )
        agent.add_argument(
            '--negotiation-data-dir',
            type=str,
            default='data/negotiations',
            help='Directory containing negotiation data files'
        )
        agent.add_argument(
            '--include-state-context',
            type=bool,
            default=True,
            help='Whether to include state context in observations'
        )
        agent.add_argument(
            '--include-metadata',
            type=bool,
            default=True,
            help='Whether to include episode metadata in observations'
        )
        agent.add_argument(
            '--shuffle-episodes',
            type=bool,
            default=True,
            help='Whether to shuffle episodes during training'
        )
        agent.add_argument(
            '--max-episode-length',
            type=int,
            default=20,
            help='Maximum number of turns per episode'
        )
        return parser
    
    def __init__(self, opt, shared=None):
        """Initialize the negotiation teacher."""
        self.opt = opt
        self.id = 'negotiation_teacher'
        
        # Data configuration
        self.data_dir = Path(opt.get('negotiation_data_dir', 'data/negotiations'))
        self.data_file = opt.get('negotiation_data_file', 'episodes.jsonl')
        self.include_state = opt.get('include_state_context', True)
        self.include_metadata = opt.get('include_metadata', True)
        self.shuffle = opt.get('shuffle_episodes', True)
        self.max_episode_length = opt.get('max_episode_length', 20)
        
        # Episode tracking
        self.episodes: List[Dict[str, Any]] = []
        self.episode_metrics: Dict[str, Any] = {}
        
        # Load data if not in shared mode
        if shared:
            # Copy shared data
            self.episodes = shared['episodes']
            self.episode_metrics = shared['episode_metrics']
        else:
            # Load episodes from JSONL
            self._load_episodes()
        
        # Initialize parent class
        super().__init__(opt, shared)
    
    def share(self):
        """Share data between teacher instances."""
        shared = super().share()
        shared['episodes'] = self.episodes
        shared['episode_metrics'] = self.episode_metrics
        return shared
    
    def _load_episodes(self):
        """Load negotiation episodes from JSONL file."""
        filepath = self.data_dir / self.data_file
        
        if not filepath.exists():
            # If file doesn't exist, create empty episodes list
            logger.warning(f"Data file {filepath} not found. Using empty dataset.")
            self.episodes = []
            return
        
        # Load episodes from JSONL
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        episode = json.loads(line)
                        self.episodes.append(episode)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
        
        # Shuffle if requested
        if self.shuffle and self.episodes:
            random.shuffle(self.episodes)
        
        logger.info(f"Loaded {len(self.episodes)} episodes from {filepath}")
    
    def _format_state_context(self, state: Dict[str, Any], side: str) -> str:
        """Format state information as context string."""
        context_parts = []
        
        # Add role
        context_parts.append(f"You are the {side} in this negotiation.")
        
        # Add item details
        if 'item' in state:
            item = state['item']
            item_desc = f"Item: {item.get('name', 'Unknown')}"
            if item.get('brand'):
                item_desc += f" ({item['brand']}"
                if item.get('model'):
                    item_desc += f" {item['model']}"
                item_desc += ")"
            if item.get('condition'):
                item_desc += f", Condition: {item['condition']}"
            if item.get('list_price'):
                item_desc += f", List Price: ${item['list_price']}"
            context_parts.append(item_desc)
        
        # Add role-specific constraints
        if side == 'buyer' and 'buyer_state' in state:
            buyer = state['buyer_state']
            context_parts.append(f"Your maximum price: ${buyer.get('max_price', 'N/A')}")
            context_parts.append(f"Your target price: ${buyer.get('target_price', 'N/A')}")
            if buyer.get('preferences'):
                context_parts.append(f"Preferences: {json.dumps(buyer['preferences'])}")
        elif side == 'seller' and 'seller_state' in state:
            seller = state['seller_state']
            context_parts.append(f"Your minimum price: ${seller.get('min_price', 'N/A')}")
            if seller.get('item_issues'):
                context_parts.append(f"Known issues: {', '.join(seller['item_issues'])}")
        
        # Add logistics info
        role_state = state.get(f'{side}_state', {})
        if 'logistics' in role_state:
            logistics = role_state['logistics']
            log_parts = []
            if logistics.get('region'):
                log_parts.append(f"Region: {logistics['region']}")
            if 'meet_ok' in logistics:
                log_parts.append(f"In-person meeting: {'Yes' if logistics['meet_ok'] else 'No'}")
            if 'ship_ok' in logistics:
                log_parts.append(f"Shipping: {'Yes' if logistics['ship_ok'] else 'No'}")
            if logistics.get('return_days'):
                log_parts.append(f"Return policy: {logistics['return_days']} days")
            if log_parts:
                context_parts.append("Logistics: " + ", ".join(log_parts))
        
        return "\n".join(context_parts)
    
    def _format_observation(self, episode: Dict[str, Any], turn_idx: int) -> Dict[str, Any]:
        """Format a single turn as a ParlAI observation."""
        dialogue = episode.get('dialogue', [])
        state = episode.get('state', {})
        metadata = episode.get('meta', {})
        
        # Get current turn
        if turn_idx >= len(dialogue):
            return {'episode_done': True}
        
        turn = dialogue[turn_idx]
        
        # Build observation
        obs = {}
        
        # Add state context if this is the first turn
        if turn_idx == 0 and self.include_state:
            side = turn.get('side', metadata.get('side', 'buyer'))
            context = self._format_state_context(state, side)
            obs['text'] = context + "\n\n" + turn.get('text', '')
        else:
            obs['text'] = turn.get('text', '')
        
        # Add labels (response)
        if 'response' in turn:
            obs['labels'] = [turn['response']]
        elif 'label' in turn:
            obs['labels'] = [turn['label']]
        
        # Add metadata
        if self.include_metadata:
            obs['episode_id'] = metadata.get('episode_id', f"ep_{id(episode)}")
            obs['dialogue_id'] = metadata.get('dialogue_id', obs['episode_id'])
            obs['turn'] = turn_idx
            obs['side'] = turn.get('side', metadata.get('side'))
            
            # Add any additional metadata
            for key, value in metadata.items():
                if key not in ['episode_id', 'dialogue_id', 'side']:
                    obs[f'meta_{key}'] = value
        
        # Mark episode done if this is the last turn
        obs['episode_done'] = (turn_idx >= len(dialogue) - 1)
        
        return obs
    
    def setup_data(self, datafile=None):
        """Set up the data for the teacher."""
        # Create examples from episodes
        for episode in self.episodes:
            dialogue = episode.get('dialogue', [])
            
            # Skip empty episodes
            if not dialogue:
                continue
            
            # Create example for each turn
            for turn_idx in range(min(len(dialogue), self.max_episode_length)):
                obs = self._format_observation(episode, turn_idx)
                
                # Only yield if we have both text and labels
                if 'text' in obs and 'labels' in obs:
                    yield obs, obs.get('episode_done', False)
    
    def get_episode_metrics(self, episode_id: str) -> Dict[str, Any]:
        """Get metrics for a specific episode."""
        return self.episode_metrics.get(episode_id, {})
    
    def update_episode_metrics(self, episode_id: str, metrics: Dict[str, Any]):
        """Update metrics for a specific episode."""
        if episode_id not in self.episode_metrics:
            self.episode_metrics[episode_id] = {}
        self.episode_metrics[episode_id].update(metrics)
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all episodes."""
        if not self.episode_metrics:
            return {}
        
        # Aggregate metrics
        total_turns = 0
        total_tokens = 0
        deals_closed = 0
        protocol_violations = 0
        
        for metrics in self.episode_metrics.values():
            total_turns += metrics.get('turns', 0)
            total_tokens += metrics.get('total_tokens', 0)
            if metrics.get('deal_closed', False):
                deals_closed += 1
            protocol_violations += len(metrics.get('protocol_violations', []))
        
        num_episodes = len(self.episode_metrics)
        
        return {
            'num_episodes': num_episodes,
            'avg_turns': total_turns / num_episodes if num_episodes > 0 else 0,
            'avg_tokens': total_tokens / num_episodes if num_episodes > 0 else 0,
            'deal_close_rate': deals_closed / num_episodes if num_episodes > 0 else 0,
            'avg_violations': protocol_violations / num_episodes if num_episodes > 0 else 0,
        }
    
    def save_episodes(self, filepath: Optional[Path] = None):
        """Save current episodes to JSONL file."""
        if filepath is None:
            filepath = self.data_dir / f"processed_{self.data_file}"
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save episodes
        with open(filepath, 'w') as f:
            for episode in self.episodes:
                # Add metrics if available
                episode_id = episode.get('meta', {}).get('episode_id')
                if episode_id and episode_id in self.episode_metrics:
                    episode['metrics'] = self.episode_metrics[episode_id]
                
                f.write(json.dumps(episode) + '\n')
        
        logger.info(f"Saved {len(self.episodes)} episodes to {filepath}")