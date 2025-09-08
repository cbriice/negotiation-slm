"""
DPO (Direct Preference Optimization) format converter.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import random
from training.converters.base import BaseConverter
from training.converters.scoring import EpisodeScorer

# Set up logging
logger = logging.getLogger(__name__)


class DPOConverter(BaseConverter):
    """Convert ParlAI episodes to DPO training format."""
    
    def __init__(self, output_dir: str = "data/training/dpo"):
        """Initialize DPO converter."""
        super().__init__(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = EpisodeScorer()
    
    def create_dpo_pair(self,
                       good_episode: Dict[str, Any],
                       bad_episode: Dict[str, Any],
                       alignment_strategy: str = "turn_by_turn") -> List[Dict[str, Any]]:
        """
        Create DPO training pairs from good/bad episodes.
        
        Args:
            good_episode: Episode with high quality negotiation
            bad_episode: Episode with poor negotiation behavior
            alignment_strategy: How to align turns between episodes:
                - "turn_by_turn": Match turns by index
                - "action_matching": Match turns by action type
        
        Returns:
            List of DPO training pairs
        """
        dpo_pairs = []
        
        # Extract components
        good_state = good_episode.get('state', {})
        bad_state = bad_episode.get('state', {})
        good_dialogue = good_episode.get('dialogue', [])
        bad_dialogue = bad_episode.get('dialogue', [])
        good_meta = good_episode.get('meta', {})
        bad_meta = bad_episode.get('meta', {})
        
        # Calculate scores
        good_score = self.scorer.calculate_score(good_episode)
        bad_score = self.scorer.calculate_score(bad_episode)
        
        # Ensure good score is better
        if good_score <= bad_score:
            logger.warning(f"Good score ({good_score:.2f}) not better than bad ({bad_score:.2f})")
            if good_score < bad_score:
                # Swap episodes
                good_episode, bad_episode = bad_episode, good_episode
                good_dialogue, bad_dialogue = bad_dialogue, good_dialogue
                good_meta, bad_meta = bad_meta, good_meta
                good_score, bad_score = bad_score, good_score
                good_state, bad_state = bad_state, good_state
        
        # Align turns
        if alignment_strategy == "action_matching":
            aligned_pairs = self._align_by_action(good_dialogue, bad_dialogue)
        else:
            aligned_pairs = self._align_by_index(good_dialogue, bad_dialogue)
        
        # Create DPO pairs
        for good_turn, bad_turn, turn_idx in aligned_pairs:
            dpo_example = self._build_dpo_example(
                good_turn, bad_turn, turn_idx,
                good_state, good_dialogue,
                good_meta, bad_meta,
                good_score, bad_score
            )
            
            if dpo_example:
                dpo_pairs.append(dpo_example)
        
        return dpo_pairs
    
    def _build_dpo_example(self,
                          good_turn: Dict,
                          bad_turn: Dict,
                          turn_idx: int,
                          state: Dict,
                          dialogue: List[Dict],
                          good_meta: Dict,
                          bad_meta: Dict,
                          good_score: float,
                          bad_score: float) -> Optional[Dict[str, Any]]:
        """Build a single DPO training example."""
        # Skip if either lacks response
        if not good_turn.get('response') or not bad_turn.get('response'):
            return None
        
        # Build history
        history = self.extract_dialogue_history(dialogue, turn_idx)
        
        # Get shared context
        partner_msg = good_turn.get('text', bad_turn.get('text', ''))
        
        # Create DPO pair
        dpo_example = {
            "context": {
                "state": state,
                "recent_convo": history,
                "partner_msg": partner_msg,
                "current_side": good_turn.get('side', good_meta.get('side', 'buyer')),
                "turn_number": turn_idx
            },
            "chosen": good_turn['response'],
            "rejected": bad_turn['response'],
            "meta": {
                "chosen_score": good_score,
                "rejected_score": bad_score,
                "score_delta": good_score - bad_score,
                "chosen_episode_id": good_meta.get('episode_id', ''),
                "rejected_episode_id": bad_meta.get('episode_id', ''),
                "turn": turn_idx,
                "chosen_action": good_turn.get('action', 'PROPOSE'),
                "rejected_action": bad_turn.get('action', 'PROPOSE'),
                "chosen_deal_closed": good_meta.get('deal_closed', False),
                "rejected_deal_closed": bad_meta.get('deal_closed', False)
            }
        }
        
        return dpo_example
    
    def _align_by_index(self, 
                       dialogue1: List[Dict],
                       dialogue2: List[Dict]) -> List[Tuple[Dict, Dict, int]]:
        """Align dialogues by turn index."""
        aligned = []
        min_length = min(len(dialogue1), len(dialogue2))
        
        for idx in range(min_length):
            aligned.append((dialogue1[idx], dialogue2[idx], idx))
        
        return aligned
    
    def _align_by_action(self,
                        dialogue1: List[Dict],
                        dialogue2: List[Dict]) -> List[Tuple[Dict, Dict, int]]:
        """Align dialogues by matching action types."""
        aligned = []
        
        # Group by action
        d1_actions = defaultdict(list)
        d2_actions = defaultdict(list)
        
        for idx, turn in enumerate(dialogue1):
            action = turn.get('action', 'PROPOSE')
            d1_actions[action].append((turn, idx))
        
        for idx, turn in enumerate(dialogue2):
            action = turn.get('action', 'PROPOSE')
            d2_actions[action].append((turn, idx))
        
        # Match by action
        for action in d1_actions:
            if action in d2_actions:
                pairs = min(len(d1_actions[action]), len(d2_actions[action]))
                for i in range(pairs):
                    turn1, idx1 = d1_actions[action][i]
                    turn2, idx2 = d2_actions[action][i]
                    aligned.append((turn1, turn2, min(idx1, idx2)))
        
        # Sort by turn index
        aligned.sort(key=lambda x: x[2])
        
        return aligned
    
    def batch_create_pairs(self,
                          good_episodes_file: str,
                          bad_episodes_file: str,
                          output_file: Optional[str] = None,
                          max_pairs: Optional[int] = None,
                          alignment_strategy: str = "turn_by_turn") -> str:
        """
        Create DPO pairs from batches of good and bad episodes.
        
        Args:
            good_episodes_file: Path to JSONL with good episodes
            bad_episodes_file: Path to JSONL with bad episodes
            output_file: Optional output path for DPO pairs
            max_pairs: Maximum number of pairs to generate
            alignment_strategy: Turn alignment strategy
        
        Returns:
            Path to output file
        """
        # Load episodes
        good_episodes = self.load_jsonl(Path(good_episodes_file))
        bad_episodes = self.load_jsonl(Path(bad_episodes_file))
        
        # Determine number of pairs
        num_pairs = min(len(good_episodes), len(bad_episodes))
        if max_pairs:
            num_pairs = min(num_pairs, max_pairs)
        
        # Shuffle for variety
        random.shuffle(good_episodes)
        random.shuffle(bad_episodes)
        
        # Generate pairs
        all_pairs = []
        for i in range(num_pairs):
            pairs = self.create_dpo_pair(
                good_episodes[i],
                bad_episodes[i],
                alignment_strategy
            )
            all_pairs.extend(pairs)
        
        # Save to file
        if output_file is None:
            output_path = self.output_dir / f"dpo_batch_{len(all_pairs)}.jsonl"
        else:
            output_path = Path(output_file)
        
        self.save_jsonl(all_pairs, output_path)
        
        logger.info(f"Created {len(all_pairs)} DPO pairs from {num_pairs} episode pairs")
        logger.info(f"Saved to {output_path}")
        
        return str(output_path)
    
    def auto_generate_pairs(self,
                           episodes_file: str,
                           output_file: Optional[str] = None,
                           score_threshold: float = 5.0) -> str:
        """
        Automatically generate DPO pairs by scoring and pairing episodes.
        
        Args:
            episodes_file: Path to JSONL with all episodes
            output_file: Optional output path
            score_threshold: Minimum score difference for pairing
        
        Returns:
            Path to output file
        """
        episodes = self.load_jsonl(Path(episodes_file))
        
        # Score and rank episodes
        ranked = self.scorer.rank_episodes(episodes)
        
        # Split into good and bad
        median_idx = len(ranked) // 2
        good_episodes = ranked[:median_idx]
        bad_episodes = ranked[median_idx:]
        
        # Generate pairs with sufficient score difference
        all_pairs = []
        for good_ep in good_episodes[:50]:  # Limit to top 50
            for bad_ep in bad_episodes[-50:]:  # Bottom 50
                score_diff = good_ep['quality_score'] - bad_ep['quality_score']
                
                if score_diff >= score_threshold:
                    pairs = self.create_dpo_pair(good_ep, bad_ep)
                    all_pairs.extend(pairs)
                    
                    if len(all_pairs) >= 1000:  # Limit total pairs
                        break
            
            if len(all_pairs) >= 1000:
                break
        
        # Save
        if output_file is None:
            output_path = self.output_dir / f"dpo_auto_{len(all_pairs)}.jsonl"
        else:
            output_path = Path(output_file)
        
        self.save_jsonl(all_pairs, output_path)
        
        logger.info(f"Auto-generated {len(all_pairs)} DPO pairs")
        logger.info(f"Saved to {output_path}")
        
        return str(output_path)