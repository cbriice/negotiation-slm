"""
SFT (Supervised Fine-Tuning) format converter.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from training.converters.base import BaseConverter


class SFTConverter(BaseConverter):
    """Convert ParlAI episodes to SFT training format."""
    
    def __init__(self, output_dir: str = "data/training/sft"):
        """Initialize SFT converter."""
        super().__init__(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def episode_to_sft(self, 
                      episode: Dict[str, Any], 
                      history_window: int = 5,
                      include_metrics: bool = True) -> List[Dict[str, Any]]:
        """
        Convert a single ParlAI episode to SFT format.
        
        Args:
            episode: ParlAI episode dictionary
            history_window: Number of recent turns to include as context
            include_metrics: Whether to include episode metrics in output
        
        Returns:
            List of SFT training examples
        """
        sft_examples = []
        
        state = episode.get('state', {})
        dialogue = episode.get('dialogue', [])
        metadata = episode.get('meta', {})
        
        for turn_idx, turn in enumerate(dialogue):
            # Skip turns without a response
            if 'response' not in turn and 'label' not in turn:
                continue
            
            # Build the SFT example
            sft_example = self._build_sft_example(
                state, dialogue, turn, turn_idx, 
                metadata, history_window, include_metrics, episode
            )
            
            sft_examples.append(sft_example)
        
        return sft_examples
    
    def _build_sft_example(self, 
                          state: Dict,
                          dialogue: List[Dict],
                          turn: Dict,
                          turn_idx: int,
                          metadata: Dict,
                          history_window: int,
                          include_metrics: bool,
                          episode: Dict) -> Dict[str, Any]:
        """Build a single SFT training example."""
        # Get conversation history
        history = self.extract_dialogue_history(dialogue, turn_idx, history_window)
        
        # Get partner message and agent response
        partner_msg = turn.get('text', '')
        agent_response = turn.get('response', turn.get('label', ''))
        
        # Build the example
        sft_example = {
            "input": {
                "state": state,
                "recent_convo": history,
                "partner_msg": partner_msg,
                "current_side": turn.get('side', metadata.get('side', 'buyer')),
                "turn_number": turn_idx
            },
            "output": agent_response,
            "meta": {
                "episode_id": metadata.get('episode_id', f"ep_{hash(str(episode))%100000}"),
                "dialogue_id": metadata.get('dialogue_id', metadata.get('episode_id')),
                "turn": turn_idx,
                "side": turn.get('side', metadata.get('side')),
                "action": turn.get('action', 'PROPOSE'),
                "deal_closed": metadata.get('deal_closed', False),
                "num_turns": len(dialogue)
            }
        }
        
        # Add metrics if requested and available
        if include_metrics and 'metrics' in episode:
            sft_example['meta']['episode_metrics'] = episode['metrics']
        
        return sft_example
    
    def batch_convert(self, 
                     episodes_file: str,
                     output_file: Optional[str] = None,
                     history_window: int = 5,
                     include_metrics: bool = True) -> str:
        """
        Convert a batch of episodes to SFT format.
        
        Args:
            episodes_file: Path to JSONL file with episodes
            output_file: Optional output path for SFT data
            history_window: Number of recent turns for context
            include_metrics: Whether to include metrics
        
        Returns:
            Path to output file
        """
        episodes = self.load_jsonl(Path(episodes_file))
        
        all_sft = []
        for episode in episodes:
            sft_examples = self.episode_to_sft(
                episode, history_window, include_metrics
            )
            all_sft.extend(sft_examples)
        
        # Determine output path
        if output_file is None:
            output_path = self.output_dir / f"sft_batch_{len(all_sft)}.jsonl"
        else:
            output_path = Path(output_file)
        
        self.save_jsonl(all_sft, output_path)
        
        print(f"Converted {len(episodes)} episodes to {len(all_sft)} SFT examples")
        print(f"Saved to {output_path}")
        
        return str(output_path)
    
    def filter_by_quality(self,
                         sft_file: str,
                         min_score: float = 0.0,
                         output_file: Optional[str] = None) -> str:
        """
        Filter SFT examples by episode quality score.
        
        Args:
            sft_file: Path to SFT JSONL file
            min_score: Minimum quality score to include
            output_file: Optional output path
        
        Returns:
            Path to filtered output file
        """
        examples = self.load_jsonl(Path(sft_file))
        
        filtered = []
        for example in examples:
            metrics = example.get('meta', {}).get('episode_metrics', {})
            score = metrics.get('quality_score', 0.0)
            
            if score >= min_score:
                filtered.append(example)
        
        if output_file is None:
            output_path = self.output_dir / f"sft_filtered_{len(filtered)}.jsonl"
        else:
            output_path = Path(output_file)
        
        self.save_jsonl(filtered, output_path)
        
        print(f"Filtered {len(examples)} to {len(filtered)} examples (min_score={min_score})")
        print(f"Saved to {output_path}")
        
        return str(output_path)