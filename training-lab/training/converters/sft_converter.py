"""
SFT (Supervised Fine-Tuning) format converter.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from training.converters.base import BaseConverter

# Set up logging
logger = logging.getLogger(__name__)


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
        
        logger.info(f"Converted {len(episodes)} episodes to {len(all_sft)} SFT examples")
        logger.info(f"Saved to {output_path}")
        
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
        
        logger.info(f"Filtered {len(examples)} to {len(filtered)} examples (min_score={min_score})")
        logger.info(f"Saved to {output_path}")
        
        return str(output_path)
    
    def episode_to_sft_separated(self,
                                 episode: Dict[str, Any],
                                 history_window: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert episode to separated SFT format for two-part response training.
        
        Args:
            episode: ParlAI episode dictionary with strategy and response data
            history_window: Number of recent turns for context
        
        Returns:
            Dictionary with 'strategy' and 'response' SFT example lists
        """
        strategy_examples = []
        response_examples = []
        
        state = episode.get('state', {})
        dialogue = episode.get('dialogue', [])
        metadata = episode.get('meta', {})
        
        for turn_idx, turn in enumerate(dialogue):
            # Skip turns without response data
            if 'response' not in turn and 'label' not in turn:
                continue
            
            # Build strategy SFT example if strategy data exists
            if 'strategy_generation' in turn:
                strategy_example = self._build_strategy_sft_example(
                    state, dialogue, turn, turn_idx, metadata, history_window
                )
                strategy_examples.append(strategy_example)
            
            # Build response SFT example if response data exists
            if 'response_generation' in turn:
                response_example = self._build_response_sft_example(
                    state, dialogue, turn, turn_idx, metadata, history_window
                )
                response_examples.append(response_example)
        
        return {
            'strategy': strategy_examples,
            'response': response_examples
        }
    
    def _build_strategy_sft_example(self,
                                   state: Dict,
                                   dialogue: List[Dict],
                                   turn: Dict,
                                   turn_idx: int,
                                   metadata: Dict,
                                   history_window: int) -> Dict[str, Any]:
        """Build SFT example for strategy generation."""
        history = self.extract_dialogue_history(dialogue, turn_idx, history_window)
        partner_msg = turn.get('text', '')
        strategy_data = turn.get('strategy_generation', {})
        
        return {
            "input": {
                "task": "strategy_generation",
                "state": state,
                "recent_convo": history,
                "partner_msg": partner_msg,
                "current_side": turn.get('side', metadata.get('side', 'buyer')),
                "turn_number": turn_idx
            },
            "output": strategy_data.get('strategy', '{}'),
            "meta": {
                "episode_id": metadata.get('episode_id'),
                "dialogue_id": metadata.get('dialogue_id'),
                "turn": turn_idx,
                "side": turn.get('side', metadata.get('side')),
                "prompt_tokens": strategy_data.get('prompt_tokens', 0),
                "output_tokens": strategy_data.get('output_tokens', 0),
                "generation_time_ms": strategy_data.get('generation_time_ms', 0)
            }
        }
    
    def _build_response_sft_example(self,
                                   state: Dict,
                                   dialogue: List[Dict],
                                   turn: Dict,
                                   turn_idx: int,
                                   metadata: Dict,
                                   history_window: int) -> Dict[str, Any]:
        """Build SFT example for response generation."""
        history = self.extract_dialogue_history(dialogue, turn_idx, history_window)
        partner_msg = turn.get('text', '')
        strategy_data = turn.get('strategy_generation', {})
        response_data = turn.get('response_generation', {})
        
        return {
            "input": {
                "task": "response_generation",
                "state": state,
                "recent_convo": history,
                "partner_msg": partner_msg,
                "strategy": strategy_data.get('strategy', '{}'),
                "current_side": turn.get('side', metadata.get('side', 'buyer')),
                "turn_number": turn_idx
            },
            "output": response_data.get('response', turn.get('response', '')),
            "meta": {
                "episode_id": metadata.get('episode_id'),
                "dialogue_id": metadata.get('dialogue_id'),
                "turn": turn_idx,
                "side": turn.get('side', metadata.get('side')),
                "action": turn.get('action', 'PROPOSE'),
                "prompt_tokens": response_data.get('prompt_tokens', 0),
                "output_tokens": response_data.get('output_tokens', 0),
                "generation_time_ms": response_data.get('generation_time_ms', 0)
            }
        }
    
    def batch_convert_separated(self,
                               episodes_file: str,
                               strategy_output: Optional[str] = None,
                               response_output: Optional[str] = None,
                               history_window: int = 5) -> Dict[str, str]:
        """
        Convert batch of episodes to separated SFT format.
        
        Args:
            episodes_file: Path to JSONL file with episodes
            strategy_output: Optional output path for strategy SFT data
            response_output: Optional output path for response SFT data
            history_window: Number of recent turns for context
        
        Returns:
            Dictionary with paths to strategy and response output files
        """
        episodes = self.load_jsonl(Path(episodes_file))
        
        all_strategy = []
        all_response = []
        
        for episode in episodes:
            separated = self.episode_to_sft_separated(episode, history_window)
            all_strategy.extend(separated['strategy'])
            all_response.extend(separated['response'])
        
        # Determine output paths
        if strategy_output is None:
            strategy_path = self.output_dir / f"sft_strategy_{len(all_strategy)}.jsonl"
        else:
            strategy_path = Path(strategy_output)
        
        if response_output is None:
            response_path = self.output_dir / f"sft_response_{len(all_response)}.jsonl"
        else:
            response_path = Path(response_output)
        
        # Save separated datasets
        self.save_jsonl(all_strategy, strategy_path)
        self.save_jsonl(all_response, response_path)
        
        logger.info(f"Converted {len(episodes)} episodes:")
        logger.info(f"  - {len(all_strategy)} strategy examples -> {strategy_path}")
        logger.info(f"  - {len(all_response)} response examples -> {response_path}")
        
        return {
            'strategy': str(strategy_path),
            'response': str(response_path)
        }