"""
Episode logging system with NDJSON format compliance.
Handles episode transformation, stage-specific directories, and batch writing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training stages for directory organization."""
    SELFCHAT = "selfchat"
    DIALOGPARTNER = "dialogpartner"
    HUMAN_EVAL = "human_eval"
    BASELINE = "baseline"
    DPO_PAIRS = "dpo_pairs"
    AUGMENTED = "augmented"


class EpisodeLogger:
    """
    Handles episode logging with NDJSON format compliance.
    Transforms episodes to match schema templates and provides batch writing.
    """
    
    def __init__(self, base_dir: str = "data/negotiations", 
                 batch_size: int = 100,
                 stage: Optional[TrainingStage] = None):
        """
        Initialize the episode logger.
        
        Args:
            base_dir: Base directory for saving episodes
            batch_size: Number of episodes to buffer before writing
            stage: Current training stage for directory organization
        """
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.stage = stage
        self.buffer: List[Dict[str, Any]] = []
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'episodes_logged': 0,
            'sft_packets_created': 0,
            'dpo_pairs_created': 0,
            'batches_written': 0
        }
    
    def get_stage_directory(self, stage: Optional[TrainingStage] = None) -> Path:
        """
        Get the directory for a specific training stage.
        
        Args:
            stage: Training stage (uses instance stage if not provided)
            
        Returns:
            Path to stage-specific directory
        """
        stage = stage or self.stage
        if stage:
            stage_dir = self.base_dir / stage.value
        else:
            stage_dir = self.base_dir / "general"
        
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    def transform_to_sft_packet(self, episode: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transform an episode into SFT training packets following schema/sft_packet_template.jsonl.
        
        Args:
            episode: Raw episode data with dialogue, state, and meta fields
            
        Returns:
            List of SFT packets (one per dialogue turn)
        """
        packets = []
        dialogue = episode.get('dialogue', [])
        state = episode.get('state', {})
        meta = episode.get('meta', {})
        
        # Build conversation history progressively
        conversation_history = []
        
        for i, turn in enumerate(dialogue):
            # Skip if no response in this turn
            if 'response' not in turn and 'label' not in turn:
                continue
            
            # Build input context
            input_packet = {
                'state': state,
                'recent_convo': conversation_history[-4:] if conversation_history else [],  # Last 2 exchanges
                'partner_msg': turn.get('text', '')
            }
            
            # Get the response
            output = turn.get('response', turn.get('label', ''))
            
            # Create SFT packet
            sft_packet = {
                'input': input_packet,
                'output': output,
                'meta': {
                    'episode_id': meta.get('episode_id', f"ep_{id(episode)}"),
                    'dialogue_id': meta.get('dialogue_id', meta.get('episode_id', '')),
                    'turn': i,
                    'side': turn.get('side', meta.get('side', 'unknown')),
                    'stage': self.stage.value if self.stage else 'unknown',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            packets.append(sft_packet)
            
            # Update conversation history
            conversation_history.append({
                'side': turn.get('side', 'unknown'),
                'text': turn.get('text', ''),
                'response': output
            })
        
        self.stats['sft_packets_created'] += len(packets)
        return packets
    
    def transform_to_dpo_pair(self, episode_chosen: Dict[str, Any], 
                             episode_rejected: Dict[str, Any],
                             turn_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Transform two episodes into a DPO training pair following schema/dpo_packet_template.jsonl.
        
        Args:
            episode_chosen: Episode with preferred response
            episode_rejected: Episode with rejected response
            turn_idx: Specific turn to create pair for (None = find best divergence point)
            
        Returns:
            DPO pair packet or None if episodes incompatible
        """
        dialogue_chosen = episode_chosen.get('dialogue', [])
        dialogue_rejected = episode_rejected.get('dialogue', [])
        state = episode_chosen.get('state', {})
        
        # If no turn specified, find divergence point
        if turn_idx is None:
            turn_idx = self._find_divergence_point(dialogue_chosen, dialogue_rejected)
            if turn_idx is None:
                logger.warning("No suitable divergence point found for DPO pair")
                return None
        
        # Validate turn exists in both dialogues
        if turn_idx >= len(dialogue_chosen) or turn_idx >= len(dialogue_rejected):
            return None
        
        # Build context up to divergence point
        context_history = []
        for i in range(turn_idx):
            if i < len(dialogue_chosen):
                turn = dialogue_chosen[i]
                context_history.append({
                    'side': turn.get('side', 'unknown'),
                    'text': turn.get('text', ''),
                    'response': turn.get('response', turn.get('label', ''))
                })
        
        # Get the diverging responses
        chosen_response = dialogue_chosen[turn_idx].get('response', 
                                                        dialogue_chosen[turn_idx].get('label', ''))
        rejected_response = dialogue_rejected[turn_idx].get('response',
                                                            dialogue_rejected[turn_idx].get('label', ''))
        
        # Create DPO packet
        dpo_packet = {
            'context': {
                'state': state,
                'recent_convo': context_history[-4:] if context_history else [],
                'partner_msg': dialogue_chosen[turn_idx].get('text', '')
            },
            'chosen': chosen_response,
            'rejected': rejected_response,
            'meta': {
                'chosen_episode_id': episode_chosen.get('meta', {}).get('episode_id', ''),
                'rejected_episode_id': episode_rejected.get('meta', {}).get('episode_id', ''),
                'turn': turn_idx,
                'stage': self.stage.value if self.stage else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'chosen_metrics': episode_chosen.get('metrics', {}),
                'rejected_metrics': episode_rejected.get('metrics', {})
            }
        }
        
        self.stats['dpo_pairs_created'] += 1
        return dpo_packet
    
    def _find_divergence_point(self, dialogue1: List[Dict], 
                               dialogue2: List[Dict]) -> Optional[int]:
        """
        Find the turn where two dialogues diverge.
        
        Args:
            dialogue1: First dialogue
            dialogue2: Second dialogue
            
        Returns:
            Turn index where dialogues diverge, or None if identical
        """
        min_length = min(len(dialogue1), len(dialogue2))
        
        for i in range(min_length):
            # Compare the responses at this turn
            response1 = dialogue1[i].get('response', dialogue1[i].get('label', ''))
            response2 = dialogue2[i].get('response', dialogue2[i].get('label', ''))
            
            if response1 != response2:
                return i
        
        # If one dialogue is longer, divergence is at the end of the shorter one
        if len(dialogue1) != len(dialogue2):
            return min_length
        
        return None
    
    def log_episode(self, episode: Dict[str, Any], 
                   format: str = 'raw',
                   stage: Optional[TrainingStage] = None) -> None:
        """
        Log a single episode to the buffer.
        
        Args:
            episode: Episode data to log
            format: Output format ('raw', 'sft', or 'dpo')
            stage: Training stage (overrides instance stage)
        """
        from utils.protocol_validator import (
            validate_episode_for_training, 
            calculate_protocol_score
        )
        
        # Validate episode for training suitability
        is_suitable, reasons = validate_episode_for_training(episode)
        
        # Calculate protocol score
        violations = episode.get('metrics', {}).get('protocol_violations', [])
        action_history = episode.get('metrics', {}).get('action_history', [])
        deal_closed = episode.get('metrics', {}).get('deal_closed', False)
        protocol_score = calculate_protocol_score(violations, action_history, deal_closed)
        
        # Add validation metadata
        episode['meta'] = episode.get('meta', {})
        episode['meta']['suitable_for_training'] = is_suitable
        episode['meta']['validation_issues'] = reasons if not is_suitable else []
        episode['meta']['protocol_score'] = protocol_score
        
        # Add to buffer based on format
        if format == 'sft':
            # Only transform to SFT if suitable for training
            if is_suitable:
                packets = self.transform_to_sft_packet(episode)
                self.buffer.extend(packets)
            else:
                logger.warning(f"Episode not suitable for SFT training: {reasons}")
        else:
            # Raw format - add metadata
            episode['meta']['logged_at'] = datetime.now().isoformat()
            current_stage = stage or self.stage
            episode['meta']['stage'] = current_stage.value if current_stage is not None else 'unknown'
            self.buffer.append(episode)
        
        self.stats['episodes_logged'] += 1
        
        # Write buffer if it's full
        if len(self.buffer) >= self.batch_size:
            self.flush(stage=stage)
    
    def log_episodes_batch(self, episodes: List[Dict[str, Any]], 
                          format: str = 'raw',
                          stage: Optional[TrainingStage] = None) -> None:
        """
        Log multiple episodes at once.
        
        Args:
            episodes: List of episodes to log
            format: Output format ('raw' or 'sft')
            stage: Training stage
        """
        for episode in episodes:
            self.log_episode(episode, format=format, stage=stage)
    
    def log_dpo_pair(self, chosen_episode: Dict[str, Any],
                    rejected_episode: Dict[str, Any],
                    turn_idx: Optional[int] = None,
                    stage: Optional[TrainingStage] = None) -> None:
        """
        Log a DPO training pair.
        
        Args:
            chosen_episode: Episode with preferred responses
            rejected_episode: Episode with rejected responses
            turn_idx: Specific turn to create pair for
            stage: Training stage
        """
        from utils.protocol_validator import validate_episode_for_training
        
        # Validate both episodes are suitable for training
        chosen_suitable, chosen_reasons = validate_episode_for_training(chosen_episode)
        rejected_suitable, rejected_reasons = validate_episode_for_training(rejected_episode)
        
        if not chosen_suitable:
            logger.warning(f"Chosen episode not suitable for DPO: {chosen_reasons}")
            return
        
        # Rejected episode can have issues (that's why it's rejected)
        # but shouldn't have critical issues like tag leaks
        if rejected_reasons:
            critical_issues = [r for r in rejected_reasons if 'leaked action tags' in r]
            if critical_issues:
                logger.warning(f"Rejected episode has critical issues: {critical_issues}")
                return
        
        dpo_pair = self.transform_to_dpo_pair(chosen_episode, rejected_episode, turn_idx)
        if dpo_pair:
            self.buffer.append(dpo_pair)
            
            # Write buffer if it's full
            if len(self.buffer) >= self.batch_size:
                self.flush(stage=stage or TrainingStage.DPO_PAIRS)
    
    def flush(self, stage: Optional[TrainingStage] = None) -> None:
        """
        Write buffered episodes to disk.
        
        Args:
            stage: Training stage for directory selection
        """
        if not self.buffer:
            return
        
        # Determine output directory and filename
        output_dir = self.get_stage_directory(stage)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate unique batch ID
        batch_id = hashlib.md5(f"{timestamp}_{id(self.buffer)}".encode()).hexdigest()[:8]
        
        # Determine filename based on content type
        if self.buffer and 'chosen' in self.buffer[0]:
            filename = f"dpo_pairs_{timestamp}_{batch_id}.jsonl"
        elif self.buffer and 'input' in self.buffer[0] and 'output' in self.buffer[0]:
            filename = f"sft_packets_{timestamp}_{batch_id}.jsonl"
        else:
            filename = f"episodes_{timestamp}_{batch_id}.jsonl"
        
        filepath = output_dir / filename
        
        # Write NDJSON
        with open(filepath, 'w') as f:
            for item in self.buffer:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Wrote {len(self.buffer)} items to {filepath}")
        
        # Update statistics
        self.stats['batches_written'] += 1
        
        # Clear buffer
        self.buffer.clear()
    
    def stream_episodes(self, filepath: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """
        Stream episodes from a JSONL file without loading all into memory.
        
        Args:
            filepath: Path to JSONL file
            
        Yields:
            Episode dictionaries
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"File {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
                        continue
    
    def select_and_log_dpo_pairs(self, episodes: List[Dict[str, Any]], 
                                 min_score_gap: float = 0.3,
                                 stage: Optional[TrainingStage] = None) -> int:
        """
        Automatically select and log DPO pairs from episodes based on protocol scores.
        
        Args:
            episodes: List of episodes to select pairs from
            min_score_gap: Minimum score difference for pair selection
            stage: Training stage
            
        Returns:
            Number of DPO pairs created
        """
        from utils.protocol_validator import select_dpo_pairs
        
        # Select DPO pairs based on protocol compliance
        pairs = select_dpo_pairs(episodes, min_score_gap=min_score_gap)
        
        # Log each pair
        for chosen, rejected in pairs:
            self.log_dpo_pair(chosen, rejected, stage=stage)
        
        logger.info(f"Created {len(pairs)} DPO pairs from {len(episodes)} episodes")
        return len(pairs)
    
    def load_episodes(self, filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load all episodes from JSONL file into memory.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of episode dictionaries
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"File {filepath} not found")
            return []
        
        episodes = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
                        continue
        
        logger.info(f"Loaded {len(episodes)} episodes from {filepath}")
        return episodes
    
    def write_episodes(self, episodes: List[Dict[str, Any]], 
                      filepath: Union[str, Path],
                      format: str = 'raw') -> None:
        """
        Write episodes directly to file without buffering.
        
        Args:
            episodes: List of episodes to write
            filepath: Output file path
            format: Output format ('raw', 'sft', or 'dpo')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Transform episodes based on format
        if format == 'sft':
            output_items = []
            for episode in episodes:
                packets = self.transform_to_sft_packet(episode)
                output_items.extend(packets)
        elif format == 'dpo':
            # Assumes episodes are already DPO pairs
            output_items = episodes
        else:
            # Raw format
            output_items = episodes
        
        # Write NDJSON
        with open(filepath, 'w') as f:
            for item in output_items:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Wrote {len(output_items)} items to {filepath}")
    
    def batch_transform_to_sft(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform multiple episodes to SFT format.
        
        Args:
            episodes: List of episodes to transform
            
        Returns:
            List of SFT packets
        """
        sft_packets = []
        
        for episode in episodes:
            # Validate episode for training
            from utils.protocol_validator import validate_episode_for_training
            is_suitable, reasons = validate_episode_for_training(episode)
            
            if is_suitable:
                packets = self.transform_to_sft_packet(episode)
                sft_packets.extend(packets)
            else:
                logger.debug(f"Episode not suitable for SFT: {reasons}")
        
        logger.info(f"Transformed {len(episodes)} episodes into {len(sft_packets)} SFT packets")
        return sft_packets
    
    def batch_create_dpo_pairs(self, good_episodes: List[Dict[str, Any]], 
                               bad_episodes: List[Dict[str, Any]],
                               max_pairs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create DPO pairs from episode lists.
        
        Args:
            good_episodes: Episodes with good responses
            bad_episodes: Episodes with bad responses  
            max_pairs: Maximum number of pairs to create
            
        Returns:
            List of DPO pair packets
        """
        from utils.protocol_validator import validate_episode_for_training
        
        dpo_pairs = []
        
        # Filter suitable episodes
        good_suitable = []
        for ep in good_episodes:
            is_suitable, _ = validate_episode_for_training(ep)
            if is_suitable:
                good_suitable.append(ep)
        
        bad_suitable = []
        for ep in bad_episodes:
            is_suitable, reasons = validate_episode_for_training(ep)
            # Bad episodes can have protocol violations but not critical issues
            critical_issues = [r for r in reasons if 'leaked action tags' in r]
            if not critical_issues:
                bad_suitable.append(ep)
        
        # Create pairs
        min_length = min(len(good_suitable), len(bad_suitable))
        if max_pairs:
            min_length = min(min_length, max_pairs)
        
        for i in range(min_length):
            pair = self.transform_to_dpo_pair(good_suitable[i], bad_suitable[i])
            if pair:
                dpo_pairs.append(pair)
        
        logger.info(f"Created {len(dpo_pairs)} DPO pairs from {len(good_episodes)} good and {len(bad_episodes)} bad episodes")
        return dpo_pairs
    
    def filter_by_quality(self, episodes: List[Dict[str, Any]], 
                         min_score: float = 0.8) -> List[Dict[str, Any]]:
        """
        Filter episodes by protocol score quality.
        
        Args:
            episodes: Episodes to filter
            min_score: Minimum protocol score (0-1)
            
        Returns:
            List of high-quality episodes
        """
        from utils.protocol_validator import calculate_protocol_score
        
        filtered = []
        for episode in episodes:
            violations = episode.get('metrics', {}).get('protocol_violations', [])
            action_history = episode.get('metrics', {}).get('action_history', [])
            deal_closed = episode.get('metrics', {}).get('deal_closed', False)
            
            score = calculate_protocol_score(violations, action_history, deal_closed)
            
            if score >= min_score:
                filtered.append(episode)
        
        logger.info(f"Filtered {len(filtered)} high-quality episodes from {len(episodes)} total (min_score={min_score})")
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'stage': self.stage.value if self.stage is not None else 'none'
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush remaining buffer."""
        self.flush()