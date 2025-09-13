"""
Main data converter orchestrator for ParlAI negotiation training.
Coordinates conversion between ParlAI episodes and training formats.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from training.converters.builtin_export import BuiltinTaskExporter
from utils.episode_logger import EpisodeLogger, TrainingStage
from utils.protocol_validator import calculate_protocol_score, select_dpo_pairs

# Set up logging
from pathlib import Path

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Configure logging with 'w' mode to clear log on each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_converter.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataConverter:
    """Main orchestrator for data conversion pipeline using EpisodeLogger."""
    
    def __init__(self, output_dir: str = "data/training"):
        """
        Initialize the data converter with EpisodeLogger as the I/O layer.
        
        Args:
            output_dir: Base directory for all converted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize episode logger as the single source of truth for data handling
        self.episode_logger = EpisodeLogger(base_dir=str(self.output_dir), batch_size=1000)
        
        # Keep builtin exporter for ParlAI task compatibility
        self.builtin_exporter = BuiltinTaskExporter(str(self.output_dir / "builtin"))
    
    def convert_episodes_to_sft(self,
                               episodes_file: str,
                               output_file: Optional[str] = None,
                               history_window: int = 5) -> str:
        """
        Convert ParlAI episodes to SFT format.
        
        Args:
            episodes_file: Path to JSONL file with episodes
            output_file: Optional output path
            history_window: Number of recent turns for context (currently unused, kept for compatibility)
            
        Returns:
            Path to output file
        """
        # Load episodes
        episodes = self.episode_logger.load_episodes(episodes_file)
        
        # Transform to SFT format
        sft_packets = self.episode_logger.batch_transform_to_sft(episodes)
        
        # Determine output path
        if output_file is None:
            output_file = str(self.output_dir / "sft" / "episodes_sft.jsonl")
        
        # Write SFT packets
        self.episode_logger.write_episodes(sft_packets, output_file, format='raw')
        
        return output_file
    
    def create_dpo_pairs(self,
                        good_episodes: str,
                        bad_episodes: str,
                        output_file: Optional[str] = None,
                        max_pairs: Optional[int] = None) -> str:
        """
        Create DPO training pairs from good and bad episodes.
        
        Args:
            good_episodes: Path to JSONL with good episodes
            bad_episodes: Path to JSONL with bad episodes
            output_file: Optional output path
            max_pairs: Maximum number of pairs to generate
            
        Returns:
            Path to output file
        """
        # Load episodes
        good_eps = self.episode_logger.load_episodes(good_episodes)
        bad_eps = self.episode_logger.load_episodes(bad_episodes)
        
        # Create DPO pairs
        dpo_pairs = self.episode_logger.batch_create_dpo_pairs(good_eps, bad_eps, max_pairs)
        
        # Determine output path
        if output_file is None:
            output_file = str(self.output_dir / "dpo" / "dpo_pairs.jsonl")
        
        # Write DPO pairs
        self.episode_logger.write_episodes(dpo_pairs, output_file, format='dpo')
        
        return output_file
    
    def auto_generate_dpo(self,
                         episodes_file: str,
                         output_file: Optional[str] = None,
                         score_threshold: float = 0.3) -> str:
        """
        Automatically generate DPO pairs by scoring episodes.
        
        Args:
            episodes_file: Path to all episodes
            output_file: Optional output path
            score_threshold: Minimum protocol score difference for pairing (0-1 scale)
            
        Returns:
            Path to output file
        """
        # Load episodes
        episodes = self.episode_logger.load_episodes(episodes_file)
        
        # Use protocol validator to select DPO pairs
        pairs = select_dpo_pairs(episodes, min_score_gap=score_threshold)
        
        # Convert to DPO packet format
        dpo_packets = []
        for chosen, rejected in pairs:
            pair = self.episode_logger.transform_to_dpo_pair(chosen, rejected)
            if pair:
                dpo_packets.append(pair)
        
        # Determine output path
        if output_file is None:
            output_file = str(self.output_dir / "dpo" / "auto_dpo_pairs.jsonl")
        
        # Write DPO pairs
        self.episode_logger.write_episodes(dpo_packets, output_file, format='dpo')
        
        logger.info(f"Generated {len(dpo_packets)} DPO pairs from {len(episodes)} episodes")
        return output_file
    
    def export_builtin_tasks(self,
                            tasks: Optional[List[str]] = None,
                            max_episodes: int = 1000,
                            convert_to_sft: bool = True) -> Dict[str, str]:
        """
        Export and convert built-in ParlAI tasks.
        
        Args:
            tasks: List of task names (default: ['dealnodeal', 'casino'])
            max_episodes: Max episodes per task
            convert_to_sft: Whether to also convert to SFT format
            
        Returns:
            Dict mapping task names to output files
        """
        if tasks is None:
            tasks = ['dealnodeal', 'casino']
        
        # Export tasks using builtin exporter
        episode_files = self.builtin_exporter.export_all_tasks(
            tasks, max_episodes
        )
        
        # Convert to SFT if requested using EpisodeLogger
        if convert_to_sft:
            sft_files = {}
            for task_name, episode_file in episode_files.items():
                # Load episodes
                episodes = self.episode_logger.load_episodes(episode_file)
                
                # Transform to SFT
                sft_packets = self.episode_logger.batch_transform_to_sft(episodes)
                
                # Write SFT file
                sft_file = str(self.output_dir / "builtin" / f"{task_name}_sft.jsonl")
                self.episode_logger.write_episodes(sft_packets, sft_file, format='raw')
                
                sft_files[f"{task_name}_sft"] = sft_file
            
            # Merge results
            episode_files.update(sft_files)
        
        return episode_files
    
    def score_episodes(self,
                      episodes_file: str,
                      output_file: Optional[str] = None) -> str:
        """
        Score episodes using protocol validation and save with scores.
        
        Args:
            episodes_file: Path to episodes JSONL
            output_file: Optional output path
            
        Returns:
            Path to scored episodes file
        """
        # Load episodes
        episodes = self.episode_logger.load_episodes(episodes_file)
        
        # Score each episode using protocol validator
        for episode in episodes:
            violations = episode.get('metrics', {}).get('protocol_violations', [])
            action_history = episode.get('metrics', {}).get('action_history', [])
            deal_closed = episode.get('metrics', {}).get('deal_closed', False)
            
            # Calculate protocol score (0-1)
            protocol_score = calculate_protocol_score(violations, action_history, deal_closed)
            
            # Add scoring metadata
            episode['meta'] = episode.get('meta', {})
            episode['meta']['protocol_score'] = protocol_score
            
            # Classify quality tier based on protocol score
            if protocol_score >= 0.9:
                quality_tier = 'excellent'
            elif protocol_score >= 0.7:
                quality_tier = 'good'
            elif protocol_score >= 0.5:
                quality_tier = 'fair'
            else:
                quality_tier = 'poor'
            
            episode['meta']['quality_tier'] = quality_tier
        
        # Sort by protocol score (highest first)
        episodes.sort(key=lambda e: e.get('meta', {}).get('protocol_score', 0), reverse=True)
        
        # Determine output path
        if output_file is None:
            output_file = str(self.output_dir / "scored" / "scored_episodes.jsonl")
        
        # Write scored episodes
        self.episode_logger.write_episodes(episodes, output_file, format='raw')
        
        # Log statistics
        logger.info(f"Scored {len(episodes)} episodes:")
        tiers = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for ep in episodes:
            tiers[ep['meta']['quality_tier']] += 1
        
        for tier, count in tiers.items():
            logger.info(f"  {tier}: {count} episodes")
        
        logger.info(f"Saved to {output_file}")
        return output_file
    
    def prepare_training_data(self,
                             episodes_file: str,
                             output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Full pipeline to prepare training data from episodes.
        
        Args:
            episodes_file: Path to raw episodes
            output_dir: Optional output directory
            
        Returns:
            Dict with paths to all generated files
        """
        if output_dir:
            base_dir = Path(output_dir)
        else:
            base_dir = self.output_dir / "prepared"
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. Score episodes
        logger.info("1. Scoring episodes...")
        scored_file = str(base_dir / "scored_episodes.jsonl")
        scored_file = self.score_episodes(episodes_file, scored_file)
        results['scored_episodes'] = scored_file
        
        # 2. Convert all to SFT
        logger.info("2. Converting to SFT format...")
        sft_file = str(base_dir / "sft" / "all_episodes.jsonl")
        sft_file = self.convert_episodes_to_sft(scored_file, sft_file)
        results['sft_all'] = sft_file
        
        # 3. Filter high-quality episodes for SFT
        logger.info("3. Filtering high-quality SFT examples...")
        
        # Load scored episodes and filter by protocol score
        episodes = self.episode_logger.load_episodes(scored_file)
        high_quality = self.episode_logger.filter_by_quality(episodes, min_score=0.8)
        
        # Transform to SFT and save
        sft_packets = self.episode_logger.batch_transform_to_sft(high_quality)
        sft_good = str(base_dir / "sft" / "high_quality.jsonl")
        self.episode_logger.write_episodes(sft_packets, sft_good, format='raw')
        results['sft_high_quality'] = sft_good
        
        # 4. Auto-generate DPO pairs
        logger.info("4. Generating DPO pairs...")
        dpo_file = str(base_dir / "dpo" / "auto_pairs.jsonl")
        dpo_file = self.auto_generate_dpo(scored_file, dpo_file, score_threshold=0.3)
        results['dpo_pairs'] = dpo_file
        
        logger.info("✓ Training data preparation complete!")
        logger.info(f"  Output directory: {base_dir}")
        
        return results


def main():
    """Command-line interface for data conversion."""
    parser = argparse.ArgumentParser(
        description="Convert ParlAI negotiation episodes to training formats"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # SFT conversion
    sft_parser = subparsers.add_parser('sft', help='Convert to SFT format')
    sft_parser.add_argument('--input', required=True, help='Input episodes file')
    sft_parser.add_argument('--output', help='Output file')
    sft_parser.add_argument('--history-window', type=int, default=5,
                           help='Number of recent turns for context')
    
    # DPO pair creation
    dpo_parser = subparsers.add_parser('dpo', help='Create DPO pairs')
    dpo_parser.add_argument('--good', required=True, help='Good episodes file')
    dpo_parser.add_argument('--bad', required=True, help='Bad episodes file')
    dpo_parser.add_argument('--output', help='Output file')
    dpo_parser.add_argument('--max-pairs', type=int, help='Maximum pairs to generate')
    
    # Auto DPO generation
    auto_dpo_parser = subparsers.add_parser('auto-dpo', help='Auto-generate DPO pairs')
    auto_dpo_parser.add_argument('--input', required=True, help='Input episodes file')
    auto_dpo_parser.add_argument('--output', help='Output file')
    auto_dpo_parser.add_argument('--threshold', type=float, default=0.3,
                                help='Minimum protocol score difference (0-1 scale)')
    
    # Built-in task export
    builtin_parser = subparsers.add_parser('builtin', help='Export built-in tasks')
    builtin_parser.add_argument('--tasks', nargs='+',
                               default=['dealnodeal', 'casino'],
                               help='Tasks to export')
    builtin_parser.add_argument('--max-episodes', type=int, default=1000,
                               help='Max episodes per task')
    builtin_parser.add_argument('--no-sft', action='store_true',
                               help='Skip SFT conversion')
    
    # Score episodes
    score_parser = subparsers.add_parser('score', help='Score episodes')
    score_parser.add_argument('--input', required=True, help='Input episodes file')
    score_parser.add_argument('--output', help='Output file')
    
    # Full pipeline
    pipeline_parser = subparsers.add_parser('prepare', help='Full data preparation')
    pipeline_parser.add_argument('--input', required=True, help='Input episodes file')
    pipeline_parser.add_argument('--output-dir', help='Output directory')
    
    # General arguments
    parser.add_argument('--data-dir', default='data/training',
                       help='Base directory for output data')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = DataConverter(output_dir=args.data_dir)
    
    # Execute command
    if args.command == 'sft':
        result = converter.convert_episodes_to_sft(
            args.input, args.output, args.history_window
        )
        logger.info(f"SFT conversion complete: {result}")
    
    elif args.command == 'dpo':
        result = converter.create_dpo_pairs(
            args.good, args.bad, args.output, args.max_pairs
        )
        logger.info(f"DPO pairs created: {result}")
    
    elif args.command == 'auto-dpo':
        result = converter.auto_generate_dpo(
            args.input, args.output, args.threshold
        )
        logger.info(f"Auto-generated DPO pairs: {result}")
    
    elif args.command == 'builtin':
        results = converter.export_builtin_tasks(
            args.tasks, args.max_episodes, not args.no_sft
        )
        logger.info("Exported files:")
        for name, path in results.items():
            logger.info(f"  {name}: {path}")
    
    elif args.command == 'score':
        result = converter.score_episodes(args.input, args.output)
        logger.info(f"Scored episodes: {result}")
    
    elif args.command == 'prepare':
        results = converter.prepare_training_data(args.input, args.output_dir)
        logger.info("Generated files:")
        for name, path in results.items():
            logger.info(f"  {name}: {path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()