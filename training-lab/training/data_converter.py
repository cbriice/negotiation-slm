"""
Main data converter orchestrator for ParlAI negotiation training.
Coordinates conversion between ParlAI episodes and training formats.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from training.converters.sft_converter import SFTConverter
from training.converters.dpo_converter import DPOConverter
from training.converters.builtin_export import BuiltinTaskExporter
from training.converters.scoring import EpisodeScorer


class DataConverter:
    """Main orchestrator for data conversion pipeline."""
    
    def __init__(self, output_dir: str = "data/training"):
        """
        Initialize the data converter with all sub-converters.
        
        Args:
            output_dir: Base directory for all converted data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize converters
        self.sft_converter = SFTConverter(str(self.output_dir / "sft"))
        self.dpo_converter = DPOConverter(str(self.output_dir / "dpo"))
        self.builtin_exporter = BuiltinTaskExporter(str(self.output_dir / "builtin"))
        self.scorer = EpisodeScorer()
    
    def convert_episodes_to_sft(self,
                               episodes_file: str,
                               output_file: Optional[str] = None,
                               history_window: int = 5) -> str:
        """
        Convert ParlAI episodes to SFT format.
        
        Args:
            episodes_file: Path to JSONL file with episodes
            output_file: Optional output path
            history_window: Number of recent turns for context
            
        Returns:
            Path to output file
        """
        return self.sft_converter.batch_convert(
            episodes_file, output_file, history_window
        )
    
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
        return self.dpo_converter.batch_create_pairs(
            good_episodes, bad_episodes, output_file, max_pairs
        )
    
    def auto_generate_dpo(self,
                         episodes_file: str,
                         output_file: Optional[str] = None,
                         score_threshold: float = 5.0) -> str:
        """
        Automatically generate DPO pairs by scoring episodes.
        
        Args:
            episodes_file: Path to all episodes
            output_file: Optional output path
            score_threshold: Minimum score difference for pairing
            
        Returns:
            Path to output file
        """
        return self.dpo_converter.auto_generate_pairs(
            episodes_file, output_file, score_threshold
        )
    
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
        
        # Export tasks
        episode_files = self.builtin_exporter.export_all_tasks(
            tasks, max_episodes
        )
        
        # Convert to SFT if requested
        if convert_to_sft:
            sft_files = {}
            for task_name, episode_file in episode_files.items():
                sft_file = self.builtin_exporter.convert_to_sft(
                    task_name, episode_file
                )
                sft_files[f"{task_name}_sft"] = sft_file
            
            # Merge results
            episode_files.update(sft_files)
        
        return episode_files
    
    def score_episodes(self,
                      episodes_file: str,
                      output_file: Optional[str] = None) -> str:
        """
        Score episodes and save with quality scores.
        
        Args:
            episodes_file: Path to episodes JSONL
            output_file: Optional output path
            
        Returns:
            Path to scored episodes file
        """
        episodes = self.sft_converter.load_jsonl(Path(episodes_file))
        
        # Score and rank episodes
        scored_episodes = self.scorer.rank_episodes(episodes)
        
        # Add quality tier classification
        for episode in scored_episodes:
            episode['quality_tier'] = self.scorer.classify_episode_quality(episode)
            episode['score_breakdown'] = self.scorer.get_score_breakdown(episode)
        
        # Save scored episodes
        if output_file is None:
            output_path = self.output_dir / "scored" / "scored_episodes.jsonl"
        else:
            output_path = Path(output_file)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sft_converter.save_jsonl(scored_episodes, output_path)
        
        # Print statistics
        print(f"\nScored {len(scored_episodes)} episodes:")
        tiers = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for ep in scored_episodes:
            tiers[ep['quality_tier']] += 1
        
        for tier, count in tiers.items():
            print(f"  {tier}: {count} episodes")
        
        print(f"\nSaved to {output_path}")
        return str(output_path)
    
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
        print("\n1. Scoring episodes...")
        scored_file = str(base_dir / "scored_episodes.jsonl")
        scored_file = self.score_episodes(episodes_file, scored_file)
        results['scored_episodes'] = scored_file
        
        # 2. Convert all to SFT
        print("\n2. Converting to SFT format...")
        sft_file = str(base_dir / "sft" / "all_episodes.jsonl")
        sft_file = self.convert_episodes_to_sft(scored_file, sft_file)
        results['sft_all'] = sft_file
        
        # 3. Filter high-quality SFT
        print("\n3. Filtering high-quality SFT examples...")
        sft_good = str(base_dir / "sft" / "high_quality.jsonl")
        sft_good = self.sft_converter.filter_by_quality(sft_file, 8.0, sft_good)
        results['sft_high_quality'] = sft_good
        
        # 4. Auto-generate DPO pairs
        print("\n4. Generating DPO pairs...")
        dpo_file = str(base_dir / "dpo" / "auto_pairs.jsonl")
        dpo_file = self.auto_generate_dpo(scored_file, dpo_file)
        results['dpo_pairs'] = dpo_file
        
        print("\n✓ Training data preparation complete!")
        print(f"  Output directory: {base_dir}")
        
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
    auto_dpo_parser.add_argument('--threshold', type=float, default=5.0,
                                help='Minimum score difference')
    
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
        print(f"SFT conversion complete: {result}")
    
    elif args.command == 'dpo':
        result = converter.create_dpo_pairs(
            args.good, args.bad, args.output, args.max_pairs
        )
        print(f"DPO pairs created: {result}")
    
    elif args.command == 'auto-dpo':
        result = converter.auto_generate_dpo(
            args.input, args.output, args.threshold
        )
        print(f"Auto-generated DPO pairs: {result}")
    
    elif args.command == 'builtin':
        results = converter.export_builtin_tasks(
            args.tasks, args.max_episodes, not args.no_sft
        )
        print("\nExported files:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    
    elif args.command == 'score':
        result = converter.score_episodes(args.input, args.output)
        print(f"Scored episodes: {result}")
    
    elif args.command == 'prepare':
        results = converter.prepare_training_data(args.input, args.output_dir)
        print("\nGenerated files:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()