"""
Integration helper for connecting metrics tracking with ParlAI teachers.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from evaluation.episode_metrics import EpisodeMetrics
from evaluation.session_metrics import SessionMetrics
from evaluation.csv_logger import CSVLogger

logger = logging.getLogger(__name__)


class MetricsIntegration:
    """
    Helper class to integrate metrics tracking with ParlAI teachers.
    Provides methods to track episodes, calculate metrics, and persist to CSV.
    """
    
    def __init__(self, logger: Optional[CSVLogger] = None, data_dir: str = 'data/metrics'):
        """
        Initialize metrics integration.
        
        Args:
            logger: Optional CSVLogger instance
            data_dir: Directory for metrics storage
        """
        self.episode_metrics = EpisodeMetrics()
        self.session_metrics = SessionMetrics()
        self.csv_logger = logger or CSVLogger(data_dir)
        self.current_episodes = []
        self.current_stage = None
    
    def set_stage(self, stage: str):
        """
        Set the current training stage for metrics organization.
        
        Args:
            stage: Training stage identifier (e.g., 'baseline', 'selfchat', 'dialog_partner')
        """
        self.current_stage = stage
        logger.info(f"Metrics tracking stage set to: {stage}")
    
    def track_episode(self, episode: Dict[str, Any], stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Track metrics for a single episode.
        
        Args:
            episode: Episode data containing dialogue, state, and metadata
            stage: Optional training stage identifier (overrides current_stage)
            
        Returns:
            Calculated metrics dictionary
        """
        # Calculate metrics
        metrics = self.episode_metrics.calculate(episode)
        
        # Add to current session
        self.current_episodes.append(metrics)
        
        # Determine stage
        stage = stage or self.current_stage
        
        # Log to CSV
        self.csv_logger.log_episode(metrics, stage)
        
        # Log summary
        logger.debug(
            f"Episode {metrics['episode_id']}: "
            f"turns={metrics['turns']}, "
            f"deal_closed={metrics['deal_closed']}, "
            f"violations={metrics['violation_count']}"
        )
        
        return metrics
    
    def track_batch(self, episodes: List[Dict[str, Any]], stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Track metrics for a batch of episodes.
        
        Args:
            episodes: List of episode data
            stage: Optional training stage identifier
            
        Returns:
            List of calculated metrics dictionaries
        """
        metrics_list = []
        for episode in episodes:
            metrics = self.track_episode(episode, stage)
            metrics_list.append(metrics)
        
        logger.info(f"Tracked batch of {len(episodes)} episodes")
        return metrics_list
    
    def finalize_session(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Finalize session and calculate aggregate metrics.
        
        Args:
            stage: Optional training stage identifier
            
        Returns:
            Session metrics dictionary
        """
        # Calculate aggregate metrics
        session_metrics = self.session_metrics.aggregate(self.current_episodes)
        
        # Determine stage
        stage = stage or self.current_stage
        
        # Log to CSV
        self.csv_logger.log_session(session_metrics, stage)
        
        # Log summary
        logger.info(
            f"Session {session_metrics['session_id']} finalized: "
            f"episodes={session_metrics['total_episodes']}, "
            f"deals_closed={session_metrics['deals_closed_pct']:.1f}%, "
            f"avg_turns={session_metrics['avg_turns']:.1f}, "
            f"tokens/s={session_metrics['tokens_per_second']:.1f}"
        )
        
        # Reset for next session
        self.current_episodes = []
        self.session_metrics = SessionMetrics()
        
        return session_metrics
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics without finalizing.
        
        Returns:
            Current session metrics
        """
        if not self.current_episodes:
            return self.session_metrics._empty_metrics()
        return self.session_metrics.aggregate(self.current_episodes)
    
    def get_episode_count(self) -> int:
        """Get count of episodes in current session."""
        return len(self.current_episodes)
    
    def get_deal_close_rate(self) -> float:
        """Get current deal close rate for session."""
        if not self.current_episodes:
            return 0.0
        deals_closed = sum(1 for ep in self.current_episodes if ep.get('deal_closed', False))
        return (deals_closed / len(self.current_episodes)) * 100
    
    def get_violation_summary(self) -> Dict[str, int]:
        """
        Get summary of protocol violations in current session.
        
        Returns:
            Dictionary mapping violation types to counts
        """
        violation_counts = {}
        for episode in self.current_episodes:
            for violation in episode.get('protocol_violations', []):
                # Extract violation type
                vtype = violation.split(':')[1].strip() if ':' in violation else violation
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        return violation_counts
    
    def export_session_data(self, filepath: Optional[Path] = None) -> Path:
        """
        Export current session episode data to JSONL file.
        
        Args:
            filepath: Optional output path
            
        Returns:
            Path to exported file
        """
        import json
        
        if filepath is None:
            filepath = Path('data/exports') / f"session_{self.session_metrics.start_time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            for episode in self.current_episodes:
                f.write(json.dumps(episode) + '\n')
        
        logger.info(f"Exported {len(self.current_episodes)} episodes to {filepath}")
        return filepath
    
    def log_summary(self):
        """Log a formatted summary of current session metrics."""
        stats = self.get_current_stats()
        violations = self.get_violation_summary()
        
        logger.info("="*50)
        logger.info("SESSION METRICS SUMMARY")
        logger.info("="*50)
        logger.info(f"Episodes: {stats['total_episodes']}")
        logger.info(f"Deals Closed: {stats['deals_closed_pct']:.1f}%")
        logger.info(f"Avg Turns: {stats['avg_turns']:.1f}")
        logger.info(f"Avg Tokens: {stats['avg_tokens']:.1f}")
        logger.info(f"Tokens/sec: {stats['tokens_per_second']:.1f}")
        logger.info(f"Avg Violations: {stats['avg_violations_per_episode']:.2f}")
        
        if stats.get('buyer_deals_closed_pct') is not None:
            logger.info("Buyer Performance:")
            logger.info(f"  Deals Closed: {stats['buyer_deals_closed_pct']:.1f}%")
            logger.info(f"  Avg Price vs Target: {stats.get('buyer_avg_price_vs_target', 0):.1f}%")
        
        if stats.get('seller_deals_closed_pct') is not None:
            logger.info("Seller Performance:")
            logger.info(f"  Deals Closed: {stats['seller_deals_closed_pct']:.1f}%")
            logger.info(f"  Avg Price vs Target: {stats.get('seller_avg_price_vs_target', 0):.1f}%")
        
        if violations:
            logger.info("Top Violations:")
            for vtype, count in sorted(violations.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {vtype}: {count}")
        
        logger.info("="*50)