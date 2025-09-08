"""
CSV logging functionality for metrics persistence using pandas.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


class CSVLogger:
    """
    Log metrics to CSV files for persistence and analysis using pandas.
    """
    
    def __init__(self, data_dir: Union[str, Path] = 'data/metrics'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for different metric types
        self.episode_file = self.data_dir / 'episodes.csv'
        self.session_file = self.data_dir / 'sessions.csv'
        
        # Initialize files with headers if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Create CSV files with headers if they don't exist."""
        # Episode metrics columns
        episode_columns = [
            'episode_id', 'dialogue_id', 'side', 'seed', 'timestamp',
            'turns', 'total_tokens', 'agent_tokens', 'deal_closed',
            'final_price', 'terms_match_pct', 'price_vs_target', 'price_vs_limit',
            'violation_count', 'num_proposals', 'num_counteroffers',
            'price_convergence', 'avg_response_length', 'negotiation_rounds',
            'verbosity_score', 'uses_reasoning', 'uses_empathy', 'uses_urgency',
            'professionalism_score', 'protocol_violations'
        ]
        
        # Session metrics columns
        session_columns = [
            'session_id', 'timestamp', 'duration_seconds', 'total_episodes',
            'avg_turns', 'avg_tokens', 'avg_agent_tokens', 'deals_closed_pct',
            'avg_violations_per_episode', 'tokens_per_second',
            'conversations_per_second', 'agent_tokens_per_second',
            'avg_terms_match', 'std_terms_match',
            'buyer_avg_price_vs_target', 'buyer_deals_closed_pct',
            'seller_avg_price_vs_target', 'seller_deals_closed_pct',
            'avg_proposals', 'avg_rounds', 'price_convergence_rate',
            'avg_verbosity', 'reasoning_usage_pct', 'empathy_usage_pct',
            'urgency_usage_pct', 'avg_professionalism', 'top_violations'
        ]
        
        # Create episode file if needed
        if not self.episode_file.exists():
            df = pd.DataFrame(columns=episode_columns)
            df.to_csv(self.episode_file, index=False)
            logger.info(f"Created episode metrics file: {self.episode_file}")
        
        # Create session file if needed
        if not self.session_file.exists():
            df = pd.DataFrame(columns=session_columns)
            df.to_csv(self.session_file, index=False)
            logger.info(f"Created session metrics file: {self.session_file}")
    
    def log_episode(self, metrics: Dict[str, Any], stage: Optional[str] = None):
        """
        Log episode metrics to CSV file.
        
        Args:
            metrics: Episode metrics dictionary
            stage: Optional training stage identifier for separate files
        """
        # Determine file path
        if stage:
            file_path = self.data_dir / f'episodes_{stage}.csv'
            # Initialize stage-specific file if needed
            if not file_path.exists():
                self._initialize_stage_file(file_path, 'episode')
        else:
            file_path = self.episode_file
        
        # Prepare metrics for pandas
        metrics_copy = metrics.copy()
        
        # Convert protocol violations list to string
        if 'protocol_violations' in metrics_copy:
            metrics_copy['protocol_violations'] = '|'.join(metrics_copy['protocol_violations'])
        
        # Convert single row dict to DataFrame
        df_new = pd.DataFrame([metrics_copy])
        
        # Read existing data if file exists and has content
        if file_path.exists() and file_path.stat().st_size > 0:
            df_existing = pd.read_csv(file_path)
            # Concatenate with existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Write back to CSV
            df_combined.to_csv(file_path, index=False)
        else:
            # Write new DataFrame
            df_new.to_csv(file_path, index=False)
        
        logger.debug(f"Logged episode {metrics.get('episode_id')} to {file_path}")
    
    def log_session(self, metrics: Dict[str, Any], stage: Optional[str] = None):
        """
        Log session metrics to CSV file.
        
        Args:
            metrics: Session metrics dictionary
            stage: Optional training stage identifier
        """
        # Determine file path
        if stage:
            file_path = self.data_dir / f'sessions_{stage}.csv'
            # Initialize stage-specific file if needed
            if not file_path.exists():
                self._initialize_stage_file(file_path, 'session')
        else:
            file_path = self.session_file
        
        # Prepare metrics for pandas
        metrics_copy = metrics.copy()
        
        # Convert top violations list to string
        if 'top_violations' in metrics_copy:
            metrics_copy['top_violations'] = '|'.join(metrics_copy['top_violations'])
        
        # Convert single row dict to DataFrame
        df_new = pd.DataFrame([metrics_copy])
        
        # Read existing data if file exists and has content
        if file_path.exists() and file_path.stat().st_size > 0:
            df_existing = pd.read_csv(file_path)
            # Concatenate with existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Write back to CSV
            df_combined.to_csv(file_path, index=False)
        else:
            # Write new DataFrame
            df_new.to_csv(file_path, index=False)
        
        logger.info(f"Logged session {metrics.get('session_id')} to {file_path}")
    
    def _initialize_stage_file(self, file_path: Path, metric_type: str):
        """Initialize a stage-specific metrics file."""
        if metric_type == 'episode':
            columns = [
                'episode_id', 'dialogue_id', 'side', 'seed', 'timestamp',
                'turns', 'total_tokens', 'agent_tokens', 'deal_closed',
                'final_price', 'terms_match_pct', 'price_vs_target', 'price_vs_limit',
                'violation_count', 'num_proposals', 'num_counteroffers',
                'price_convergence', 'avg_response_length', 'negotiation_rounds',
                'verbosity_score', 'uses_reasoning', 'uses_empathy', 'uses_urgency',
                'professionalism_score', 'protocol_violations'
            ]
        else:  # session
            columns = [
                'session_id', 'timestamp', 'duration_seconds', 'total_episodes',
                'avg_turns', 'avg_tokens', 'avg_agent_tokens', 'deals_closed_pct',
                'avg_violations_per_episode', 'tokens_per_second',
                'conversations_per_second', 'agent_tokens_per_second',
                'avg_terms_match', 'std_terms_match',
                'buyer_avg_price_vs_target', 'buyer_deals_closed_pct',
                'seller_avg_price_vs_target', 'seller_deals_closed_pct',
                'avg_proposals', 'avg_rounds', 'price_convergence_rate',
                'avg_verbosity', 'reasoning_usage_pct', 'empathy_usage_pct',
                'urgency_usage_pct', 'avg_professionalism', 'top_violations'
            ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        logger.info(f"Created stage-specific {metric_type} file: {file_path}")
    
    def read_episodes(self, stage: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Read episode metrics from CSV file.
        
        Args:
            stage: Optional stage identifier to read stage-specific file
            limit: Optional limit on number of records to read
            
        Returns:
            DataFrame of episode metrics
        """
        file_path = self.data_dir / f'episodes_{stage}.csv' if stage else self.episode_file
        
        if not file_path.exists():
            logger.warning(f"Episode file {file_path} does not exist")
            return pd.DataFrame()
        
        # Read CSV with pandas
        df = pd.read_csv(file_path, nrows=limit)
        
        # Convert protocol violations back to list
        if 'protocol_violations' in df.columns:
            df['protocol_violations'] = df['protocol_violations'].fillna('').apply(
                lambda x: x.split('|') if x else []
            )
        
        # Ensure boolean columns are proper type
        bool_columns = ['deal_closed', 'price_convergence', 'uses_reasoning', 
                       'uses_empathy', 'uses_urgency']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
    
    def read_sessions(self, stage: Optional[str] = None) -> pd.DataFrame:
        """
        Read session metrics from CSV file.
        
        Args:
            stage: Optional stage identifier to read stage-specific file
            
        Returns:
            DataFrame of session metrics
        """
        file_path = self.data_dir / f'sessions_{stage}.csv' if stage else self.session_file
        
        if not file_path.exists():
            logger.warning(f"Session file {file_path} does not exist")
            return pd.DataFrame()
        
        # Read CSV with pandas
        df = pd.read_csv(file_path)
        
        # Convert top violations back to list
        if 'top_violations' in df.columns:
            df['top_violations'] = df['top_violations'].fillna('').apply(
                lambda x: x.split('|') if x else []
            )
        
        return df
    
    def get_episode_summary(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for episode metrics.
        
        Args:
            stage: Optional stage identifier
            
        Returns:
            Dictionary of summary statistics
        """
        df = self.read_episodes(stage)
        
        if df.empty:
            return {}
        
        summary = {
            'total_episodes': len(df),
            'deals_closed': df['deal_closed'].sum() if 'deal_closed' in df.columns else 0,
            'avg_turns': df['turns'].mean() if 'turns' in df.columns else 0,
            'avg_tokens': df['total_tokens'].mean() if 'total_tokens' in df.columns else 0,
            'avg_final_price': df[df['deal_closed'] == True]['final_price'].mean() if 'final_price' in df.columns else 0,
            'avg_violations': df['violation_count'].mean() if 'violation_count' in df.columns else 0,
        }
        
        # Side-specific stats
        if 'side' in df.columns:
            for side in ['buyer', 'seller']:
                side_df = df[df['side'] == side]
                if not side_df.empty:
                    summary[f'{side}_episodes'] = len(side_df)
                    summary[f'{side}_deals_closed'] = side_df['deal_closed'].sum() if 'deal_closed' in side_df.columns else 0
        
        return summary
    
    def get_session_summary(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for session metrics.
        
        Args:
            stage: Optional stage identifier
            
        Returns:
            Dictionary of summary statistics
        """
        df = self.read_sessions(stage)
        
        if df.empty:
            return {}
        
        summary = {
            'total_sessions': len(df),
            'total_episodes_processed': df['total_episodes'].sum() if 'total_episodes' in df.columns else 0,
            'avg_deal_close_rate': df['deals_closed_pct'].mean() if 'deals_closed_pct' in df.columns else 0,
            'avg_tokens_per_second': df['tokens_per_second'].mean() if 'tokens_per_second' in df.columns else 0,
            'total_duration_hours': df['duration_seconds'].sum() / 3600 if 'duration_seconds' in df.columns else 0,
        }
        
        return summary