"""Metrics tracking for QLoRA training, integrated with evaluation metrics."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch
import psutil
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Import existing evaluation metrics
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.episode_metrics import EpisodeMetrics
from evaluation.session_metrics import SessionMetrics
from evaluation.csv_logger import CSVLogger


class TrainingMetricsTracker:
    """
    Tracks training-specific metrics and integrates with evaluation metrics.
    Extends the evaluation system with training-specific tracking using pandas.
    """
    
    def __init__(self, log_dir: str = "logs", stage: str = "training"):
        """
        Initialize training metrics tracker.
        
        Args:
            log_dir: Directory for saving logs
            stage: Training stage (baseline, sft, dpo)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        
        logger.info(f"TrainingMetricsTracker initialized: stage={stage}, log_dir={log_dir}")
        
        # Initialize evaluation metric trackers
        self.episode_metrics = EpisodeMetrics()
        self.session_metrics = SessionMetrics()
        self.csv_logger = CSVLogger(data_dir=self.log_dir / "metrics")
        
        # Training metrics DataFrame
        self.training_df = pd.DataFrame()
        
        # Session tracking
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_csv = self.log_dir / f"training_{self.stage}_{self.current_session}.csv"
        logger.debug(f"Session ID: {self.current_session}")
        logger.debug(f"Training CSV: {self.training_csv}")
        
        # Episode buffer for batch metrics
        self.episode_buffer = []
        self.last_step_time = datetime.now()
        
        # Initialize training CSV with columns
        self._init_training_csv()
        
    def _init_training_csv(self):
        """Initialize training-specific CSV file with pandas."""
        columns = [
            "timestamp", "step", "epoch", "train_loss", "eval_loss",
            "learning_rate", "gradient_norm", "tokens_per_second",
            "vram_gb", "cpu_percent", "gpu_util",
            "reward_margin", "kl_divergence", "reward_chosen", "reward_rejected",
            "step_duration"
        ]
        
        # Create empty DataFrame with columns
        self.training_df = pd.DataFrame(columns=columns)
        # Save empty CSV with headers
        self.training_df.to_csv(self.training_csv, index=False)
        logger.debug(f"Initialized CSV with {len(columns)} columns")
    
    def log_training_step(
        self,
        step: int,
        epoch: float,
        metrics: Dict[str, Any],
        log_to_console: bool = True
    ):
        """
        Log metrics for a training step using pandas.
        
        Args:
            step: Current training step
            epoch: Current epoch (can be fractional)
            metrics: Dictionary of metrics to log
            log_to_console: Whether to print to console
        """
        timestamp = datetime.now()
        step_duration = (timestamp - self.last_step_time).total_seconds()
        self.last_step_time = timestamp
        
        logger.debug(f"Logging training step {step}, epoch {epoch:.2f}")
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Combine all metrics
        full_metrics = {
            "timestamp": timestamp.isoformat(),
            "step": step,
            "epoch": epoch,
            "step_duration": step_duration,
            **metrics,
            **system_metrics
        }
        
        # Append to DataFrame - handle empty DataFrame properly
        new_row = pd.DataFrame([full_metrics])
        if self.training_df.empty:
            self.training_df = new_row.copy()
            # First row - write with headers
            self.training_df.to_csv(self.training_csv, index=False)
            logger.debug(f"Wrote first row with headers to CSV")
        else:
            self.training_df = pd.concat([self.training_df, new_row], ignore_index=True)
            # Append without headers
            new_row.to_csv(self.training_csv, mode='a', header=False, index=False)
            logger.debug(f"Appended row {len(self.training_df)} to CSV")
        
        # Console output
        if log_to_console:
            self._print_training_metrics(step, epoch, metrics, system_metrics, step_duration)
        
        # Log key metrics
        if "train_loss" in metrics:
            logger.info(f"Step {step}: train_loss={metrics['train_loss']:.4f}")
        if "eval_loss" in metrics:
            logger.info(f"Step {step}: eval_loss={metrics['eval_loss']:.4f}")
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """
        Log episode metrics using evaluation system.
        
        Args:
            episode_data: Episode data from ParlAI world
        """
        logger.debug(f"Logging episode data")
        
        # Calculate episode metrics using evaluation system
        metrics = self.episode_metrics.calculate(episode_data)
        logger.debug(f"Episode metrics calculated: turns={metrics.get('turns', 0)}, deal_closed={metrics.get('deal_closed', False)}")
        
        # Add training stage to metrics
        metrics['training_stage'] = self.stage
        
        # Log to CSV via evaluation system (uses pandas internally)
        self.csv_logger.log_episode(metrics, stage=self.stage)
        
        # Buffer for session aggregation
        self.episode_buffer.append(metrics)
        
        return metrics
    
    def log_session_summary(self, additional_metrics: Optional[Dict] = None):
        """
        Log session summary using buffered episodes.
        
        Args:
            additional_metrics: Additional training-specific metrics
        """
        logger.info(f"Generating session summary for {len(self.episode_buffer)} episodes")
        
        if not self.episode_buffer:
            logger.warning("No episodes in buffer for session summary")
            return {}
        
        # Calculate session metrics using evaluation system
        session_metrics = self.session_metrics.aggregate(self.episode_buffer)
        
        # Add training-specific metrics
        session_metrics['training_stage'] = self.stage
        
        if additional_metrics:
            session_metrics.update(additional_metrics)
        
        # Add training performance metrics from DataFrame
        if not self.training_df.empty:
            if 'train_loss' in self.training_df.columns:
                train_losses = self.training_df['train_loss'].dropna()
                if not train_losses.empty:
                    session_metrics['final_train_loss'] = train_losses.iloc[-1]
                    session_metrics['best_train_loss'] = train_losses.min()
                    session_metrics['avg_train_loss'] = train_losses.mean()
            
            if 'tokens_per_second' in self.training_df.columns:
                tps = self.training_df['tokens_per_second'].dropna()
                if not tps.empty:
                    session_metrics['avg_training_tps'] = tps.mean()
                    session_metrics['max_training_tps'] = tps.max()
        
        # Log to CSV via evaluation system (uses pandas internally)
        self.csv_logger.log_session(session_metrics, stage=self.stage)
        
        # Clear episode buffer
        self.episode_buffer = []
        
        logger.info(f"Session summary logged: {len(session_metrics)} metrics")
        logger.debug(f"Key session metrics: deals_closed={session_metrics.get('deals_closed_pct', 0):.1f}%, "
                    f"avg_turns={session_metrics.get('avg_turns', 0):.1f}")
        
        return session_metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        metrics = {}
        
        # CPU usage
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        
        # GPU/VRAM usage
        if torch.cuda.is_available():
            # Memory in GB
            metrics["vram_gb"] = torch.cuda.memory_allocated() / 1024**3
            
            # Try to get GPU utilization
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assuming single GPU
                    metrics["gpu_util"] = gpu.load * 100
                    logger.debug(f"GPU utilization: {metrics['gpu_util']:.1f}%")
            except ImportError:
                # Fallback if GPUtil not available
                metrics["gpu_util"] = 0
                logger.debug("GPUtil not available, GPU utilization set to 0")
        else:
            metrics["vram_gb"] = 0
            metrics["gpu_util"] = 0
        
        return metrics
    
    def _print_training_metrics(
        self,
        step: int,
        epoch: float,
        metrics: Dict[str, Any],
        system_metrics: Dict[str, Any],
        step_duration: float
    ):
        """Print training metrics to console."""
        output = f"[{self.stage.upper()}] Step {step:5d} | Epoch {epoch:.2f} | "
        
        # Training metrics
        if "train_loss" in metrics:
            output += f"Loss: {metrics['train_loss']:.4f} | "
        if "eval_loss" in metrics:
            output += f"Eval: {metrics['eval_loss']:.4f} | "
        if "learning_rate" in metrics:
            output += f"LR: {metrics['learning_rate']:.2e} | "
        
        # DPO specific
        if "reward_margin" in metrics:
            output += f"Margin: {metrics['reward_margin']:.3f} | "
        if "kl_divergence" in metrics:
            output += f"KL: {metrics['kl_divergence']:.3f} | "
        
        # Performance metrics
        if "tokens_per_second" in metrics:
            output += f"TPS: {metrics['tokens_per_second']:.0f} | "
        
        # System metrics
        if "vram_gb" in system_metrics:
            output += f"VRAM: {system_metrics['vram_gb']:.1f}GB | "
        if "gpu_util" in system_metrics:
            output += f"GPU: {system_metrics['gpu_util']:.0f}% | "
        
        # Timing
        output += f"Time: {step_duration:.1f}s"
        
        print(output)
    
    def calculate_tokens_per_second(
        self,
        num_tokens: int,
        elapsed_time: float
    ) -> float:
        """
        Calculate tokens processed per second.
        Matches evaluation system calculation.
        """
        if elapsed_time > 0:
            return num_tokens / elapsed_time
        return 0.0
    
    def save_training_summary(self):
        """Save comprehensive training summary using pandas."""
        summary_path = self.log_dir / f"summary_{self.stage}_{self.current_session}.json"
        logger.info(f"Saving training summary to {summary_path}")
        
        summary = {
            "stage": self.stage,
            "session": self.current_session,
            "total_steps": len(self.training_df),
            "total_episodes": len(self.episode_buffer),
            "final_metrics": {},
            "best_metrics": {},
            "averages": {},
            "training_csv": str(self.training_csv)
        }
        
        # Use pandas to calculate statistics
        if not self.training_df.empty:
            # Get final values
            final_row = self.training_df.iloc[-1]
            for col in self.training_df.columns:
                if col not in ["timestamp", "step"]:
                    val = final_row[col]
                    if pd.notna(val):
                        summary["final_metrics"][col] = float(val) if isinstance(val, (int, float)) else val
            
            # Get best values and averages
            numeric_cols = self.training_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ["step", "epoch"]:
                    series = self.training_df[col].dropna()
                    if not series.empty:
                        if "loss" in col or "kl" in col:
                            summary["best_metrics"][col] = float(series.min())
                        elif "reward" in col or "tokens" in col or "tps" in col:
                            summary["best_metrics"][col] = float(series.max())
                        
                        summary["averages"][col] = float(series.mean())
        
        # Add episode metrics summary if available
        if self.episode_buffer:
            episode_summary = self.session_metrics.aggregate(self.episode_buffer)
            summary["episode_metrics"] = episode_summary
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to: {summary_path}")  # Keep print for visibility
            logger.info(f"Training summary saved successfully: {len(summary)} top-level keys")
            logger.debug(f"Summary includes: total_steps={summary.get('total_steps', 0)}, "
                        f"total_episodes={summary.get('total_episodes', 0)}")
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")
            raise
        
        return summary
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics from DataFrame."""
        if self.training_df.empty:
            return {}
        
        latest = {}
        final_row = self.training_df.iloc[-1]
        for col in self.training_df.columns:
            if col not in ["timestamp"]:
                val = final_row[col]
                if pd.notna(val):
                    latest[col] = float(val) if isinstance(val, (int, float)) else val
        
        return latest
    
    def should_early_stop(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        metric: str = "train_loss"
    ) -> bool:
        """
        Check if training should stop early based on metrics.
        
        Args:
            patience: Steps to wait for improvement
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            
        Returns:
            True if should stop early
        """
        if self.training_df.empty or metric not in self.training_df.columns:
            logger.debug(f"Cannot check early stopping: DataFrame empty or metric '{metric}' not found")
            return False
        
        series = self.training_df[metric].dropna()
        if len(series) < patience + 1:
            logger.debug(f"Not enough data for early stopping check: {len(series)} < {patience + 1}")
            return False
        
        recent = series.tail(patience).values
        best_recent = recent.min() if "loss" in metric else recent.max()
        
        # Check if we're not improving
        improving = False
        for i in range(1, len(recent)):
            if "loss" in metric:
                if recent[i] < best_recent - min_delta:
                    improving = True
                    break
            else:
                if recent[i] > best_recent + min_delta:
                    improving = True
                    break
        
        if not improving:
            logger.warning(f"Early stopping triggered: No improvement in {metric} for {patience} steps")
            logger.info(f"Best recent {metric}: {best_recent:.4f}, min_delta: {min_delta}")
            return True
        
        return False
    
    def get_training_dataframe(self) -> pd.DataFrame:
        """Return the training metrics DataFrame for analysis."""
        return self.training_df.copy()
    
    def plot_metrics(self, metrics_to_plot: List[str] = [], save_path: str = ""):
        """
        Create plots of training metrics using pandas plot functionality.
        
        Args:
            metrics_to_plot: List of metric names to plot (defaults to common metrics)
            save_path: Optional path to save plot
        """
        if self.training_df.empty:
            logger.warning("No training data to plot")
            print("No training data to plot")  # Keep print for visibility
            return
        
        logger.info(f"Generating training metrics plot")
        
        try:
            import matplotlib.pyplot as plt
            
            if metrics_to_plot is None:
                # Default metrics to plot
                available_metrics = []
                for metric in ["train_loss", "eval_loss", "learning_rate", "tokens_per_second"]:
                    if metric in self.training_df.columns and self.training_df[metric].notna().any():
                        available_metrics.append(metric)
                metrics_to_plot = available_metrics
            
            if not metrics_to_plot:
                logger.warning("No valid metrics to plot")
                print("No valid metrics to plot")  # Keep print for visibility
                return
            
            logger.debug(f"Plotting metrics: {metrics_to_plot}")
            
            # Create subplots
            n_plots = len(metrics_to_plot)
            fig, axes = plt.subplots(
                (n_plots + 1) // 2, 
                2 if n_plots > 1 else 1,
                figsize=(12, 4 * ((n_plots + 1) // 2))
            )
            
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot each metric using pandas
            for i, metric in enumerate(metrics_to_plot):
                self.training_df[metric].dropna().plot(
                    ax=axes[i],
                    title=metric.replace('_', ' ').title(),
                    xlabel='Step',
                    ylabel=metric
                )
            
            # Remove extra subplots
            for i in range(n_plots, len(axes)):
                fig.delaxes(axes[i])
            
            plt.suptitle(f"Training Metrics - {self.stage.upper()} - {self.current_session}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                default_path = self.log_dir / f"metrics_plot_{self.stage}_{self.current_session}.png"
                plt.savefig(default_path)
                logger.info(f"Plot saved to {default_path}")
            
            plt.close()
            
        except ImportError:
            logger.error("Matplotlib not available, skipping plot generation")
            print("Matplotlib not available, skipping plot generation")  # Keep print for visibility