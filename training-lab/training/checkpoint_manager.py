"""Checkpoint management for QLoRA training."""

import os
import json
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch

# Set up logging
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with versioning and metadata."""
    
    def __init__(self, base_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            base_dir: Base directory for storing checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint = None
        self.training_start_time = None
        self.rollback_history = []
        logger.info(f"CheckpointManager initialized with base_dir: {self.base_dir}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively convert non-JSON-serializable objects to serializable forms.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        # Handle basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        
        # Handle torch tensors
        if torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        
        # Handle numpy arrays
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        
        # Handle lists and tuples recursively
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # For any other type (including Mock objects), convert to string
        return str(obj)
        
    def save_checkpoint(
        self,
        model,
        tokenizer,
        stage: str,
        version: int,
        metrics: Dict[str, Any],
        training_args: Optional[Dict] = None,
        dataset_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint with metadata.

        Args:
            model: PEFT model to save
            tokenizer: Tokenizer to save
            stage: Training stage (baseline, sft, dpo)
            version: Version number
            metrics: Performance metrics
            training_args: Training arguments used
            dataset_stats: Dataset statistics (num_examples, avg_length, etc.)

        Returns:
            Path to saved checkpoint
        """
        # Track training duration if start time is set
        training_duration = None
        if self.training_start_time:
            training_duration = time.time() - self.training_start_time
        # Create checkpoint directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"negotiation_{stage}_v{version}_{timestamp}"
        checkpoint_path = self.base_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        logger.debug(f"Stage: {stage}, Version: {version}")
        logger.debug(f"Metrics: {metrics}")
        
        # Save model adapters only (not full model)
        logger.debug("Saving model adapters")
        try:
            model.save_pretrained(checkpoint_path)
            logger.debug("Model adapters saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model adapters: {e}")
            raise
        
        # Save tokenizer
        logger.debug("Saving tokenizer")
        try:
            tokenizer.save_pretrained(checkpoint_path)
            logger.debug("Tokenizer saved successfully")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")
            raise
        
        # Get checkpoint size info
        checkpoint_size_mb = None
        try:
            # Calculate size after saving model
            total_size = sum(
                f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file()
            ) if checkpoint_path.exists() else 0
            checkpoint_size_mb = total_size / (1024 * 1024)
        except:
            pass

        # Create metadata - use recursive serialization helper
        metadata = {
            "stage": stage,
            "version": version,
            "timestamp": timestamp,
            "metrics": self._make_json_serializable(metrics),
            "training_args": self._make_json_serializable(training_args or {}),
            "dataset_stats": self._make_json_serializable(dataset_stats or {}),
            "model_type": "LoRA adapters",
            "base_model": self._make_json_serializable(
                getattr(model, "name_or_path", "mistralai/Mistral-7B-v0.1")
            ),
            "training_duration_seconds": training_duration,
            "checkpoint_size_mb": checkpoint_size_mb
        }
        
        # Try to add git commit hash
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.base_dir.parent
            ).decode().strip()
            metadata["git_commit"] = git_hash
            logger.debug(f"Git commit hash: {git_hash}")
        except Exception as e:
            logger.debug(f"Could not get git commit hash: {e}")
        
        # Save metadata
        metadata_path = checkpoint_path / "checkpoint_info.json"
        logger.debug(f"Saving metadata to {metadata_path}")
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
            
        self.current_checkpoint = checkpoint_path
        print(f"Checkpoint saved to: {checkpoint_path}")  # Keep print for visibility
        logger.info(f"Checkpoint saved successfully to: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint metadata and verify files.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dict with checkpoint info
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        path_obj = Path(checkpoint_path)

        if not path_obj.exists():
            # Try to find in base directory
            path_obj = self.base_dir / checkpoint_path
            logger.debug(f"Checkpoint not found, trying base directory: {path_obj}")

        if not path_obj.exists():
            logger.error(f"Checkpoint not found: {path_obj}")
            raise ValueError(f"Checkpoint not found: {path_obj}")

        # Load metadata
        metadata_path = path_obj / "checkpoint_info.json"
        logger.debug(f"Loading metadata from {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.debug(f"Metadata loaded: stage={metadata.get('stage')}, version={metadata.get('version')}")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

        # Verify required files exist
        required_files = [
            "adapter_model.bin",
            "adapter_config.json"
        ]

        for file in required_files:
            if not (path_obj / file).exists():
                warning_msg = f"{file} not found in checkpoint"
                print(f"Warning: {warning_msg}")  # Keep print for visibility
                logger.warning(warning_msg)
            else:
                logger.debug(f"Verified file exists: {file}")

        metadata["checkpoint_path"] = str(path_obj)
        logger.info(f"Checkpoint loaded successfully from {path_obj}")
        return metadata
    
    def get_best_checkpoint(self, stage: str, metric: str = "loss") -> Optional[str]:
        """
        Find best checkpoint for a given stage based on metric.
        
        Args:
            stage: Training stage to search for
            metric: Metric to compare (lower is better for loss)
            
        Returns:
            Path to best checkpoint or None
        """
        logger.info(f"Searching for best checkpoint: stage={stage}, metric={metric}")
        best_checkpoint = None
        best_value = float('inf')
        checkpoints_checked = 0
        
        for checkpoint_dir in self.base_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
                
            if f"negotiation_{stage}" not in checkpoint_dir.name:
                continue
                
            try:
                with open(checkpoint_dir / "checkpoint_info.json", "r") as f:
                    info = json.load(f)
                    
                if metric in info.get("metrics", {}):
                    value = info["metrics"][metric]
                    checkpoints_checked += 1
                    logger.debug(f"Checkpoint {checkpoint_dir.name}: {metric}={value}")
                    if value < best_value:
                        best_value = value
                        best_checkpoint = str(checkpoint_dir)
                        logger.debug(f"New best checkpoint: {checkpoint_dir.name}")
            except Exception as e:
                logger.debug(f"Failed to read checkpoint {checkpoint_dir.name}: {e}")
                continue
        
        if best_checkpoint:
            logger.info(f"Best checkpoint found: {Path(best_checkpoint).name} ({metric}={best_value})")
        else:
            logger.warning(f"No valid checkpoints found for stage={stage}")
        logger.debug(f"Checked {checkpoints_checked} checkpoints")
        
        return best_checkpoint
    
    def list_checkpoints(self, stage: Optional[str] = None) -> list:
        """
        List all available checkpoints.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of checkpoint info dicts
        """
        logger.info(f"Listing checkpoints" + (f" for stage={stage}" if stage else ""))
        checkpoints = []
        
        for checkpoint_dir in sorted(self.base_dir.iterdir()):
            if not checkpoint_dir.is_dir():
                continue
                
            if stage and f"negotiation_{stage}" not in checkpoint_dir.name:
                continue
                
            try:
                with open(checkpoint_dir / "checkpoint_info.json", "r") as f:
                    info = json.load(f)
                    info["path"] = str(checkpoint_dir)
                    checkpoints.append(info)
            except Exception as e:
                logger.debug(f"Failed to read checkpoint {checkpoint_dir.name}: {e}")
                continue
        
        logger.info(f"Found {len(checkpoints)} checkpoints")
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_n: int = 5, stage: Optional[str] = None):
        """
        Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_n: Number of checkpoints to keep
            stage: Optional stage filter
        """
        logger.info(f"Cleaning up old checkpoints: keeping {keep_n} most recent" + 
                   (f" for stage={stage}" if stage else ""))
        checkpoints = self.list_checkpoints(stage)
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Delete old checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_n:]:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint_path.name}")  # Keep print for visibility
                    logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint_path.name}: {e}")
        
        logger.info(f"Cleanup complete: removed {deleted_count} checkpoints")

    def rollback_to_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Rollback to a previous checkpoint.

        Args:
            checkpoint_path: Path to checkpoint to rollback to

        Returns:
            Dict with rollback information
        """
        logger.info(f"Rolling back to checkpoint: {checkpoint_path}")

        # Backup current checkpoint if exists
        if self.current_checkpoint:
            self.rollback_history.append({
                "from": str(self.current_checkpoint),
                "to": checkpoint_path,
                "timestamp": datetime.now().isoformat()
            })
            logger.debug(f"Added rollback to history: {self.current_checkpoint} -> {checkpoint_path}")

        # Load the checkpoint to verify it exists
        checkpoint_info = self.load_checkpoint(checkpoint_path)

        # Update current checkpoint
        self.current_checkpoint = Path(checkpoint_info["checkpoint_path"])

        logger.info(f"Successfully rolled back to: {self.current_checkpoint}")
        return {
            "rolled_back_to": str(self.current_checkpoint),
            "checkpoint_info": checkpoint_info,
            "rollback_history": self.rollback_history
        }

    def get_checkpoint_history(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get history of checkpoints for potential rollback.

        Args:
            stage: Optional filter by stage

        Returns:
            List of checkpoint info sorted by timestamp (newest first)
        """
        logger.info(f"Getting checkpoint history" + (f" for stage={stage}" if stage else ""))
        checkpoints = self.list_checkpoints(stage)

        # Sort by timestamp, newest first
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Add rollback status
        for checkpoint in checkpoints:
            checkpoint["is_current"] = (
                str(self.current_checkpoint) == checkpoint.get("path")
                if self.current_checkpoint else False
            )

        logger.debug(f"Found {len(checkpoints)} checkpoints in history")
        return checkpoints

    def get_next_version(self, stage: str) -> int:
        """
        Get the next version number for a given stage.

        Args:
            stage: Training stage

        Returns:
            Next version number to use
        """
        logger.info(f"Getting next version for stage={stage}")
        max_version = 0

        for checkpoint_dir in self.base_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            if f"negotiation_{stage}" not in checkpoint_dir.name:
                continue

            # Extract version from name pattern: negotiation_{stage}_v{version}_{timestamp}
            try:
                parts = checkpoint_dir.name.split('_')
                for part in parts:
                    if part.startswith('v') and part[1:].isdigit():
                        version = int(part[1:])
                        if version > max_version:
                            max_version = version
                        break
            except Exception as e:
                logger.debug(f"Could not parse version from {checkpoint_dir.name}: {e}")
                continue

        next_version = max_version + 1
        logger.info(f"Next version for stage={stage}: {next_version}")
        return next_version

    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str) -> Dict[str, Any]:
        """
        Compare two checkpoints to show differences.

        Args:
            checkpoint1: Path to first checkpoint
            checkpoint2: Path to second checkpoint

        Returns:
            Dict showing differences between checkpoints
        """
        logger.info(f"Comparing checkpoints: {checkpoint1} vs {checkpoint2}")

        info1 = self.load_checkpoint(checkpoint1)
        info2 = self.load_checkpoint(checkpoint2)

        comparison = {
            "checkpoint1": {
                "path": checkpoint1,
                "stage": info1.get("stage"),
                "version": info1.get("version"),
                "timestamp": info1.get("timestamp")
            },
            "checkpoint2": {
                "path": checkpoint2,
                "stage": info2.get("stage"),
                "version": info2.get("version"),
                "timestamp": info2.get("timestamp")
            },
            "metrics_diff": {},
            "dataset_stats_diff": {},
            "training_args_diff": {}
        }

        # Compare metrics
        metrics1 = info1.get("metrics", {})
        metrics2 = info2.get("metrics", {})
        all_metric_keys = set(metrics1.keys()) | set(metrics2.keys())

        for key in all_metric_keys:
            val1 = metrics1.get(key)
            val2 = metrics2.get(key)
            if val1 != val2:
                comparison["metrics_diff"][key] = {
                    "checkpoint1": val1,
                    "checkpoint2": val2,
                    "change": val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
                }

        # Compare dataset stats
        stats1 = info1.get("dataset_stats", {})
        stats2 = info2.get("dataset_stats", {})
        all_stats_keys = set(stats1.keys()) | set(stats2.keys())

        for key in all_stats_keys:
            val1 = stats1.get(key)
            val2 = stats2.get(key)
            if val1 != val2:
                comparison["dataset_stats_diff"][key] = {
                    "checkpoint1": val1,
                    "checkpoint2": val2
                }

        logger.info(f"Comparison complete: {len(comparison['metrics_diff'])} metric differences found")
        return comparison

    def export_checkpoint_summary(self, output_path: Optional[str] = None) -> str:
        """
        Export a summary of all checkpoints to a report file.

        Args:
            output_path: Optional path for the report (defaults to checkpoints_summary.json)

        Returns:
            Path to the generated report
        """
        if output_path is None:
            report_path = self.base_dir / "checkpoints_summary.json"
        else:
            report_path = Path(output_path)

        logger.info(f"Exporting checkpoint summary to {report_path}")

        summary = {
            "generated_at": datetime.now().isoformat(),
            "base_directory": str(self.base_dir),
            "current_checkpoint": str(self.current_checkpoint) if self.current_checkpoint else None,
            "rollback_history": self.rollback_history,
            "checkpoints_by_stage": {},
            "total_checkpoints": 0,
            "total_size_mb": 0
        }

        # Group checkpoints by stage
        all_checkpoints = self.list_checkpoints()

        for checkpoint in all_checkpoints:
            stage = checkpoint.get("stage", "unknown")
            if stage not in summary["checkpoints_by_stage"]:
                summary["checkpoints_by_stage"][stage] = []

            summary["checkpoints_by_stage"][stage].append({
                "version": checkpoint.get("version"),
                "timestamp": checkpoint.get("timestamp"),
                "metrics": checkpoint.get("metrics", {}),
                "size_mb": checkpoint.get("checkpoint_size_mb"),
                "path": checkpoint.get("path")
            })

            summary["total_checkpoints"] += 1
            if checkpoint.get("checkpoint_size_mb"):
                summary["total_size_mb"] += checkpoint.get("checkpoint_size_mb")

        # Sort checkpoints within each stage
        for stage in summary["checkpoints_by_stage"]:
            summary["checkpoints_by_stage"][stage].sort(
                key=lambda x: x.get("version", 0)
            )

        # Write summary
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary exported: {summary['total_checkpoints']} checkpoints, "
                   f"{summary['total_size_mb']:.2f} MB total")
        print(f"Checkpoint summary exported to: {report_path}")

        return str(report_path)

    def set_training_start_time(self):
        """Set the training start time for duration tracking."""
        self.training_start_time = time.time()
        logger.debug(f"Training start time set: {datetime.fromtimestamp(self.training_start_time)}")