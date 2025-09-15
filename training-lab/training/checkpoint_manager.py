"""Checkpoint management for QLoRA training."""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
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
        training_args: Optional[Dict] = None
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
            
        Returns:
            Path to saved checkpoint
        """
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
        
        # Create metadata - use recursive serialization helper
        metadata = {
            "stage": stage,
            "version": version,
            "timestamp": timestamp,
            "metrics": self._make_json_serializable(metrics),
            "training_args": self._make_json_serializable(training_args or {}),
            "model_type": "LoRA adapters",
            "base_model": self._make_json_serializable(
                getattr(model, "name_or_path", "mistralai/Mistral-7B-v0.1")
            )
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