"""Per-step training metrics: CSV log, console line, and an end-of-run JSON summary."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import torch
import psutil
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingMetricsTracker:
    """Records training metrics to a CSV (one row per logged step) using pandas."""

    def __init__(self, log_dir: str = "logs", stage: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage

        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_csv = self.log_dir / f"training_{self.stage}_{self.current_session}.csv"
        self.training_df = pd.DataFrame()
        self.last_step_time = datetime.now()

        logger.info(f"TrainingMetricsTracker initialized: stage={stage}, csv={self.training_csv}")

    def log_training_step(
        self,
        step: int,
        epoch: float,
        metrics: Dict[str, Any],
        log_to_console: bool = True
    ):
        """Append one row of metrics (plus system usage) to the CSV."""
        timestamp = datetime.now()
        step_duration = (timestamp - self.last_step_time).total_seconds()
        self.last_step_time = timestamp

        system_metrics = self._get_system_metrics()
        full_metrics = {
            "timestamp": timestamp.isoformat(),
            "step": step,
            "epoch": epoch,
            "step_duration": step_duration,
            **metrics,
            **system_metrics
        }

        new_row = pd.DataFrame([full_metrics])
        if self.training_df.empty:
            self.training_df = new_row.copy()
            self.training_df.to_csv(self.training_csv, index=False)
        else:
            self.training_df = pd.concat([self.training_df, new_row], ignore_index=True)
            new_row.to_csv(self.training_csv, mode='a', header=False, index=False)

        if log_to_console:
            self._print_training_metrics(step, epoch, metrics, system_metrics, step_duration)

        if "train_loss" in metrics:
            logger.info(f"Step {step}: train_loss={metrics['train_loss']:.4f}")
        if "eval_loss" in metrics:
            logger.info(f"Step {step}: eval_loss={metrics['eval_loss']:.4f}")

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Current CPU and VRAM usage."""
        metrics = {"cpu_percent": psutil.cpu_percent(interval=0.1)}
        metrics["vram_gb"] = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        return metrics

    def _print_training_metrics(
        self,
        step: int,
        epoch: float,
        metrics: Dict[str, Any],
        system_metrics: Dict[str, Any],
        step_duration: float
    ):
        output = f"[{self.stage.upper()}] Step {step:5d} | Epoch {epoch:.2f} | "
        if "train_loss" in metrics:
            output += f"Loss: {metrics['train_loss']:.4f} | "
        if "eval_loss" in metrics:
            output += f"Eval: {metrics['eval_loss']:.4f} | "
        if "learning_rate" in metrics:
            output += f"LR: {metrics['learning_rate']:.2e} | "
        if "tokens_per_second" in metrics:
            output += f"TPS: {metrics['tokens_per_second']:.0f} | "
        output += f"VRAM: {system_metrics['vram_gb']:.1f}GB | Time: {step_duration:.1f}s"
        print(output)

    def save_training_summary(self):
        """Write final / best / average values for every numeric column to JSON."""
        summary_path = self.log_dir / f"summary_{self.stage}_{self.current_session}.json"

        summary = {
            "stage": self.stage,
            "session": self.current_session,
            "total_steps": len(self.training_df),
            "final_metrics": {},
            "best_metrics": {},
            "averages": {},
            "training_csv": str(self.training_csv)
        }

        if not self.training_df.empty:
            final_row = self.training_df.iloc[-1]
            for col in self.training_df.columns:
                if col not in ["timestamp", "step"]:
                    val = final_row[col]
                    if pd.notna(val):
                        summary["final_metrics"][col] = float(val) if isinstance(val, (int, float, np.number)) else val

            numeric_cols = self.training_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ["step", "epoch"]:
                    series = self.training_df[col].dropna()
                    if not series.empty:
                        if "loss" in col:
                            summary["best_metrics"][col] = float(series.min())
                        elif "tokens" in col:
                            summary["best_metrics"][col] = float(series.max())
                        summary["averages"][col] = float(series.mean())

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to: {summary_path}")

        return summary
