"""
Comprehensive logging configuration for negotiation simulation.

This module provides multi-level logging with structured output for debugging,
monitoring, and analysis of negotiation simulations.
"""

import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
import sys
from typing import Optional, Dict, Any


class SimulationLogger:
    """Centralized logging system for negotiation simulations."""

    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize the logging system.

        Args:
            log_dir: Directory for log files
            log_level: Default logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize different loggers for different aspects
        self.sim_logger = self._setup_logger(
            "simulation",
            "simulation.log",
            log_level
        )

        self.llm_logger = self._setup_logger(
            "llm",
            "llm_interactions.log",
            logging.DEBUG
        )

        self.control_logger = self._setup_logger(
            "control",
            "control_function.log",
            logging.DEBUG
        )

        self.error_logger = self._setup_logger(
            "errors",
            "errors.log",
            logging.ERROR
        )

        # Metrics logger for structured JSON output
        self.metrics_logger = self._setup_json_logger(
            "metrics",
            "metrics.jsonl"
        )

        # Console handler for important messages
        self._setup_console_handler()

    def _setup_logger(self, name: str, filename: str, level: int) -> logging.Logger:
        """
        Set up a logger with file overwrite mode.

        Args:
            name: Logger name
            filename: Log file name
            level: Logging level

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # File handler in write mode to overwrite existing logs
        file_handler = logging.FileHandler(
            self.log_dir / filename,
            mode='w'  # Overwrite mode
        )

        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False

        return logger

    def _setup_json_logger(self, name: str, filename: str) -> logging.Logger:
        """
        Set up a logger for structured JSON output.

        Args:
            name: Logger name
            filename: Log file name

        Returns:
            Configured logger for JSON output
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # File handler in write mode for JSONL format
        file_handler = logging.FileHandler(self.log_dir / filename, mode='w')

        # Minimal formatter - just the message (which will be JSON)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.propagate = False

        return logger

    def _setup_console_handler(self):
        """Add console handler to main simulation logger for visibility."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Simpler console formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        self.sim_logger.addHandler(console_handler)

    def log_simulation_start(self, sim_id: str, seed_id: int,
                           buyer_personality: str, seller_personality: str):
        """Log simulation start with context."""
        self.sim_logger.info(
            f"Starting simulation {sim_id}: seed={seed_id}, "
            f"buyer={buyer_personality}, seller={seller_personality}"
        )

        # Also log to metrics
        self.log_metrics({
            "event": "simulation_start",
            "sim_id": sim_id,
            "seed_id": seed_id,
            "buyer_personality": buyer_personality,
            "seller_personality": seller_personality,
            "timestamp": datetime.utcnow().isoformat()
        })

    def log_simulation_end(self, sim_id: str, outcome: Dict[str, Any],
                         duration: float):
        """Log simulation completion with outcome."""
        status = "SUCCESS" if outcome.get("deal_closed") else "NO_DEAL"
        self.sim_logger.info(
            f"Completed simulation {sim_id}: status={status}, "
            f"turns={outcome.get('turns_taken')}, duration={duration:.2f}s"
        )

        # Detailed metrics
        self.log_metrics({
            "event": "simulation_end",
            "sim_id": sim_id,
            "outcome": outcome,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat()
        })

    def log_turn(self, sim_id: str, turn: int, speaker: str, action: str,
                message: str = None):
        """Log individual turn details."""
        self.sim_logger.debug(
            f"[{sim_id}] Turn {turn} - {speaker} - Action: {action}"
        )
        if message:
            self.llm_logger.debug(
                f"[{sim_id}] Turn {turn} - {speaker}: {message[:100]}..."
            )

    def log_control_decision(self, sim_id: str, turn: int,
                           control_values: Dict[str, Any]):
        """Log control function decision details."""
        self.control_logger.debug(
            f"[{sim_id}] Turn {turn} - Control decision: "
            f"action={control_values.get('recommended_action')}, "
            f"s={control_values.get('features', {}).get('s', 0):.3f}, "
            f"probs={control_values.get('action_probs')}"
        )

    def log_llm_call(self, sim_id: str, prompt: str, response: str,
                    duration: float, success: bool = True):
        """Log LLM interaction details."""
        self.llm_logger.debug(
            f"[{sim_id}] LLM call - Duration: {duration:.2f}s, "
            f"Success: {success}, Prompt length: {len(prompt)}, "
            f"Response length: {len(response) if response else 0}"
        )

        if not success:
            self.error_logger.error(
                f"[{sim_id}] LLM generation failed after {duration:.2f}s"
            )

    def log_error(self, sim_id: str, error: Exception, context: str = ""):
        """Log error with context."""
        self.error_logger.error(
            f"[{sim_id}] Error in {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )

        # Also log to metrics for analysis
        self.log_metrics({
            "event": "error",
            "sim_id": sim_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        })

    def log_warning(self, sim_id: str, message: str, context: str = ""):
        """Log warning conditions."""
        self.sim_logger.warning(f"[{sim_id}] Warning in {context}: {message}")

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log structured metrics as JSON."""
        self.metrics_logger.info(json.dumps(metrics))

    def get_logger(self, name: str) -> logging.Logger:
        """Get a specific logger by name."""
        loggers = {
            "simulation": self.sim_logger,
            "llm": self.llm_logger,
            "control": self.control_logger,
            "errors": self.error_logger,
            "metrics": self.metrics_logger
        }
        return loggers.get(name, self.sim_logger)


# Singleton instance for global access
_logger_instance: Optional[SimulationLogger] = None


def get_logger() -> SimulationLogger:
    """Get or create the singleton logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SimulationLogger()
    return _logger_instance


def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> SimulationLogger:
    """
    Initialize and return the logger instance.

    Args:
        log_dir: Directory for log files
        log_level: Default logging level

    Returns:
        Configured SimulationLogger instance
    """
    global _logger_instance
    _logger_instance = SimulationLogger(log_dir, log_level)
    return _logger_instance