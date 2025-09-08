"""
Centralized logging configuration for the training-lab project.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str = "",
    level: int = logging.INFO,
    log_file: str = "",
    console: bool = True,
    clear_log: bool = True
) -> logging.Logger:
    """
    Set up logging for a module.
    
    Args:
        name: Logger name (default: root logger)
        level: Logging level
        log_file: Optional file to write logs to
        console: Whether to also log to console
        clear_log: Whether to clear the log file on initialization (default: True)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Use 'w' mode to clear the file on each new instance if clear_log is True
        file_mode = 'w' if clear_log else 'a'
        file_handler = logging.FileHandler(log_path, mode=file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the project's standard configuration.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger
    """
    # Determine log file based on module name
    if 'data_converter' in name or 'converters' in name:
        log_file = 'logs/data_converter.log'
    elif 'hf_wrapper' in name or 'agents' in name:
        log_file = 'logs/agent_logs.log'
    elif 'teacher' in name or 'tasks' in name:
        log_file = 'logs/teachers.log'
    else:
        log_file = 'logs/general.log'
    
    return setup_logging(name, log_file=log_file)


# Configure root logger to prevent duplicate logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)