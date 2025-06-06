"""
Logging utilities for the TTS Trainer project
Provides structured logging with color coding and file output
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logging(verbose: bool = False, log_file: Optional[str] = None, 
                  log_level: Optional[str] = None) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Path to log file, defaults to logs/tts_trainer.log
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    # Determine log level
    if log_level:
        level = getattr(logging, log_level.upper())
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file
    if not log_file:
        log_file = log_dir / "tts_trainer.log"
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max, 5 backups
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Always debug to file
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}, File: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, name: str, total_steps: int):
        self.logger = get_logger(name)
        self.total_steps = total_steps
        self.current_step = 0
        
    def update(self, step: int, message: str = ""):
        """Update progress and log message."""
        self.current_step = step
        progress = (step / self.total_steps) * 100
        self.logger.info(f"Progress: {progress:.1f}% ({step}/{self.total_steps}) {message}")
        
    def increment(self, message: str = ""):
        """Increment progress by one step."""
        self.update(self.current_step + 1, message)
        
    def finish(self, message: str = "Completed"):
        """Mark operation as finished."""
        self.logger.info(f"✅ {message} - {self.total_steps}/{self.total_steps} steps")


class MetricsLogger:
    """Logger for tracking training and validation metrics."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.metrics")
        self.metrics_history = []
        
    def log_metrics(self, epoch: int, metrics: dict):
        """Log metrics for a training epoch."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metrics_str}")
        
        # Store for history
        self.metrics_history.append({
            'epoch': epoch,
            **metrics
        })
        
    def log_validation(self, epoch: int, metrics: dict):
        """Log validation metrics."""
        metrics_str = " | ".join([f"val_{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metrics_str}")
        
    def get_best_metric(self, metric_name: str, higher_is_better: bool = True):
        """Get the best value for a specific metric."""
        if not self.metrics_history:
            return None
            
        values = [m.get(metric_name) for m in self.metrics_history if metric_name in m]
        if not values:
            return None
            
        return max(values) if higher_is_better else min(values) 