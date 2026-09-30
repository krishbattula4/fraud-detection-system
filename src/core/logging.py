"""Structured logging initialization and retrieval module."""
import logging
import sys
from typing import Optional


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with structured console formatting."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name or "fraud_system")
