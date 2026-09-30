"""Unit tests for logging system setup."""
import logging
from src.core.logging import setup_logging, get_logger


def test_get_logger():
    """Verify get_logger returns a Logger instance with requested name."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_setup_logging_level():
    """Verify setup_logging sets root logger level."""
    setup_logging(log_level="DEBUG")
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
