"""Logging utilities for the application."""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import config


def setup_logger(
    name: str = "tree_species_identifier",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses config setting.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Get settings from config
    if level is None:
        level = config.get("logging.level", "INFO")

    if log_file is None:
        log_file = config.get("logging.file", "logs/app.log")

    log_format = config.get(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
