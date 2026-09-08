"""
PhysioRANO Logging Infrastructure.

Configures structured console and file logging for training and evaluation pipelines.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "PhysioRANO",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a logger instance with console and optional file handlers.

    Args:
        name (str): Logger module name.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[Path]): File path to save output logs.

    Returns:
        logging.Logger: Configured Python Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "PhysioRANO") -> logging.Logger:
    """Retrieves an existing logger or initializes a default instance.

    Args:
        name (str): Logger identifier.

    Returns:
        logging.Logger: Python Logger instance.
    """
    return logging.getLogger(name)
