"""
Logging configuration for ML Model Drift Monitor.

Provides structured logging with configurable levels, formats,
and output handlers for monitoring system components.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that produces structured log output."""

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        level = record.levelname
        module = record.module
        func = record.funcName
        message = record.getMessage()

        if self.include_timestamp:
            base = f"[{timestamp}] [{level:8s}] [{module}.{func}] {message}"
        else:
            base = f"[{level:8s}] [{module}.{func}] {message}"

        if record.exc_info and record.exc_info[1]:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
) -> None:
    """
    Configure the root logger for the monitoring system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        structured: Whether to use structured formatting.
    """
    root_logger = logging.getLogger("drift_monitor")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    root_logger.handlers.clear()

    if structured:
        formatter = StructuredFormatter(include_timestamp=True)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance under the drift_monitor namespace.

    Args:
        name: Logger name, typically the module name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"drift_monitor.{name}")
