"""
Темірадам — Structured logging.

Sets up separate log files for assistant, tools, and errors,
with console output for development.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_INITIALIZED = False


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 3,
) -> None:
    """
    Configure structured logging for Темірадам.

    Creates three log files:
    - assistant.log: General assistant operations
    - tools.log: Tool execution logs
    - errors.log: Error-level messages only

    Plus console output with colored level names.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger for temiradam
    root_logger = logging.getLogger("temiradam")
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    # Format
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-35s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "\033[90m%(asctime)s\033[0m │ %(levelname)-8s │ \033[36m%(name)-25s\033[0m │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # Main assistant log
    assistant_handler = RotatingFileHandler(
        log_dir / "assistant.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    assistant_handler.setLevel(log_level)
    assistant_handler.setFormatter(file_fmt)
    root_logger.addHandler(assistant_handler)

    # Error-only log
    error_handler = RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_fmt)
    root_logger.addHandler(error_handler)

    # Tools-specific log
    tools_logger = logging.getLogger("temiradam.tools")
    tools_handler = RotatingFileHandler(
        log_dir / "tools.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    tools_handler.setLevel(log_level)
    tools_handler.setFormatter(file_fmt)
    tools_logger.addHandler(tools_handler)

    root_logger.info("Logging initialized: level=%s, dir=%s", level, log_dir)
