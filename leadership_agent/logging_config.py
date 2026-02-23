"""
logging_config.py — Centralized logging configuration.

Call `setup_logging()` once at application entry points (cli.py, app.py).
All modules should use: logger = logging.getLogger(__name__)
"""

import logging
import logging.handlers
import sys
from pathlib import Path


LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with:
      - Console handler (stdout)
      - Rotating file handler → logs/agent.log (10 MB × 5 backups)

    Args:
        level: Logging level string — "INFO" or "DEBUG"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid adding duplicate handlers if called multiple times
    if root.handlers:
        return

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────────
    log_file = _LOGS_DIR / "agent.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

    root.info("Logging initialised — level=%s, file=%s", level, log_file)


def get_logger(name: str) -> logging.Logger:
    """Convenience helper. Prefer `logging.getLogger(__name__)` in modules."""
    return logging.getLogger(name)
