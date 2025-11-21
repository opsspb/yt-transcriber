"""Logging helpers for yt_diarizer."""

import os
from typing import Optional

from .constants import LOG_FILE_NAME

LOG_FILE_PATH: Optional[str] = None


def set_log_file(script_dir: str) -> None:
    """Configure global log file path (log.txt next to the script)."""
    global LOG_FILE_PATH
    LOG_FILE_PATH = os.path.join(script_dir, LOG_FILE_NAME)


def _append_to_log_file(msg: str) -> None:
    """Append a single line to the log file (best-effort, never crashes)."""
    global LOG_FILE_PATH
    if not LOG_FILE_PATH:
        return
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        # Logging must never break the pipeline
        pass


def log_line(msg: str) -> None:
    """Print to stdout and append the same line to log.txt."""
    print(msg)
    _append_to_log_file(msg)


def debug(msg: str) -> None:
    """Debug log helper."""
    log_line(f"[yt-diarizer] {msg}")
