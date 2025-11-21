"""Logging helpers for yt_diarizer."""

import datetime


def set_log_file(_: str) -> None:  # pragma: no cover - kept for compatibility
    """No-op compatibility hook. Logging now only prints to stdout."""


def log_line(msg: str) -> None:
    """Print a timestamped log line to stdout."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)


def debug(msg: str) -> None:
    """Debug log helper."""
    log_line(f"[yt-diarizer] {msg}")
