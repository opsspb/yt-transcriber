"""Subprocess helpers with logging."""

import subprocess
import sys
from typing import List, Optional, Tuple

from .logging_utils import debug, log_line


def run_logged_subprocess(
    cmd: List[str],
    description: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> Tuple[int, List[str]]:
    """
    Run a subprocess, streaming combined stdout/stderr to console and log file.

    Returns:
      (returncode, list_of_output_lines)
    """
    debug(f"Running subprocess ({description}): {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    lines: List[str] = []
    buffer = ""
    assert process.stdout is not None

    # Stream raw output to the console to preserve ANSI colors and progress
    # animations, while accumulating full lines for logging and error handling.
    while True:
        chunk = process.stdout.read(1024)
        if not chunk:
            break

        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()

        try:
            text_chunk = chunk.decode("utf-8", errors="replace")
        except AttributeError:
            # If chunk is already a string (unlikely), keep it as is.
            text_chunk = chunk

        buffer += text_chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            clean_line = line.rstrip("\r")
            if clean_line:
                log_line(clean_line)
            lines.append(clean_line)

    if buffer.strip():
        clean_line = buffer.rstrip("\r")
        log_line(clean_line)
        lines.append(clean_line)

    process.wait()
    return process.returncode, lines
