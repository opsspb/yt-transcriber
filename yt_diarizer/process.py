"""Subprocess helpers with logging."""

import subprocess
from typing import List, Optional, Tuple

from .logging_utils import debug, log_line


def run_logged_subprocess(
    cmd: List[str], description: str, cwd: Optional[str] = None
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    lines: List[str] = []
    last_progress_line: Optional[str] = None
    progress_displayed = False
    assert process.stdout is not None
    for raw_line in process.stdout:
        raw_line = raw_line.rstrip("\n")

        # yt-dlp prints download progress with carriage returns ("\r"), which
        # causes the progress line to be rewritten in-place instead of
        # flooding the terminal. Capture this behavior even though stdout is not
        # a TTY by treating progress lines specially.
        parts = raw_line.split("\r")
        for part in parts:
            if not part:
                continue

            is_progress = part.startswith("[download]")

            if is_progress:
                last_progress_line = part.rstrip()
                print(f"\r{last_progress_line}", end="", flush=True)
                progress_displayed = True
                continue

            if progress_displayed:
                # Ensure regular output starts on a fresh line after an inline
                # progress update.
                print()
                progress_displayed = False

            line = part.rstrip()
            if line:
                log_line(line)
            lines.append(line)

    if progress_displayed:
        print()

    if last_progress_line:
        log_line(last_progress_line)
        lines.append(last_progress_line)
    process.wait()
    return process.returncode, lines
