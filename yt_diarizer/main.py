"""Entry point wiring for the yt_diarizer package."""

import argparse
import datetime
import os
import shutil
import sys
from typing import Optional

from .constants import (
    ENV_MPS_CONVERT_VAR,
    ENV_STAGE_VAR,
    ENV_URL_VAR,
    ENV_WORKDIR_VAR,
)
from .exceptions import (
    DependencyError,
    DependencyInstallationError,
    PipelineError,
)
from .logging_utils import debug, log_line, set_log_file
from .pipeline import (
    ensure_pkg_config_available,
    run_pipeline_inside_venv,
    setup_and_run_in_venv,
)


def main(script_dir: Optional[str] = None, entrypoint_path: Optional[str] = None) -> None:
    script_dir = script_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    entrypoint_path = entrypoint_path or os.path.abspath(__file__)

    parser = argparse.ArgumentParser(
        description=(
            "Download a YouTube video's audio and produce a diarized transcript with WhisperX."
        )
    )
    parser.add_argument("url", nargs="?", help="YouTube URL to transcribe")
    parser.add_argument(
        "-c",
        "--cookies",
        dest="cookies",
        help="Path to cookies.txt for yt-dlp (optional override for YT_DIARIZER_COOKIES)",
    )
    parser.add_argument(
        "--mps-convert",
        action="store_true",
        help="Use Whisper transcription on Apple Silicon (MPS) without diarization",
    )

    # Configure logging and mark run header
    set_log_file(script_dir)
    run_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(f"=== yt-diarizer run started at {run_id} ===")

    stage = os.environ.get(ENV_STAGE_VAR, "outer")
    args: Optional[argparse.Namespace] = None
    if stage != "inner":
        args = parser.parse_args()
        if args.cookies:
            os.environ.setdefault("YT_DIARIZER_COOKIES", args.cookies)
        if args.mps_convert:
            os.environ.setdefault(ENV_MPS_CONVERT_VAR, "1")

        ensure_pkg_config_available()

    if stage == "inner":
        # Inner stage: workspace and venv already set up.
        work_dir = os.environ.get(ENV_WORKDIR_VAR)
        try:
            run_pipeline_inside_venv(script_dir, work_dir)
        except (DependencyError, PipelineError) as exc:
            log_line(f"ERROR: {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            log_line("Interrupted by user.")
            sys.exit(130)
        return

    # Outer stage: create workspace, venv, and clean everything after.
    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(script_dir, f".yt_diarizer_work_{timestamp_tag}")
    os.makedirs(work_dir, exist_ok=True)

    exit_code = 0
    cleanup_workspace = True
    try:
        if args and args.url:
            os.environ[ENV_URL_VAR] = args.url
        exit_code = setup_and_run_in_venv(
            script_dir,
            work_dir,
            entrypoint_path,
            mps_convert=bool(os.environ.get(ENV_MPS_CONVERT_VAR)),
        )
    except DependencyInstallationError as exc:
        log_line(f"ERROR: {exc}")
        exit_code = 1
        cleanup_workspace = False
        log_line(f"Workspace preserved for inspection at {work_dir}")
    except (DependencyError, PipelineError) as exc:
        log_line(f"ERROR: {exc}")
        exit_code = 1
    except KeyboardInterrupt:
        log_line("Interrupted by user.")
        exit_code = 130
    finally:
        if cleanup_workspace and os.path.isdir(work_dir):
            debug(f"Cleaning up workspace {work_dir} ...")
            shutil.rmtree(work_dir, ignore_errors=True)

    sys.exit(exit_code)
