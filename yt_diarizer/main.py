"""Entry point wiring for the yt_diarizer package."""

import datetime
import os
import shutil
import sys

from .constants import ENV_STAGE_VAR, ENV_WORKDIR_VAR
from .exceptions import DependencyError, PipelineError
from .logging_utils import debug, log_line, set_log_file
from .pipeline import run_pipeline_inside_venv, setup_and_run_in_venv


def main(script_dir: str | None = None, entrypoint_path: str | None = None) -> None:
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    entrypoint_path = entrypoint_path or os.path.abspath(__file__)

    # Configure logging to log.txt and mark run header
    set_log_file(script_dir)
    run_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line(f"=== yt-diarizer run started at {run_id} ===")

    stage = os.environ.get(ENV_STAGE_VAR, "outer")

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
    try:
        exit_code = setup_and_run_in_venv(script_dir, work_dir, entrypoint_path)
    except (DependencyError, PipelineError) as exc:
        log_line(f"ERROR: {exc}")
        exit_code = 1
    except KeyboardInterrupt:
        log_line("Interrupted by user.")
        exit_code = 130
    finally:
        if os.path.isdir(work_dir):
            debug(f"Cleaning up workspace {work_dir} ...")
            shutil.rmtree(work_dir, ignore_errors=True)

    sys.exit(exit_code)
