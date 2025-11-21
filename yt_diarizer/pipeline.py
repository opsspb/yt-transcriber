"""High-level orchestration for the diarization pipeline."""

import datetime
import os
import shutil
import subprocess
import sys
from typing import Dict, List

from .constants import ENV_STAGE_VAR, ENV_WORKDIR_VAR, TOKEN_FILENAME
from .deps import download_ffmpeg_if_missing, ensure_dependencies
from .downloader import download_audio_to_wav
from .exceptions import DependencyError, PipelineError
from .logging_utils import debug, log_line
from .process import run_logged_subprocess
from .transcriber import build_diarized_transcript_from_json, run_whisperx_cli


def load_hf_token(script_dir: str) -> str:
    """Load Hugging Face token from token.txt next to the script."""
    token_path = os.path.join(script_dir, TOKEN_FILENAME)
    if not os.path.isfile(token_path):
        raise DependencyError(
            f"token.txt not found at {token_path}."
        )
    with open(token_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise DependencyError("token.txt is empty.")
    return token


def _configure_cache_dirs(work_dir: str) -> None:
    """Point cache-related env vars into the workspace for automatic cleanup."""

    cache_root = os.path.join(work_dir, "cache")
    os.makedirs(cache_root, exist_ok=True)

    defaults = {
        "HF_HOME": os.path.join(cache_root, "hf"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "transformers"),
        "XDG_CACHE_HOME": cache_root,
        "PYANNOTE_CACHE": os.path.join(cache_root, "pyannote"),
        "TORCH_HOME": os.path.join(cache_root, "torch"),
    }

    for env_var, path in defaults.items():
        if not os.environ.get(env_var):
            os.environ[env_var] = path

    debug(
        "Caching directories redirected to workspace for cleanup: "
        + ", ".join(f"{k}={os.environ[k]}" for k in defaults)
    )


def prompt_for_youtube_url() -> str:
    """Ask for YouTube URL and log the prompt."""
    log_line("Paste YouTube video URL:")
    url = input().strip()
    if not url:
        raise PipelineError("Empty URL provided.")
    return url


def save_final_outputs(
    transcript_lines: List[str],
    json_path: str,
    script_dir: str,
) -> Dict[str, str]:
    """
    Save the diarized transcript (.txt) and raw WhisperX JSON into script_dir.

    Returns:
      {"txt": <txt_path>, "json": <json_path>}
    """
    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"diarized_transcript_{timestamp_tag}"

    txt_path = os.path.join(script_dir, base_name + ".txt")
    json_target = os.path.join(script_dir, base_name + ".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for line in transcript_lines:
            f.write(line + "\n")

    shutil.move(json_path, json_target)

    debug(f"Saved TXT transcript to {txt_path}")
    debug(f"Saved JSON output to {json_target}")

    return {"txt": txt_path, "json": json_target}


def install_python_dependencies(venv_python: str) -> None:
    """
    Install required Python packages (whisperx, yt-dlp) into the venv.
    """
    debug("Installing Python dependencies (whisperx, yt-dlp) inside venv ...")

    rc, lines = run_logged_subprocess(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        "pip upgrade",
    )
    if rc != 0:
        snippet = "\n".join([ln for ln in lines if ln][-50:])
        raise DependencyError(
            f"pip upgrade failed with exit code {rc}.\nLast output snippet:\n{snippet}"
        )

    rc, lines = run_logged_subprocess(
        [venv_python, "-m", "pip", "install", "whisperx", "yt-dlp"],
        "pip install whisperx yt-dlp",
    )
    if rc != 0:
        snippet = "\n".join([ln for ln in lines if ln][-50:])
        raise DependencyError(
            f"pip install failed with exit code {rc}.\nLast output snippet:\n{snippet}"
        )


def run_pipeline_inside_venv(script_dir: str, work_dir: str) -> None:
    """Inner stage: actual diarization pipeline inside the venv."""
    if not work_dir:
        raise PipelineError("Internal error: workspace directory not provided.")

    debug(f"Workspace inside venv: {work_dir}")

    _configure_cache_dirs(work_dir)

    hf_token = load_hf_token(script_dir)
    os.environ.setdefault("HF_TOKEN", hf_token)

    ffmpeg_path = download_ffmpeg_if_missing(work_dir)

    deps = ensure_dependencies()
    yt_downloader = deps["yt_downloader"]
    whisperx_bin = deps["whisperx"]

    url = prompt_for_youtube_url()
    wav_path = download_audio_to_wav(
        yt_downloader, url, work_dir, script_dir, ffmpeg_path
    )
    json_result_path = run_whisperx_cli(whisperx_bin, wav_path, hf_token, work_dir)

    transcript_lines = build_diarized_transcript_from_json(json_result_path)
    outputs = save_final_outputs(transcript_lines, json_result_path, script_dir)

    log_line("")
    log_line("=== Done ===")
    log_line(f"Diarized transcript (TXT): {outputs['txt']}")
    log_line(f"Raw WhisperX output (JSON): {outputs['json']}")


def setup_and_run_in_venv(script_dir: str, work_dir: str, entrypoint_path: str) -> int:
    """
    Outer stage: create temporary venv in work_dir, install deps, then re-run this
    script inside that venv. Finally, return the exit code from the inner run.
    """
    debug(f"Creating temporary virtualenv in {work_dir} ...")

    venv_dir = os.path.join(work_dir, "venv")
    rc, lines = run_logged_subprocess(
        [sys.executable, "-m", "venv", venv_dir],
        "create virtualenv",
    )
    if rc != 0:
        snippet = "\n".join([ln for ln in lines if ln][-50:])
        raise PipelineError(
            f"Failed to create virtualenv: exit code {rc}.\nLast output snippet:\n{snippet}"
        )

    if os.name == "nt":
        venv_bin = os.path.join(venv_dir, "Scripts")
        venv_python = os.path.join(venv_bin, "python.exe")
    else:
        venv_bin = os.path.join(venv_dir, "bin")
        venv_python = os.path.join(venv_bin, "python")

    if not os.path.isfile(venv_python):
        raise PipelineError(f"Could not locate venv python at {venv_python}")

    install_python_dependencies(venv_python)

    env = os.environ.copy()
    env[ENV_STAGE_VAR] = "inner"
    env[ENV_WORKDIR_VAR] = work_dir
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    debug("Re-running script inside venv...")
    completed = subprocess.run([venv_python, os.path.abspath(entrypoint_path)], env=env)
    return completed.returncode
