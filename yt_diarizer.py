#!/usr/bin/env python3
"""
yt_diarizer.py

End-to-end pipeline:

1. Ask user for a YouTube URL.
2. Create a temporary workspace + venv in the script directory.
3. Inside the venv:
   - pip-install WhisperX + yt-dlp (tools are downloaded locally into venv).
   - Ensure ffmpeg is available; if not, download a macOS universal build.
4. Use yt-dlp to download best audio and convert to WAV.
   - Tries multiple strategies (plain, cookies.txt, Safari/Chrome cookies).
5. Run WhisperX (large-v3, diarization enabled, float32) to get a diarized JSON.
6. Post-process JSON into a clean diarized transcript.
7. Save outputs (TXT + JSON) next to this script.
8. Delete the workspace + venv so the directory is clean again.

Logging:
- Every run appends everything the script prints (including subprocess output
  from pip, yt-dlp and whisperx) to log.txt in the same directory as this script.
- The same messages are printed to the terminal.

Assumptions:
- Running on macOS (Apple Silicon) with system Python 3.9.6.
- No brew / global installs; everything is inside a temporary venv.
- token.txt (HF token for segmentation 3.0 + speaker-diarization 3.1)
  lives in the same directory as this script.

Optional for restricted videos:
- Export your logged-in browser cookies to a Netscape-format cookies.txt
  and put it next to this script (or set YT_DIARIZER_COOKIES=/full/path/to/cookies.txt).
"""

import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Any, Dict, List, Optional

ENV_STAGE_VAR = "YT_DIARIZER_STAGE"
ENV_WORKDIR_VAR = "YT_DIARIZER_WORK_DIR"

LOG_FILE_PATH: Optional[str] = None


def set_log_file(script_dir: str) -> None:
    """Configure global log file path (log.txt next to this script)."""
    global LOG_FILE_PATH
    LOG_FILE_PATH = os.path.join(script_dir, "log.txt")


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


class DependencyError(RuntimeError):
    """Raised when a required external dependency is missing or broken."""


class PipelineError(RuntimeError):
    """Raised when any step of the pipeline fails."""


def run_logged_subprocess(
    cmd: List[str],
    description: str,
    cwd: Optional[str] = None,
) -> (int, List[str]):
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
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        if line:
            log_line(line)
        lines.append(line)
    process.wait()
    return process.returncode, lines


def format_timestamp(seconds: Optional[float]) -> str:
    """Format seconds as HH:MM:SS.mmm (zero-padded)."""
    if seconds is None:
        return "00:00:00.000"
    try:
        total_ms = int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        total_ms = 0
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def build_diarized_transcript_lines_from_data(data: Dict[str, Any]) -> List[str]:
    """
    Convert WhisperX JSON-like transcription result into diarized transcript lines.

    Each line looks like:
      "[00:00:01.000 --> 00:00:03.500] SPEAKER_00: Hello world"
    """
    segments = data.get("segments") or []
    lines: List[str] = []
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker") or "UNKNOWN"
        start_ts = format_timestamp(start)
        end_ts = format_timestamp(end)
        line = f"[{start_ts} --> {end_ts}] {speaker}: {text}"
        lines.append(line)
    return lines


def load_hf_token(script_dir: str) -> str:
    """Load Hugging Face token from token.txt next to the script."""
    token_path = os.path.join(script_dir, "token.txt")
    if not os.path.isfile(token_path):
        raise DependencyError(
            f"token.txt not found at {token_path}."
        )
    with open(token_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise DependencyError("token.txt is empty.")
    return token


def download_ffmpeg_if_missing(work_dir: str) -> None:
    """
    Ensure ffmpeg is in PATH.
    If missing, download a macOS universal binary into work_dir and prepend its bin dir to PATH.
    """
    import urllib.request

    if shutil.which("ffmpeg"):
        debug("ffmpeg already available in PATH.")
        return

    if sys.platform != "darwin":
        raise DependencyError("ffmpeg not found and auto-download is only implemented on macOS.")

    ffmpeg_zip_url = (
        "https://github.com/ColorsWind/FFmpeg-macOS/releases/download/"
        "n5.0.1-patch3/FFmpeg-shared-n5.0.1-OSX-universal.zip"
    )
    archive_path = os.path.join(work_dir, "ffmpeg_macos_universal.zip")
    unpack_dir = os.path.join(work_dir, "ffmpeg_unpacked")

    debug(f"Downloading ffmpeg binary from {ffmpeg_zip_url} ...")
    urllib.request.urlretrieve(ffmpeg_zip_url, archive_path)

    debug(f"Extracting ffmpeg archive to {unpack_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(unpack_dir)

    ffmpeg_path = None
    for root, _, files in os.walk(unpack_dir):
        if "ffmpeg" in files:
            ffmpeg_path = os.path.join(root, "ffmpeg")
            break
    if not ffmpeg_path:
        raise DependencyError(
            "Downloaded ffmpeg archive but could not find 'ffmpeg' binary inside."
        )

    try:
        os.chmod(ffmpeg_path, 0o755)
    except Exception:
        # Best-effort; if chmod fails but it's already executable, that's fine.
        pass

    bin_dir = os.path.dirname(ffmpeg_path)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    debug(f"Using downloaded ffmpeg at {ffmpeg_path}")


def find_executable(candidates: List[str]) -> str:
    """Return first executable found in PATH from the list, or raise."""
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    raise DependencyError(f"Executables not found: {', '.join(candidates)}")


def ensure_dependencies() -> Dict[str, str]:
    """Ensure ffmpeg, yt-dlp and whisperx CLIs are available in PATH."""
    deps: Dict[str, str] = {}
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise DependencyError("ffmpeg executable not found in PATH even after attempted download.")
    deps["ffmpeg"] = ffmpeg_path
    deps["yt_downloader"] = find_executable(["yt-dlp", "yt_dlp"])
    deps["whisperx"] = find_executable(["whisperx"])
    return deps


def prompt_for_youtube_url() -> str:
    """Ask for YouTube URL and log the prompt."""
    log_line("Paste YouTube video URL:")
    url = input().strip()
    if not url:
        raise PipelineError("Empty URL provided.")
    return url


def build_yt_dlp_command_variants(
    downloader_bin: str,
    url: str,
    work_dir: str,
    script_dir: str,
) -> List[List[str]]:
    """
    Build a list of yt-dlp command variants to try.

    Order:
      1) plain public download
      2) with cookies from cookies.txt / YT_DIARIZER_COOKIES (if present)
      3) cookies from Safari
      4) cookies from Chrome
    """
    output_template = os.path.join(work_dir, "audio.%(ext)s")
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    )

    base_cmd = [
        downloader_bin,
        "--ignore-config",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--force-ipv4",
        "--user-agent",
        user_agent,
        "-o",
        output_template,
        url,
    ]

    commands: List[List[str]] = []

    # 1) plain
    commands.append(list(base_cmd))

    # 2) cookies.txt if present
    cookies_path: Optional[str] = None
    env_cookies = os.environ.get("YT_DIARIZER_COOKIES")
    if env_cookies and os.path.isfile(env_cookies):
        cookies_path = env_cookies
    else:
        candidate = os.path.join(script_dir, "cookies.txt")
        if os.path.isfile(candidate):
            cookies_path = candidate

    if cookies_path:
        commands.append(base_cmd + ["--cookies", cookies_path])

    # 3) Safari cookies (if Terminal has Full Disk Access)
    commands.append(base_cmd + ["--cookies-from-browser", "safari"])

    # 4) Chrome cookies
    commands.append(base_cmd + ["--cookies-from-browser", "chrome"])

    return commands


def download_audio_to_wav(
    downloader_bin: str,
    url: str,
    work_dir: str,
    script_dir: str,
) -> str:
    """
    Use yt-dlp to grab best audio and convert to WAV via ffmpeg.

    Tries multiple strategies and raises a detailed error if all fail.
    """
    debug("Starting audio download via yt-dlp...")
    commands = build_yt_dlp_command_variants(downloader_bin, url, work_dir, script_dir)
    last_err_msg: Optional[str] = None

    for idx, cmd in enumerate(commands, start=1):
        debug(f"Trying yt-dlp variant #{idx}: {' '.join(cmd)}")
        rc, lines = run_logged_subprocess(cmd, f"yt-dlp variant #{idx}")
        if rc == 0:
            debug("yt-dlp download succeeded.")
            break

        snippet = "\n".join([ln for ln in lines if ln][-50:])
        last_err_msg = f"yt-dlp exited with code {rc}. Last output snippet:\n{snippet}"
        debug(last_err_msg)
    else:
        # No variant succeeded
        raise PipelineError(
            "Audio download/conversion failed after trying multiple strategies.\n"
            "If this video is age/region/login restricted, export your browser cookies\n"
            "to a Netscape-format cookies.txt and place it next to this script, or set\n"
            "YT_DIARIZER_COOKIES=/full/path/to/cookies.txt, then rerun.\n"
            "On macOS with Safari, Terminal may also need 'Full Disk Access' in\n"
            "System Settings → Privacy & Security.\n"
            + (last_err_msg or "")
        )

    expected = os.path.join(work_dir, "audio.wav")
    if os.path.isfile(expected):
        debug(f"Audio saved to {expected}")
        return expected

    wav_candidates = glob.glob(os.path.join(work_dir, "*.wav"))
    if not wav_candidates:
        raise PipelineError(
            "yt-dlp reported success but no .wav files were found in the workspace."
        )

    wav_candidates.sort()
    chosen = wav_candidates[0]
    debug(f"Using WAV file: {chosen}")
    return chosen


def run_whisperx_cli(
    whisperx_bin: str,
    audio_path: str,
    hf_token: str,
    work_dir: str,
) -> str:
    """
    Run WhisperX CLI to produce a diarized JSON transcription.

    High-quality settings:
      - model: large-v3
      - compute_type: float32
      - beam_size: 5
      - diarization: pyannote
    """
    debug("Running WhisperX diarization with large-v3 model (high quality)...")

    threads = os.cpu_count() or 1

    cmd = [
        whisperx_bin,
        audio_path,
        "--model",
        "large-v3",
        "--diarize",
        "--hf_token",
        hf_token,
        "--batch_size",
        "8",
        "--beam_size",
        "5",
        "--compute_type",
        "float32",
        "--device",
        "cpu",
        "--threads",
        str(threads),
        "--vad_method",
        "pyannote",
        "--output_format",
        "json",
        "--output_dir",
        work_dir,
        "--verbose",
        "True",
        "--print_progress",
        "True",
    ]

    rc, lines = run_logged_subprocess(cmd, "whisperx diarization")
    if rc != 0:
        snippet = "\n".join([ln for ln in lines if ln][-50:])
        raise PipelineError(
            f"WhisperX diarization failed with exit code {rc}.\nLast output snippet:\n{snippet}"
        )

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(work_dir, f"{basename}.json")
    if not os.path.isfile(json_path):
        json_candidates = glob.glob(os.path.join(work_dir, "*.json"))
        if not json_candidates:
            raise PipelineError(
                "WhisperX completed but no JSON output was found in the workspace."
            )
        json_candidates.sort()
        json_path = json_candidates[0]

    debug(f"WhisperX JSON output: {json_path}")
    return json_path


def build_diarized_transcript_from_json(json_path: str) -> List[str]:
    """Load WhisperX JSON file and build diarized transcript lines."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return build_diarized_transcript_lines_from_data(data)


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


def run_pipeline_inside_venv(script_dir: str, work_dir: str) -> None:
    """Inner stage: actual diarization pipeline inside the venv."""
    if not work_dir:
        raise PipelineError("Internal error: workspace directory not provided.")

    debug(f"Workspace inside venv: {work_dir}")

    hf_token = load_hf_token(script_dir)
    os.environ.setdefault("HF_TOKEN", hf_token)

    download_ffmpeg_if_missing(work_dir)

    deps = ensure_dependencies()
    yt_downloader = deps["yt_downloader"]
    whisperx_bin = deps["whisperx"]

    url = prompt_for_youtube_url()
    wav_path = download_audio_to_wav(yt_downloader, url, work_dir, script_dir)
    json_result_path = run_whisperx_cli(whisperx_bin, wav_path, hf_token, work_dir)

    transcript_lines = build_diarized_transcript_from_json(json_result_path)
    outputs = save_final_outputs(transcript_lines, json_result_path, script_dir)

    log_line("")
    log_line("=== Done ===")
    log_line(f"Diarized transcript (TXT): {outputs['txt']}")
    log_line(f"Raw WhisperX output (JSON): {outputs['json']}")


def install_python_dependencies(venv_python: str) -> None:
    """
    Install required Python packages (whisperx, yt-dlp) into the venv.

    All logs from pip are also mirrored into log.txt.
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


def setup_and_run_in_venv(script_dir: str, work_dir: str) -> int:
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
    completed = subprocess.run([venv_python, os.path.abspath(__file__)], env=env)
    return completed.returncode


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
        exit_code = setup_and_run_in_venv(script_dir, work_dir)
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


if __name__ == "__main__":
    main()
