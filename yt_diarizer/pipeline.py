"""High-level orchestration for the diarization pipeline."""

import datetime
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .constants import ENV_STAGE_VAR, ENV_URL_VAR, ENV_WORKDIR_VAR, TOKEN_FILENAME
from .deps import ensure_dependencies
from .downloader import download_best_audio
from .exceptions import (
    DependencyError,
    DependencyInstallationError,
    PipelineError,
)
from .logging_utils import debug, log_line
from .process import run_logged_subprocess
from .transcriber import build_diarized_transcript_from_json, run_whisperx_cli


def _token_search_paths(script_dir: str) -> List[str]:
    candidates = []

    script_dir = os.path.abspath(script_dir)
    candidates.append(script_dir)

    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    if os.path.basename(script_dir) == "yt_diarizer" and parent_dir not in candidates:
        candidates.insert(0, parent_dir)

    return [os.path.join(path, TOKEN_FILENAME) for path in candidates]


def load_hf_token(script_dir: str) -> str:
    """Load Hugging Face token, preferring the repository root when run as a module."""

    for token_path in _token_search_paths(script_dir):
        if os.path.isfile(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if not token:
                raise DependencyError("token.txt is empty.")
            return token

    raise DependencyError(
        "token.txt not found. Place it in the repository root next to yt_diarizer.py "
        "or inside the yt_diarizer package."
    )


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


def _resolve_youtube_url() -> str:
    from_env = os.environ.get(ENV_URL_VAR)
    if from_env:
        debug("Using YouTube URL provided on the command line.")
        return from_env
    return prompt_for_youtube_url()


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


def ensure_pkg_config_available() -> None:
    """Ensure pkg-config is present when required for building PyAV."""

    # PyAV (pulled in by faster-whisper) relies on pkg-config to discover
    # system libraries when building wheels from source. Skip the check on
    # Windows, where pkg-config is uncommon and prebuilt wheels are used.
    if sys.platform.startswith("win"):
        return

    if os.environ.get("YT_DIARIZER_ALLOW_MISSING_PKG_CONFIG"):
        debug(
            "Skipping pkg-config preflight because YT_DIARIZER_ALLOW_MISSING_PKG_CONFIG is set."
        )
        return

    if sys.platform == "darwin":
        if shutil.which("pkg-config"):
            return

        debug(
            "pkg-config not found on macOS; skipping preflight. If pip later fails with "
            "'pkg-config is required for building pyav', adjust pinned dependencies to "
            "use a PyAV wheel."
        )
        return

    if shutil.which("pkg-config"):
        return

    raise DependencyInstallationError(
        "pkg-config not found in PATH. Install pkg-config (e.g., `brew install "
        "pkg-config` on macOS or `sudo apt-get install pkg-config` on "
        "Debian/Ubuntu) and rerun, or set YT_DIARIZER_ALLOW_MISSING_PKG_CONFIG=1 "
        "to bypass this preflight if you know pkg-config is unnecessary for your environment.",
    )


def install_python_dependencies(venv_python: str) -> None:
    """
    Install required Python packages (whisperx stack + yt-dlp) into the venv.
    We pin only yt-dlp and av; WhisperX and its own dependencies are allowed
    to resolve their versions to avoid conflicts (in particular around NumPy).
    """
    debug("Installing Python dependencies (WhisperX stack) inside venv ...")

    # On macOS, PyAV may require pkg-config to be present; we keep this preflight.
    ensure_pkg_config_available()

    pinned_versions = {
        # Keep these pinned for reproducibility; the rest is resolved by WhisperX.
        "av": "12.3.0",
        "yt-dlp": "2024.11.18",
    }

    def _run(cmd: List[str], description: str) -> None:
        rc, log_lines = run_logged_subprocess(cmd, description)
        if rc != 0:
            last_snippet = "\n".join(log_lines[-20:])
            raise RuntimeError(
                f"ERROR: {description} failed with exit code {rc}.\n"
                f"Last output snippet:\n{last_snippet}"
            )

    # Upgrade pip first to get a modern dependency resolver.
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ],
        "pip upgrade",
    )

    # Install WhisperX from GitHub together with pinned yt-dlp and av.
    # NumPy, ctranslate2, faster-whisper etc. are resolved automatically
    # according to WhisperX's pyproject metadata, so we don't manually
    # constrain them and avoid version conflicts.
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "git+https://github.com/m-bain/whisperX.git",
            f"yt-dlp=={pinned_versions['yt-dlp']}",
            f"av=={pinned_versions['av']}",
        ],
        "install WhisperX (from GitHub), yt-dlp and av",
    )


# ---------------------------------------------------------------------------
# ffmpeg resolution order:
# 1) YT_DIARIZER_FFMPEG / YT_DIARIZER_FFPROBE environment overrides
# 2) Existing ffmpeg/ffprobe on PATH
# 3) macOS: download from the previously working ColorsWind GitHub release
# 4) Linux/Windows: download from yt-dlp/FFmpeg-Builds
# If downloads fail, instruct the user to install ffmpeg manually or set env vars.
# ---------------------------------------------------------------------------


def _log_error(message: str) -> None:
    """Log errors consistently using the debug logger."""

    debug(message)


def _validate_binary(path: str, description: str) -> str:
    if not path:
        raise DependencyError(f"{description} was empty.")
    if not os.path.isfile(path):
        raise DependencyError(f"{description} points to a missing file: {path}")
    return path


def _ffmpeg_from_env() -> Optional[Dict[str, str]]:
    """Return ffmpeg/ffprobe paths if provided via env overrides.

    When *either* override is present we require both binaries to resolve
    successfully. This prevents silently falling back to downloads when the
    user explicitly provided paths.
    """

    def _resolve_path(value: str, exe_name: str) -> str:
        if os.path.isdir(value):
            candidate = os.path.join(value, exe_name)
        else:
            candidate = value
        return _validate_binary(candidate, f"Environment override for {exe_name}")

    ffmpeg_env = os.environ.get("YT_DIARIZER_FFMPEG") or os.environ.get(
        "YT_DIARIZER_FFMPEG_PATH"
    )
    ffprobe_env = os.environ.get("YT_DIARIZER_FFPROBE") or os.environ.get(
        "YT_DIARIZER_FFPROBE_PATH"
    )

    if not ffmpeg_env and not ffprobe_env:
        return None

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"

    ffmpeg_path = _resolve_path(ffmpeg_env, exe_name) if ffmpeg_env else ""

    if ffprobe_env:
        ffprobe_path = _resolve_path(ffprobe_env, probe_name)
    elif ffmpeg_path:
        sibling = os.path.join(os.path.dirname(ffmpeg_path), probe_name)
        if os.path.isfile(sibling):
            ffprobe_path = sibling
        else:
            raise DependencyError(
                "YT_DIARIZER_FFMPEG was provided but ffprobe was not found next to it. "
                "Set YT_DIARIZER_FFPROBE or point to a directory containing both binaries."
            )
    else:
        raise DependencyError(
            "YT_DIARIZER_FFPROBE must be set when YT_DIARIZER_FFMPEG is absent."
        )

    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    probe_dir = os.path.dirname(ffprobe_path)
    if ffmpeg_dir != probe_dir:
        raise DependencyError(
            "YT_DIARIZER_FFMPEG and YT_DIARIZER_FFPROBE must reside in the same directory "
            "so yt-dlp can load both via --ffmpeg-location."
        )

    debug(f"Using ffmpeg from environment override: {ffmpeg_path}")
    return {"ffmpeg": ffmpeg_path, "ffprobe": ffprobe_path}


def _ffmpeg_from_path() -> Optional[Dict[str, str]]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        debug(
            "Using ffmpeg/ffprobe from system PATH: "
            f"ffmpeg={ffmpeg_path}, ffprobe={ffprobe_path}"
        )
        return {"ffmpeg": ffmpeg_path, "ffprobe": ffprobe_path}
    return None


def _download_and_extract_archive(urls: List[str], archive_path: str, unpack_dir: str) -> None:
    import urllib.request
    from urllib.error import HTTPError, URLError

    attempted: List[str] = []
    last_error: Optional[Exception] = None

    for url in urls:
        attempted.append(url)
        debug(f"Attempting ffmpeg download from {url}")
        try:
            urllib.request.urlretrieve(url, archive_path)
            last_error = None
            break
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            _log_error(f"ffmpeg download failed from {url}: {exc}")

    if last_error:
        raise RuntimeError(
            "Automatic ffmpeg download failed; tried URLs: "
            + ", ".join(attempted)
            + ". "
            "Install ffmpeg so it is on PATH or set YT_DIARIZER_FFMPEG_PATH."
        )

    os.makedirs(unpack_dir, exist_ok=True)
    debug(f"Extracting ffmpeg archive to {unpack_dir} ...")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(unpack_dir)
        elif archive_path.endswith((".tar.xz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(unpack_dir)
        else:
            raise RuntimeError(f"Unsupported ffmpeg archive format: {archive_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract ffmpeg archive '{archive_path}': {exc}"
        ) from exc


def _find_ffmpeg_binaries(
    root: Path, log: Callable[[str], None]
) -> Tuple[Optional[Path], Optional[Path]]:
    """Recursively search for ffmpeg/ffprobe binaries under root (case-insensitive)."""

    log(f"[yt-diarizer] Searching for ffmpeg/ffprobe under {root}")

    ffmpeg_path: Optional[Path] = None
    ffprobe_path: Optional[Path] = None

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        log(f"[yt-diarizer] Checking file: {path}")

        name = path.name.lower()
        if name == "ffmpeg" and ffmpeg_path is None:
            ffmpeg_path = path
            log(f"[yt-diarizer] Found ffmpeg candidate: {path}")
        elif name == "ffprobe" and ffprobe_path is None:
            ffprobe_path = path
            log(f"[yt-diarizer] Found ffprobe candidate: {path}")

        if ffmpeg_path and ffprobe_path:
            break

    log(
        "[yt-diarizer] ffmpeg search results: ffmpeg=%s, ffprobe=%s"
        % (ffmpeg_path, ffprobe_path)
    )
    if ffmpeg_path is None:
        raise RuntimeError(
            f"Could not locate 'ffmpeg' binary after scanning extracted archive under {root}"
        )

    return ffmpeg_path, ffprobe_path


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _prepare_macos_ffmpeg(
    extract_dir: Path, workspace_dir: Path, log: Callable[[str], None]
) -> Dict[str, Optional[str]]:
    ffmpeg_src, ffprobe_src = _find_ffmpeg_binaries(extract_dir, log)

    ffmpeg_dir = workspace_dir / "ffmpeg_macos" / "bin"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_dst = ffmpeg_dir / "ffmpeg"
    log(f"[yt-diarizer] Copying ffmpeg to {ffmpeg_dst}")
    shutil.copy2(ffmpeg_src, ffmpeg_dst)
    _make_executable(ffmpeg_dst)

    ffprobe_dst = ffmpeg_dir / "ffprobe"
    if ffprobe_src is not None:
        log(f"[yt-diarizer] Copying ffprobe to {ffprobe_dst}")
        shutil.copy2(ffprobe_src, ffprobe_dst)
        _make_executable(ffprobe_dst)
    else:
        log(
            "[WARN] [yt-diarizer] ffprobe not found in downloaded archive; yt-dlp may still "
            "fail and you may need a different build"
        )

    log(f"[yt-diarizer] Using downloaded ffmpeg at {ffmpeg_dst}")
    return {
        "ffmpeg": str(ffmpeg_dst),
        "ffprobe": str(ffprobe_dst) if ffprobe_dst.exists() else None,
    }


def _find_binary(unpack_dir: str, name: str) -> str:
    for root, _, files in os.walk(unpack_dir):
        if name in files:
            return os.path.join(root, name)
    raise RuntimeError(f"Downloaded ffmpeg archive missing '{name}' binary")


def download_ffmpeg_for_macos(work_dir: str) -> Dict[str, Optional[str]]:
    """Download ffmpeg/ffprobe for macOS using the previously working release."""

    colorswind_urls = [
        "https://github.com/ColorsWind/FFmpeg-macOS/releases/download/"
        "n5.0.1-patch3/FFmpeg-shared-n5.0.1-OSX-universal.zip",
        "https://github.com/ColorsWind/FFmpeg-macOS/releases/download/"
        "n5.0.1-patch3/FFmpeg-n5.0.1-OSX-universal.zip",
    ]

    filename = os.path.basename(colorswind_urls[0])
    archive_path = os.path.join(work_dir, filename)
    unpack_dir = os.path.join(work_dir, "ffmpeg_macos_unpack")

    _download_and_extract_archive(colorswind_urls, archive_path, unpack_dir)

    return _prepare_macos_ffmpeg(Path(unpack_dir), Path(work_dir), debug)


def _build_ffmpeg_urls_for_other_platforms() -> List[str]:
    system = sys.platform

    if system.startswith("linux"):
        base_names = ["ffmpeg-master-latest-linux64-gpl", "ffmpeg-latest-linux64-gpl"]
        suffixes = [".tar.xz", ".zip"]
    elif system.startswith("win"):
        base_names = ["ffmpeg-master-latest-win64-gpl", "ffmpeg-latest-win64-gpl"]
        suffixes = [".zip"]
    else:
        base_names = ["ffmpeg-master-latest-macos-universal"]
        suffixes = [".zip"]

    urls: List[str] = []
    for base in base_names:
        for suffix in suffixes:
            name = f"{base}{suffix}"
            urls.append(
                f"https://github.com/yt-dlp/FFmpeg-Builds/releases/latest/download/{name}"
            )
    return urls


def download_ffmpeg_for_other_platforms(work_dir: str) -> Dict[str, str]:
    urls = _build_ffmpeg_urls_for_other_platforms()
    if not urls:
        raise RuntimeError("No ffmpeg download URLs defined for this platform")

    filename = os.path.basename(urls[0])
    archive_path = os.path.join(work_dir, filename)
    unpack_dir = os.path.join(work_dir, "ffmpeg_other")

    _download_and_extract_archive(urls, archive_path, unpack_dir)

    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffmpeg_path = _find_binary(unpack_dir, ffmpeg_name)
    ffprobe_path = _find_binary(unpack_dir, ffprobe_name)

    for binary_path in (ffmpeg_path, ffprobe_path):
        try:
            os.chmod(binary_path, 0o755)
        except Exception:
            pass

    return {"ffmpeg": ffmpeg_path, "ffprobe": ffprobe_path}


def ensure_ffmpeg(work_dir: str) -> Dict[str, Optional[str]]:
    """
    Ensure ffmpeg and ffprobe are available.

    Priority order:
    1. Environment overrides (YT_DIARIZER_FFMPEG_PATH / YT_DIARIZER_FFPROBE_PATH).
    2. Binaries already on PATH.
    3. macOS: download from the previously working ColorsWind release.
    4. Other platforms: download from yt-dlp/FFmpeg-Builds.
    """

    env_paths = _ffmpeg_from_env()
    if env_paths:
        bin_dir = os.path.dirname(env_paths["ffmpeg"])
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        debug(
            "Respecting environment overrides for ffmpeg/ffprobe; skipping auto-download."
        )
        return {
            "ffmpeg": env_paths["ffmpeg"],
            "ffprobe": env_paths.get("ffprobe"),
            "location_dir": bin_dir,
        }

    path_paths = _ffmpeg_from_path()
    if path_paths:
        bin_dir = os.path.dirname(path_paths["ffmpeg"])
        return {
            "ffmpeg": path_paths["ffmpeg"],
            "ffprobe": path_paths.get("ffprobe"),
            "location_dir": bin_dir,
        }

    if sys.platform == "darwin":
        try:
            download_paths = download_ffmpeg_for_macos(work_dir)
        except RuntimeError as exc:
            raise RuntimeError(
                "Automatic macOS ffmpeg download failed. Install ffmpeg so it appears on PATH "
                "or set YT_DIARIZER_FFMPEG_PATH/FFPROBE_PATH."
            ) from exc
    else:
        try:
            download_paths = download_ffmpeg_for_other_platforms(work_dir)
        except RuntimeError as exc:
            raise RuntimeError(
                "Automatic ffmpeg download failed. Install ffmpeg so it appears on PATH or set "
                "YT_DIARIZER_FFMPEG_PATH/FFPROBE_PATH."
            ) from exc

    bin_dir = os.path.dirname(download_paths["ffmpeg"])
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    debug(
        f"Using downloaded ffmpeg at {download_paths['ffmpeg']} "
        f"(ffprobe={download_paths.get('ffprobe')})"
    )
    return {
        "ffmpeg": download_paths["ffmpeg"],
        "ffprobe": download_paths.get("ffprobe"),
        "location_dir": bin_dir,
    }


def run_pipeline_inside_venv(script_dir: str, work_dir: str) -> None:
    """Inner stage: actual diarization pipeline inside the venv."""
    if not work_dir:
        raise PipelineError("Internal error: workspace directory not provided.")

    debug(f"Workspace inside venv: {work_dir}")

    _configure_cache_dirs(work_dir)

    hf_token = load_hf_token(script_dir)
    os.environ.setdefault("HF_TOKEN", hf_token)

    ffmpeg_paths = ensure_ffmpeg(work_dir)
    ffmpeg_location = ffmpeg_paths["location_dir"]

    deps = ensure_dependencies()
    yt_downloader = deps["yt_downloader"]
    whisperx_bin = deps["whisperx"]

    url = _resolve_youtube_url()
    audio_path = download_best_audio(
        yt_downloader, url, work_dir, script_dir, ffmpeg_location
    )
    json_result_path = run_whisperx_cli(
        whisperx_bin, audio_path, hf_token, work_dir
    )

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
    ensure_pkg_config_available()

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
