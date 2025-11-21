"""High-level orchestration for the diarization pipeline."""

import datetime
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import ENV_STAGE_VAR, ENV_URL_VAR, ENV_WORKDIR_VAR, TOKEN_FILENAME
from .deps import ensure_dependencies
from .downloader import download_audio_to_wav
from .exceptions import DependencyError, PipelineError
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


def install_python_dependencies(venv_python: str) -> None:
    """
    Install required Python packages (whisperx stack + yt-dlp) into the venv.
    """
    debug("Installing Python dependencies (pinned WhisperX stack) inside venv ...")

    pinned_versions = {
        "torch": "2.1.2",
        "torchaudio": "2.1.2",
        "whisperx": "3.1.1",
        "yt-dlp": "2024.11.18",
    }

    def _run(cmd: List[str], description: str) -> None:
        rc, lines = run_logged_subprocess(cmd, description)
        if rc != 0:
            snippet = "\n".join([ln for ln in lines if ln][-50:])
            raise DependencyError(
                f"{description} failed with exit code {rc}.\nLast output snippet:\n{snippet}"
            )

    _run(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        "pip upgrade",
    )

    torch_index = "https://download.pytorch.org/whl/cpu"
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            f"torch=={pinned_versions['torch']}",
            f"torchaudio=={pinned_versions['torchaudio']}",
            "--index-url",
            torch_index,
        ],
        "install PyTorch CPU wheels",
    )

    constraint_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            constraint_path = tmp.name
            tmp.write(
                f"torch=={pinned_versions['torch']}\n"
                f"torchaudio=={pinned_versions['torchaudio']}\n"
            )

        _run(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                f"whisperx=={pinned_versions['whisperx']}",
                f"yt-dlp=={pinned_versions['yt-dlp']}",
                "--constraint",
                constraint_path,
            ],
            "install WhisperX and yt-dlp",
        )
    finally:
        if constraint_path and os.path.exists(constraint_path):
            try:
                os.remove(constraint_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# ffmpeg resolution order:
# 1) YT_DIARIZER_FFMPEG_PATH / YT_DIARIZER_FFPROBE_PATH environment overrides
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

    Accepts both current (YT_DIARIZER_FFMPEG_PATH/FFPROBE_PATH) and legacy
    (YT_DIARIZER_FFMPEG/YT_DIARIZER_FFPROBE) variable names.
    """

    def _resolve_path(value: str, exe_name: str) -> str:
        if os.path.isdir(value):
            candidate = os.path.join(value, exe_name)
        else:
            candidate = value
        return _validate_binary(candidate, f"Environment override for {exe_name}")

    ffmpeg_env = os.environ.get("YT_DIARIZER_FFMPEG_PATH") or os.environ.get(
        "YT_DIARIZER_FFMPEG"
    )
    ffprobe_env = os.environ.get("YT_DIARIZER_FFPROBE_PATH") or os.environ.get(
        "YT_DIARIZER_FFPROBE"
    )

    if not ffmpeg_env and not ffprobe_env:
        return None

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_path = _resolve_path(ffmpeg_env, exe_name) if ffmpeg_env else ""

    probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    if ffprobe_env:
        ffprobe_path = _resolve_path(ffprobe_env, probe_name)
    elif ffmpeg_path:
        sibling = os.path.join(os.path.dirname(ffmpeg_path), probe_name)
        if os.path.isfile(sibling):
            ffprobe_path = sibling
        else:
            raise DependencyError(
                "YT_DIARIZER_FFMPEG_PATH was provided but ffprobe was not found next to it. "
                "Set YT_DIARIZER_FFPROBE_PATH or point to a directory containing both binaries."
            )
    else:
        raise DependencyError(
            "YT_DIARIZER_FFPROBE_PATH must be set when YT_DIARIZER_FFMPEG_PATH is absent."
        )

    debug(f"Using ffmpeg from YT_DIARIZER_FFMPEG_PATH: {ffmpeg_path}")
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


def _find_ffmpeg_binaries(root: Path, logger) -> Tuple[Optional[Path], Optional[Path]]:
    """Recursively search for ffmpeg/ffprobe binaries under root (case-insensitive)."""

    ffmpeg_path: Optional[Path] = None
    ffprobe_path: Optional[Path] = None

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()
        if name == "ffmpeg" and ffmpeg_path is None:
            ffmpeg_path = path
        elif name == "ffprobe" and ffprobe_path is None:
            ffprobe_path = path

        if ffmpeg_path and ffprobe_path:
            break

    logger.info(
        "[yt-diarizer] ffmpeg search results: ffmpeg=%s, ffprobe=%s",
        ffmpeg_path,
        ffprobe_path,
    )
    return ffmpeg_path, ffprobe_path


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _prepare_macos_ffmpeg(
    extract_dir: Path, workspace_dir: Path, logger
) -> Dict[str, Optional[str]]:
    ffmpeg_src, ffprobe_src = _find_ffmpeg_binaries(extract_dir, logger)
    if ffmpeg_src is None:
        raise RuntimeError(
            f"Downloaded ffmpeg archive in {extract_dir} does not contain an 'ffmpeg' binary"
        )

    ffmpeg_dir = workspace_dir / "ffmpeg_macos" / "bin"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_dst = ffmpeg_dir / "ffmpeg"
    shutil.copy2(ffmpeg_src, ffmpeg_dst)
    _make_executable(ffmpeg_dst)

    ffprobe_dst = ffmpeg_dir / "ffprobe"
    if ffprobe_src is not None:
        shutil.copy2(ffprobe_src, ffprobe_dst)
        _make_executable(ffprobe_dst)
    else:
        logger.warning(
            "[yt-diarizer] ffprobe not found in downloaded archive; yt-dlp may still fail "
            "and you may need a different build"
        )

    logger.info("[yt-diarizer] Using downloaded ffmpeg at %s", ffmpeg_dst)
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

    try:
        env_paths = _ffmpeg_from_env()
        if env_paths:
            bin_dir = os.path.dirname(env_paths["ffmpeg"])
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return env_paths
    except DependencyError as exc:
        _log_error(str(exc))

    path_paths = _ffmpeg_from_path()
    if path_paths:
        return path_paths

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
        "Using downloaded ffmpeg at %s (ffprobe=%s)",
        download_paths["ffmpeg"],
        download_paths.get("ffprobe"),
    )
    return download_paths


def run_pipeline_inside_venv(script_dir: str, work_dir: str) -> None:
    """Inner stage: actual diarization pipeline inside the venv."""
    if not work_dir:
        raise PipelineError("Internal error: workspace directory not provided.")

    debug(f"Workspace inside venv: {work_dir}")

    _configure_cache_dirs(work_dir)

    hf_token = load_hf_token(script_dir)
    os.environ.setdefault("HF_TOKEN", hf_token)

    ffmpeg_paths = ensure_ffmpeg(work_dir)
    ffmpeg_path = ffmpeg_paths["ffmpeg"]

    deps = ensure_dependencies()
    yt_downloader = deps["yt_downloader"]
    whisperx_bin = deps["whisperx"]

    url = _resolve_youtube_url()
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
