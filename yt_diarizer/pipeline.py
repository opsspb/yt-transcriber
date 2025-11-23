"""High-level orchestration for the diarization pipeline."""

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

from .constants import (
    ENV_MPS_CONVERT_VAR,
    ENV_STAGE_VAR,
    ENV_URL_VAR,
    ENV_WORKDIR_VAR,
    TOKEN_FILENAME,
)
from .deps import ensure_dependencies, find_executable
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


def _build_output_base_name_from_url(url: str) -> str:
    """Derive a filesystem-friendly base name from the provided YouTube URL."""

    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")

    parts: List[str] = []

    if parsed.netloc:
        parts.append(parsed.netloc)

    query = parse_qs(parsed.query)
    video_id = None
    if query.get("v") and query["v"][0]:
        video_id = query["v"][0]
    elif parsed.path:
        path_parts = [segment for segment in parsed.path.split("/") if segment]
        if path_parts:
            video_id = path_parts[-1]

    if video_id:
        parts.append(video_id)

    if parsed.fragment:
        parts.append(parsed.fragment)

    if not parts:
        parts.append("video")

    raw_name = "_".join(parts)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("._-")
    if not safe_name:
        safe_name = "video"

    return f"diarized_transcript_{safe_name}"


def save_final_outputs(
    transcript_lines: List[str],
    json_path: str,
    script_dir: str,
    url: str,
) -> Dict[str, str]:
    """
    Save the diarized transcript (.txt) and raw WhisperX JSON into script_dir.

    Returns:
      {"txt": <txt_path>, "json": <json_path>}
    """
    base_name = _build_output_base_name_from_url(url)

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


def install_python_dependencies(venv_python: str, mps_convert: bool = False) -> None:
    """
    Install required Python packages into the venv.

    When *mps_convert* is True we install a Whisper-only stack suitable for
    Apple Silicon (MPS) transcription without diarization. Otherwise we install
    the default WhisperX stack.
    """
    stack_label = "Whisper (MPS)" if mps_convert else "WhisperX"
    debug(f"Installing Python dependencies ({stack_label} stack) inside venv ...")

    # On macOS, PyAV may require pkg-config to be present; we keep this preflight.
    ensure_pkg_config_available()

    pinned_versions = {
        "numpy": "1.26.4",  # WhisperX deps expect numpy<2
        "torch": "2.3.1",
        "torchaudio": "2.3.1",
        "whisperx": "3.2.0",  # upgraded from 3.1.1
        "whisper": "20240930",
        "yt-dlp": "2024.11.18",
    }

    venv_dir = os.path.dirname(os.path.dirname(venv_python))
    pip_env = os.environ.copy()
    pip_env.setdefault("VIRTUAL_ENV", venv_dir)
    pip_env["PYTHONNOUSERSITE"] = "1"
    pip_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip_env["PIP_NO_CACHE_DIR"] = "1"

    def _run(cmd: List[str], description: str) -> None:
        rc, log_lines = run_logged_subprocess(cmd, description, env=pip_env)
        if rc != 0:
            last_snippet = "\n".join(log_lines[-20:])
            raise DependencyError(
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

    # Pin numpy below 2.x until WhisperX updates its requirements.
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            f"numpy=={pinned_versions['numpy']}",
        ],
        "install numpy below 2.x for WhisperX dependencies",
    )

    torch_cmd = [
        venv_python,
        "-m",
        "pip",
        "install",
        f"torch=={pinned_versions['torch']}",
        f"torchaudio=={pinned_versions['torchaudio']}",
    ]
    if not mps_convert:
        torch_cmd.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])

    _run(torch_cmd, "install PyTorch wheels")

    if mps_convert:
        _run(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                f"openai-whisper=={pinned_versions['whisper']}",
                f"yt-dlp=={pinned_versions['yt-dlp']}",
            ],
            "install Whisper (MPS) transcription dependencies",
        )
    else:
        _run(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                f"whisperx=={pinned_versions['whisperx']}",
                f"yt-dlp=={pinned_versions['yt-dlp']}",
                "--constraint",
                "https://raw.githubusercontent.com/m-bain/whisperX/v3.2.0/requirements.txt",
            ],
            "install WhisperX, yt-dlp and supporting dependencies",
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
    return {"ffmpeg": str(ffmpeg_path), "ffprobe": str(ffprobe_path)}


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


def _prepare_macos_ffmpeg(unpack_dir: Path, work_dir: Path, debug: bool = False) -> tuple[Path, Path]:
    """
    Prepare ffmpeg/ffprobe for macOS from the ColorsWind archive.

    We keep everything inside work_dir/ffmpeg_macos (no system-wide install) and
    copy all .dylib files next to the binaries, then fix their load paths so they
    no longer point to the original build machine path.
    """
    ffmpeg_macos_dir = work_dir / "ffmpeg_macos"
    bin_dir = ffmpeg_macos_dir / "bin"
    lib_dir = ffmpeg_macos_dir / "lib"

    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_candidates = _find_binary(unpack_dir, "ffmpeg")
    ffprobe_candidates = _find_binary(unpack_dir, "ffprobe")

    if not ffmpeg_candidates or not ffprobe_candidates:
        raise RuntimeError(
            "FFmpeg or ffprobe not found in the unpacked macOS archive. "
            "Please install FFmpeg manually and set FFMPEG_PATH and FFPROBE_PATH."
        )

    ffmpeg_path = bin_dir / "ffmpeg"
    ffprobe_path = bin_dir / "ffprobe"

    shutil.copy2(ffmpeg_candidates[0], ffmpeg_path)
    shutil.copy2(ffprobe_candidates[0], ffprobe_path)

    # Copy all shared libraries from the archive into our local lib directory.
    # The prebuilt ColorsWind binaries are linked against the build machine path
    # (/Users/runner/work/FFmpeg-macOS/FFmpeg-macOS/...), so we must keep the
    # .dylibs next to the binaries and rewrite their load paths.
    for dylib in Path(unpack_dir).rglob("*.dylib"):
        dest = lib_dir / dylib.name
        if not dest.exists():
            shutil.copy2(dylib, dest)

    ffmpeg_path.chmod(ffmpeg_path.stat().st_mode | stat.S_IEXEC)
    ffprobe_path.chmod(ffprobe_path.stat().st_mode | stat.S_IEXEC)

    # Fix dyld load commands so that ffmpeg/ffprobe and the libs use the local
    # ffmpeg_macos/lib directory instead of the original build path.
    _fix_macos_ffmpeg_install_names(ffmpeg_macos_dir, debug=debug)

    if debug:
        print(f"[yt-diarizer] ffmpeg copied to: {ffmpeg_path}")
        print(f"[yt-diarizer] ffprobe copied to: {ffprobe_path}")

    return ffmpeg_path, ffprobe_path


def _fix_macos_ffmpeg_install_names(ffmpeg_macos_dir: Path, debug: bool = False) -> None:
    """
    Adjust shared library load paths for the locally copied ffmpeg/ffprobe
    binaries on macOS.

    This avoids runtime errors like:
      Library not loaded: /Users/runner/work/FFmpeg-macOS/FFmpeg-macOS/libavdevice.59.dylib
    by rewriting the load commands to use the local ffmpeg_macos/lib directory.
    """
    lib_dir = ffmpeg_macos_dir / "lib"
    bin_dir = ffmpeg_macos_dir / "bin"

    if not lib_dir.is_dir() or not bin_dir.is_dir():
        return

    # Map library file name -> absolute path inside ffmpeg_macos/lib
    libs = {p.name: p for p in lib_dir.glob("*.dylib")}
    if not libs:
        return

    try:
        import subprocess  # noqa: F401
    except Exception:
        # If subprocess/otool/install_name_tool are not available, just skip.
        return

    def _run(cmd: list[str]) -> None:
        if debug:
            print(f"[yt-diarizer] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:  # best-effort only
            if debug:
                print(f"[yt-diarizer] install_name_tool failed for {cmd}: {exc}")

    def _patch_binary(binary: Path, use_executable_path: bool) -> None:
        """Rewrite load commands for one Mach-O binary (ffmpeg, ffprobe or .dylib)."""
        if not binary.is_file():
            return

        try:
            output = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        except Exception as exc:
            if debug:
                print(f"[yt-diarizer] otool -L failed for {binary}: {exc}")
            return

        lines = output.splitlines()[1:]
        for line in lines:
            if not line.strip():
                continue

            dep = line.strip().split(" ", 1)[0]
            dep_name = os.path.basename(dep)

            # Only touch dependencies that correspond to the libs we just copied.
            if dep_name not in libs:
                continue

            target_lib = libs[dep_name]
            rel = os.path.relpath(target_lib, start=binary.parent)
            prefix = "@executable_path" if use_executable_path else "@loader_path"
            new_dep = f"{prefix}/{rel}"

            if new_dep == dep:
                continue

            _run(["install_name_tool", "-change", dep, new_dep, str(binary)])

    # 1) Point ffmpeg and ffprobe to ../lib relative to their bin directory
    for name in ("ffmpeg", "ffprobe"):
        _patch_binary(bin_dir / name, use_executable_path=True)

    # 2) Ensure the .dylib files reference each other via @loader_path
    for lib in libs.values():
        _patch_binary(lib, use_executable_path=False)


def _find_binary(unpack_dir: Union[str, Path], name: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(unpack_dir):
        if name in files:
            matches.append(Path(root) / name)
    return matches


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

    ffmpeg_path, ffprobe_path = _prepare_macos_ffmpeg(
        Path(unpack_dir), Path(work_dir), debug=bool(os.environ.get("YT_DIARIZER_DEBUG"))
    )

    return {"ffmpeg": str(ffmpeg_path), "ffprobe": str(ffprobe_path)}


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
    ffmpeg_candidates = _find_binary(unpack_dir, ffmpeg_name)
    ffprobe_candidates = _find_binary(unpack_dir, ffprobe_name)

    if not ffmpeg_candidates or not ffprobe_candidates:
        raise RuntimeError("Downloaded ffmpeg archive missing ffmpeg/ffprobe binary")

    ffmpeg_path = ffmpeg_candidates[0]
    ffprobe_path = ffprobe_candidates[0]

    for binary_path in (ffmpeg_path, ffprobe_path):
        try:
            os.chmod(binary_path, 0o755)
        except Exception:
            pass

    return {"ffmpeg": str(ffmpeg_path), "ffprobe": str(ffprobe_path)}


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

    mps_convert = os.environ.get(ENV_MPS_CONVERT_VAR) == "1"

    _configure_cache_dirs(work_dir)

    ffmpeg_paths = ensure_ffmpeg(work_dir)
    ffmpeg_location = ffmpeg_paths["location_dir"]

    url = _resolve_youtube_url()
    if mps_convert:
        _run_mps_transcription(script_dir, work_dir, url, ffmpeg_location)
        return

    hf_token = load_hf_token(script_dir)
    os.environ.setdefault("HF_TOKEN", hf_token)

    deps = ensure_dependencies()
    yt_downloader = deps["yt_downloader"]
    whisperx_bin = deps["whisperx"]

    audio_path = download_best_audio(
        yt_downloader, url, work_dir, script_dir, ffmpeg_location
    )
    json_result_path = run_whisperx_cli(
        whisperx_bin, audio_path, hf_token, work_dir
    )

    transcript_lines = build_diarized_transcript_from_json(json_result_path)
    outputs = save_final_outputs(transcript_lines, json_result_path, script_dir, url)

    log_line("")
    log_line("=== Done ===")
    log_line(f"Diarized transcript (TXT): {outputs['txt']}")
    log_line(f"Raw WhisperX output (JSON): {outputs['json']}")


def _run_mps_transcription(
    script_dir: str, work_dir: str, url: str, ffmpeg_location: Optional[str]
) -> None:
    """Run the Whisper-only transcription path optimized for Apple Silicon."""

    from .mps_convert import transcribe_audio_with_mps_whisper

    yt_downloader = find_executable(["yt-dlp", "yt_dlp"])
    audio_path = download_best_audio(
        yt_downloader, url, work_dir, script_dir, ffmpeg_location
    )
    json_result_path, transcript_lines = transcribe_audio_with_mps_whisper(
        audio_path, work_dir
    )

    outputs = save_final_outputs(transcript_lines, json_result_path, script_dir, url)

    log_line("")
    log_line("=== Done (Whisper MPS) ===")
    log_line(f"Transcript (TXT): {outputs['txt']}")
    log_line(f"Raw Whisper output (JSON): {outputs['json']}")


def setup_and_run_in_venv(
    script_dir: str, work_dir: str, entrypoint_path: str, mps_convert: bool = False
) -> int:
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

    install_python_dependencies(venv_python, mps_convert=mps_convert)

    env = os.environ.copy()
    env[ENV_STAGE_VAR] = "inner"
    env[ENV_WORKDIR_VAR] = work_dir
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    debug("Re-running script inside venv...")
    completed = subprocess.run([venv_python, os.path.abspath(entrypoint_path)], env=env)
    return completed.returncode
