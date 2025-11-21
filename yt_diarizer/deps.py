"""Dependency and download utilities."""

import os
import shutil
import sys
import zipfile
from typing import Dict, List

from .exceptions import DependencyError
from .logging_utils import debug


def download_ffmpeg_if_missing(work_dir: str) -> str:
    """
    Ensure ffmpeg is in PATH.
    If missing, download a macOS universal binary into work_dir and prepend its bin dir to PATH.

    Returns the ffmpeg binary path (either existing or downloaded).
    """
    import urllib.request

    if shutil.which("ffmpeg"):
        debug("ffmpeg already available in PATH.")
        ffmpeg_existing = shutil.which("ffmpeg")
        assert ffmpeg_existing is not None
        return ffmpeg_existing

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
    return ffmpeg_path


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
