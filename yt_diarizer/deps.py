"""Dependency and download utilities."""

import os
import platform
import shutil
import sys
import zipfile
from typing import Dict, List, Tuple

from .exceptions import DependencyError
from .logging_utils import debug


def _macos_ffmpeg_download() -> Tuple[str, str]:
    """Return URL and archive name for macOS ffmpeg+ffprobe build."""
    # Use yt-dlp maintained static builds that include both ffmpeg and ffprobe.
    arch = platform.machine().lower()
    if "arm" in arch or arch == "aarch64":
        filename = "ffmpeg-master-latest-macos-arm64-static.zip"
    else:
        filename = "ffmpeg-master-latest-macos64-static.zip"
    # The yt-dlp FFmpeg builds expose the latest assets under the
    # `releases/latest/download/<asset>` path. The previous
    # `releases/download/latest/<asset>` form now returns 404s, so use the
    # supported location.
    base = "https://github.com/yt-dlp/FFmpeg-Builds/releases/latest/download"
    return f"{base}/{filename}", filename


def _download_and_extract_zip(url: str, archive_path: str, unpack_dir: str) -> None:
    import urllib.request
    from urllib.error import HTTPError, URLError

    debug(f"Downloading ffmpeg binary from {url} ...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except (HTTPError, URLError) as exc:
        raise DependencyError(
            "Failed to download ffmpeg/ffprobe static build; "
            "please check your network connection or provide ffmpeg manually."
        ) from exc

    debug(f"Extracting ffmpeg archive to {unpack_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(unpack_dir)


def _find_binary(unpack_dir: str, name: str) -> str:
    for root, _, files in os.walk(unpack_dir):
        if name in files:
            return os.path.join(root, name)
    raise DependencyError(
        f"Downloaded ffmpeg archive but could not find '{name}' binary inside."
    )


def download_ffmpeg_if_missing(work_dir: str) -> str:
    """
    Ensure ffmpeg + ffprobe are in PATH.
    If missing, download a macOS static build into work_dir and prepend its bin dir to PATH.

    Returns the ffmpeg binary path (either existing or downloaded).
    """
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        debug("ffmpeg/ffprobe already available in PATH.")
        ffmpeg_existing = shutil.which("ffmpeg")
        assert ffmpeg_existing is not None
        return ffmpeg_existing

    if sys.platform != "darwin":
        raise DependencyError(
            "ffmpeg/ffprobe not found and auto-download is only implemented on macOS."
        )

    ffmpeg_zip_url, filename = _macos_ffmpeg_download()
    archive_path = os.path.join(work_dir, filename)
    unpack_dir = os.path.join(work_dir, "ffmpeg_unpacked")

    _download_and_extract_zip(ffmpeg_zip_url, archive_path, unpack_dir)

    ffmpeg_path = _find_binary(unpack_dir, "ffmpeg")
    ffprobe_path = _find_binary(unpack_dir, "ffprobe")

    for binary_path in (ffmpeg_path, ffprobe_path):
        try:
            os.chmod(binary_path, 0o755)
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
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise DependencyError(
            "ffmpeg/ffprobe executables not found in PATH even after attempted download."
        )
    deps["ffmpeg"] = ffmpeg_path
    deps["ffprobe"] = ffprobe_path
    deps["yt_downloader"] = find_executable(["yt-dlp", "yt_dlp"])
    deps["whisperx"] = find_executable(["whisperx"])
    return deps
