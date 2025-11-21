"""Dependency and download utilities."""

import json
import os
import platform
import shutil
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

from .exceptions import DependencyError
from .logging_utils import debug


def _macos_ffmpeg_download() -> Tuple[List[str], str]:
    """Return candidate URLs and archive name for macOS ffmpeg+ffprobe build."""
    # Use yt-dlp maintained static builds that include both ffmpeg and ffprobe.
    arch = platform.machine().lower()
    if "arm" in arch or arch == "aarch64":
        filename = "ffmpeg-master-latest-macos-arm64-static.zip"
    else:
        filename = "ffmpeg-master-latest-macos64-static.zip"

    # yt-dlp FFmpeg builds expose the latest assets under
    # `releases/latest/download/<asset>`, but some environments still resolve
    # `releases/download/latest/<asset>` successfully. Try both to reduce
    # spurious download failures.
    api_url = "https://api.github.com/repos/yt-dlp/FFmpeg-Builds/releases/latest"
    api_urls: List[str] = []

    # Try resolving the latest asset name dynamically via the GitHub API to
    # avoid breakages when the upstream project rotates filenames.
    try:
        import urllib.request

        with urllib.request.urlopen(api_url, timeout=10) as resp:  # type: ignore[attr-defined]
            data = json.load(resp)
        assets = data.get("assets") or []
        for asset in assets:
            name = asset.get("name", "")
            # The upstream repository ships macOS static builds as ZIPs.
            if not name.endswith(".zip"):
                continue
            if "macos" not in name or "static" not in name:
                continue
            if "arm64" in name and "arm" not in arch and arch != "aarch64":
                continue
            if "arm64" not in name and ("arm" in arch or arch == "aarch64"):
                continue
            api_urls.append(asset.get("browser_download_url", ""))
            filename = name
            break
    except Exception as exc:  # pragma: no cover - best effort network call
        debug(f"Failed to resolve latest ffmpeg asset from GitHub API: {exc}")

    bases = [
        "https://github.com/yt-dlp/FFmpeg-Builds/releases/latest/download",
        "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest",
    ]

    fallback_urls = [f"{base}/{filename}" for base in bases]
    return [url for url in api_urls + fallback_urls if url], filename


def _download_and_extract_zip(urls: List[str], archive_path: str, unpack_dir: str) -> None:
    import urllib.request
    from urllib.error import HTTPError, URLError

    last_error: Optional[Exception] = None
    for url in urls:
        debug(f"Downloading ffmpeg binary from {url} ...")
        try:
            urllib.request.urlretrieve(url, archive_path)
            break
        except (HTTPError, URLError) as exc:
            last_error = exc
            debug(f"ffmpeg download failed from {url}: {exc}")
    else:
        raise DependencyError(
            "Failed to download ffmpeg/ffprobe static build after trying multiple URLs; "
            "please check your network connection or provide ffmpeg manually."
        ) from last_error

    debug(f"Extracting ffmpeg archive to {unpack_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(unpack_dir)


def _resolve_manual_ffmpeg() -> Tuple[str, str]:
    """Return ffmpeg/ffprobe paths from YT_DIARIZER_FFMPEG/FFPROBE if set."""

    def _validate(path: str, name: str) -> str:
        if not path:
            raise DependencyError(
                f"Environment override for {name} was provided but is empty."
            )
        if not os.path.isfile(path):
            raise DependencyError(
                f"Environment override for {name} points to a missing file: {path}"
            )
        return path

    ffmpeg_env = os.environ.get("YT_DIARIZER_FFMPEG")
    ffprobe_env = os.environ.get("YT_DIARIZER_FFPROBE")

    if not ffmpeg_env and not ffprobe_env:
        return "", ""

    ffmpeg_path = _validate(ffmpeg_env, "YT_DIARIZER_FFMPEG") if ffmpeg_env else ""
    if ffprobe_env:
        ffprobe_path = _validate(ffprobe_env, "YT_DIARIZER_FFPROBE")
    else:
        # Try locating ffprobe next to ffmpeg if only ffmpeg was provided.
        if not ffmpeg_path:
            raise DependencyError(
                "YT_DIARIZER_FFPROBE must be set when YT_DIARIZER_FFMPEG is not provided."
            )
        candidate = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if not os.path.isfile(candidate):
            raise DependencyError(
                "YT_DIARIZER_FFMPEG was provided but ffprobe was not found next to it. "
                "Set YT_DIARIZER_FFPROBE to the ffprobe binary."
            )
        ffprobe_path = candidate

    return ffmpeg_path, ffprobe_path


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
    manual_ffmpeg, manual_ffprobe = _resolve_manual_ffmpeg()
    if manual_ffmpeg and manual_ffprobe:
        bin_dir = os.path.dirname(manual_ffmpeg)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        debug(
            "Using ffmpeg/ffprobe from environment overrides "
            "YT_DIARIZER_FFMPEG/YT_DIARIZER_FFPROBE."
        )
        return manual_ffmpeg

    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        debug("ffmpeg/ffprobe already available in PATH.")
        ffmpeg_existing = shutil.which("ffmpeg")
        assert ffmpeg_existing is not None
        return ffmpeg_existing

    if sys.platform != "darwin":
        raise DependencyError(
            "ffmpeg/ffprobe not found and auto-download is only implemented on macOS."
        )

    ffmpeg_zip_urls, filename = _macos_ffmpeg_download()
    archive_path = os.path.join(work_dir, filename)
    unpack_dir = os.path.join(work_dir, "ffmpeg_unpacked")

    _download_and_extract_zip(ffmpeg_zip_urls, archive_path, unpack_dir)

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
