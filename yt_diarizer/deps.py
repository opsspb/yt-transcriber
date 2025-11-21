"""Dependency and download utilities."""

import json
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from typing import Dict, List, Optional, Tuple

from .exceptions import DependencyError
from .logging_utils import debug


def _macos_ffmpeg_download() -> Tuple[List[str], str]:
    """Return candidate URLs and archive name for macOS ffmpeg+ffprobe build."""
    # Use yt-dlp maintained static builds that include both ffmpeg and ffprobe.
    arch = platform.machine().lower()
    is_arm = "arm" in arch or arch == "aarch64"

    # Upstream has recently renamed macOS ARM builds to "apple-silicon"; keep both
    # historical and current patterns as fallbacks.
    base_names = (
        ["ffmpeg-master-latest-macos-arm64", "ffmpeg-master-latest-macos-apple-silicon"]
        if is_arm
        else ["ffmpeg-master-latest-macos64", "ffmpeg-master-latest-macos-intel"]
    )

    # Newer releases rotate between "static", "gpl" and "lgpl" flavors and may use
    # either ZIP or tar.xz archives. Try a handful of plausible names to remain
    # resilient to upstream filename changes.
    suffixes = [
        "-static.zip",
        "-gpl.zip",
        "-lgpl.zip",
        "-gpl.tar.xz",
        "-lgpl.tar.xz",
    ]
    candidate_names = [f"{base}{suffix}" for base in base_names for suffix in suffixes]
    filename = candidate_names[0]

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

        def _is_arm_build(name: str) -> bool:
            return any(token in name for token in ("arm64", "aarch64", "apple-silicon"))

        for asset in assets:
            name = asset.get("name", "")
            if not name.endswith((".zip", ".tar.xz")):
                continue
            if "macos" not in name:
                continue
            is_asset_arm = _is_arm_build(name)
            if is_asset_arm and not is_arm:
                continue
            if not is_asset_arm and is_arm:
                continue
            if not any(flavor in name for flavor in ("static", "gpl", "lgpl")):
                continue

            api_urls.append(asset.get("browser_download_url", ""))
            filename = name
            # Prefer the first matching asset exposed by the API; others remain
            # available as fallbacks below.
            break
    except Exception as exc:  # pragma: no cover - best effort network call
        debug(f"Failed to resolve latest ffmpeg asset from GitHub API: {exc}")

    bases = [
        "https://github.com/yt-dlp/FFmpeg-Builds/releases/latest/download",
        "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest",
    ]

    fallback_urls = [f"{base}/{name}" for base in bases for name in candidate_names]
    return [url for url in api_urls + fallback_urls if url], filename


def _download_and_extract_archive(urls: List[str], archive_path: str, unpack_dir: str) -> None:
    import urllib.request
    from urllib.error import HTTPError, URLError

    attempted_urls: List[str] = []
    last_error: Optional[Exception] = None
    for url in urls:
        debug(f"Downloading ffmpeg binary from {url} ...")
        attempted_urls.append(url)
        try:
            urllib.request.urlretrieve(url, archive_path)
            break
        except (HTTPError, URLError) as exc:
            last_error = exc
            debug(f"ffmpeg download failed from {url}: {exc}")
    else:
        details = (
            f"Failed to download ffmpeg/ffprobe static build after trying multiple URLs. "
            f"Tried: {attempted_urls}. "
            f"Last error: {last_error}"
        )
        raise DependencyError(
            details + "; please check your network connection or provide ffmpeg manually."
        ) from last_error

    debug(f"Extracting ffmpeg archive to {unpack_dir} ...")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(unpack_dir)
        elif archive_path.endswith((".tar.xz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(unpack_dir)
        else:
            raise DependencyError(
                f"Unsupported ffmpeg archive format for {archive_path}; provide ffmpeg manually."
            )
    except Exception as exc:
        raise DependencyError(
            f"Failed to extract ffmpeg archive '{archive_path}' to '{unpack_dir}': {exc}"
        ) from exc


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

    _download_and_extract_archive(ffmpeg_zip_urls, archive_path, unpack_dir)

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
