"""Audio download helpers using yt-dlp."""

import glob
import os
from typing import List, Optional

from .exceptions import PipelineError
from .logging_utils import debug
from .process import run_logged_subprocess


def build_yt_dlp_command_variants(
    downloader_bin: str,
    url: str,
    work_dir: str,
    script_dir: str,
    ffmpeg_path: Optional[str] = None,
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
        "--user-agent",
        user_agent,
        "--extractor-args",
        "youtube:player_client=web",
        "-o",
        output_template,
        url,
    ]

    if ffmpeg_path:
        base_cmd.extend(["--ffmpeg-location", os.path.dirname(ffmpeg_path)])

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
    ffmpeg_path: Optional[str],
) -> str:
    """
    Use yt-dlp to grab best audio and convert to WAV via ffmpeg.

    Tries multiple strategies and raises a detailed error if all fail.
    """
    debug("Starting audio download via yt-dlp...")
    commands = build_yt_dlp_command_variants(
        downloader_bin, url, work_dir, script_dir, ffmpeg_path
    )
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
