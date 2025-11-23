"""WhisperX transcription helpers and formatting utilities."""

import glob
import json
import os
from typing import Any, Dict, List, Optional

from .exceptions import PipelineError
from .logging_utils import debug
from .process import run_logged_subprocess


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


def smooth_speaker_labels(
    segments: List[Dict[str, Any]],
    max_short: float = 0.7,
) -> None:
    """
    In-place smoothing heuristic for diarization results.

    If a very short segment has a different speaker than its immediate
    neighbours, and both neighbours share the same speaker label, we
    relabel the short segment to match them.

    This helps remove single-word "speaker ping-pong" artefacts.
    """
    if len(segments) < 3:
        return

    # Ensure chronological order just in case
    segments.sort(key=lambda s: float(s.get("start", 0.0)))

    for i in range(1, len(segments) - 1):
        cur = segments[i]
        prev_seg = segments[i - 1]
        next_seg = segments[i + 1]

        try:
            start = float(cur.get("start", 0.0))
            end = float(cur.get("end", start))
        except (TypeError, ValueError):
            continue

        duration = max(0.0, end - start)
        if duration > max_short:
            continue

        cur_speaker = cur.get("speaker")
        prev_speaker = prev_seg.get("speaker")
        next_speaker = next_seg.get("speaker")

        if not prev_speaker or not next_speaker:
            continue

        if (
            prev_speaker == next_speaker
            and cur_speaker is not None
            and cur_speaker != prev_speaker
        ):
            debug(
                f"Smoothing speaker label at index {i}: "
                f"{cur_speaker!r} -> {prev_speaker!r} "
                f"(duration={duration:.3f}s)."
            )
            cur["speaker"] = prev_speaker


def build_diarized_transcript_lines_from_data(data: Dict[str, Any]) -> List[str]:
    """
    Convert WhisperX JSON-like transcription result into diarized transcript lines.

    Each line looks like:
      "[00:00:01.000 --> 00:00:03.500] SPEAKER_00: Hello world"
    """
    segments = data.get("segments") or []
    if not isinstance(segments, list):
        segments = []

    smooth_speaker_labels(segments)

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
    language = os.environ.get("YT_DIARIZER_LANGUAGE") or None
    initial_prompt = os.environ.get("YT_DIARIZER_INITIAL_PROMPT") or None
    min_speakers = os.environ.get("YT_DIARIZER_MIN_SPEAKERS") or None
    max_speakers = os.environ.get("YT_DIARIZER_MAX_SPEAKERS") or None

    if language:
        debug(f"WhisperX language hint: {language}")
    if min_speakers or max_speakers:
        debug(
            f"WhisperX diarization speakers bounds: "
            f"min={min_speakers or 'auto'}, max={max_speakers or 'auto'}"
        )
    if initial_prompt:
        debug("WhisperX will use an initial prompt for decoding.")

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
        "--output_format",
        "json",
        "--output_dir",
        work_dir,
        "--verbose",
        "True",
        "--print_progress",
        "True",
    ]

    if language:
        cmd.extend(["--language", language])

    if min_speakers:
        cmd.extend(["--min_speakers", str(min_speakers)])

    if max_speakers:
        cmd.extend(["--max_speakers", str(max_speakers)])

    if initial_prompt:
        cmd.extend(["--initial_prompt", initial_prompt])

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
