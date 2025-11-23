"""Whisper-only transcription path optimized for Apple Silicon (MPS)."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
import whisper

from .exceptions import PipelineError
from .logging_utils import debug
from .transcriber import format_timestamp


def _ensure_mps_available() -> None:
    """Validate that the MPS backend is accessible before running Whisper."""

    if not torch.backends.mps.is_available():
        raise PipelineError(
            "MPS backend is not available. Ensure you are running on Apple Silicon "
            "with a recent PyTorch build that includes MPS support."
        )


def _build_transcript_lines(result: dict) -> List[str]:
    lines: List[str] = []
    for segment in result.get("segments", []):
        start_ts = format_timestamp(segment.get("start"))
        end_ts = format_timestamp(segment.get("end"))
        text = (segment.get("text") or "").strip()
        lines.append(f"[{start_ts} --> {end_ts}] {text}")
    return lines


def transcribe_audio_with_mps_whisper(audio_path: str, work_dir: str) -> Tuple[str, List[str]]:
    """Run Whisper transcription on MPS and persist the raw JSON output."""

    if not os.path.isfile(audio_path):
        raise PipelineError(f"Audio path not found: {audio_path}")

    _ensure_mps_available()

    device = "mps"
    debug("Running Whisper transcription with large-v3 model on MPS (no diarization)...")

    def _run_whisper(target_device: str) -> dict:
        model = whisper.load_model("large-v3", device=target_device)
        return model.transcribe(audio_path, verbose=True)

    try:
        result = _run_whisper(device)
    except (NotImplementedError, RuntimeError) as exc:
        debug(f"Whisper on MPS failed: {exc!r}")
        unsupported_runtime = isinstance(exc, RuntimeError) and "unsupported" in str(exc).lower()

        if isinstance(exc, NotImplementedError) or unsupported_runtime:
            raise PipelineError(
                "Whisper failed to execute on MPS due to an unsupported operation. "
                "Update PyTorch/Whisper to a build with full MPS support or rerun "
                "with a compatible GPU."
            ) from exc
        raise

    json_path = os.path.join(work_dir, f"{Path(audio_path).stem}_whisper.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    transcript_lines = _build_transcript_lines(result)
    debug(f"Whisper MPS JSON output: {json_path}")

    return json_path, transcript_lines
