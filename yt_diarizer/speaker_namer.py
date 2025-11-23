"""Interactive helper to rename diarized speakers in transcript outputs."""

import argparse
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .transcriber import format_timestamp

CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


SPEAKER_RE = re.compile(r"\b(SPEAKER_\d{2})\b")

PREVIEW_LIMIT = 20


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def transliterate_to_english(name: str) -> str:
    """Transliterate a provided name to ASCII-friendly uppercase with underscores."""
    transliterated: List[str] = []
    for char in name:
        lower = char.lower()
        if lower in CYRILLIC_TO_LATIN:
            transliterated.append(CYRILLIC_TO_LATIN[lower])
        else:
            transliterated.append(char)
    ascii_like = _strip_diacritics("".join(transliterated))
    sanitized = re.sub(r"[^\w\s-]", " ", ascii_like)
    collapsed = re.sub(r"[-\s]+", "_", sanitized).strip("_")
    return collapsed.upper()


def collect_speaker_lines(lines: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    speaker_lines: Dict[str, List[str]] = {}
    ordered: List[str] = []
    for line in lines:
        for match in SPEAKER_RE.findall(line):
            speaker_lines.setdefault(match, []).append(line)
            if match not in ordered:
                ordered.append(match)
    return speaker_lines, ordered


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_speaker_score(segment: Dict[str, Any], speaker: str) -> float:
    """Extract a diarization/ASR confidence score for the given speaker.

    Heuristics (in priority order):

    1. Segment-level speaker-specific scores:
       - "speaker_prob"
       - "speaker_probs"[speaker]
       - generic segment-level "score" / "confidence"
    2. Word-level scores for this speaker (or words without explicit speaker):
       - "speaker_prob"
       - "prob"
       - "probability"
       - "score"
       - "confidence"
    3. Segment-level ASR scores:
       - "avg_logprob"  -> exp(logprob), clamped to [0, 1]
       - "no_speech_prob" -> 1 - no_speech_prob
    4. Fallback: segment duration (end - start), if available.

    If nothing usable is found, return 0.0 to keep sorting deterministic.
    """

    # 1) Explicit segment-level probabilities for this speaker
    prob = _safe_float(segment.get("speaker_prob"))
    if prob is not None:
        return prob

    speaker_probs = segment.get("speaker_probs")
    if isinstance(speaker_probs, dict):
        prob = _safe_float(speaker_probs.get(speaker))
        if prob is not None:
            return prob

    # 1b) Generic segment-level confidence/score
    for key in ("score", "confidence"):
        prob = _safe_float(segment.get(key))
        if prob is not None:
            return prob

    # 2) Word-level scores (more common in whisperx / faster-whisper outputs)
    words = segment.get("words")
    if isinstance(words, list) and words:
        scores: List[float] = []
        for word in words:
            if not isinstance(word, dict):
                continue

            word_speaker = word.get("speaker")
            if word_speaker is not None and word_speaker != speaker:
                # If a word is explicitly tagged with a different speaker, skip it
                continue

            # Try a range of possible probability/score keys
            word_prob = None
            for key in (
                "speaker_prob",
                "prob",
                "probability",
                "score",
                "confidence",
            ):
                val = word.get(key)
                word_prob = _safe_float(val)
                if word_prob is not None:
                    break

            if word_prob is not None:
                scores.append(word_prob)

        if scores:
            return sum(scores) / len(scores)

    # 3) ASR-level scores (avg logprob, no_speech_prob)
    avg_logprob = _safe_float(segment.get("avg_logprob"))
    if avg_logprob is not None:
        # avg_logprob is in log-space; convert to a rough [0, 1] proxy
        # and clamp to keep things sane.
        try:
            prob_val = math.exp(avg_logprob)
        except (OverflowError, ValueError):
            prob_val = 0.0
        if prob_val < 0.0:
            prob_val = 0.0
        if prob_val > 1.0:
            prob_val = 1.0
        return prob_val

    no_speech_prob = _safe_float(segment.get("no_speech_prob"))
    if no_speech_prob is not None:
        # The lower the no_speech probability, the more confident we are it's real speech.
        # Clamp to [0, 1].
        value = 1.0 - no_speech_prob
        if value < 0.0:
            value = 0.0
        if value > 1.0:
            value = 1.0
        return value

    # 4) Fallback: duration-based score (longer segments tend to carry more info)
    start = _safe_float(segment.get("start"))
    end = _safe_float(segment.get("end"))
    if start is not None and end is not None:
        duration = end - start
        if duration > 0:
            return duration

    # Final fallback: no usable info
    return 0.0


def collect_scored_segments_by_speaker(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group segments by speaker, preserving timing/text and adding scores."""

    scored_segments: Dict[str, List[Dict[str, Any]]] = {}
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue

        speaker = seg.get("speaker")
        if not isinstance(speaker, str):
            continue

        start = _safe_float(seg.get("start"))
        end = _safe_float(seg.get("end"))
        text = (seg.get("text") or "").strip()
        score = extract_speaker_score(seg, speaker)

        scored_segments.setdefault(speaker, []).append(
            {
                "start": start,
                "end": end,
                "text": text,
                "score": score,
            }
        )

    return scored_segments


def build_preview_lines(
    speaker: str,
    text_lines: List[str],
    scored_segments: List[Dict[str, Any]],
    limit: int = PREVIEW_LIMIT,
) -> List[str]:
    """Return preview lines prioritizing highest-confidence JSON segments.

    Segments are first sorted by score (descending) to select the top-N samples
    for the speaker, then ordered chronologically for readability. If no scored
    segments are available, the function falls back to transcript lines.
    """

    if scored_segments:
        top_segments = sorted(
            scored_segments, key=lambda seg: seg.get("score", 0.0), reverse=True
        )[:limit]
        top_segments.sort(key=lambda seg: seg.get("start") or 0.0)

        previews: List[str] = []
        for seg in top_segments:
            start_ts = format_timestamp(seg.get("start"))
            end_ts = format_timestamp(seg.get("end"))
            score = seg.get("score") or 0.0
            text = seg.get("text") or ""
            previews.append(
                f"[{start_ts} --> {end_ts}] {speaker}: {text} (score={score:.3f})"
            )

        return previews

    return text_lines[:limit]


def prompt_for_name(speaker: str) -> str:
    while True:
        raw_name = input(f"\nEnter the real name for {speaker}: ").strip()
        if not raw_name:
            print("Name cannot be empty. Please try again.")
            continue
        normalized = transliterate_to_english(raw_name)
        print(f"Name will be saved as: {normalized}")
        while True:
            confirmation = input("Enter 'y' to confirm or 'e' to edit: ").strip().lower()
            if confirmation == "y":
                return normalized
            if confirmation == "e":
                break
            print("Please enter either 'y' or 'e'.")


def replace_speakers_in_text(text: str, mapping: Dict[str, str]) -> str:
    def replacer(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return SPEAKER_RE.sub(replacer, text)


def replace_speakers_in_json(data: Dict, mapping: Dict[str, str]) -> Dict:
    segments = data.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, dict):
                speaker = seg.get("speaker")
                if speaker in mapping:
                    seg["speaker"] = mapping[speaker]
    return data


def build_named_path(path: str) -> str:
    directory, filename = os.path.split(path)
    return os.path.join(directory, f"NAMED_{filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename SPEAKER_XX placeholders in transcript and JSON diarization outputs."
        )
    )
    parser.add_argument(
        "transcript",
        help="Path to the diarized .txt file containing SPEAKER_XX entries",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path to the associated transcription JSON file",
    )
    args = parser.parse_args()

    transcript_path = os.path.abspath(args.transcript)
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Unable to find text file: {transcript_path}")

    json_path = args.json_path
    if not json_path:
        candidate = os.path.splitext(transcript_path)[0] + ".json"
        if os.path.isfile(candidate):
            json_path = candidate

    with open(transcript_path, "r", encoding="utf-8") as f:
        text_content = f.read()
    lines = text_content.splitlines()

    scored_segments: Dict[str, List[Dict[str, Any]]] = {}
    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        scored_segments = collect_scored_segments_by_speaker(json_data)

    speaker_lines, speaker_order = collect_speaker_lines(lines)
    if not speaker_order:
        print("No SPEAKER_XX tags were found in the provided file.")
        return

    mapping: Dict[str, str] = {}
    for speaker in speaker_order:
        speaker_scored = scored_segments.get(speaker, [])
        preview = build_preview_lines(
            speaker,
            speaker_lines.get(speaker, []),
            speaker_scored,
        )

        # If we have any non-zero scores, treat this as a "top by score" preview
        has_non_zero_scores = any(
            isinstance(seg, dict) and (seg.get("score") or 0.0) != 0.0
            for seg in speaker_scored
        )
        if has_non_zero_scores:
            header = (
                f"\nExamples for {speaker} "
                f"(top {min(len(preview), PREVIEW_LIMIT)} by score):"
            )
        else:
            header = f"\nExamples for {speaker} (up to {PREVIEW_LIMIT} lines):"

        print(header)
        if not preview:
            print("No examples found for this speaker.")
        for example in preview:
            print(example)

        mapping[speaker] = prompt_for_name(speaker)

    print("\nAll speakers processed. Creating named files...")

    named_text_path = build_named_path(transcript_path)
    named_text_content = replace_speakers_in_text(text_content, mapping)
    with open(named_text_path, "w", encoding="utf-8") as f:
        f.write(named_text_content)
    print(f"Created file: {named_text_path}")

    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        updated = replace_speakers_in_json(data, mapping)
        named_json_path = build_named_path(json_path)
        with open(named_json_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        print(f"Created file: {named_json_path}")
    else:
        print("JSON file not found; skipping JSON copy.")

    print("Done.")


if __name__ == "__main__":
    main()
