"""Interactive helper to rename diarized speakers in transcript outputs."""

import argparse
import json
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
    """Extract a diarization confidence score for the given speaker.

    The function attempts to read segment-level probabilities first, then falls
    back to averaging word-level speaker probabilities. If no usable numeric
    value is available, zero is returned to ensure deterministic sorting.
    """

    prob = _safe_float(segment.get("speaker_prob"))
    if prob is not None:
        return prob

    speaker_probs = segment.get("speaker_probs")
    if isinstance(speaker_probs, dict):
        prob = _safe_float(speaker_probs.get(speaker))
        if prob is not None:
            return prob

    words = segment.get("words")
    if isinstance(words, list):
        scores: List[float] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            word_speaker = word.get("speaker")
            if word_speaker is not None and word_speaker != speaker:
                continue
            word_prob = _safe_float(
                word.get("speaker_prob")
                if "speaker_prob" in word
                else word.get("prob")
            )
            if word_prob is not None:
                scores.append(word_prob)
        if scores:
            return sum(scores) / len(scores)

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
        raw_name = input(f"\nВведите реальное имя для {speaker}: ").strip()
        if not raw_name:
            print("Имя не может быть пустым. Попробуйте снова.")
            continue
        normalized = transliterate_to_english(raw_name)
        print(f"Имя будет сохранено как: {normalized}")
        while True:
            confirmation = input("Введите 'y' для подтверждения или 'e' для правки: ").strip().lower()
            if confirmation == "y":
                return normalized
            if confirmation == "e":
                break
            print("Пожалуйста, введите 'y' или 'e'.")


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
            "Переименование участников SPEAKER_XX в файлах транскрипции и JSON выводе."
        )
    )
    parser.add_argument(
        "transcript",
        help="Путь к diarized .txt файлу, содержащему SPEAKER_XX записи",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Необязательный путь к связанному JSON файлу транскрипции",
    )
    args = parser.parse_args()

    transcript_path = os.path.abspath(args.transcript)
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Не удалось найти текстовый файл: {transcript_path}")

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
        print("SPEAKER_XX метки не найдены в указанном файле.")
        return

    mapping: Dict[str, str] = {}
    for speaker in speaker_order:
        preview = build_preview_lines(
            speaker,
            speaker_lines.get(speaker, []),
            scored_segments.get(speaker, []),
        )
        print(f"\nПримеры для {speaker} (до {PREVIEW_LIMIT} строк):")
        if not preview:
            print("Примеры не найдены для этого участника.")
        for example in preview:
            print(example)
        mapping[speaker] = prompt_for_name(speaker)

    print("\nВсе участники успешно обработаны. Формируем именованные файлы...")

    named_text_path = build_named_path(transcript_path)
    named_text_content = replace_speakers_in_text(text_content, mapping)
    with open(named_text_path, "w", encoding="utf-8") as f:
        f.write(named_text_content)
    print(f"Создан файл: {named_text_path}")

    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        updated = replace_speakers_in_json(data, mapping)
        named_json_path = build_named_path(json_path)
        with open(named_json_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
        print(f"Создан файл: {named_json_path}")
    else:
        print("JSON файл не найден, пропускаем копию JSON.")

    print("Готово.")


if __name__ == "__main__":
    main()
