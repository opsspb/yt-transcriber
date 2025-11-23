"""Interactive helper to rename diarized speakers in transcript outputs."""

import argparse
import json
import os
import re
import unicodedata
from typing import Dict, List, Tuple

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

    speaker_lines, speaker_order = collect_speaker_lines(lines)
    if not speaker_order:
        print("SPEAKER_XX метки не найдены в указанном файле.")
        return

    mapping: Dict[str, str] = {}
    for speaker in speaker_order:
        preview = speaker_lines.get(speaker, [])[:10]
        print(f"\nПримеры для {speaker} (до 10 строк):")
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
