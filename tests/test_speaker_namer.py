import json

import yt_diarizer.speaker_namer as sn


def test_transliterate_to_english_handles_russian_and_spacing():
    assert sn.transliterate_to_english("Анна-Мария Иванова") == "ANNA_MARIYA_IVANOVA"
    assert sn.transliterate_to_english("John Doe") == "JOHN_DOE"


def test_collect_speaker_lines_preserves_order_and_examples():
    lines = [
        "[00:00:01.000 --> 00:00:03.500] SPEAKER_01: Hello",
        "[00:00:04.000 --> 00:00:05.500] SPEAKER_02: Hi",
        "[00:00:06.000 --> 00:00:07.500] SPEAKER_01: Welcome",
    ]
    speaker_lines, order = sn.collect_speaker_lines(lines)
    assert order == ["SPEAKER_01", "SPEAKER_02"]
    assert speaker_lines["SPEAKER_01"] == [lines[0], lines[2]]
    assert speaker_lines["SPEAKER_02"] == [lines[1]]


def test_replace_speakers_in_text_swaps_all_instances():
    text = """
[00:00:01.000 --> 00:00:03.500] SPEAKER_01: Hello
[00:00:04.000 --> 00:00:05.500] SPEAKER_02: Hi
[00:00:06.000 --> 00:00:07.500] SPEAKER_01: Welcome
""".strip()
    mapping = {"SPEAKER_01": "HOST", "SPEAKER_02": "GUEST"}
    result = sn.replace_speakers_in_text(text, mapping)
    assert "SPEAKER_01" not in result
    assert "SPEAKER_02" not in result
    assert "HOST" in result
    assert "GUEST" in result


def test_replace_speakers_in_json_updates_segments_only():
    data = {
        "segments": [
            {"speaker": "SPEAKER_01", "text": "Hello"},
            {"speaker": "SPEAKER_02", "text": "Hi"},
        ],
        "other": "keep"
    }
    mapping = {"SPEAKER_01": "HOST", "SPEAKER_02": "GUEST"}
    updated = sn.replace_speakers_in_json(json.loads(json.dumps(data)), mapping)
    assert updated["segments"][0]["speaker"] == "HOST"
    assert updated["segments"][1]["speaker"] == "GUEST"
    assert updated["other"] == "keep"
