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


def test_collect_scored_segments_prioritizes_confident_samples():
    data = {
        "segments": [
            {
                "speaker": "SPEAKER_01",
                "start": 5,
                "end": 6,
                "text": "High confidence",
                "speaker_probs": {"SPEAKER_01": 0.9},
            },
            {
                "speaker": "SPEAKER_01",
                "start": 10,
                "end": 12,
                "text": "Low confidence",
                "speaker_prob": 0.3,
            },
            {
                "speaker": "SPEAKER_02",
                "start": 1,
                "end": 2,
                "text": "Other speaker",
                "speaker_prob": 0.95,
            },
        ]
    }

    scored = sn.collect_scored_segments_by_speaker(data)
    assert list(scored.keys()) == ["SPEAKER_01", "SPEAKER_02"]
    assert len(scored["SPEAKER_01"]) == 2

    previews = sn.build_preview_lines("SPEAKER_01", [], scored["SPEAKER_01"], limit=1)
    assert len(previews) == 1
    assert previews[0].startswith("[00:00:05.000")
    assert "High confidence" in previews[0]


def test_collect_scored_segments_uses_word_level_scores_when_segment_scores_missing():
    data = {
        "segments": [
            {
                "speaker": "SPEAKER_01",
                "start": 1.0,
                "end": 2.0,
                "text": "Low score sample",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.5, "score": 0.2},
                    {"word": "world", "start": 1.5, "end": 2.0, "score": 0.3},
                ],
            },
            {
                "speaker": "SPEAKER_01",
                "start": 3.0,
                "end": 4.0,
                "text": "High score sample",
                "words": [
                    {"word": "good", "start": 3.0, "end": 3.5, "score": 0.8},
                    {"word": "day", "start": 3.5, "end": 4.0, "score": 0.9},
                ],
            },
        ]
    }

    scored = sn.collect_scored_segments_by_speaker(data)
    assert list(scored.keys()) == ["SPEAKER_01"]
    scores = [seg["score"] for seg in scored["SPEAKER_01"]]

    # Second segment should have a higher average word-level score
    assert scores[1] > scores[0]

    # build_preview_lines should pick the high-score segment first
    previews = sn.build_preview_lines("SPEAKER_01", [], scored["SPEAKER_01"], limit=1)
    assert len(previews) == 1
    assert "High score sample" in previews[0]


def test_build_preview_lines_falls_back_to_transcript_lines():
    previews = sn.build_preview_lines(
        "SPEAKER_03", ["line1", "line2", "line3"], [], limit=2
    )
    assert previews == ["line1", "line2"]
