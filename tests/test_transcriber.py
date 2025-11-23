import yt_diarizer.transcriber as tr


def test_format_timestamp_basic():
    assert tr.format_timestamp(0.0) == "00:00:00.000"
    assert tr.format_timestamp(1.234) == "00:00:01.234"


def test_smooth_speaker_labels_merges_short_outlier_segment():
    segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "hello"},
        {"start": 2.0, "end": 2.4, "speaker": "SPEAKER_01", "text": "да"},
        {"start": 2.4, "end": 5.0, "speaker": "SPEAKER_00", "text": "world"},
    ]

    tr.smooth_speaker_labels(segments, max_short=0.7)

    assert all(seg["speaker"] == "SPEAKER_00" for seg in segments)


def test_build_diarized_transcript_lines_from_data_uses_smoothed_labels():
    data = {
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "hello"},
            {"start": 2.0, "end": 2.3, "speaker": "SPEAKER_01", "text": "да"},
            {"start": 2.3, "end": 4.0, "speaker": "SPEAKER_00", "text": "world"},
        ]
    }

    lines = tr.build_diarized_transcript_lines_from_data(data)
    assert all("SPEAKER_00" in line for line in lines)


def test_run_whisperx_cli_adds_language_and_speaker_hints(monkeypatch, tmp_path):
    invoked_cmds = []

    def _fake_run(cmd, description, env=None):
        invoked_cmds.append(cmd)
        return 0, ["ok"]

    monkeypatch.setenv("YT_DIARIZER_LANGUAGE", "ru")
    monkeypatch.setenv("YT_DIARIZER_MIN_SPEAKERS", "2")
    monkeypatch.setenv("YT_DIARIZER_MAX_SPEAKERS", "3")
    monkeypatch.setenv("YT_DIARIZER_INITIAL_PROMPT", "Привет")

    monkeypatch.setattr(tr, "run_logged_subprocess", _fake_run)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("dummy")

    json_path = tmp_path / "audio.json"
    json_path.write_text("{}")

    returned = tr.run_whisperx_cli(
        whisperx_bin="whisperx",
        audio_path=str(audio_path),
        hf_token="token",
        work_dir=str(tmp_path),
    )

    assert returned == str(json_path)
    assert invoked_cmds, "Subprocess should have been invoked"

    cmd = invoked_cmds[0]
    assert "--language" in cmd and "ru" in cmd
    assert "--min_speakers" in cmd and "2" in cmd
    assert "--max_speakers" in cmd and "3" in cmd
    assert "--initial_prompt" in cmd and "Привет" in cmd

    for env_var in [
        "YT_DIARIZER_LANGUAGE",
        "YT_DIARIZER_MIN_SPEAKERS",
        "YT_DIARIZER_MAX_SPEAKERS",
        "YT_DIARIZER_INITIAL_PROMPT",
    ]:
        monkeypatch.delenv(env_var, raising=False)
