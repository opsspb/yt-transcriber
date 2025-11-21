# yt-transcriber

A one-command CLI that downloads a YouTube video, transcribes it with WhisperX, and saves a diarized transcript with speaker labels. The script creates a temporary virtual environment for each run, installs dependencies inside it, and cleans up after itself while keeping the final transcript and logs.

## Requirements

- Python 3.9.6 or newer available on the command line (confirmed to work on macOS 26.1).
- macOS is the primary target (Apple Silicon recommended). ffmpeg is auto-downloaded on macOS when missing; on other platforms it must already be in `PATH`. You can also point the tool to existing binaries with `YT_DIARIZER_FFMPEG=/full/path/to/ffmpeg` and (optionally) `YT_DIARIZER_FFPROBE=/full/path/to/ffprobe`.
- A Hugging Face access token saved to `token.txt` in the repository root (used for WhisperX/pyannote diarization). Module execution (`python -m yt_diarizer`) also looks for this file in the repository root before falling back to `yt_diarizer/token.txt`.
- Internet access to download YouTube audio and WhisperX models.

## Quick start

1. Clone the repository and open a terminal in the project root.
2. Create `token.txt` containing your Hugging Face token (no quotes or extra spaces).
3. (Optional) If the video requires authentication, place a Netscape-format `cookies.txt` file next to the script or set `YT_DIARIZER_COOKIES=/full/path/to/cookies.txt`.
4. Run the tool. You can provide the YouTube URL on the command line or wait for the interactive prompt:
   ```bash
   python3 yt_diarizer.py "https://www.youtube.com/watch?v=..."
   # or
   python -m yt_diarizer "https://www.youtube.com/watch?v=..."
   ```
5. When prompted, paste the YouTube URL and press **Enter**. The script will:
   - create a temporary workspace and virtual environment under `.yt_diarizer_work_*`,
   - install pinned versions of WhisperX, PyTorch CPU wheels, pyannote, and `yt-dlp` inside the venv,
   - download best-quality audio via `yt-dlp` (falling back to your cookies or browser cookies if needed),
   - transcribe with WhisperX using the `large-v3` model and pyannote diarization on CPU,
   - print progress and debug messages to the terminal.

The temporary workspace is deleted automatically after the run. If the process is interrupted, you can safely delete any leftover `.yt_diarizer_work_*` directories.

## Outputs

- A diarized transcript in `diarized_transcript_YYYYMMDD_HHMMSS.txt` with lines like:
  ```
  [00:00:01.000 --> 00:00:03.500] SPEAKER_00: Hello world
  ```
- The raw WhisperX JSON in a matching `diarized_transcript_YYYYMMDD_HHMMSS.json` file.

## Tips and troubleshooting

- First run may take time while WhisperX dependencies download.
- If `yt-dlp` fails on restricted videos, provide cookies as noted above or grant your terminal “Full Disk Access” on macOS so `--cookies-from-browser` can read Safari/Chrome cookies.
- Ensure ffmpeg is in `PATH` on non-macOS platforms; otherwise the run will fail early. You can avoid repeated model downloads between runs by keeping the default cache locations, but by default the script now stores Hugging Face, Transformers, pyannote, and Torch caches inside the temporary workspace so they are cleaned up automatically.
