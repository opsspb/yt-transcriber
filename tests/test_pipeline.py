import os
import tempfile
import unittest
from unittest import mock
import sys

from yt_diarizer import pipeline
from yt_diarizer.exceptions import DependencyError, PipelineError


class InstallPythonDependenciesTests(unittest.TestCase):
    def test_install_python_dependencies_uses_pinned_commands(self) -> None:
        calls = []

        def _fake_run(cmd, description):
            calls.append((cmd, description))
            return 0, ["ok"]

        with mock.patch("yt_diarizer.pipeline.run_logged_subprocess", side_effect=_fake_run):
            pipeline.install_python_dependencies("/venv/python")

        self.assertEqual(len(calls), 3)
        torch_cmd, torch_desc = calls[1]
        self.assertIn("install PyTorch CPU wheels", torch_desc)
        self.assertIn("torch==2.1.2", torch_cmd)
        self.assertIn("torchaudio==2.1.2", torch_cmd)
        self.assertIn("--index-url", torch_cmd)

        whisper_cmd, whisper_desc = calls[2]
        self.assertIn("install WhisperX", whisper_desc)
        self.assertIn("whisperx==3.1.1", whisper_cmd)
        self.assertIn("yt-dlp==2024.11.18", whisper_cmd)
        self.assertIn("--constraint", whisper_cmd)
        self.assertNotIn("pyannote.audio", " ".join(whisper_cmd))

    def test_install_python_dependencies_failure_reports_snippet(self) -> None:
        with mock.patch(
            "yt_diarizer.pipeline.run_logged_subprocess",
            side_effect=[(0, ["ok"]), (1, ["line1", "line2", "line3"]), (0, [])],
        ):
            with self.assertRaises(DependencyError) as ctx:
                pipeline.install_python_dependencies("/venv/python")

        message = str(ctx.exception)
        self.assertIn("line2", message)
        self.assertIn("exit code 1", message)


class WorkspaceCleanupTests(unittest.TestCase):
    def test_workspace_removed_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_entrypoint = os.path.join(tmpdir, "entry.py")
            with open(dummy_entrypoint, "w", encoding="utf-8") as f:
                f.write("print('noop')")

            with mock.patch(
                "yt_diarizer.main.setup_and_run_in_venv",
                side_effect=PipelineError("boom"),
            ), mock.patch("sys.argv", ["yt_diarizer"]):
                with self.assertRaises(SystemExit):
                    from yt_diarizer import main as main_func

                    main_func(script_dir=tmpdir, entrypoint_path=dummy_entrypoint)

            leftovers = [p for p in os.listdir(tmpdir) if p.startswith(".yt_diarizer_work_")]
            self.assertEqual(leftovers, [])


class TokenResolutionTests(unittest.TestCase):
    def test_token_found_in_repository_root_when_running_as_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = os.path.join(tmpdir, "yt_diarizer")
            os.makedirs(pkg_dir, exist_ok=True)
            token_path = os.path.join(tmpdir, "token.txt")
            with open(token_path, "w", encoding="utf-8") as f:
                f.write("secret-token")

            token = pipeline.load_hf_token(pkg_dir)
            self.assertEqual(token, "secret-token")


class FfmpegChecksTests(unittest.TestCase):
    def test_non_macos_without_ffmpeg_fails_fast(self) -> None:
        with mock.patch("yt_diarizer.deps.sys.platform", "linux"), mock.patch(
            "shutil.which", return_value=None
        ):
            with self.assertRaises(DependencyError) as ctx:
                from yt_diarizer import deps

                deps.download_ffmpeg_if_missing("/tmp/work")

        self.assertIn("auto-download is only implemented on macOS", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
