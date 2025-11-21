import os
import tempfile
import unittest
from unittest import mock
import sys

from yt_diarizer import pipeline
from yt_diarizer.exceptions import (
    DependencyError,
    DependencyInstallationError,
    PipelineError,
)


class InstallPythonDependenciesTests(unittest.TestCase):
    def test_install_python_dependencies_uses_pinned_commands(self) -> None:
        calls = []

        def _fake_run(cmd, description):
            calls.append((cmd, description))
            return 0, ["ok"]

        with mock.patch("yt_diarizer.pipeline.run_logged_subprocess", side_effect=_fake_run), mock.patch(
            "shutil.which", return_value="/usr/bin/pkg-config"
        ):
            pipeline.install_python_dependencies("/venv/python")

        self.assertEqual(len(calls), 4)
        numpy_cmd, numpy_desc = calls[1]
        self.assertIn("numpy==1.26.4", numpy_cmd)
        self.assertIn("below 2.x", numpy_desc)

        torch_cmd, torch_desc = calls[2]
        self.assertIn("install PyTorch CPU wheels", torch_desc)
        self.assertIn("torch==2.3.1", torch_cmd)
        self.assertIn("torchaudio==2.3.1", torch_cmd)
        self.assertIn("--index-url", torch_cmd)

        whisper_cmd, whisper_desc = calls[3]
        self.assertIn("install WhisperX", whisper_desc)
        self.assertIn("whisperx==3.1.1", whisper_cmd)
        self.assertIn("yt-dlp==2024.11.18", whisper_cmd)
        self.assertIn("--constraint", whisper_cmd)
        self.assertNotIn("pyannote.audio", " ".join(whisper_cmd))

    def test_install_python_dependencies_failure_reports_snippet(self) -> None:
        with mock.patch(
            "yt_diarizer.pipeline.run_logged_subprocess",
            side_effect=[(0, ["ok"]), (1, ["line1", "line2", "line3"]), (0, [])],
        ), mock.patch("shutil.which", return_value="/usr/bin/pkg-config"):
            with self.assertRaises(DependencyError) as ctx:
                pipeline.install_python_dependencies("/venv/python")

        message = str(ctx.exception)
        self.assertIn("line2", message)
        self.assertIn("exit code 1", message)

    def test_install_python_dependencies_requires_pkg_config_on_unix(self) -> None:
        with mock.patch("sys.platform", "darwin"), mock.patch(
            "shutil.which", return_value=None
        ), mock.patch("yt_diarizer.pipeline.run_logged_subprocess") as mocked_run:
            with self.assertRaises(DependencyInstallationError) as ctx:
                pipeline.install_python_dependencies("/venv/python")

        self.assertIn("pkg-config not found", str(ctx.exception))
        mocked_run.assert_not_called()


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
    def test_env_override_used_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_path = os.path.join(tmpdir, "ffmpeg")
            ffprobe_path = os.path.join(tmpdir, "ffprobe")
            for path in (ffmpeg_path, ffprobe_path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("#!/bin/sh\n")

            with mock.patch.dict(
                os.environ,
                {
                    "YT_DIARIZER_FFMPEG_PATH": ffmpeg_path,
                    "YT_DIARIZER_FFPROBE_PATH": ffprobe_path,
                },
                clear=False,
            ):
                paths = pipeline.ensure_ffmpeg(tmpdir)

            self.assertEqual(paths["ffmpeg"], ffmpeg_path)
            self.assertEqual(paths["ffprobe"], ffprobe_path)

    def test_path_detection_short_circuits_download(self) -> None:
        with mock.patch("shutil.which", side_effect=["/usr/bin/ffmpeg", "/usr/bin/ffprobe"]):
            paths = pipeline.ensure_ffmpeg("/tmp/work")

        self.assertEqual(paths["ffmpeg"], "/usr/bin/ffmpeg")
        self.assertEqual(paths["ffprobe"], "/usr/bin/ffprobe")

    def test_download_failure_on_non_macos_reports_runtime_error(self) -> None:
        with mock.patch("sys.platform", "linux"), mock.patch(
            "yt_diarizer.pipeline.download_ffmpeg_for_other_platforms",
            side_effect=RuntimeError("network error"),
        ), mock.patch("yt_diarizer.pipeline._ffmpeg_from_path", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                pipeline.ensure_ffmpeg("/tmp/work")

        self.assertIn("Automatic ffmpeg download failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
