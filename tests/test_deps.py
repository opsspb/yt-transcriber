import os
import tempfile
import unittest
from unittest import mock

from yt_diarizer import deps


class MacOsFfmpegDownloadTests(unittest.TestCase):
    def _run_with_machine(self, machine: str):
        with mock.patch("platform.machine", return_value=machine):
            # Avoid real network calls to the GitHub API.
            with mock.patch("urllib.request.urlopen", side_effect=Exception("no net")):
                return deps._macos_ffmpeg_download()

    def test_universal_candidates_included_for_arm(self) -> None:
        urls, _ = self._run_with_machine("arm64")
        self.assertTrue(
            any("macos-universal" in url for url in urls),
            "Universal macOS ffmpeg assets should be considered for ARM machines",
        )

    def test_universal_candidates_included_for_intel(self) -> None:
        urls, _ = self._run_with_machine("x86_64")
        self.assertTrue(
            any("macos-universal" in url for url in urls),
            "Universal macOS ffmpeg assets should be considered for Intel machines",
        )


class ManualFfmpegOverrideTests(unittest.TestCase):
    def test_directory_override_resolves_both_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_path = os.path.join(tmpdir, "ffmpeg")
            ffprobe_path = os.path.join(tmpdir, "ffprobe")

            for path in (ffmpeg_path, ffprobe_path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("#!/bin/sh\n")

            with mock.patch.dict(
                os.environ,
                {"YT_DIARIZER_FFMPEG_PATH": tmpdir},
                clear=False,
            ):
                resolved_path = deps.download_ffmpeg_if_missing(tmpdir)

                self.assertEqual(resolved_path, ffmpeg_path)
                self.assertIn(tmpdir, os.environ["PATH"])


if __name__ == "__main__":
    unittest.main()
