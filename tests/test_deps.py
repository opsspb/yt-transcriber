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


if __name__ == "__main__":
    unittest.main()
