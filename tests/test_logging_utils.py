import io
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout

from yt_diarizer.logging_utils import log_line, set_log_file


class LoggingUtilsTests(unittest.TestCase):
    def test_set_log_file_does_not_create_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log.txt")

            set_log_file(tmpdir)

            self.assertFalse(
                os.path.exists(log_path),
                "set_log_file should not create a log file on disk",
            )

    def test_log_line_writes_timestamped_output(self) -> None:
        buffer = io.StringIO()
        message = "test message"

        with redirect_stdout(buffer):
            log_line(message)

        output = buffer.getvalue().strip()
        self.assertTrue(output.startswith("["))
        self.assertIn(message, output)
        self.assertRegex(output, re.compile(r"\[\d{4}-\d{2}-\d{2} "))


if __name__ == "__main__":
    unittest.main()
