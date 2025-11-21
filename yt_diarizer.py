#!/usr/bin/env python3
"""Console entrypoint for yt_diarizer."""

import os

from yt_diarizer.main import main


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    entrypoint_path = os.path.abspath(__file__)
    main(script_dir=script_dir, entrypoint_path=entrypoint_path)
