"""
test_video_compressor.py – Tests for video compression module.

Note: These tests are skipped if FFmpeg is not available on the system.
"""

import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.video_compressor import is_video, compress_video, _find_ffmpeg


# Check if FFmpeg is available for tests
try:
    _find_ffmpeg()
    HAS_FFMPEG = True
except FileNotFoundError:
    HAS_FFMPEG = False

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not available")


class TestIsVideo:
    """Tests for the is_video() file detection function."""

    def test_valid_extensions(self):
        assert is_video("clip.mp4") is True
        assert is_video("clip.avi") is True
        assert is_video("clip.mov") is True
        assert is_video("clip.mkv") is True
        assert is_video("clip.webm") is True

    def test_invalid_extensions(self):
        assert is_video("photo.jpg") is False
        assert is_video("document.pdf") is False
        assert is_video("noext") is False

    def test_case_insensitive(self):
        assert is_video("VIDEO.MP4") is True
        assert is_video("clip.AVI") is True
