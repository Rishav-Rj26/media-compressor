"""
test_batch_processor.py – Tests for the batch processing engine.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.batch_processor import _collect_files, run_batch


class TestCollectFiles:
    """Tests for the file collection/classification function."""

    def test_finds_images(self, sample_batch_dir):
        images, videos = _collect_files(sample_batch_dir)
        assert len(images) == 3
        assert len(videos) == 0

    def test_empty_dir(self, test_dir):
        empty = os.path.join(test_dir, "empty_batch")
        os.makedirs(empty, exist_ok=True)
        images, videos = _collect_files(empty)
        assert len(images) == 0
        assert len(videos) == 0


class TestRunBatch:
    """Tests for the full batch processing pipeline."""

    def test_batch_produces_report(self, sample_batch_dir, test_dir):
        out = os.path.join(test_dir, "batch_output")
        report = os.path.join(out, "report.csv")
        csv_path = run_batch(
            input_dir=sample_batch_dir,
            output_dir=out,
            report_path=report,
            formats=["JPEG"],
            qualities=[50],
            codecs=[],
            max_workers=2,
        )
        assert os.path.isfile(csv_path)

    def test_batch_creates_compressed_files(self, sample_batch_dir, test_dir):
        out = os.path.join(test_dir, "batch_output2")
        report = os.path.join(out, "report2.csv")
        run_batch(
            input_dir=sample_batch_dir,
            output_dir=out,
            report_path=report,
            formats=["JPEG"],
            qualities=[50],
            codecs=[],
            max_workers=2,
        )
        # Should have 3 images × 1 format × 1 quality = 3 compressed files
        jpeg_dir = os.path.join(out, "images", "jpeg")
        assert os.path.isdir(jpeg_dir)
        files = [f for f in os.listdir(jpeg_dir) if f.endswith(".jpg")]
        assert len(files) == 3

    def test_batch_with_progress_callback(self, sample_batch_dir, test_dir):
        out = os.path.join(test_dir, "batch_progress")
        report = os.path.join(out, "report.csv")
        progress_log = []

        def on_progress(done, total, msg):
            progress_log.append((done, total, msg))

        run_batch(
            input_dir=sample_batch_dir,
            output_dir=out,
            report_path=report,
            formats=["WEBP"],
            qualities=[70],
            codecs=[],
            max_workers=2,
            progress_callback=on_progress,
        )
        assert len(progress_log) == 3  # 3 images × 1 format × 1 quality

    def test_batch_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            run_batch(
                input_dir="/nonexistent/dir",
                output_dir="/tmp/out",
                report_path="/tmp/rep.csv",
            )

    def test_batch_empty_dir_returns_empty(self, test_dir):
        empty = os.path.join(test_dir, "empty_for_batch")
        os.makedirs(empty, exist_ok=True)
        result = run_batch(
            input_dir=empty,
            output_dir=os.path.join(test_dir, "empty_out"),
            report_path=os.path.join(test_dir, "empty_out", "r.csv"),
        )
        assert result == ""
