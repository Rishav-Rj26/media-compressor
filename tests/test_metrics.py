"""
test_metrics.py – Tests for the quality metrics module.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (
    compression_ratio,
    file_sizes,
    compression_speed,
    compute_image_psnr,
    compute_image_ssim,
    compute_image_ms_ssim,
)
from src.image_compressor import compress_image


class TestFileSizeMetrics:
    """Tests for file size and ratio calculations."""

    def test_compression_ratio(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "ratio_test.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=50)
        ratio = compression_ratio(sample_image_path, out)
        assert ratio > 1.0, "Compression ratio should be > 1 for PNG→JPEG@Q50"

    def test_file_sizes(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "sizes_test.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=50)
        orig, comp = file_sizes(sample_image_path, out)
        assert orig > 0
        assert comp > 0
        assert orig > comp

    def test_compression_speed(self, sample_image_path):
        speed = compression_speed(sample_image_path, 0.01)
        assert speed > 0


class TestImagePSNR:
    """Tests for PSNR computation."""

    def test_identical_images_have_infinite_psnr(self, sample_image_path):
        psnr = compute_image_psnr(sample_image_path, sample_image_path)
        assert psnr == float("inf"), "Identical images should have infinite PSNR"

    def test_psnr_is_positive(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "psnr_test.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=50)
        psnr = compute_image_psnr(sample_image_path, out)
        assert psnr > 0, "PSNR should be positive"

    def test_higher_quality_means_higher_psnr(self, sample_image_path, output_dir):
        out_30 = os.path.join(output_dir, "psnr_q30.jpg")
        out_90 = os.path.join(output_dir, "psnr_q90.jpg")
        compress_image(sample_image_path, out_30, fmt="JPEG", quality=30, size_guard=False)
        compress_image(sample_image_path, out_90, fmt="JPEG", quality=90, size_guard=False)
        psnr_30 = compute_image_psnr(sample_image_path, out_30)
        psnr_90 = compute_image_psnr(sample_image_path, out_90)
        assert psnr_90 > psnr_30, f"PSNR@Q90 ({psnr_90}) should exceed PSNR@Q30 ({psnr_30})"

    def test_psnr_in_reasonable_range(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "psnr_range.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=70)
        psnr = compute_image_psnr(sample_image_path, out)
        assert 20 < psnr < 80, f"PSNR={psnr} is outside reasonable range for Q70 JPEG"

    def test_missing_file_raises(self, sample_image_path):
        with pytest.raises(FileNotFoundError):
            compute_image_psnr(sample_image_path, "/nonexistent.jpg")


class TestImageSSIM:
    """Tests for SSIM computation."""

    def test_identical_images_have_ssim_one(self, sample_image_path):
        ssim_val = compute_image_ssim(sample_image_path, sample_image_path)
        assert abs(ssim_val - 1.0) < 0.001, f"Identical images should have SSIM ≈ 1.0, got {ssim_val}"

    def test_ssim_between_zero_and_one(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "ssim_test.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=30)
        ssim_val = compute_image_ssim(sample_image_path, out)
        assert 0 < ssim_val <= 1.0, f"SSIM should be in (0, 1], got {ssim_val}"

    def test_higher_quality_means_higher_ssim(self, sample_image_path, output_dir):
        out_30 = os.path.join(output_dir, "ssim_q30.jpg")
        out_90 = os.path.join(output_dir, "ssim_q90.jpg")
        compress_image(sample_image_path, out_30, fmt="JPEG", quality=30, size_guard=False)
        compress_image(sample_image_path, out_90, fmt="JPEG", quality=90, size_guard=False)
        ssim_30 = compute_image_ssim(sample_image_path, out_30)
        ssim_90 = compute_image_ssim(sample_image_path, out_90)
        assert ssim_90 > ssim_30, f"SSIM@Q90 ({ssim_90}) should exceed SSIM@Q30 ({ssim_30})"


class TestMSSSIM:
    """Tests for Multi-Scale SSIM."""

    def test_ms_ssim_between_zero_and_one(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "msssim_test.jpg")
        compress_image(sample_image_path, out, fmt="JPEG", quality=50)
        ms_ssim = compute_image_ms_ssim(sample_image_path, out)
        assert 0 < ms_ssim <= 1.0, f"MS-SSIM should be in (0, 1], got {ms_ssim}"

    def test_ms_ssim_correlates_with_quality(self, sample_image_path, output_dir):
        """Higher quality should generally produce higher MS-SSIM."""
        out_30 = os.path.join(output_dir, "msssim_q30.jpg")
        out_90 = os.path.join(output_dir, "msssim_q90.jpg")
        compress_image(sample_image_path, out_30, fmt="JPEG", quality=30)
        compress_image(sample_image_path, out_90, fmt="JPEG", quality=90)
        ms_30 = compute_image_ms_ssim(sample_image_path, out_30)
        ms_90 = compute_image_ms_ssim(sample_image_path, out_90)
        assert ms_90 >= ms_30, f"MS-SSIM@Q90 ({ms_90}) should be >= MS-SSIM@Q30 ({ms_30})"
