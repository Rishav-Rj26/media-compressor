"""
test_image_compressor.py – Tests for image compression module.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.image_compressor import compress_image, is_image, validate_image, SUPPORTED_FORMATS


class TestIsImage:
    """Tests for the is_image() file detection function."""

    def test_valid_extensions(self):
        assert is_image("photo.jpg") is True
        assert is_image("photo.jpeg") is True
        assert is_image("photo.png") is True
        assert is_image("photo.bmp") is True
        assert is_image("photo.webp") is True
        assert is_image("photo.tiff") is True
        assert is_image("photo.avif") is True

    def test_invalid_extensions(self):
        assert is_image("video.mp4") is False
        assert is_image("document.pdf") is False
        assert is_image("archive.zip") is False
        assert is_image("noextension") is False

    def test_case_insensitive(self):
        assert is_image("PHOTO.JPG") is True
        assert is_image("photo.PNG") is True


class TestValidateImage:
    """Tests for image validation."""

    def test_valid_image(self, sample_image_path):
        assert validate_image(sample_image_path) is True

    def test_invalid_path(self):
        assert validate_image("/nonexistent/path.png") is False

    def test_non_image_file(self, test_dir):
        txt_path = os.path.join(test_dir, "not_an_image.png")
        with open(txt_path, "w") as f:
            f.write("this is not an image")
        assert validate_image(txt_path) is False


class TestCompressImage:
    """Tests for the core compress_image() function."""

    def test_jpeg_compression(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "test_q70.jpg")
        result = compress_image(sample_image_path, out, fmt="JPEG", quality=70)
        assert os.path.isfile(result["output_path"])
        assert os.path.getsize(result["output_path"]) > 0
        assert result["elapsed_seconds"] > 0

    def test_webp_compression(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "test_q70.webp")
        result = compress_image(sample_image_path, out, fmt="WEBP", quality=70)
        assert os.path.isfile(result["output_path"])
        assert os.path.getsize(result["output_path"]) > 0

    def test_compressed_smaller_than_original(self, sample_image_path, output_dir):
        """At quality 30, the output should be smaller than the PNG original."""
        out = os.path.join(output_dir, "test_small.jpg")
        result = compress_image(sample_image_path, out, fmt="JPEG", quality=30)
        orig_size = os.path.getsize(sample_image_path)
        comp_size = os.path.getsize(result["output_path"])
        assert comp_size < orig_size, f"Expected compressed ({comp_size}) < original ({orig_size})"

    def test_lower_quality_means_smaller_file(self, sample_image_path, output_dir):
        """Quality 30 should produce a smaller file than quality 90."""
        out_30 = os.path.join(output_dir, "q30.jpg")
        out_90 = os.path.join(output_dir, "q90.jpg")
        compress_image(sample_image_path, out_30, fmt="JPEG", quality=30, size_guard=False)
        compress_image(sample_image_path, out_90, fmt="JPEG", quality=90, size_guard=False)
        assert os.path.getsize(out_30) < os.path.getsize(out_90)

    def test_rgba_to_jpeg(self, sample_rgba_image_path, output_dir):
        """RGBA images should be converted to RGB for JPEG output."""
        out = os.path.join(output_dir, "rgba_to_jpeg.jpg")
        result = compress_image(sample_rgba_image_path, out, fmt="JPEG", quality=70)
        assert os.path.isfile(result["output_path"])

    def test_size_guard(self, sample_image_path, output_dir):
        """Size guard should refuse to produce output larger than original."""
        out = os.path.join(output_dir, "guarded.jpg")
        result = compress_image(
            sample_image_path, out, fmt="JPEG", quality=95, size_guard=True
        )
        orig_size = os.path.getsize(sample_image_path)
        comp_size = os.path.getsize(result["output_path"])
        # Either smaller or the guard kicked in
        assert comp_size <= orig_size or result["size_guarded"] is True

    def test_invalid_format_raises(self, sample_image_path, output_dir):
        out = os.path.join(output_dir, "bad.gif")
        with pytest.raises(ValueError, match="Unsupported format"):
            compress_image(sample_image_path, out, fmt="GIF", quality=70)

    def test_missing_file_raises(self, output_dir):
        out = os.path.join(output_dir, "should_fail.jpg")
        with pytest.raises(FileNotFoundError):
            compress_image("/nonexistent/image.png", out, fmt="JPEG", quality=70)

    def test_creates_output_directory(self, sample_image_path, test_dir):
        """Output directory should be created automatically."""
        nested = os.path.join(test_dir, "deep", "nested", "dir", "out.jpg")
        result = compress_image(sample_image_path, nested, fmt="JPEG", quality=50)
        assert os.path.isfile(result["output_path"])
