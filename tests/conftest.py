"""
conftest.py – Shared pytest fixtures for the test suite.

Creates temporary test images and videos for compression testing.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test artifacts. Cleaned up after tests."""
    d = tempfile.mkdtemp(prefix="mcs_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_image_path(test_dir):
    """Create a 512×512 synthetic test image (PNG) with varied content."""
    path = os.path.join(test_dir, "test_image.png")
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)

    # Gradient background
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for x in range(512):
            arr[y, x] = (x % 256, y % 256, (x + y) % 256)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Add shapes for structural content (important for SSIM testing)
    draw.ellipse([50, 50, 400, 400], fill=(255, 128, 0), outline="white")
    draw.rectangle([200, 30, 450, 200], fill=(0, 128, 255))
    for i in range(0, 500, 30):
        draw.line([(i, 0), (0, i)], fill=(200, 200, 200), width=1)

    img.save(path, format="PNG")
    return path


@pytest.fixture(scope="session")
def sample_rgba_image_path(test_dir):
    """Create an RGBA image to test alpha channel handling."""
    path = os.path.join(test_dir, "test_rgba.png")
    img = Image.new("RGBA", (128, 128), (200, 100, 50, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill=(0, 255, 0, 200))
    img.save(path, format="PNG")
    return path


@pytest.fixture(scope="session")
def tiny_image_path(test_dir):
    """Create a very small 8×8 image for edge case testing."""
    path = os.path.join(test_dir, "tiny.png")
    img = Image.new("RGB", (8, 8), (100, 150, 200))
    img.save(path, format="PNG")
    return path


@pytest.fixture(scope="session")
def output_dir(test_dir):
    """Create an output subdirectory."""
    d = os.path.join(test_dir, "output")
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def sample_batch_dir(test_dir):
    """Create a directory with multiple test images for batch processing."""
    d = os.path.join(test_dir, "batch_input")
    os.makedirs(d, exist_ok=True)

    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        img = Image.new("RGB", (128, 128), color)
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 108, 108], fill=(255, 255, 255))
        img.save(os.path.join(d, f"batch_{i}.png"), format="PNG")

    return d
