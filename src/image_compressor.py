"""
image_compressor.py – JPEG & WebP Image Compression
Compresses a single image at a given quality level without resizing.
"""

import os
import tempfile
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image(filepath: str) -> bool:
    """Check if file has a supported image extension."""
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTENSIONS


def compress_image(input_path: str, output_path: str,
                   fmt: str = "JPEG", quality: int = 70) -> str:
    """
    Compress an image to the specified format and quality.

    Parameters
    ----------
    input_path  : path to the original image
    output_path : path to write compressed image
    fmt         : 'JPEG' or 'WEBP'
    quality     : 1-100

    Returns
    -------
    output_path on success
    """
    img = Image.open(input_path)

    # Convert RGBA → RGB for JPEG
    if fmt.upper() == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_kwargs = {"quality": quality, "optimize": True}
    if fmt.upper() == "JPEG":
        save_kwargs["subsampling"] = 2   # 4:2:0
        save_kwargs["progressive"] = True
    elif fmt.upper() == "WEBP":
        save_kwargs["method"] = 4        # balance speed / compression

    img.save(output_path, format=fmt.upper(), **save_kwargs)
    return output_path
