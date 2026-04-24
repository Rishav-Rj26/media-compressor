"""
image_compressor.py – JPEG, WebP & AVIF Image Compression
Compresses a single image at a given quality level without resizing.
Includes size guard, input validation, and compression speed tracking.
"""

import os
import time
import tempfile
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".avif"}
SUPPORTED_FORMATS = {"JPEG", "WEBP", "AVIF"}
FORMAT_EXT = {"JPEG": ".jpg", "WEBP": ".webp", "AVIF": ".avif"}


def is_image(filepath: str) -> bool:
    """Check if file has a supported image extension."""
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTENSIONS


def validate_image(filepath: str) -> bool:
    """Verify the file is a valid, readable image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def compress_image(
    input_path: str,
    output_path: str,
    fmt: str = "JPEG",
    quality: int = 70,
    size_guard: bool = True,
) -> dict:
    """
    Compress an image to the specified format and quality.

    Parameters
    ----------
    input_path  : path to the original image
    output_path : path to write compressed image
    fmt         : 'JPEG', 'WEBP', or 'AVIF'
    quality     : 1-100
    size_guard  : if True, re-encode at lower quality when output > input

    Returns
    -------
    dict with keys: output_path, elapsed_seconds, size_guarded
    """
    fmt = fmt.upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from {SUPPORTED_FORMATS}")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    start = time.perf_counter()

    img = Image.open(input_path)

    # Convert RGBA → RGB for JPEG (JPEG doesn't support alpha)
    if fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    # Ensure RGB for AVIF
    if fmt == "AVIF" and img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_kwargs = {"quality": quality, "optimize": True}
    if fmt == "JPEG":
        save_kwargs["subsampling"] = 2      # 4:2:0 chroma subsampling
        save_kwargs["progressive"] = True
    elif fmt == "WEBP":
        save_kwargs["method"] = 4           # balance speed / compression
    elif fmt == "AVIF":
        # Pillow >= 10.1 supports AVIF natively, or via pillow-avif-plugin
        save_kwargs["speed"] = 6            # 0 (slowest/best) to 10 (fastest)
        save_kwargs.pop("optimize", None)   # AVIF doesn't use 'optimize'

    img.save(output_path, format=fmt, **save_kwargs)

    # ── Size guard: if output is bigger than input, try harder ────────────
    size_guarded = False
    orig_size = os.path.getsize(input_path)
    comp_size = os.path.getsize(output_path)

    if size_guard and comp_size >= orig_size and quality > 10:
        # Try progressively lower quality until we beat original size
        for fallback_q in range(quality - 10, 9, -10):
            fallback_kwargs = dict(save_kwargs, quality=fallback_q)
            img.save(output_path, format=fmt, **fallback_kwargs)
            comp_size = os.path.getsize(output_path)
            if comp_size < orig_size:
                size_guarded = True
                break

    elapsed = time.perf_counter() - start

    return {
        "output_path": output_path,
        "elapsed_seconds": round(elapsed, 4),
        "size_guarded": size_guarded,
    }
