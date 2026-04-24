"""
generate_samples.py – Create synthetic test images for the compression system.
Run this once to populate the samples/ folder if you don't have real media files.

Usage:
    python generate_samples.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def create_gradient_image(path: str, width=1920, height=1080):
    """Create a smooth RGB gradient image (good for testing compression artifacts)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = int(255 * x / width)          # R gradient →
            arr[y, x, 1] = int(255 * y / height)          # G gradient ↓
            arr[y, x, 2] = int(255 * (1 - x / width))     # B gradient ←
    img = Image.fromarray(arr)
    img.save(path)
    print(f"  Created: {path}")


def create_photo_like_image(path: str, width=1920, height=1080):
    """Create a complex synthetic image with shapes, text, and noise."""
    img = Image.new("RGB", (width, height), (30, 30, 50))
    draw = ImageDraw.Draw(img)

    # Draw overlapping colored circles
    np.random.seed(42)
    for _ in range(15):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        r = np.random.randint(50, 300)
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white")

    # Draw rectangles
    for _ in range(10):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = x1 + np.random.randint(50, 400), y1 + np.random.randint(50, 300)
        color = tuple(np.random.randint(30, 200, size=3).tolist())
        draw.rectangle([x1, y1, x2, y2], fill=color, outline="white", width=2)

    # Add some text
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((100, 100), "COMPRESSION TEST IMAGE", fill="white", font=font)
    draw.text((100, 200), "Media Compression System v1.0", fill=(200, 200, 200), font=font)

    # Add noise to make it more realistic
    arr = np.array(img)
    noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    print(f"  Created: {path}")


def create_high_detail_image(path: str, width=1920, height=1080):
    """Create an image with fine details (lines, patterns) to stress-test compression."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Grid pattern
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)

    # Diagonal lines
    for i in range(0, width + height, 40):
        draw.line([(i, 0), (0, i)], fill=(100, 150, 200), width=2)

    # Concentric circles
    cx, cy = width // 2, height // 2
    for r in range(10, min(width, height) // 2, 15):
        color = (r % 256, (r * 2) % 256, (r * 3) % 256)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=1)

    img.save(path)
    print(f"  Created: {path}")


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print("Generating sample images...\n")

    create_gradient_image(os.path.join(SAMPLES_DIR, "gradient_1080p.png"))
    create_photo_like_image(os.path.join(SAMPLES_DIR, "synthetic_photo.png"))
    create_high_detail_image(os.path.join(SAMPLES_DIR, "detailed_pattern.bmp"))

    print(f"\n[OK] Sample images created in: {SAMPLES_DIR}")
    print("   Place any additional images/videos in this folder before running main.py")


if __name__ == "__main__":
    main()
