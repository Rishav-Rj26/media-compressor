"""
metrics.py – Quality Metrics Calculator
Computes PSNR, SSIM, and compression ratio for image/video files.
"""

import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def compression_ratio(original_path: str, compressed_path: str) -> float:
    """Return compression ratio = original_size / compressed_size."""
    orig = os.path.getsize(original_path)
    comp = os.path.getsize(compressed_path)
    if comp == 0:
        return float("inf")
    return round(orig / comp, 4)


def file_sizes(original_path: str, compressed_path: str) -> tuple:
    """Return (original_bytes, compressed_bytes)."""
    return os.path.getsize(original_path), os.path.getsize(compressed_path)


# ── Image Metrics ────────────────────────────────────────────────────────────

def compute_image_psnr(original_path: str, compressed_path: str) -> float:
    """Compute PSNR between two images (loaded via OpenCV)."""
    img_orig = cv2.imread(original_path)
    img_comp = cv2.imread(compressed_path)
    if img_orig is None or img_comp is None:
        raise FileNotFoundError("Could not read one of the image files.")
    # Resize compressed to original dimensions if they differ (shouldn't normally)
    if img_orig.shape != img_comp.shape:
        img_comp = cv2.resize(img_comp, (img_orig.shape[1], img_orig.shape[0]))
    mse = np.mean((img_orig.astype(np.float64) - img_comp.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return round(10 * np.log10(255.0 ** 2 / mse), 4)


def compute_image_ssim(original_path: str, compressed_path: str) -> float:
    """Compute SSIM between two images (converted to grayscale)."""
    img_orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    img_comp = cv2.imread(compressed_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None or img_comp is None:
        raise FileNotFoundError("Could not read one of the image files.")
    if img_orig.shape != img_comp.shape:
        img_comp = cv2.resize(img_comp, (img_orig.shape[1], img_orig.shape[0]))
    score = ssim(img_orig, img_comp)
    return round(score, 6)


# ── Video Metrics ────────────────────────────────────────────────────────────

def _sample_frames(video_path: str, max_frames: int = 30) -> list:
    """Extract up to `max_frames` evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def compute_video_psnr(original_path: str, compressed_path: str,
                       max_frames: int = 30) -> float:
    """Average PSNR across sampled frames of two videos."""
    orig_frames = _sample_frames(original_path, max_frames)
    comp_frames = _sample_frames(compressed_path, max_frames)
    n = min(len(orig_frames), len(comp_frames))
    if n == 0:
        return 0.0
    psnrs = []
    for i in range(n):
        o = orig_frames[i].astype(np.float64)
        c = comp_frames[i].astype(np.float64)
        if o.shape != c.shape:
            c = cv2.resize(comp_frames[i], (o.shape[1], o.shape[0])).astype(np.float64)
        mse = np.mean((o - c) ** 2)
        if mse == 0:
            psnrs.append(100.0)  # cap
        else:
            psnrs.append(10 * np.log10(255.0 ** 2 / mse))
    return round(float(np.mean(psnrs)), 4)


def compute_video_ssim(original_path: str, compressed_path: str,
                       max_frames: int = 30) -> float:
    """Average SSIM across sampled frames of two videos."""
    orig_frames = _sample_frames(original_path, max_frames)
    comp_frames = _sample_frames(compressed_path, max_frames)
    n = min(len(orig_frames), len(comp_frames))
    if n == 0:
        return 0.0
    ssims = []
    for i in range(n):
        o = cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2GRAY)
        c = cv2.cvtColor(comp_frames[i], cv2.COLOR_BGR2GRAY)
        if o.shape != c.shape:
            c = cv2.resize(c, (o.shape[1], o.shape[0]))
        ssims.append(ssim(o, c))
    return round(float(np.mean(ssims)), 6)
