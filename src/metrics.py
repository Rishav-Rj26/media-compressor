"""
metrics.py – Quality Metrics Calculator
Computes PSNR, SSIM, MS-SSIM, and compression ratio for image/video files.
Includes edge-case handling and compression speed metric.
"""

import os
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# ── File-Level Metrics ───────────────────────────────────────────────────────

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


def compression_speed(original_path: str, elapsed_seconds: float) -> float:
    """
    Return compression throughput in MB/s.
    elapsed_seconds should come from the compressor's timing output.
    """
    if elapsed_seconds <= 0:
        return 0.0
    size_mb = os.path.getsize(original_path) / (1024 * 1024)
    return round(size_mb / elapsed_seconds, 4)


# ── Image Metrics ────────────────────────────────────────────────────────────

def _load_image_pair(original_path: str, compressed_path: str, grayscale: bool = False):
    """
    Load two images via OpenCV, resize compressed to match original if needed.
    Returns (img_orig, img_comp).
    Raises FileNotFoundError / ValueError on bad inputs.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img_orig = cv2.imread(original_path, flag)
    img_comp = cv2.imread(compressed_path, flag)

    if img_orig is None:
        raise FileNotFoundError(f"Cannot read original image: {original_path}")
    if img_comp is None:
        raise FileNotFoundError(f"Cannot read compressed image: {compressed_path}")

    if img_orig.size == 0 or img_comp.size == 0:
        raise ValueError("One of the images has zero pixels.")

    # Resize compressed to original dimensions if they differ
    if img_orig.shape != img_comp.shape:
        if grayscale:
            img_comp = cv2.resize(img_comp, (img_orig.shape[1], img_orig.shape[0]))
        else:
            img_comp = cv2.resize(img_comp, (img_orig.shape[1], img_orig.shape[0]))

    return img_orig, img_comp


def compute_image_psnr(original_path: str, compressed_path: str) -> float:
    """Compute PSNR between two images (loaded via OpenCV)."""
    img_orig, img_comp = _load_image_pair(original_path, compressed_path)
    mse = np.mean((img_orig.astype(np.float64) - img_comp.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return round(10 * np.log10(255.0 ** 2 / mse), 4)


def compute_image_ssim(original_path: str, compressed_path: str) -> float:
    """Compute SSIM between two images (converted to grayscale)."""
    img_orig, img_comp = _load_image_pair(original_path, compressed_path, grayscale=True)
    score = ssim(img_orig, img_comp)
    return round(score, 6)


def compute_image_ms_ssim(original_path: str, compressed_path: str) -> float:
    """
    Compute Multi-Scale SSIM between two images.
    
    MS-SSIM evaluates structural similarity at multiple resolutions,
    providing a more robust quality metric than single-scale SSIM.
    Returns a value in [0, 1] — higher is better.
    """
    img_orig, img_comp = _load_image_pair(original_path, compressed_path, grayscale=True)

    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    levels = len(weights)

    ms_ssim_values = []
    cs_values = []

    for level in range(levels):
        # SSIM at current scale
        score, S = ssim(img_orig, img_comp, full=True)

        # Luminance and contrast-structure components
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu_orig = cv2.GaussianBlur(img_orig.astype(np.float64), (11, 11), 1.5)
        mu_comp = cv2.GaussianBlur(img_comp.astype(np.float64), (11, 11), 1.5)

        mu_orig_sq = mu_orig ** 2
        mu_comp_sq = mu_comp ** 2
        mu_orig_comp = mu_orig * mu_comp

        sigma_orig_sq = cv2.GaussianBlur(img_orig.astype(np.float64) ** 2, (11, 11), 1.5) - mu_orig_sq
        sigma_comp_sq = cv2.GaussianBlur(img_comp.astype(np.float64) ** 2, (11, 11), 1.5) - mu_comp_sq
        sigma_orig_comp = cv2.GaussianBlur(
            img_orig.astype(np.float64) * img_comp.astype(np.float64), (11, 11), 1.5
        ) - mu_orig_comp

        cs = (2 * sigma_orig_comp + C2) / (sigma_orig_sq + sigma_comp_sq + C2)
        cs_values.append(np.mean(cs))
        ms_ssim_values.append(score)

        # Downsample for next level
        if level < levels - 1:
            img_orig = cv2.resize(
                img_orig,
                (img_orig.shape[1] // 2, img_orig.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            img_comp = cv2.resize(
                img_comp,
                (img_comp.shape[1] // 2, img_comp.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            # Stop if images become too small
            if img_orig.shape[0] < 16 or img_orig.shape[1] < 16:
                # Fill remaining levels with current values
                for remaining in range(level + 1, levels):
                    cs_values.append(cs_values[-1])
                    ms_ssim_values.append(ms_ssim_values[-1])
                break

    # Combine: product of CS^weight at each scale, times luminance^weight at final scale
    result = 1.0
    for i in range(levels - 1):
        result *= max(cs_values[i], 0) ** weights[i]
    result *= max(ms_ssim_values[-1], 0) ** weights[-1]

    return round(float(result), 6)


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
