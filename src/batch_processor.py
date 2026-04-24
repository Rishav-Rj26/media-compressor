"""
batch_processor.py – Core Batch Processing Engine
Scans an input folder and runs all compression experiments with parallel execution.
"""

import os
import time
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.image_compressor import is_image, compress_image
from src.video_compressor import is_video, compress_video
from src.metrics import (
    file_sizes, compression_ratio, compression_speed,
    compute_image_psnr, compute_image_ssim,
    compute_video_psnr, compute_video_ssim,
)
from src.report import generate_csv, analyze_results

logger = logging.getLogger(__name__)

# ── Experiment parameters ────────────────────────────────────────────────────
IMAGE_FORMATS = ["JPEG", "WEBP"]
IMAGE_QUALITIES = [30, 50, 70, 90]
VIDEO_CRF_VALUES = [18, 23, 28, 32]
VIDEO_CODECS = ["libx264", "libx265"]
FORMAT_EXT = {"JPEG": ".jpg", "WEBP": ".webp", "AVIF": ".avif"}


def _collect_files(input_dir: str) -> tuple[list[str], list[str]]:
    """Walk the input directory and classify files into images / videos."""
    images, videos = [], []
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            if is_image(fpath):
                images.append(fpath)
            elif is_video(fpath):
                videos.append(fpath)
    return images, videos


def _run_single_image_experiment(
    img_path: str, output_dir: str, fmt: str, quality: int
) -> dict | None:
    """Run a single image compression experiment. Returns result dict or None."""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    ext = FORMAT_EXT.get(fmt, ".jpg")
    out_name = f"{basename}_q{quality}{ext}"
    out_path = os.path.join(output_dir, "images", fmt.lower(), out_name)

    try:
        info = compress_image(img_path, out_path, fmt=fmt, quality=quality)
        elapsed = info["elapsed_seconds"]
        orig_sz, comp_sz = file_sizes(img_path, out_path)
        ratio = compression_ratio(img_path, out_path)
        psnr = compute_image_psnr(img_path, out_path)
        ssim_val = compute_image_ssim(img_path, out_path)
        speed = compression_speed(img_path, elapsed)

        result = {
            "Filename": os.path.basename(img_path),
            "Format": fmt,
            "Quality_CRF": quality,
            "Original_Size_Bytes": orig_sz,
            "Compressed_Size_Bytes": comp_sz,
            "Compression_Ratio": ratio,
            "PSNR": psnr,
            "SSIM": ssim_val,
            "Speed_MBps": speed,
            "Time_Seconds": elapsed,
        }
        logger.info(
            f"  [OK] {out_name}  ratio={ratio:.2f}x  PSNR={psnr:.2f}  "
            f"SSIM={ssim_val:.4f}  {speed:.1f} MB/s"
        )
        return result
    except Exception as e:
        logger.error(f"  [FAIL] {out_name}: {e}")
        return None


def _run_single_video_experiment(
    vid_path: str, output_dir: str, codec: str, crf: int
) -> dict | None:
    """Run a single video compression experiment. Returns result dict or None."""
    basename = os.path.splitext(os.path.basename(vid_path))[0]
    codec_label = "H.264" if codec == "libx264" else "H.265"
    out_name = f"{basename}_{codec_label}_crf{crf}.mp4"
    out_path = os.path.join(output_dir, "videos", codec_label.lower(), out_name)

    try:
        info = compress_video(vid_path, out_path, crf=crf, codec=codec)
        elapsed = info["elapsed_seconds"]
        orig_sz, comp_sz = file_sizes(vid_path, out_path)
        ratio = compression_ratio(vid_path, out_path)
        psnr = compute_video_psnr(vid_path, out_path)
        ssim_val = compute_video_ssim(vid_path, out_path)
        speed = compression_speed(vid_path, elapsed)

        result = {
            "Filename": os.path.basename(vid_path),
            "Format": codec_label,
            "Quality_CRF": crf,
            "Original_Size_Bytes": orig_sz,
            "Compressed_Size_Bytes": comp_sz,
            "Compression_Ratio": ratio,
            "PSNR": psnr,
            "SSIM": ssim_val,
            "Speed_MBps": speed,
            "Time_Seconds": elapsed,
        }
        logger.info(
            f"  [OK] {out_name}  ratio={ratio:.2f}x  PSNR={psnr:.2f}  "
            f"SSIM={ssim_val:.4f}  {elapsed:.1f}s"
        )
        return result
    except Exception as e:
        logger.error(f"  [FAIL] {out_name}: {e}")
        return None


def run_batch(
    input_dir: str,
    output_dir: str = "output",
    report_path: str = "output/compression_report.csv",
    codecs: Optional[list[str]] = None,
    formats: Optional[list[str]] = None,
    qualities: Optional[list[int]] = None,
    crfs: Optional[list[int]] = None,
    max_workers: int = 4,
    progress_callback=None,
) -> str:
    """
    Run the full batch compression experiment with parallel image processing.

    Parameters
    ----------
    input_dir   : folder containing source images / videos
    output_dir  : folder for compressed outputs
    report_path : where to write the CSV report
    codecs      : list of video codecs to test (default: H.264 + H.265)
    formats     : list of image formats (default: JPEG + WEBP)
    qualities   : list of image quality levels (default: 30, 50, 70, 90)
    crfs        : list of video CRF values (default: 18, 23, 28, 32)
    max_workers : max threads for parallel image compression
    progress_callback : callable(done, total, message) for progress updates

    Returns
    -------
    Path to the generated CSV report.
    """
    if codecs is None:
        codecs = VIDEO_CODECS
    if formats is None:
        formats = IMAGE_FORMATS
    if qualities is None:
        qualities = IMAGE_QUALITIES
    if crfs is None:
        crfs = VIDEO_CRF_VALUES

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images, videos = _collect_files(input_dir)
    total_files = len(images) + len(videos)
    logger.info(f"Found {len(images)} image(s) and {len(videos)} video(s) in '{input_dir}'")
    if total_files == 0:
        logger.warning("No supported files found. Exiting.")
        return ""

    results: list[dict] = []
    total_experiments = (
        len(images) * len(formats) * len(qualities) +
        len(videos) * len(codecs) * len(crfs)
    )
    done = 0

    # ── Image experiments (parallel) ─────────────────────────────────────
    if images:
        logger.info(f"Processing {len(images)} images with {max_workers} threads...")
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for img_path in images:
                for fmt in formats:
                    for q in qualities:
                        future = pool.submit(
                            _run_single_image_experiment,
                            img_path, output_dir, fmt, q,
                        )
                        futures[future] = (os.path.basename(img_path), fmt, q)

            for future in as_completed(futures):
                name, fmt, q = futures[future]
                result = future.result()
                if result:
                    results.append(result)
                done += 1
                if progress_callback:
                    progress_callback(done, total_experiments, f"{name} → {fmt} Q{q}")
                logger.info(f"[{done}/{total_experiments}] Done: {name} {fmt} Q{q}")

    # ── Video experiments (sequential – FFmpeg is CPU-intensive) ──────────
    for vid_path in videos:
        basename = os.path.basename(vid_path)
        for codec in codecs:
            for crf in crfs:
                result = _run_single_video_experiment(
                    vid_path, output_dir, codec, crf,
                )
                if result:
                    results.append(result)
                done += 1
                codec_label = "H.264" if codec == "libx264" else "H.265"
                if progress_callback:
                    progress_callback(done, total_experiments, f"{basename} → {codec_label} CRF{crf}")
                logger.info(f"[{done}/{total_experiments}] Done: {basename} {codec_label} CRF{crf}")

    # ── Report ───────────────────────────────────────────────────────────
    if results:
        csv_path = generate_csv(results, report_path)
        logger.info(f"\nCSV report saved to: {csv_path}")
        analysis = analyze_results(csv_path)
        # Save analysis to text file alongside CSV
        analysis_path = os.path.splitext(report_path)[0] + "_analysis.txt"
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        logger.info(f"Analysis saved to: {analysis_path}")
        print("\n" + analysis)
        return csv_path
    else:
        logger.warning("No results collected.")
        return ""
