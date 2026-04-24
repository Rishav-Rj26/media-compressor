"""
batch_processor.py – Core Batch Processing Engine
Scans an input folder and runs all compression experiments automatically.
"""

import os
import time
import logging
from typing import Optional

from src.image_compressor import is_image, compress_image
from src.video_compressor import is_video, compress_video
from src.metrics import (
    file_sizes, compression_ratio,
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
FORMAT_EXT = {"JPEG": ".jpg", "WEBP": ".webp"}


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


def run_batch(
    input_dir: str,
    output_dir: str = "output",
    report_path: str = "output/compression_report.csv",
    codecs: Optional[list[str]] = None,
) -> str:
    """
    Run the full batch compression experiment.

    Parameters
    ----------
    input_dir   : folder containing source images / videos
    output_dir  : folder for compressed outputs
    report_path : where to write the CSV report
    codecs      : list of video codecs to test (default: H.264 + H.265)

    Returns
    -------
    Path to the generated CSV report.
    """
    if codecs is None:
        codecs = VIDEO_CODECS

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images, videos = _collect_files(input_dir)
    total_files = len(images) + len(videos)
    logger.info(f"Found {len(images)} image(s) and {len(videos)} video(s) in '{input_dir}'")
    if total_files == 0:
        logger.warning("No supported files found. Exiting.")
        return ""

    results: list[dict] = []
    processed = 0

    # ── Image experiments ────────────────────────────────────────────────
    for img_path in images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        for fmt in IMAGE_FORMATS:
            for q in IMAGE_QUALITIES:
                out_name = f"{basename}_q{q}{FORMAT_EXT[fmt]}"
                out_path = os.path.join(output_dir, "images", fmt.lower(), out_name)
                try:
                    compress_image(img_path, out_path, fmt=fmt, quality=q)
                    orig_sz, comp_sz = file_sizes(img_path, out_path)
                    ratio = compression_ratio(img_path, out_path)
                    psnr = compute_image_psnr(img_path, out_path)
                    ssim_val = compute_image_ssim(img_path, out_path)
                    results.append({
                        "Filename": os.path.basename(img_path),
                        "Format": fmt,
                        "Quality_CRF": q,
                        "Original_Size_Bytes": orig_sz,
                        "Compressed_Size_Bytes": comp_sz,
                        "Compression_Ratio": ratio,
                        "PSNR": psnr,
                        "SSIM": ssim_val,
                    })
                    logger.info(f"  [OK] {out_name}  ratio={ratio:.2f}x  PSNR={psnr:.2f}  SSIM={ssim_val:.4f}")
                except Exception as e:
                    logger.error(f"  [FAIL] {out_name}: {e}")
        processed += 1
        logger.info(f"[{processed}/{total_files}] Done: {os.path.basename(img_path)}")

    # ── Video experiments ────────────────────────────────────────────────
    for vid_path in videos:
        basename = os.path.splitext(os.path.basename(vid_path))[0]
        for codec in codecs:
            codec_label = "H.264" if codec == "libx264" else "H.265"
            for crf in VIDEO_CRF_VALUES:
                out_name = f"{basename}_{codec_label}_crf{crf}.mp4"
                out_path = os.path.join(output_dir, "videos", codec_label.lower(), out_name)
                try:
                    compress_video(vid_path, out_path, crf=crf, codec=codec)
                    orig_sz, comp_sz = file_sizes(vid_path, out_path)
                    ratio = compression_ratio(vid_path, out_path)
                    psnr = compute_video_psnr(vid_path, out_path)
                    ssim_val = compute_video_ssim(vid_path, out_path)
                    results.append({
                        "Filename": os.path.basename(vid_path),
                        "Format": codec_label,
                        "Quality_CRF": crf,
                        "Original_Size_Bytes": orig_sz,
                        "Compressed_Size_Bytes": comp_sz,
                        "Compression_Ratio": ratio,
                        "PSNR": psnr,
                        "SSIM": ssim_val,
                    })
                    logger.info(f"  [OK] {out_name}  ratio={ratio:.2f}x  PSNR={psnr:.2f}  SSIM={ssim_val:.4f}")
                except Exception as e:
                    logger.error(f"  [FAIL] {out_name}: {e}")
        processed += 1
        logger.info(f"[{processed}/{total_files}] Done: {os.path.basename(vid_path)}")

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
