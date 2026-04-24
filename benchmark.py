"""
benchmark.py – Kodak Dataset Benchmark Script

Downloads a subset of the Kodak Lossless True Color Image Suite and runs
a full compression sweep to produce reproducible benchmark results.

Usage:
    python benchmark.py
    python benchmark.py --formats JPEG WEBP AVIF --qualities 30 50 70 90
"""

import os
import sys
import time
import argparse
import urllib.request
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.image_compressor import compress_image, SUPPORTED_FORMATS
from src.metrics import (
    file_sizes, compression_ratio,
    compute_image_psnr, compute_image_ssim, compute_image_ms_ssim,
)
from src.report import generate_csv, analyze_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Kodak dataset: 24 uncompressed PNG images (768×512 / 512×768)
# We use a subset of 5 for speed; extend the list for full benchmarking
KODAK_URLS = {
    "kodim01.png": "https://r0k.us/graphics/kodak/kodak/kodim01.png",
    "kodim04.png": "https://r0k.us/graphics/kodak/kodak/kodim04.png",
    "kodim08.png": "https://r0k.us/graphics/kodak/kodak/kodim08.png",
    "kodim13.png": "https://r0k.us/graphics/kodak/kodak/kodim13.png",
    "kodim23.png": "https://r0k.us/graphics/kodak/kodak/kodim23.png",
}

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
KODAK_DIR = os.path.join(BENCHMARK_DIR, "kodak_images")
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "output")


def download_kodak_images(target_dir: str = KODAK_DIR) -> list[str]:
    """Download Kodak images if not already present. Returns list of local paths."""
    os.makedirs(target_dir, exist_ok=True)
    paths = []
    for name, url in KODAK_URLS.items():
        local_path = os.path.join(target_dir, name)
        if os.path.isfile(local_path):
            logger.info(f"  [CACHED] {name}")
        else:
            logger.info(f"  [DOWNLOADING] {name} ...")
            try:
                urllib.request.urlretrieve(url, local_path)
                logger.info(f"  [OK] {name} ({os.path.getsize(local_path) / 1024:.0f} KB)")
            except Exception as e:
                logger.error(f"  [FAIL] Could not download {name}: {e}")
                continue
        paths.append(local_path)
    return paths


def run_benchmark(
    image_paths: list[str],
    formats: list[str] = None,
    qualities: list[int] = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Run compression benchmark on a set of images.

    Returns path to the CSV report.
    """
    if formats is None:
        formats = ["JPEG", "WEBP"]
    if qualities is None:
        qualities = [10, 30, 50, 70, 90]

    results = []
    total = len(image_paths) * len(formats) * len(qualities)
    done = 0

    fmt_ext = {"JPEG": ".jpg", "WEBP": ".webp", "AVIF": ".avif"}

    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        for fmt in formats:
            for q in qualities:
                ext = fmt_ext.get(fmt, ".jpg")
                out_name = f"{basename}_{fmt.lower()}_q{q}{ext}"
                out_path = os.path.join(output_dir, fmt.lower(), out_name)

                try:
                    info = compress_image(img_path, out_path, fmt=fmt, quality=q, size_guard=False)
                    elapsed = info["elapsed_seconds"]
                    orig_sz, comp_sz = file_sizes(img_path, out_path)
                    ratio = compression_ratio(img_path, out_path)
                    psnr = compute_image_psnr(img_path, out_path)
                    ssim_val = compute_image_ssim(img_path, out_path)

                    # Also compute MS-SSIM for benchmark
                    try:
                        ms_ssim = compute_image_ms_ssim(img_path, out_path)
                    except Exception:
                        ms_ssim = None

                    results.append({
                        "Filename": os.path.basename(img_path),
                        "Format": fmt,
                        "Quality_CRF": q,
                        "Original_Size_Bytes": orig_sz,
                        "Compressed_Size_Bytes": comp_sz,
                        "Compression_Ratio": ratio,
                        "PSNR": psnr,
                        "SSIM": ssim_val,
                        "MS_SSIM": ms_ssim,
                        "Speed_MBps": round(orig_sz / (1024 * 1024) / elapsed, 2) if elapsed > 0 else 0,
                        "Time_Seconds": elapsed,
                    })

                    done += 1
                    logger.info(
                        f"  [{done}/{total}] {out_name}  "
                        f"ratio={ratio:.2f}x  PSNR={psnr:.1f}  SSIM={ssim_val:.4f}"
                    )
                except Exception as e:
                    logger.error(f"  [FAIL] {out_name}: {e}")
                    done += 1

    # Generate report
    report_path = os.path.join(output_dir, "kodak_benchmark.csv")
    if results:
        generate_csv(results, report_path)
        analysis = analyze_results(report_path)

        analysis_path = os.path.join(output_dir, "kodak_analysis.txt")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(analysis)

        # Print summary
        print("\n" + analysis)

        # Print MS-SSIM summary table
        import pandas as pd
        df = pd.DataFrame(results)
        if "MS_SSIM" in df.columns and df["MS_SSIM"].notna().any():
            print("\n── MS-SSIM Summary ──")
            for fmt in formats:
                sub = df[(df["Format"] == fmt) & df["MS_SSIM"].notna()]
                if not sub.empty:
                    print(f"  {fmt}: avg MS-SSIM = {sub['MS_SSIM'].mean():.6f}")
            print()

        return report_path
    else:
        logger.warning("No results collected.")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Kodak Image Benchmark for the Media Compression System",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["JPEG", "WEBP"],
        choices=list(SUPPORTED_FORMATS),
        help="Image formats to benchmark (default: JPEG WEBP)",
    )
    parser.add_argument(
        "--qualities", nargs="+", type=int, default=[10, 30, 50, 70, 90],
        help="Quality levels to test (default: 10 30 50 70 90)",
    )
    parser.add_argument(
        "--all-kodak", action="store_true",
        help="Download all 24 Kodak images instead of the default 5",
    )
    args = parser.parse_args()

    # Optionally extend to all 24 images
    if args.all_kodak:
        for i in range(1, 25):
            name = f"kodim{i:02d}.png"
            if name not in KODAK_URLS:
                KODAK_URLS[name] = f"https://r0k.us/graphics/kodak/kodak/{name}"

    print()
    print("+--------------------------------------------------------------+")
    print("|     KODAK IMAGE BENCHMARK                                    |")
    print("|     Media Compression System                                 |")
    print("+--------------------------------------------------------------+")
    print()

    # Step 1: Download
    logger.info("Step 1: Downloading Kodak images...")
    image_paths = download_kodak_images()
    if not image_paths:
        logger.error("No images downloaded. Check your internet connection.")
        return 1

    # Step 2: Benchmark
    logger.info(f"\nStep 2: Running benchmark ({len(image_paths)} images × "
                f"{len(args.formats)} formats × {len(args.qualities)} qualities)...")
    report = run_benchmark(image_paths, formats=args.formats, qualities=args.qualities)

    if report:
        print(f"\n[OK] Benchmark complete!")
        print(f"   Report: {os.path.abspath(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
