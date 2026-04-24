#!/usr/bin/env python3
"""
main.py – CLI Entry Point for the Media Compression System

Usage:
    python main.py <input_folder> [--output <output_folder>] [--codec h264|h265|both]

Examples:
    python main.py ./samples
    python main.py ./samples --output ./results --codec h264
    python main.py ./samples --codec both
"""

import argparse
import logging
import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.batch_processor import run_batch


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Local File & Media Compression System – Batch Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", type=str,
        help="Path to folder containing images and/or videos to compress.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory for compressed files & reports (default: ./output).",
    )
    parser.add_argument(
        "--codec", type=str, default="both",
        choices=["h264", "h265", "both"],
        help="Video codec to test: h264, h265, or both (default: both).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    codec_map = {
        "h264": ["libx264"],
        "h265": ["libx265"],
        "both": ["libx264", "libx265"],
    }
    codecs = codec_map[args.codec]

    report_path = os.path.join(args.output, "compression_report.csv")

    print()
    print("+--------------------------------------------------------------+")
    print("|     LOCAL FILE & MEDIA COMPRESSION SYSTEM                    |")
    print("|     Batch Processor v1.0                                     |")
    print("+--------------------------------------------------------------+")
    print()
    print(f"  Input folder : {os.path.abspath(args.input)}")
    print(f"  Output folder: {os.path.abspath(args.output)}")
    print(f"  Video codecs : {', '.join(codecs)}")
    print(f"  Report       : {os.path.abspath(report_path)}")
    print()

    start = time.time()
    csv = run_batch(
        input_dir=args.input,
        output_dir=args.output,
        report_path=report_path,
        codecs=codecs,
    )
    elapsed = time.time() - start

    if csv:
        print(f"\n[OK] Batch processing complete in {elapsed:.1f}s")
        print(f"   Report: {os.path.abspath(csv)}")
    else:
        print("\n[WARNING] No files were processed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
