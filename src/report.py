"""
report.py - CSV Report Generation & Experimental Analysis
Generates detailed reports and prints analysis conclusions to the console.
"""

import os
import pandas as pd


def generate_csv(results: list[dict], output_path: str) -> str:
    """
    Write a list of result dictionaries to a CSV file.

    Expected keys per dict:
        Filename, Format, Quality_CRF, Original_Size_Bytes,
        Compressed_Size_Bytes, Compression_Ratio, PSNR, SSIM

    Returns the output path.
    """
    columns = [
        "Filename", "Format", "Quality_CRF",
        "Original_Size_Bytes", "Compressed_Size_Bytes",
        "Compression_Ratio", "PSNR", "SSIM"
    ]
    df = pd.DataFrame(results, columns=columns)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def analyze_results(csv_path: str) -> str:
    """
    Read the CSV report and produce an experimental analysis string.

    Comparisons:
    - JPEG vs WebP (avg SSIM, PSNR, compression ratio)
    - CRF values for video
    - Recommendations for optimal parameters
    """
    df = pd.read_csv(csv_path)
    lines = []
    lines.append("=" * 70)
    lines.append("  EXPERIMENTAL ANALYSIS & CONCLUSIONS")
    lines.append("=" * 70)

    # ── Image analysis ───────────────────────────────────────────────────
    img_df = df[df["Format"].isin(["JPEG", "WEBP"])]
    if not img_df.empty:
        lines.append("\n-- IMAGE COMPRESSION ANALYSIS --\n")
        for fmt in ["JPEG", "WEBP"]:
            subset = img_df[img_df["Format"] == fmt]
            if subset.empty:
                continue
            lines.append(f"  {fmt}:")
            lines.append(f"    Avg SSIM             : {subset['SSIM'].mean():.6f}")
            lines.append(f"    Avg PSNR (dB)        : {subset['PSNR'].mean():.2f}")
            lines.append(f"    Avg Compression Ratio: {subset['Compression_Ratio'].mean():.2f}x")
            # Best quality setting (highest SSIM with ratio > 1.5)
            good = subset[subset["Compression_Ratio"] > 1.5]
            if not good.empty:
                best = good.loc[good["SSIM"].idxmax()]
                lines.append(f"    * Best Quality@Ratio>1.5x: Q={int(best['Quality_CRF'])} "
                             f"(SSIM={best['SSIM']:.6f}, Ratio={best['Compression_Ratio']:.2f}x)")
            lines.append("")

        # Head-to-head
        jpeg_df = img_df[img_df["Format"] == "JPEG"]
        webp_df = img_df[img_df["Format"] == "WEBP"]
        if not jpeg_df.empty and not webp_df.empty:
            lines.append("  HEAD-TO-HEAD (JPEG vs WebP):")
            j_ratio = jpeg_df["Compression_Ratio"].mean()
            w_ratio = webp_df["Compression_Ratio"].mean()
            lines.append(f"    Avg Ratio - JPEG: {j_ratio:.2f}x  |  WebP: {w_ratio:.2f}x")
            if w_ratio > j_ratio:
                pct = ((w_ratio - j_ratio) / j_ratio) * 100
                lines.append(f"    -> WebP achieves {pct:.1f}% better compression on average.")
            else:
                lines.append(f"    -> JPEG achieves equal or better compression.")
            j_ssim = jpeg_df["SSIM"].mean()
            w_ssim = webp_df["SSIM"].mean()
            lines.append(f"    Avg SSIM - JPEG: {j_ssim:.6f}  |  WebP: {w_ssim:.6f}")
            lines.append("")

    # ── Video analysis ───────────────────────────────────────────────────
    vid_df = df[~df["Format"].isin(["JPEG", "WEBP"])]
    if not vid_df.empty:
        lines.append("-- VIDEO COMPRESSION ANALYSIS --\n")
        for codec in vid_df["Format"].unique():
            subset = vid_df[vid_df["Format"] == codec]
            lines.append(f"  Codec: {codec}")
            for _, row in subset.iterrows():
                lines.append(
                    f"    CRF {int(row['Quality_CRF']):>2}: "
                    f"Ratio={row['Compression_Ratio']:.2f}x  "
                    f"PSNR={row['PSNR']:.2f} dB  "
                    f"SSIM={row['SSIM']:.6f}"
                )
            # Best CRF
            good = subset[subset["SSIM"] >= 0.85]
            if not good.empty:
                best = good.loc[good["Compression_Ratio"].idxmax()]
                lines.append(f"    * Best CRF (SSIM>=0.85): CRF={int(best['Quality_CRF'])} "
                             f"(Ratio={best['Compression_Ratio']:.2f}x)")
            lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────
    lines.append("-- RECOMMENDATIONS --\n")
    lines.append("  - Use WebP for web delivery - superior compression at equal quality.")
    lines.append("  - Use JPEG where broad compatibility is required (email, legacy systems).")
    lines.append("  - Use H.264 (libx264) for fast encoding & wide device support.")
    lines.append("  - Use H.265 (libx265) when storage savings outweigh encoding time.")
    lines.append("  - Quality 70 (images) and CRF 23 (video) offer the best quality-to-size balance.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report
