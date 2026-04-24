"""
report.py - CSV Report Generation & Experimental Analysis
Generates detailed reports and prints analysis conclusions to the console.
"""

import os
import pandas as pd


# Extended columns including speed/time metrics
COLUMNS = [
    "Filename", "Format", "Quality_CRF",
    "Original_Size_Bytes", "Compressed_Size_Bytes",
    "Compression_Ratio", "PSNR", "SSIM",
    "Speed_MBps", "Time_Seconds",
]

# Fallback columns (without speed metrics, for backward compatibility)
COLUMNS_LEGACY = [
    "Filename", "Format", "Quality_CRF",
    "Original_Size_Bytes", "Compressed_Size_Bytes",
    "Compression_Ratio", "PSNR", "SSIM",
]


def generate_csv(results: list[dict], output_path: str) -> str:
    """
    Write a list of result dictionaries to a CSV file.
    Automatically detects whether speed/time columns are present.

    Returns the output path.
    """
    if not results:
        return output_path

    # Use extended columns if the data includes Speed_MBps
    sample = results[0]
    columns = COLUMNS if "Speed_MBps" in sample else COLUMNS_LEGACY

    df = pd.DataFrame(results, columns=columns)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def analyze_results(csv_path: str) -> str:
    """
    Read the CSV report and produce an experimental analysis string.

    Comparisons:
    - JPEG vs WebP vs AVIF (avg SSIM, PSNR, compression ratio)
    - CRF values for video
    - Speed analysis
    - Recommendations for optimal parameters
    """
    df = pd.read_csv(csv_path)
    lines = []
    lines.append("=" * 70)
    lines.append("  EXPERIMENTAL ANALYSIS & CONCLUSIONS")
    lines.append("=" * 70)

    # ── Image analysis ───────────────────────────────────────────────────
    image_formats = {"JPEG", "WEBP", "AVIF"}
    img_df = df[df["Format"].isin(image_formats)]
    if not img_df.empty:
        lines.append("\n-- IMAGE COMPRESSION ANALYSIS --\n")
        for fmt in ["JPEG", "WEBP", "AVIF"]:
            subset = img_df[img_df["Format"] == fmt]
            if subset.empty:
                continue
            lines.append(f"  {fmt}:")
            lines.append(f"    Avg SSIM             : {subset['SSIM'].mean():.6f}")
            lines.append(f"    Avg PSNR (dB)        : {subset['PSNR'].mean():.2f}")
            lines.append(f"    Avg Compression Ratio: {subset['Compression_Ratio'].mean():.2f}x")
            if "Speed_MBps" in subset.columns and subset["Speed_MBps"].notna().any():
                lines.append(f"    Avg Speed            : {subset['Speed_MBps'].mean():.1f} MB/s")
            # Best quality setting (highest SSIM with ratio > 1.5)
            good = subset[subset["Compression_Ratio"] > 1.5]
            if not good.empty:
                best = good.loc[good["SSIM"].idxmax()]
                lines.append(f"    * Best Quality@Ratio>1.5x: Q={int(best['Quality_CRF'])} "
                             f"(SSIM={best['SSIM']:.6f}, Ratio={best['Compression_Ratio']:.2f}x)")
            lines.append("")

        # Head-to-head comparisons
        present_formats = [f for f in ["JPEG", "WEBP", "AVIF"] if not img_df[img_df["Format"] == f].empty]
        if len(present_formats) >= 2:
            lines.append("  HEAD-TO-HEAD COMPARISON:")
            for fmt in present_formats:
                sub = img_df[img_df["Format"] == fmt]
                lines.append(
                    f"    {fmt:>4s}: Ratio={sub['Compression_Ratio'].mean():.2f}x  "
                    f"SSIM={sub['SSIM'].mean():.6f}  PSNR={sub['PSNR'].mean():.2f} dB"
                )
            # Find winner by compression ratio
            ratios = {f: img_df[img_df["Format"] == f]["Compression_Ratio"].mean() for f in present_formats}
            winner = max(ratios, key=ratios.get)
            runner_up = sorted(ratios, key=ratios.get, reverse=True)[1] if len(ratios) > 1 else None
            if runner_up:
                pct = ((ratios[winner] - ratios[runner_up]) / ratios[runner_up]) * 100
                lines.append(f"    -> {winner} achieves {pct:.1f}% better compression than {runner_up} on average.")
            lines.append("")

    # ── Video analysis ───────────────────────────────────────────────────
    vid_df = df[~df["Format"].isin(image_formats)]
    # Also exclude autoencoder rows
    vid_df = vid_df[~vid_df["Format"].str.contains("Autoencoder", case=False, na=False)]
    if not vid_df.empty:
        lines.append("-- VIDEO COMPRESSION ANALYSIS --\n")
        for codec in vid_df["Format"].unique():
            subset = vid_df[vid_df["Format"] == codec]
            lines.append(f"  Codec: {codec}")
            for _, row in subset.iterrows():
                line = (
                    f"    CRF {int(row['Quality_CRF']):>2}: "
                    f"Ratio={row['Compression_Ratio']:.2f}x  "
                    f"PSNR={row['PSNR']:.2f} dB  "
                    f"SSIM={row['SSIM']:.6f}"
                )
                if "Time_Seconds" in row and pd.notna(row.get("Time_Seconds")):
                    line += f"  ({row['Time_Seconds']:.1f}s)"
                lines.append(line)
            # Best CRF
            good = subset[subset["SSIM"] >= 0.85]
            if not good.empty:
                best = good.loc[good["Compression_Ratio"].idxmax()]
                lines.append(f"    * Best CRF (SSIM>=0.85): CRF={int(best['Quality_CRF'])} "
                             f"(Ratio={best['Compression_Ratio']:.2f}x)")
            lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────
    lines.append("-- RECOMMENDATIONS --\n")
    if not img_df[img_df["Format"] == "AVIF"].empty:
        lines.append("  - Use AVIF for best compression efficiency (modern browsers & apps).")
    lines.append("  - Use WebP for web delivery — superior compression at equal quality.")
    lines.append("  - Use JPEG where broad compatibility is required (email, legacy systems).")
    lines.append("  - Use H.264 (libx264) for fast encoding & wide device support.")
    lines.append("  - Use H.265 (libx265) when storage savings outweigh encoding time.")
    lines.append("  - Quality 70 (images) and CRF 23 (video) offer the best quality-to-size balance.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report
