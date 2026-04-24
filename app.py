"""
app.py - Media Compression System Web UI
Beautiful Streamlit-based interface for image & video compression.
Refactored: utility functions and chart builders extracted to src/ui_helpers.py.
"""

import os
import io
import sys
import time
import shutil
import streamlit as st
import pandas as pd
from PIL import Image

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.image_compressor import compress_image, is_image, SUPPORTED_EXTENSIONS as IMG_EXT
from src.video_compressor import compress_video, is_video, SUPPORTED_EXTENSIONS as VID_EXT
from src.metrics import (
    file_sizes, compression_ratio, compression_speed,
    compute_image_psnr, compute_image_ssim,
    compute_video_psnr, compute_video_ssim,
)
from src.report import generate_csv, analyze_results
from src.ui_helpers import (
    format_bytes, create_zip, save_upload, scan_folder, img_to_b64,
    chart_compression_ratio_by_quality, chart_ssim_by_quality,
    chart_ssim_vs_ratio, chart_psnr_by_quality,
    chart_video_ratio_vs_crf, chart_video_ssim_vs_crf,
    chart_speed_comparison,
    build_slider_html,
)

# Neural compression (optional – requires PyTorch)
try:
    from src.autoencoder import (
        TORCH_AVAILABLE, train_autoencoder, compress_with_autoencoder,
        get_latent_size, save_model, load_model,
    )
except ImportError:
    TORCH_AVAILABLE = False

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Media Compression Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global – exclude Material Icons so Streamlit icon-font glyphs render correctly */
html, body, [class*="st-"]:not([class*="material"]):not([data-testid="stIconMaterial"]) {
    font-family: 'Inter', sans-serif;
}
/* Ensure Material Icons keep their own font */
.material-symbols-rounded,
.material-icons,
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* Hide Streamlit branding but keep sidebar toggle visible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}
/* Hide the header decoration bar but keep the collapse button */
header[data-testid="stHeader"]::after {
    display: none;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hero header */
.hero-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(124, 58, 237, 0.3);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), 0 0 40px rgba(124, 58, 237, 0.1);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #7c3aed, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    position: relative;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    position: relative;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #1a2332);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(124, 58, 237, 0.2);
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.metric-card:hover {
    border-color: rgba(124, 58, 237, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-info {
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

/* File upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(124, 58, 237, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(124, 58, 237, 0.03) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124, 58, 237, 0.6) !important;
    background: rgba(124, 58, 237, 0.06) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(124, 58, 237, 0.15);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #a78bfa;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(124, 58, 237, 0.3);
}

/* Result row */
.result-row {
    background: linear-gradient(145deg, #1e293b, #1a2332);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s ease;
}
.result-row:hover {
    border-color: rgba(124, 58, 237, 0.3);
}

/* Progress animation */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}
.shimmer {
    background: linear-gradient(90deg, #1e293b 25%, #2d3a4f 50%, #1e293b 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite linear;
    border-radius: 8px;
    height: 20px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    background: rgba(124, 58, 237, 0.05);
    border: 1px solid rgba(124, 58, 237, 0.15);
}
.stTabs [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.2) !important;
    border-color: rgba(124, 58, 237, 0.5) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #7c3aed);
    box-shadow: 0 6px 25px rgba(124, 58, 237, 0.4);
    transform: translateY(-1px);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #34d399, #10b981) !important;
    box-shadow: 0 6px 25px rgba(16, 185, 129, 0.4) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 1rem;
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Before/After Slider */
.ba-slider {
    position: relative;
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
    border: 1px solid rgba(124, 58, 237, 0.3);
}
.ba-slider img {
    display: block;
    width: 100%;
    height: auto;
}
.ba-slider .ba-after {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    overflow: hidden;
    border-right: 3px solid #7c3aed;
}
.ba-slider .ba-after img {
    display: block;
    height: 100%;
    width: auto;
    max-width: none;
}

/* Savings badge */
.savings-card {
    background: linear-gradient(135deg, #065f46, #064e3b);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    border: 1px solid rgba(16, 185, 129, 0.4);
    text-align: center;
    margin: 1rem 0;
}
.savings-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #34d399;
}
.savings-label {
    font-size: 0.9rem;
    color: #a7f3d0;
    margin-top: 0.3rem;
}

/* Autoencoder badge */
.ae-badge {
    background: linear-gradient(135deg, #7c3aed20, #6d28d920);
    border: 1px solid #7c3aed50;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #c4b5fd;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
# Helper functions (format_bytes, create_zip, save_upload, scan_folder, charts)
# are now imported from src.ui_helpers

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Media Compression Studio</div>
    <div class="hero-subtitle">
        Compress images & videos with precision. Compare JPEG vs WebP, tune CRF values,
        and analyze quality metrics -- all from your browser.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    st.markdown("### Image Compression")
    img_formats = st.multiselect(
        "Formats",
        ["JPEG", "WEBP", "AVIF"],
        default=["JPEG", "WEBP"],
        help="Select image output formats (AVIF requires Pillow 10.1+)"
    )
    img_qualities = st.multiselect(
        "Quality Levels",
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        default=[30, 50, 70, 90],
        help="Quality parameter (1-100)"
    )

    st.markdown("---")
    st.markdown("### Video Compression")
    vid_codecs = st.multiselect(
        "Codecs",
        ["H.264 (libx264)", "H.265 (libx265)"],
        default=["H.264 (libx264)"],
        help="Video encoding codecs"
    )
    vid_crfs = st.multiselect(
        "CRF Values",
        [14, 18, 23, 28, 32, 36, 40],
        default=[18, 23, 28, 32],
        help="Constant Rate Factor (lower = better quality)"
    )

    st.markdown("---")
    st.markdown("### 🧠 Neural Compression")
    if TORCH_AVAILABLE:
        enable_autoencoder = st.checkbox(
            "Enable Autoencoder",
            value=False,
            help="Train a neural network to learn compression on your images"
        )
        if enable_autoencoder:
            ae_bottleneck = st.slider(
                "Bottleneck Size",
                min_value=2, max_value=32, value=8, step=2,
                help="Fewer channels = more compression, lower quality"
            )
            ae_epochs = st.slider(
                "Training Epochs",
                min_value=10, max_value=200, value=50, step=10,
                help="More epochs = better quality, longer training"
            )
    else:
        enable_autoencoder = False
        st.markdown(
            "<div class='ae-badge'>⚠️ Install PyTorch to enable neural compression:<br>"
            "<code>pip install torch torchvision</code></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "<div style='color:#64748b; font-size:0.85rem;'>"
        "Built for offline media compression experiments. "
        "Supports traditional codecs + neural autoencoder with PSNR, SSIM analysis."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main Content Tabs ────────────────────────────────────────────────────────
tab_compress, tab_results, tab_analysis, tab_compare = st.tabs([
    "Compress Files",
    "Results & Metrics",
    "Analysis & Charts",
    "Visual Compare",
])

# ── TAB 1: Compress Files ────────────────────────────────────────────────────
with tab_compress:
    st.markdown('<div class="section-header">Upload Your Media Files</div>', unsafe_allow_html=True)

    # ── Input Mode Toggle ────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input Source",
        ["📁 Upload Files", "📂 Folder Path"],
        horizontal=True,
        help="Upload individual files or point to an entire folder on your machine.",
    )

    # scan_folder is now imported from src.ui_helpers

    # ── Collect files from chosen input mode ─────────────────────────────────
    # These will be populated with (display_name, source_path) tuples
    image_entries = []   # list of (name, abs_path)
    video_entries = []   # list of (name, abs_path)
    uploaded_files = []  # only used for the upload mode
    folder_mode = (input_mode == "📂 Folder Path")
    has_files = False

    if folder_mode:
        st.markdown(
            "<div style='color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;'>"
            "Enter the full path to a folder on your computer containing images and/or videos."
            "</div>",
            unsafe_allow_html=True,
        )
        col_path, col_recurse = st.columns([4, 1])
        with col_path:
            folder_input = st.text_input(
                "Folder Path",
                placeholder=r"e.g. C:\Users\You\Pictures\VacationPhotos",
                label_visibility="collapsed",
            )
        with col_recurse:
            scan_recursive = st.checkbox("Include subfolders", value=True)

        if folder_input:
            folder_input = folder_input.strip().strip('"').strip("'")
            if not os.path.isdir(folder_input):
                st.error(f"Folder not found: **{folder_input}**")
            else:
                img_paths, vid_paths = scan_folder(folder_input, IMG_EXT, VID_EXT, recursive=scan_recursive)
                image_entries = [(os.path.basename(p), p) for p in sorted(img_paths)]
                video_entries = [(os.path.basename(p), p) for p in sorted(vid_paths)]
                has_files = bool(image_entries or video_entries)

                if not has_files:
                    st.warning("No supported media files found in this folder.")
    else:
        uploaded_files = st.file_uploader(
            "Drag and drop images or videos here",
            type=list({ext.lstrip(".") for ext in IMG_EXT | VID_EXT}),
            accept_multiple_files=True,
            help="Supported: JPG, PNG, BMP, TIFF, WebP, MP4, AVI, MOV, MKV, WMV, FLV, WebM",
        )
        if uploaded_files:
            for f in uploaded_files:
                ext = f".{f.name.rsplit('.', 1)[-1].lower()}"
                if ext in IMG_EXT:
                    image_entries.append((f.name, f))   # UploadedFile obj
                elif ext in VID_EXT:
                    video_entries.append((f.name, f))
            has_files = True

    # ── Show summary + preview ───────────────────────────────────────────────
    total_files = len(image_entries) + len(video_entries)
    if has_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{total_files}</div>'
                f'<div class="metric-label">Total Files</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(image_entries)}</div>'
                f'<div class="metric-label">Images</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(video_entries)}</div>'
                f'<div class="metric-label">Videos</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── File list (folder mode) ──────────────────────────────────────
        if folder_mode and has_files:
            with st.expander(f"📋 Found {total_files} media files", expanded=False):
                listing_data = []
                for name, path in image_entries:
                    listing_data.append({"File": name, "Type": "🖼️ Image", "Size": format_bytes(os.path.getsize(path))})
                for name, path in video_entries:
                    listing_data.append({"File": name, "Type": "🎬 Video", "Size": format_bytes(os.path.getsize(path))})
                st.dataframe(pd.DataFrame(listing_data), width='stretch', hide_index=True, height=300)

        # ── Image previews ───────────────────────────────────────────────
        preview_entries = image_entries[:4]
        if preview_entries:
            with st.expander("Preview Images", expanded=True):
                preview_cols = st.columns(min(len(preview_entries), 4))
                for i, (name, source) in enumerate(preview_entries):
                    with preview_cols[i]:
                        if folder_mode:
                            img = Image.open(source)
                            file_sz = os.path.getsize(source)
                        else:
                            img = Image.open(source)
                            file_sz = source.size
                        st.image(img, caption=name, width="stretch")
                        st.markdown(
                            f"<div style='text-align:center; color:#64748b; font-size:0.8rem;'>"
                            f"{img.size[0]}x{img.size[1]} | {format_bytes(file_sz)}</div>",
                            unsafe_allow_html=True,
                        )

        # ── Compress button ──────────────────────────────────────────────
        st.markdown("")
        compress_clicked = st.button(
            "🚀 Start Compression",
            width='stretch',
            type="primary",
        )

        if compress_clicked:
            if not img_formats and not vid_codecs:
                st.error("Please select at least one format/codec in the sidebar.")
            elif not img_qualities and image_entries:
                st.error("Please select at least one quality level in the sidebar.")
            elif not vid_crfs and video_entries:
                st.error("Please select at least one CRF value in the sidebar.")
            else:
                # Clean previous output
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR, ignore_errors=True)

                results = []
                compressed_files = []
                total_experiments = (
                    len(image_entries) * len(img_formats) * len(img_qualities) +
                    len(video_entries) * len(vid_codecs) * len(vid_crfs)
                )

                progress_bar = st.progress(0, text="Starting compression...")
                status_text = st.empty()
                done = 0

                # ── Process Images ───────────────────────────────────────
                for img_name, img_source in image_entries:
                    # Resolve source path: folder mode → already a path; upload → save to temp
                    if folder_mode:
                        src_path = img_source
                    else:
                        src_path = save_upload(img_source, TEMP_DIR)
                    basename = os.path.splitext(img_name)[0]

                    for fmt in img_formats:
                        ext = {"JPEG": ".jpg", "WEBP": ".webp", "AVIF": ".avif"}.get(fmt, ".jpg")
                        for q in sorted(img_qualities):
                            out_name = f"{basename}_q{q}{ext}"
                            out_path = os.path.join(OUTPUT_DIR, "images", fmt.lower(), out_name)
                            try:
                                status_text.markdown(
                                    f"<span class='status-badge badge-info'>"
                                    f"Processing</span> **{img_name}** → {fmt} Q{q}",
                                    unsafe_allow_html=True,
                                )
                                info = compress_image(src_path, out_path, fmt=fmt, quality=q)
                                elapsed = info["elapsed_seconds"]
                                orig_sz, comp_sz = file_sizes(src_path, out_path)
                                ratio = compression_ratio(src_path, out_path)
                                psnr = compute_image_psnr(src_path, out_path)
                                ssim_val = compute_image_ssim(src_path, out_path)
                                speed = compression_speed(src_path, elapsed)
                                results.append({
                                    "Filename": img_name,
                                    "Format": fmt,
                                    "Quality_CRF": q,
                                    "Original_Size_Bytes": orig_sz,
                                    "Compressed_Size_Bytes": comp_sz,
                                    "Compression_Ratio": ratio,
                                    "PSNR": psnr,
                                    "SSIM": ssim_val,
                                    "Speed_MBps": speed,
                                    "Time_Seconds": elapsed,
                                })
                                compressed_files.append(out_path)
                            except Exception as e:
                                st.warning(f"Failed: {out_name} — {e}")
                            done += 1
                            progress_bar.progress(
                                done / total_experiments,
                                text=f"Processing… ({done}/{total_experiments})",
                            )

                # ── Process Videos ────────────────────────────────────────
                codec_map = {
                    "H.264 (libx264)": ("libx264", "H.264"),
                    "H.265 (libx265)": ("libx265", "H.265"),
                }
                for vid_name, vid_source in video_entries:
                    if folder_mode:
                        src_path = vid_source
                    else:
                        src_path = save_upload(vid_source, TEMP_DIR)
                    basename = os.path.splitext(vid_name)[0]

                    for codec_choice in vid_codecs:
                        codec, codec_label = codec_map[codec_choice]
                        for crf in sorted(vid_crfs):
                            out_name = f"{basename}_{codec_label}_crf{crf}.mp4"
                            out_path = os.path.join(
                                OUTPUT_DIR, "videos", codec_label.lower(), out_name
                            )
                            try:
                                status_text.markdown(
                                    f"<span class='status-badge badge-info'>"
                                    f"Encoding</span> **{vid_name}** → {codec_label} CRF {crf}",
                                    unsafe_allow_html=True,
                                )
                                info = compress_video(src_path, out_path, crf=crf, codec=codec)
                                elapsed = info["elapsed_seconds"]
                                orig_sz, comp_sz = file_sizes(src_path, out_path)
                                ratio = compression_ratio(src_path, out_path)
                                psnr = compute_video_psnr(src_path, out_path)
                                ssim_val = compute_video_ssim(src_path, out_path)
                                speed = compression_speed(src_path, elapsed)
                                results.append({
                                    "Filename": vid_name,
                                    "Format": codec_label,
                                    "Quality_CRF": crf,
                                    "Original_Size_Bytes": orig_sz,
                                    "Compressed_Size_Bytes": comp_sz,
                                    "Compression_Ratio": ratio,
                                    "PSNR": psnr,
                                    "SSIM": ssim_val,
                                    "Speed_MBps": speed,
                                    "Time_Seconds": elapsed,
                                })
                                compressed_files.append(out_path)
                            except Exception as e:
                                st.warning(f"Failed: {out_name} — {e}")
                            done += 1
                            progress_bar.progress(
                                done / total_experiments,
                                text=f"Processing… ({done}/{total_experiments})",
                            )

                progress_bar.progress(1.0, text="✅ Traditional compression complete!")
                status_text.markdown(
                    "<span class='status-badge badge-success'>Complete</span> "
                    "All codec-based files processed!",
                    unsafe_allow_html=True,
                )

                # ── Autoencoder Compression (if enabled) ──────────────────
                ae_results = []
                if enable_autoencoder and image_entries:
                    st.markdown("")
                    st.markdown('<div class="section-header">🧠 Neural Autoencoder Compression</div>', unsafe_allow_html=True)

                    ae_status = st.empty()
                    ae_progress = st.progress(0, text="Training autoencoder...")

                    # Collect image paths for training
                    train_paths = []
                    for img_name, img_source in image_entries:
                        if folder_mode:
                            train_paths.append(img_source)
                        else:
                            p = os.path.join(TEMP_DIR, img_name)
                            if os.path.isfile(p):
                                train_paths.append(p)

                    # Train
                    def ae_progress_cb(epoch, total, loss):
                        ae_progress.progress(
                            epoch / total,
                            text=f"Training epoch {epoch}/{total} — loss: {loss:.6f}",
                        )

                    try:
                        ae_model = train_autoencoder(
                            train_paths,
                            bottleneck_channels=ae_bottleneck,
                            epochs=ae_epochs,
                            progress_callback=ae_progress_cb,
                        )
                        ae_progress.progress(1.0, text="✅ Training complete!")

                        # Save model
                        model_path = os.path.join(OUTPUT_DIR, "autoencoder_model.pth")
                        save_model(ae_model, model_path)

                        # Compress each image with the autoencoder
                        for i, (img_name, img_source) in enumerate(image_entries):
                            if folder_mode:
                                src_path = img_source
                            else:
                                src_path = os.path.join(TEMP_DIR, img_name)

                            basename = os.path.splitext(img_name)[0]
                            out_name = f"{basename}_autoencoder.png"
                            out_path = os.path.join(OUTPUT_DIR, "images", "autoencoder", out_name)

                            ae_status.markdown(
                                f"<span class='status-badge badge-info'>Neural</span> **{img_name}** → Autoencoder (bottleneck={ae_bottleneck})",
                                unsafe_allow_html=True,
                            )

                            compress_with_autoencoder(ae_model, src_path, out_path)
                            orig_sz, comp_sz = file_sizes(src_path, out_path)
                            latent_sz = get_latent_size(ae_model, src_path)
                            psnr = compute_image_psnr(src_path, out_path)
                            ssim_val = compute_image_ssim(src_path, out_path)

                            ae_row = {
                                "Filename": img_name,
                                "Format": f"Autoencoder (b={ae_bottleneck})",
                                "Quality_CRF": ae_bottleneck,
                                "Original_Size_Bytes": orig_sz,
                                "Compressed_Size_Bytes": latent_sz,  # true compressed size
                                "Compression_Ratio": round(orig_sz / latent_sz, 4) if latent_sz > 0 else 0,
                                "PSNR": psnr,
                                "SSIM": ssim_val,
                            }
                            results.append(ae_row)
                            ae_results.append(ae_row)
                            compressed_files.append(out_path)

                        ae_status.markdown(
                            "<span class='status-badge badge-success'>Complete</span> "
                            "Autoencoder compression finished!",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"Autoencoder failed: {e}")

                # Save results
                if results:
                    csv_path = os.path.join(OUTPUT_DIR, "compression_report.csv")
                    generate_csv(results, csv_path)
                    st.session_state["results"] = results
                    st.session_state["csv_path"] = csv_path
                    st.session_state["compressed_files"] = compressed_files

                    # Store source paths for visual comparison
                    source_map = {}
                    for img_name, img_source in image_entries:
                        if folder_mode:
                            source_map[img_name] = img_source
                        else:
                            source_map[img_name] = os.path.join(TEMP_DIR, img_name)
                    st.session_state["source_map"] = source_map

                    # ── Total Savings Summary ─────────────────────────────
                    df = pd.DataFrame(results)
                    total_original = df.groupby("Filename")["Original_Size_Bytes"].first().sum()
                    best_per_file = df.loc[df.groupby("Filename")["Compressed_Size_Bytes"].idxmin()]
                    total_best_compressed = best_per_file["Compressed_Size_Bytes"].sum()
                    total_saved = total_original - total_best_compressed
                    savings_pct = (total_saved / total_original * 100) if total_original > 0 else 0

                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.markdown(
                            f'<div class="savings-card">'
                            f'<div class="savings-value">{format_bytes(total_saved)}</div>'
                            f'<div class="savings-label">Total Space Saved (Best Config)</div></div>',
                            unsafe_allow_html=True,
                        )
                    with sc2:
                        st.markdown(
                            f'<div class="savings-card">'
                            f'<div class="savings-value">{savings_pct:.1f}%</div>'
                            f'<div class="savings-label">Size Reduction</div></div>',
                            unsafe_allow_html=True,
                        )
                    with sc3:
                        st.markdown(
                            f'<div class="savings-card">'
                            f'<div class="savings-value">{len(results)}</div>'
                            f'<div class="savings-label">Total Experiments Run</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Summary metrics
                    st.markdown("")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{len(results)}</div>'
                            f'<div class="metric-label">Experiments</div></div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        avg_ratio = df["Compression_Ratio"].mean()
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{avg_ratio:.1f}x</div>'
                            f'<div class="metric-label">Avg Compression</div></div>',
                            unsafe_allow_html=True,
                        )
                    with c3:
                        avg_psnr = df["PSNR"].mean()
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{avg_psnr:.1f} dB</div>'
                            f'<div class="metric-label">Avg PSNR</div></div>',
                            unsafe_allow_html=True,
                        )
                    with c4:
                        avg_ssim = df["SSIM"].mean()
                        st.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-value">{avg_ssim:.4f}</div>'
                            f'<div class="metric-label">Avg SSIM</div></div>',
                            unsafe_allow_html=True,
                        )

                    # Download buttons
                    st.markdown("")
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        zip_data = create_zip(compressed_files)
                        st.download_button(
                            "📦 Download All Compressed Files (ZIP)",
                            data=zip_data,
                            file_name="compressed_files.zip",
                            mime="application/zip",
                            width='stretch',
                        )
                    with dl2:
                        csv_data = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📊 Download CSV Report",
                            data=csv_data,
                            file_name="compression_report.csv",
                            mime="text/csv",
                            width='stretch',
                        )

                    st.success("Switch to the **Results & Metrics**, **Analysis & Charts**, or **Visual Compare** tab for detailed insights!")

    else:
        # Empty state
        if folder_mode:
            st.markdown(
                "<div style='text-align:center; padding: 3rem; color: #475569;'>"
                "<div style='font-size: 3rem; margin-bottom: 1rem;'>📂</div>"
                "<div style='font-size: 1.1rem;'>Enter a folder path above to scan for media files.</div>"
                "<div style='font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;'>"
                "All images and videos in the folder will be compressed at once.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center; padding: 3rem; color: #475569;'>"
                "<div style='font-size: 3rem; margin-bottom: 1rem;'>Upload Files Above</div>"
                "<div style='font-size: 1.1rem;'>Drop your images or videos to get started.</div>"
                "<div style='font-size: 0.9rem; margin-top: 0.5rem; color: #64748b;'>"
                "Supports JPG, PNG, BMP, TIFF, WebP, MP4, AVI, MOV, MKV & more</div>"
                "</div>",
                unsafe_allow_html=True,
            )

# ── TAB 2: Results & Metrics ────────────────────────────────────────────────
with tab_results:
    if "results" in st.session_state and st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"])

        st.markdown('<div class="section-header">Compression Results</div>', unsafe_allow_html=True)

        # Format columns for display
        display_df = df.copy()
        display_df["Original Size"] = display_df["Original_Size_Bytes"].apply(format_bytes)
        display_df["Compressed Size"] = display_df["Compressed_Size_Bytes"].apply(format_bytes)
        display_df["Ratio"] = display_df["Compression_Ratio"].apply(lambda x: f"{x:.2f}x")
        display_df["PSNR (dB)"] = display_df["PSNR"].apply(lambda x: f"{x:.2f}")
        display_df["SSIM"] = display_df["SSIM"].apply(lambda x: f"{x:.6f}")

        # Filter controls
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            filter_file = st.selectbox(
                "Filter by file",
                ["All Files"] + list(df["Filename"].unique())
            )
        with col_filter2:
            filter_fmt = st.selectbox(
                "Filter by format",
                ["All Formats"] + list(df["Format"].unique())
            )

        filtered = display_df.copy()
        if filter_file != "All Files":
            filtered = filtered[filtered["Filename"] == filter_file]
        if filter_fmt != "All Formats":
            filtered = filtered[filtered["Format"] == filter_fmt]

        st.dataframe(
            filtered[["Filename", "Format", "Quality_CRF", "Original Size",
                       "Compressed Size", "Ratio", "PSNR (dB)", "SSIM"]],
            width='stretch',
            height=400,
        )

        # Per-file download
        st.markdown('<div class="section-header">Download Individual Files</div>', unsafe_allow_html=True)
        if "compressed_files" in st.session_state:
            comp_files = st.session_state["compressed_files"]
            cols_per_row = 3
            for i in range(0, len(comp_files), cols_per_row):
                row_files = comp_files[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for j, fpath in enumerate(row_files):
                    if os.path.isfile(fpath):
                        with cols[j]:
                            fname = os.path.basename(fpath)
                            fsize = format_bytes(os.path.getsize(fpath))
                            with open(fpath, "rb") as f:
                                st.download_button(
                                    f"{fname} ({fsize})",
                                    data=f.read(),
                                    file_name=fname,
                                    width='stretch',
                                )
    else:
        st.info("No results yet. Go to the **Compress Files** tab and process some files first.")

# ── TAB 3: Analysis & Charts ────────────────────────────────────────────────
with tab_analysis:
    if "results" in st.session_state and st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"])

        st.markdown('<div class="section-header">Experimental Analysis</div>', unsafe_allow_html=True)

        # ── Image Analysis ───────────────────────────────────────────
        img_df = df[df["Format"].isin(["JPEG", "WEBP", "AVIF"])]
        if not img_df.empty:
            st.markdown("### Image Compression Comparison")

            chart1, chart2 = st.columns(2)

            with chart1:
                st.plotly_chart(chart_compression_ratio_by_quality(img_df), width='stretch')

            with chart2:
                st.plotly_chart(chart_ssim_by_quality(img_df), width='stretch')

            chart3, chart4 = st.columns(2)

            with chart3:
                st.plotly_chart(chart_ssim_vs_ratio(img_df), width='stretch')

            with chart4:
                st.plotly_chart(chart_psnr_by_quality(img_df), width='stretch')

            # Speed comparison chart (if available)
            speed_fig = chart_speed_comparison(img_df)
            if speed_fig:
                st.markdown("### Compression Speed")
                st.plotly_chart(speed_fig, width='stretch')

            # Head-to-head summary
            st.markdown("### Format Comparison Summary")
            summary_data = []
            for fmt in ["JPEG", "WEBP", "AVIF"]:
                subset = img_df[img_df["Format"] == fmt]
                if not subset.empty:
                    row_data = {
                        "Format": fmt,
                        "Avg SSIM": f"{subset['SSIM'].mean():.6f}",
                        "Avg PSNR (dB)": f"{subset['PSNR'].mean():.2f}",
                        "Avg Compression Ratio": f"{subset['Compression_Ratio'].mean():.2f}x",
                        "Best Quality (SSIM)": f"Q={int(subset.loc[subset['SSIM'].idxmax(), 'Quality_CRF'])}",
                    }
                    if "Speed_MBps" in subset.columns and subset["Speed_MBps"].notna().any():
                        row_data["Avg Speed (MB/s)"] = f"{subset['Speed_MBps'].mean():.1f}"
                    summary_data.append(row_data)
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

            # Winner badge
            if len(summary_data) == 2:
                jpeg_df_sub = img_df[img_df["Format"] == "JPEG"]
                webp_df_sub = img_df[img_df["Format"] == "WEBP"]
                if webp_df_sub["Compression_Ratio"].mean() > jpeg_df_sub["Compression_Ratio"].mean():
                    pct = (
                        (webp_df_sub["Compression_Ratio"].mean() - jpeg_df_sub["Compression_Ratio"].mean())
                        / jpeg_df_sub["Compression_Ratio"].mean() * 100
                    )
                    st.success(f"WebP achieves **{pct:.1f}% better compression** on average with comparable quality.")
                else:
                    st.info("JPEG achieves equal or better compression for these images.")

        # ── Video Analysis ───────────────────────────────────────────
        vid_df = df[~df["Format"].isin(["JPEG", "WEBP", "AVIF"])]
        vid_df = vid_df[~vid_df["Format"].str.contains("Autoencoder", case=False, na=False)]
        if not vid_df.empty:
            st.markdown("### Video Compression Analysis")

            vc1, vc2 = st.columns(2)
            with vc1:
                st.plotly_chart(chart_video_ratio_vs_crf(vid_df), width='stretch')

            with vc2:
                st.plotly_chart(chart_video_ssim_vs_crf(vid_df), width='stretch')

        # ── Recommendations ──────────────────────────────────────────
        st.markdown("### Recommendations")
        rec1, rec2 = st.columns(2)
        with rec1:
            st.markdown(
                """
                <div class="metric-card" style="text-align:left;">
                <h4 style="color:#a78bfa; margin-top:0;">Image Guidelines</h4>
                <ul style="color:#cbd5e1; line-height:2;">
                <li><b>AVIF</b> for maximum compression efficiency</li>
                <li><b>WebP</b> for web delivery (broad modern support)</li>
                <li><b>JPEG</b> for email & legacy systems</li>
                <li><b>Quality 70</b> offers the best quality-to-size balance</li>
                <li><b>Quality 90</b> for near-lossless preservation</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rec2:
            st.markdown(
                """
                <div class="metric-card" style="text-align:left;">
                <h4 style="color:#a78bfa; margin-top:0;">Video Guidelines</h4>
                <ul style="color:#cbd5e1; line-height:2;">
                <li><b>H.264</b> for fast encoding & wide support</li>
                <li><b>H.265</b> for maximum storage savings</li>
                <li><b>CRF 23</b> is the sweet spot for most content</li>
                <li><b>CRF 18</b> for high-fidelity archival</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No results yet. Go to the **Compress Files** tab and process some files first.")

# ── TAB 4: Visual Compare ────────────────────────────────────────────────────
with tab_compare:
    if "results" in st.session_state and st.session_state["results"]:
        df = pd.DataFrame(st.session_state["results"])
        source_map = st.session_state.get("source_map", {})
        compressed_files = st.session_state.get("compressed_files", [])

        # Only show image comparisons (video compare would need frame extraction)
        img_results = df[df["Format"].isin(["JPEG", "WEBP", "AVIF"]) | df["Format"].str.contains("Autoencoder", case=False, na=False)]

        if not img_results.empty and source_map:
            st.markdown('<div class="section-header">🔍 Original vs Compressed</div>', unsafe_allow_html=True)

            # File selector
            available_files = list(img_results["Filename"].unique())
            selected_file = st.selectbox("Select Image", available_files, key="compare_file")

            if selected_file and selected_file in source_map:
                orig_path = source_map[selected_file]
                file_results = img_results[img_results["Filename"] == selected_file]

                # Format selector for this file
                available_formats = list(file_results["Format"].unique())
                selected_format = st.selectbox("Select Format", available_formats, key="compare_format")

                format_results = file_results[file_results["Format"] == selected_format]

                if not format_results.empty:
                    # Quality selector
                    available_qualities = sorted(format_results["Quality_CRF"].unique())
                    selected_quality = st.select_slider(
                        "Quality / Bottleneck",
                        options=available_qualities,
                        value=available_qualities[len(available_qualities) // 2],
                        key="compare_quality",
                    )

                    row = format_results[format_results["Quality_CRF"] == selected_quality].iloc[0]

                    # Find the compressed file
                    basename = os.path.splitext(selected_file)[0]
                    if "Autoencoder" in selected_format:
                        comp_name = f"{basename}_autoencoder.png"
                        comp_dir = os.path.join(OUTPUT_DIR, "images", "autoencoder")
                    elif selected_format == "JPEG":
                        comp_name = f"{basename}_q{int(selected_quality)}.jpg"
                        comp_dir = os.path.join(OUTPUT_DIR, "images", "jpeg")
                    elif selected_format == "AVIF":
                        comp_name = f"{basename}_q{int(selected_quality)}.avif"
                        comp_dir = os.path.join(OUTPUT_DIR, "images", "avif")
                    elif selected_format == "WEBP":
                        comp_name = f"{basename}_q{int(selected_quality)}.webp"
                        comp_dir = os.path.join(OUTPUT_DIR, "images", "webp")
                    else:
                        comp_name = None
                        comp_dir = None

                    comp_path = os.path.join(comp_dir, comp_name) if comp_dir and comp_name else None

                    if comp_path and os.path.isfile(comp_path) and os.path.isfile(orig_path):
                        orig_img = Image.open(orig_path)
                        comp_img = Image.open(comp_path)

                        # Metrics for this specific comparison
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-value">{row["Compression_Ratio"]:.1f}x</div>'
                                f'<div class="metric-label">Compression</div></div>',
                                unsafe_allow_html=True,
                            )
                        with m2:
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-value">{row["PSNR"]:.1f} dB</div>'
                                f'<div class="metric-label">PSNR</div></div>',
                                unsafe_allow_html=True,
                            )
                        with m3:
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-value">{row["SSIM"]:.4f}</div>'
                                f'<div class="metric-label">SSIM</div></div>',
                                unsafe_allow_html=True,
                            )
                        with m4:
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-value">{format_bytes(int(row["Compressed_Size_Bytes"]))}</div>'
                                f'<div class="metric-label">Compressed Size</div></div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("")

                        # ── Side-by-Side Comparison ──────────────────
                        st.markdown("#### Side-by-Side")
                        col_orig, col_comp = st.columns(2)
                        with col_orig:
                            st.markdown(
                                f"<div style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;'>"
                                f"<b>Original</b> — {format_bytes(int(row['Original_Size_Bytes']))}</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(orig_img, width="stretch")
                        with col_comp:
                            st.markdown(
                                f"<div style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;'>"
                                f"<b>{selected_format} Q{int(selected_quality)}</b> — "
                                f"{format_bytes(int(row['Compressed_Size_Bytes']))}</div>",
                                unsafe_allow_html=True,
                            )
                            st.image(comp_img, width="stretch")

                        # ── Before/After Slider ──────────────────────
                        st.markdown("")
                        st.markdown("#### Before / After Slider")
                        st.markdown(
                            "<div style='color:#64748b; font-size:0.85rem; margin-bottom:0.5rem;'>"
                            "Drag the slider to compare original (left) vs compressed (right).</div>",
                            unsafe_allow_html=True,
                        )

                        slider_html, display_h = build_slider_html(orig_img, comp_img)
                        st.components.v1.html(slider_html, height=display_h + 20)

                    else:
                        st.warning("Compressed file not found. Run compression first.")
        else:
            st.info("No image comparison data available. Compress some images first.")
    else:
        st.info("No results yet. Go to the **Compress Files** tab and process some files first.")


# ── Cleanup on session end ───────────────────────────────────────────────────
# Temp files are cleaned at the start of each new compression run
