"""
app.py - Media Compression System Web UI
Beautiful Streamlit-based interface for image & video compression.
"""

import os
import io
import sys
import time
import shutil
import zipfile
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.image_compressor import compress_image, is_image, SUPPORTED_EXTENSIONS as IMG_EXT
from src.video_compressor import compress_video, is_video, SUPPORTED_EXTENSIONS as VID_EXT
from src.metrics import (
    file_sizes, compression_ratio,
    compute_image_psnr, compute_image_ssim,
    compute_video_psnr, compute_video_ssim,
)
from src.report import generate_csv, analyze_results

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

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

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
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ─────────────────────────────────────────────────────────

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def save_upload(uploaded_file) -> str:
    """Save an uploaded file to temp and return its path."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def format_bytes(size: int) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def create_zip(file_paths: list, zip_name: str = "compressed_files.zip") -> bytes:
    """Create a ZIP archive from a list of file paths."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if os.path.isfile(fp):
                zf.write(fp, os.path.basename(fp))
    buffer.seek(0)
    return buffer.read()


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
        ["JPEG", "WEBP"],
        default=["JPEG", "WEBP"],
        help="Select image output formats"
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
    st.markdown("### About")
    st.markdown(
        "<div style='color:#64748b; font-size:0.85rem;'>"
        "Built for offline media compression experiments. "
        "Supports batch processing with PSNR, SSIM & compression ratio analysis."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main Content Tabs ────────────────────────────────────────────────────────
tab_compress, tab_results, tab_analysis = st.tabs([
    "Compress Files",
    "Results & Metrics",
    "Analysis & Charts"
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

    # ── Helper: scan a folder for media files ────────────────────────────────
    def scan_folder(folder_path: str, recursive: bool = True):
        """Walk a folder and return lists of image/video absolute paths."""
        img_paths, vid_paths = [], []
        all_exts = IMG_EXT | VID_EXT
        walker = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
        for root, _dirs, files in walker:
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                full = os.path.join(root, fname)
                if ext in IMG_EXT:
                    img_paths.append(full)
                elif ext in VID_EXT:
                    vid_paths.append(full)
        return img_paths, vid_paths

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
                img_paths, vid_paths = scan_folder(folder_input, recursive=scan_recursive)
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
                        src_path = save_upload(img_source)
                    basename = os.path.splitext(img_name)[0]

                    for fmt in img_formats:
                        ext = ".jpg" if fmt == "JPEG" else ".webp"
                        for q in sorted(img_qualities):
                            out_name = f"{basename}_q{q}{ext}"
                            out_path = os.path.join(OUTPUT_DIR, "images", fmt.lower(), out_name)
                            try:
                                status_text.markdown(
                                    f"<span class='status-badge badge-info'>"
                                    f"Processing</span> **{img_name}** → {fmt} Q{q}",
                                    unsafe_allow_html=True,
                                )
                                compress_image(src_path, out_path, fmt=fmt, quality=q)
                                orig_sz, comp_sz = file_sizes(src_path, out_path)
                                ratio = compression_ratio(src_path, out_path)
                                psnr = compute_image_psnr(src_path, out_path)
                                ssim_val = compute_image_ssim(src_path, out_path)
                                results.append({
                                    "Filename": img_name,
                                    "Format": fmt,
                                    "Quality_CRF": q,
                                    "Original_Size_Bytes": orig_sz,
                                    "Compressed_Size_Bytes": comp_sz,
                                    "Compression_Ratio": ratio,
                                    "PSNR": psnr,
                                    "SSIM": ssim_val,
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
                        src_path = save_upload(vid_source)
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
                                compress_video(src_path, out_path, crf=crf, codec=codec)
                                orig_sz, comp_sz = file_sizes(src_path, out_path)
                                ratio = compression_ratio(src_path, out_path)
                                psnr = compute_video_psnr(src_path, out_path)
                                ssim_val = compute_video_ssim(src_path, out_path)
                                results.append({
                                    "Filename": vid_name,
                                    "Format": codec_label,
                                    "Quality_CRF": crf,
                                    "Original_Size_Bytes": orig_sz,
                                    "Compressed_Size_Bytes": comp_sz,
                                    "Compression_Ratio": ratio,
                                    "PSNR": psnr,
                                    "SSIM": ssim_val,
                                })
                                compressed_files.append(out_path)
                            except Exception as e:
                                st.warning(f"Failed: {out_name} — {e}")
                            done += 1
                            progress_bar.progress(
                                done / total_experiments,
                                text=f"Processing… ({done}/{total_experiments})",
                            )

                progress_bar.progress(1.0, text="✅ Compression complete!")
                status_text.markdown(
                    "<span class='status-badge badge-success'>Complete</span> "
                    "All files processed successfully!",
                    unsafe_allow_html=True,
                )

                # Save results
                if results:
                    csv_path = os.path.join(OUTPUT_DIR, "compression_report.csv")
                    generate_csv(results, csv_path)
                    st.session_state["results"] = results
                    st.session_state["csv_path"] = csv_path
                    st.session_state["compressed_files"] = compressed_files

                    # Summary metrics
                    st.markdown("")
                    df = pd.DataFrame(results)
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

                    st.success("Switch to the **Results & Metrics** or **Analysis & Charts** tab for detailed insights!")

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

        # Color theme for charts
        colors = {
            "JPEG": "#f59e0b",
            "WEBP": "#8b5cf6",
            "H.264": "#3b82f6",
            "H.265": "#10b981",
        }

        # ── Image Analysis ───────────────────────────────────────────
        img_df = df[df["Format"].isin(["JPEG", "WEBP"])]
        if not img_df.empty:
            st.markdown("### Image Compression Comparison")

            chart1, chart2 = st.columns(2)

            with chart1:
                fig = px.bar(
                    img_df,
                    x="Quality_CRF",
                    y="Compression_Ratio",
                    color="Format",
                    barmode="group",
                    color_discrete_map=colors,
                    title="Compression Ratio by Quality Level",
                    labels={"Quality_CRF": "Quality", "Compression_Ratio": "Compression Ratio (x)"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig, width='stretch')

            with chart2:
                fig = px.line(
                    img_df,
                    x="Quality_CRF",
                    y="SSIM",
                    color="Format",
                    markers=True,
                    color_discrete_map=colors,
                    title="SSIM vs Quality Level",
                    labels={"Quality_CRF": "Quality", "SSIM": "SSIM Score"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig, width='stretch')

            chart3, chart4 = st.columns(2)

            with chart3:
                fig = px.scatter(
                    img_df,
                    x="Compression_Ratio",
                    y="SSIM",
                    color="Format",
                    size="PSNR",
                    hover_data=["Filename", "Quality_CRF", "PSNR"],
                    color_discrete_map=colors,
                    title="SSIM vs Compression Ratio (bubble size = PSNR)",
                    labels={"Compression_Ratio": "Compression Ratio (x)", "SSIM": "SSIM"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig, width='stretch')

            with chart4:
                fig = px.bar(
                    img_df,
                    x="Quality_CRF",
                    y="PSNR",
                    color="Format",
                    barmode="group",
                    color_discrete_map=colors,
                    title="PSNR by Quality Level",
                    labels={"Quality_CRF": "Quality", "PSNR": "PSNR (dB)"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(t=50, b=60),
                )
                st.plotly_chart(fig, width='stretch')

            # Head-to-head summary
            st.markdown("### Format Comparison Summary")
            summary_data = []
            for fmt in ["JPEG", "WEBP"]:
                subset = img_df[img_df["Format"] == fmt]
                if not subset.empty:
                    summary_data.append({
                        "Format": fmt,
                        "Avg SSIM": f"{subset['SSIM'].mean():.6f}",
                        "Avg PSNR (dB)": f"{subset['PSNR'].mean():.2f}",
                        "Avg Compression Ratio": f"{subset['Compression_Ratio'].mean():.2f}x",
                        "Best Quality (SSIM)": f"Q={int(subset.loc[subset['SSIM'].idxmax(), 'Quality_CRF'])}",
                    })
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
        vid_df = df[~df["Format"].isin(["JPEG", "WEBP"])]
        if not vid_df.empty:
            st.markdown("### Video Compression Analysis")

            vc1, vc2 = st.columns(2)
            with vc1:
                fig = px.line(
                    vid_df,
                    x="Quality_CRF",
                    y="Compression_Ratio",
                    color="Format",
                    markers=True,
                    color_discrete_map=colors,
                    title="Compression Ratio vs CRF",
                    labels={"Quality_CRF": "CRF", "Compression_Ratio": "Compression Ratio (x)"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, width='stretch')

            with vc2:
                fig = px.line(
                    vid_df,
                    x="Quality_CRF",
                    y="SSIM",
                    color="Format",
                    markers=True,
                    color_discrete_map=colors,
                    title="SSIM vs CRF",
                    labels={"Quality_CRF": "CRF", "SSIM": "SSIM Score"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter"),
                    title_font_size=16,
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig, width='stretch')

        # ── Recommendations ──────────────────────────────────────────
        st.markdown("### Recommendations")
        rec1, rec2 = st.columns(2)
        with rec1:
            st.markdown(
                """
                <div class="metric-card" style="text-align:left;">
                <h4 style="color:#a78bfa; margin-top:0;">Image Guidelines</h4>
                <ul style="color:#cbd5e1; line-height:2;">
                <li><b>WebP</b> for web delivery (better compression)</li>
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


# ── Cleanup on session end ───────────────────────────────────────────────────
# Temp files are cleaned at the start of each new compression run
