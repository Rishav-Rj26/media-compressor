"""
ui_helpers.py – Extracted utility functions for the Streamlit UI.

Contains compression runner logic, chart builders, and helper functions
that were previously inlined in app.py.
"""

import os
import io
import zipfile
import base64
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── File Utilities ───────────────────────────────────────────────────────────

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


def img_to_b64(pil_img, fmt="PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def save_upload(uploaded_file, temp_dir: str) -> str:
    """Save a Streamlit uploaded file to temp and return its path."""
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def scan_folder(folder_path: str, img_exts: set, vid_exts: set, recursive: bool = True):
    """Walk a folder and return lists of image/video absolute paths."""
    img_paths, vid_paths = [], []
    walker = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
    for root, _dirs, files in walker:
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            full = os.path.join(root, fname)
            if ext in img_exts:
                img_paths.append(full)
            elif ext in vid_exts:
                vid_paths.append(full)
    return img_paths, vid_paths


# ── Chart Builders ───────────────────────────────────────────────────────────

# Color theme for charts
CHART_COLORS = {
    "JPEG": "#f59e0b",
    "WEBP": "#8b5cf6",
    "AVIF": "#ec4899",
    "H.264": "#3b82f6",
    "H.265": "#10b981",
}

CHART_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
    title_font_size=16,
    legend=dict(orientation="h", y=-0.2),
    margin=dict(t=50, b=60),
)


def chart_compression_ratio_by_quality(img_df: pd.DataFrame):
    """Bar chart: Compression Ratio grouped by Quality Level."""
    fig = px.bar(
        img_df,
        x="Quality_CRF",
        y="Compression_Ratio",
        color="Format",
        barmode="group",
        color_discrete_map=CHART_COLORS,
        title="Compression Ratio by Quality Level",
        labels={"Quality_CRF": "Quality", "Compression_Ratio": "Compression Ratio (x)"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_ssim_by_quality(img_df: pd.DataFrame):
    """Line chart: SSIM vs Quality Level."""
    fig = px.line(
        img_df,
        x="Quality_CRF",
        y="SSIM",
        color="Format",
        markers=True,
        color_discrete_map=CHART_COLORS,
        title="SSIM vs Quality Level",
        labels={"Quality_CRF": "Quality", "SSIM": "SSIM Score"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_ssim_vs_ratio(img_df: pd.DataFrame):
    """Scatter: SSIM vs Compression Ratio (bubble size = PSNR)."""
    fig = px.scatter(
        img_df,
        x="Compression_Ratio",
        y="SSIM",
        color="Format",
        size="PSNR",
        hover_data=["Filename", "Quality_CRF", "PSNR"],
        color_discrete_map=CHART_COLORS,
        title="SSIM vs Compression Ratio (bubble size = PSNR)",
        labels={"Compression_Ratio": "Compression Ratio (x)", "SSIM": "SSIM"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_psnr_by_quality(img_df: pd.DataFrame):
    """Bar chart: PSNR grouped by Quality Level."""
    fig = px.bar(
        img_df,
        x="Quality_CRF",
        y="PSNR",
        color="Format",
        barmode="group",
        color_discrete_map=CHART_COLORS,
        title="PSNR by Quality Level",
        labels={"Quality_CRF": "Quality", "PSNR": "PSNR (dB)"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_video_ratio_vs_crf(vid_df: pd.DataFrame):
    """Line chart: Video Compression Ratio vs CRF."""
    fig = px.line(
        vid_df,
        x="Quality_CRF",
        y="Compression_Ratio",
        color="Format",
        markers=True,
        color_discrete_map=CHART_COLORS,
        title="Compression Ratio vs CRF",
        labels={"Quality_CRF": "CRF", "Compression_Ratio": "Compression Ratio (x)"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_video_ssim_vs_crf(vid_df: pd.DataFrame):
    """Line chart: Video SSIM vs CRF."""
    fig = px.line(
        vid_df,
        x="Quality_CRF",
        y="SSIM",
        color="Format",
        markers=True,
        color_discrete_map=CHART_COLORS,
        title="SSIM vs CRF",
        labels={"Quality_CRF": "CRF", "SSIM": "SSIM Score"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


def chart_speed_comparison(df: pd.DataFrame):
    """Bar chart: Average compression speed by format."""
    if "Speed_MBps" not in df.columns or df["Speed_MBps"].isna().all():
        return None
    speed_df = df.groupby("Format")["Speed_MBps"].mean().reset_index()
    fig = px.bar(
        speed_df,
        x="Format",
        y="Speed_MBps",
        color="Format",
        color_discrete_map=CHART_COLORS,
        title="Average Compression Speed by Format",
        labels={"Speed_MBps": "Speed (MB/s)", "Format": "Format"},
        template="plotly_dark",
    )
    fig.update_layout(**CHART_LAYOUT_DEFAULTS)
    return fig


# ── Before/After Slider HTML ────────────────────────────────────────────────

def build_slider_html(
    orig_img: Image.Image,
    comp_img: Image.Image,
    max_width: int = 800,
) -> tuple[str, int]:
    """
    Build an interactive before/after slider HTML component.
    Returns (html_string, display_height).
    """
    display_w = min(max_width, orig_img.width)
    scale = display_w / orig_img.width
    display_h = int(orig_img.height * scale)
    orig_resized = orig_img.resize((display_w, display_h), Image.LANCZOS)
    comp_resized = comp_img.resize((display_w, display_h), Image.LANCZOS)

    b64_orig = img_to_b64(orig_resized)
    b64_comp = img_to_b64(comp_resized)

    slider_html = f"""
    <div style="position:relative; width:{display_w}px; height:{display_h}px;
                border-radius:12px; overflow:hidden; margin:0 auto;
                border:1px solid rgba(124,58,237,0.4);">
        <img src="data:image/png;base64,{b64_comp}"
             style="position:absolute; top:0; left:0; width:{display_w}px; height:{display_h}px;" />
        <div id="ba-clip" style="position:absolute; top:0; left:0;
             width:{display_w // 2}px; height:{display_h}px; overflow:hidden;
             border-right: 3px solid #7c3aed;">
            <img src="data:image/png;base64,{b64_orig}"
                 style="width:{display_w}px; height:{display_h}px;" />
        </div>
        <div id="ba-handle" style="position:absolute; top:0; left:{display_w // 2}px;
             width:3px; height:100%; background:#7c3aed; cursor:col-resize; z-index:10;">
            <div style="position:absolute; top:50%; left:-18px; width:38px; height:38px;
                 background:#7c3aed; border-radius:50%; transform:translateY(-50%);
                 display:flex; align-items:center; justify-content:center;
                 font-size:14px; color:white; box-shadow:0 2px 10px rgba(0,0,0,0.3);">⇔</div>
        </div>
        <div style="position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.6);
             color:white; padding:4px 10px; border-radius:6px; font-size:12px; z-index:5;">Original</div>
        <div style="position:absolute; top:10px; right:10px; background:rgba(124,58,237,0.7);
             color:white; padding:4px 10px; border-radius:6px; font-size:12px; z-index:5;">Compressed</div>
    </div>
    <script>
    (function() {{
        const container = document.querySelector('[id="ba-handle"]').parentElement;
        const clip = document.getElementById('ba-clip');
        const handle = document.getElementById('ba-handle');
        let dragging = false;
        handle.addEventListener('mousedown', () => dragging = true);
        document.addEventListener('mouseup', () => dragging = false);
        container.addEventListener('mousemove', (e) => {{
            if (!dragging) return;
            const rect = container.getBoundingClientRect();
            let x = e.clientX - rect.left;
            x = Math.max(0, Math.min(x, rect.width));
            clip.style.width = x + 'px';
            handle.style.left = x + 'px';
        }});
        container.addEventListener('click', (e) => {{
            const rect = container.getBoundingClientRect();
            let x = e.clientX - rect.left;
            clip.style.width = x + 'px';
            handle.style.left = x + 'px';
        }});
    }})();
    </script>
    """
    return slider_html, display_h
