# Media Compression Studio

A fully offline media compression system for images and videos with an interactive Streamlit web UI. Supports batch processing, parameter experimentation, quality metrics (PSNR, SSIM), and automated report generation.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?style=flat&logo=streamlit&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-007808?style=flat&logo=ffmpeg&logoColor=white)

---

## Features

- **Interactive Web UI** — Beautiful Streamlit-based dashboard with dark theme and real-time progress
- **Image Compression** — JPEG & WebP at configurable quality levels (10–100)
- **Video Compression** — H.264 & H.265 via FFmpeg at configurable CRF values
- **Quality Metrics** — PSNR, SSIM, and compression ratio computed automatically
- **Batch Processing** — Upload files or point to a folder; process everything at once
- **CSV Reports** — Full metrics for every compression experiment
- **Interactive Charts** — Plotly-powered analysis with format comparisons and recommendations
- **CLI Mode** — Headless batch processing via command line

---

## Project Structure

```
Media-Compression System/
├── app.py                   # Streamlit web UI entry point
├── main.py                  # CLI entry point
├── generate_samples.py      # Creates synthetic test images
├── requirements.txt         # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── samples/                 # Sample input images (auto-generated)
│   ├── gradient_1080p.png
│   ├── synthetic_photo.png
│   └── detailed_pattern.bmp
├── src/
│   ├── __init__.py
│   ├── image_compressor.py  # JPEG & WebP compression (Pillow)
│   ├── video_compressor.py  # FFmpeg-based video compression
│   ├── metrics.py           # PSNR, SSIM, compression ratio
│   ├── batch_processor.py   # Core batch engine (CLI mode)
│   └── report.py            # CSV generation & analysis
└── output/                  # Generated after running
    ├── images/
    │   ├── jpeg/
    │   └── webp/
    ├── videos/
    │   ├── h.264/
    │   └── h.265/
    └── compression_report.csv
```

---

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure FFmpeg is Available (for video compression)

Place `ffmpeg.exe` in the project root, or install FFmpeg system-wide and add it to your PATH.

### 3. Generate Sample Images (optional)

```bash
python generate_samples.py
```

---

## Usage

### Web UI (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

**Features of the Web UI:**
- **Upload Files** — Drag & drop images/videos directly into the browser
- **Folder Path** — Point to any folder on your machine for batch processing
- **Sidebar Controls** — Choose formats, quality levels, codecs, and CRF values
- **3 Tabs:**
  - **Compress Files** — Upload, preview, and start compression
  - **Results & Metrics** — Filterable data table with download buttons
  - **Analysis & Charts** — Interactive Plotly charts with format comparisons

### CLI Mode

```bash
# Basic usage – process all files in a folder
python main.py ./samples

# Specify output directory
python main.py ./samples --output ./results

# Use only H.264 for video
python main.py ./samples --codec h264

# Use only H.265 for video
python main.py ./samples --codec h265

# Verbose logging
python main.py ./samples -v
```

---

## Supported Formats

| Type   | Input Formats                              | Output Formats   |
|--------|--------------------------------------------|------------------|
| Images | JPG, JPEG, PNG, BMP, TIFF, TIF, WebP      | JPEG, WebP       |
| Videos | MP4, AVI, MOV, MKV, WMV, FLV, WebM        | MP4 (H.264/H.265)|

---

## Quality Metrics

| Metric            | Description                                     | Good Range        |
|-------------------|-------------------------------------------------|-------------------|
| **PSNR** (dB)     | Peak Signal-to-Noise Ratio                      | > 30 dB           |
| **SSIM**          | Structural Similarity Index (0–1)               | > 0.90            |
| **Compression Ratio** | Original size ÷ Compressed size            | Higher = smaller  |

---

## Output

The system generates:
1. **Compressed files** organized by format in the output directory
2. **compression_report.csv** with columns:
   - Filename, Format, Quality/CRF, Original Size, Compressed Size, Compression Ratio, PSNR, SSIM
3. **Interactive charts** (Web UI) comparing formats and quality levels

---

## Decision Guidelines

| Use Case                  | Recommendation                          |
|---------------------------|-----------------------------------------|
| Web delivery              | WebP (better compression at same quality)|
| Broad compatibility       | JPEG                                    |
| Fast video encoding       | H.264 (libx264)                         |
| Maximum video compression | H.265 (libx265)                         |
| Best quality-to-size ratio| Quality 70 (images), CRF 23 (video)    |
| Near-lossless archival    | Quality 90 (images), CRF 18 (video)    |

---

## Technology Stack

- **Python 3.10+** — Core language
- **Streamlit** — Web UI framework
- **Pillow** — Image compression (JPEG, WebP)
- **FFmpeg** — Video compression (H.264, H.265)
- **OpenCV** — Image/video frame reading for metrics
- **scikit-image** — SSIM computation
- **NumPy** — Numerical operations
- **Pandas** — Data handling and CSV reports
- **Plotly** — Interactive charts and visualizations
