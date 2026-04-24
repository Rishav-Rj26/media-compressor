# Media Compression Studio

A fully offline media compression system combining **traditional codecs** (JPEG, WebP, AVIF, H.264, H.265) with a **neural autoencoder** for learned compression. Features an interactive Streamlit web UI with batch processing, quality metrics (PSNR, SSIM, MS-SSIM), interactive charts, visual before/after comparison, and Kodak dataset benchmarking.

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?style=flat&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Optional-ee4c2c?style=flat&logo=pytorch&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-007808?style=flat&logo=ffmpeg&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-40+-22c55e?style=flat&logo=pytest&logoColor=white)

---

## Features

### Compression Engines
- **JPEG & WebP** — Traditional image codecs via Pillow at configurable quality (10–100)
- **AVIF** — Next-gen image format with superior compression efficiency (Pillow 10.1+)
- **H.264 & H.265** — FFmpeg-based video compression at configurable CRF values
- **🧠 Neural Autoencoder** — Convolutional autoencoder with perceptual SSIM+L1 loss for learned compression

### Analysis & Visualization
- **Quality Metrics** — PSNR, SSIM, MS-SSIM, and compression ratio computed automatically
- **Speed Metrics** — Compression throughput (MB/s) tracked per experiment
- **Interactive Charts** — Plotly-powered format comparisons, speed charts, and recommendations
- **Visual Compare** — Side-by-side original vs compressed + interactive before/after slider
- **Savings Summary** — Total space saved with best configuration

### Workflow
- **Batch Processing** — Upload files or point to a folder; process everything at once
- **Parallel Execution** — Image compression runs in parallel via ThreadPoolExecutor
- **CSV Reports** — Full metrics for every compression experiment
- **Dual Interface** — Beautiful Web UI (Streamlit) + headless CLI mode

### Quality Assurance
- **40+ Unit Tests** — Comprehensive test suite covering compression, metrics, reporting, and batch processing
- **Kodak Benchmark** — Reproducible benchmark against the Kodak Lossless True Color Image Suite
- **Size Guard** — Prevents compressed output from being larger than the original
- **Input Validation** — Verifies files are valid images/videos before processing

---

## Project Structure

```
Media-Compression System/
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── benchmark.py             # Kodak dataset benchmark script
├── generate_samples.py      # Creates synthetic test images
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit theme
├── samples/                 # Sample test images
├── src/
│   ├── __init__.py
│   ├── image_compressor.py  # JPEG, WebP & AVIF compression (Pillow)
│   ├── video_compressor.py  # FFmpeg-based video compression
│   ├── autoencoder.py       # Neural autoencoder (PyTorch) with perceptual loss
│   ├── metrics.py           # PSNR, SSIM, MS-SSIM, compression ratio
│   ├── batch_processor.py   # Parallel batch engine (CLI)
│   ├── report.py            # CSV generation & analysis
│   └── ui_helpers.py        # Chart builders & UI utilities (extracted from app.py)
├── tests/
│   ├── conftest.py          # Shared test fixtures
│   ├── test_image_compressor.py
│   ├── test_video_compressor.py
│   ├── test_metrics.py
│   ├── test_report.py
│   └── test_batch_processor.py
├── benchmark/               # Generated benchmark results (after running benchmark.py)
└── output/                  # Generated after running compression
```

---

## Setup

### 1. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 2. Enable Neural Compression (Optional)

```bash
pip install torch torchvision
```

### 3. Ensure FFmpeg is Available (for video)

Place `ffmpeg.exe` in the project root, or install FFmpeg system-wide.

### 4. Generate Sample Images (optional)

```bash
python generate_samples.py
```

---

## Usage

### Web UI (Recommended)

```bash
streamlit run app.py
```

**4 Tabs:**
| Tab | Description |
|-----|-------------|
| **Compress Files** | Upload/browse files, configure settings, run compression |
| **Results & Metrics** | Filterable data table with per-file downloads |
| **Analysis & Charts** | Interactive Plotly charts comparing formats + speed analysis |
| **Visual Compare** | Side-by-side + before/after slider comparison |

### CLI Mode

```bash
python main.py ./samples                     # Process all files
python main.py ./samples --output ./results  # Custom output dir
python main.py ./samples --codec h265 -v     # H.265 only, verbose
```

### Kodak Benchmark

Run a standardized compression benchmark against the Kodak image dataset:

```bash
python benchmark.py                                 # Default (5 images, JPEG + WebP)
python benchmark.py --formats JPEG WEBP AVIF         # Include AVIF
python benchmark.py --all-kodak                      # All 24 Kodak images
python benchmark.py --qualities 10 30 50 70 90 100   # Custom quality range
```

Results are saved to `benchmark/output/kodak_benchmark.csv`.

### Running Tests

```bash
python -m pytest tests/ -v              # Run all tests
python -m pytest tests/ -v --tb=short   # Shorter output
python -m pytest tests/test_metrics.py  # Run specific test file
```

---

## Neural Autoencoder

The system includes a **convolutional autoencoder** for learned image compression with improved training:

### Standard Architecture (3-layer)
```
Encoder:  Input(3ch) → Conv→BN→ReLU(64) → Conv→BN→ReLU(32) → Conv→BN→ReLU(bottleneck)
Decoder:  Bottleneck → ConvT→BN→ReLU(32) → ConvT→BN→ReLU(64) → ConvT→Sigmoid(3ch)
```

### Deep Architecture (5-layer, optional)
```
Encoder:  3→64→128→64→32→bottleneck  (spatial reduction: H/32)
Decoder:  bottleneck→32→64→128→64→3
```

**Improvements over baseline:**
- **Combined L1 + SSIM loss** for perceptual quality (not just MSE)
- **Cosine annealing LR schedule** for better convergence
- **Deeper architecture option** for higher quality
- **Bottleneck size** controls compression aggressiveness (2=max compression, 32=high quality)
- Trained on your own images — no pre-trained weights needed

---

## Quality Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **PSNR** (dB) | Peak Signal-to-Noise Ratio | > 30 dB |
| **SSIM** | Structural Similarity Index (0–1) | > 0.90 |
| **MS-SSIM** | Multi-Scale SSIM (more robust) | > 0.90 |
| **Compression Ratio** | Original ÷ Compressed size | Higher = better |
| **Speed** (MB/s) | Compression throughput | Higher = faster |

---

## Decision Guidelines

| Use Case | Recommendation |
|----------|----------------|
| Maximum compression | AVIF (best ratio at equal quality) |
| Web delivery | WebP (broad modern browser support) |
| Broad compatibility | JPEG |
| Fast video encoding | H.264 (libx264) |
| Max video compression | H.265 (libx265) |
| Best quality-to-size | Quality 70 (images), CRF 23 (video) |
| Learned compression | Autoencoder with bottleneck=8 |

---

## Technology Stack

- **Python 3.10+** — Core language
- **Streamlit** — Web UI framework
- **Pillow** — Image compression (JPEG, WebP, AVIF)
- **FFmpeg** — Video compression (H.264, H.265)
- **PyTorch** — Neural autoencoder (optional)
- **OpenCV + scikit-image** — PSNR, SSIM & MS-SSIM metrics
- **Pandas** — Data handling and CSV reports
- **Plotly** — Interactive charts
- **pytest** — Unit testing framework
