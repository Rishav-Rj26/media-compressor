# Project Report: Media Compression System

## 1. Executive Summary

The **Media Compression System** (or Media Compression Studio) is a fully offline, comprehensive application designed to compress, analyze, and evaluate various media formats. By combining traditional codecs (JPEG, WebP, AVIF, H.264, H.265) with a neural autoencoder for learned compression, the system offers an exhaustive platform for media optimization. 

A standout feature of this system is its dual-interface capability, offering both a beautiful, interactive Streamlit web UI and a headless CLI batch processor. Furthermore, it incorporates advanced quality metrics (PSNR, SSIM, MS-SSIM) to provide data-driven insights into compression efficiency versus quality loss.

## 2. Key Features

- **Multi-Format Image Compression:** Supports advanced formats like WebP and AVIF alongside standard JPEG, with customizable quality levels.
- **Video Compression:** Utilizes FFmpeg for robust video encoding using H.264 (libx264) and H.265 (libx265) codecs, with configurable Constant Rate Factor (CRF).
- **Neural Autoencoder:** Integrates a basic and deep PyTorch-based convolutional autoencoder for learned image compression, leveraging a combined L1 and SSIM-based perceptual loss.
- **Batch Processing Engine:** Parses entire folders for media, efficiently dispatching image compressions to a parallel ThreadPoolExecutor while handling video processing sequentially.
- **Quality Metrics & Analysis:** Automatically calculates Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), Multi-Scale SSIM (MS-SSIM), compression speed (MB/s), and compression ratios.
- **Interactive Web UI:** Built with Streamlit, providing file uploads, data table reports, Plotly charts for metric analysis, and an interactive before/after visual comparison slider.
- **Detailed CSV Reporting:** Generates comprehensive benchmarking CSV reports and textual analysis for deeper insights into the experiments run.

## 3. Project Architecture

The project maintains a modular architecture separating the user interface from core compression and metric calculation logic.

### Directory Structure
```
Media-Compression System/
├── app.py                   # Streamlit web UI & main entry for the web app
├── main.py                  # CLI entry point for headless batch processing
├── benchmark.py             # Script for standardized Kodak dataset benchmarking
├── generate_samples.py      # Script to create synthetic media for testing
├── requirements.txt         # Project dependencies
├── samples/                 # Sample images/videos for testing
├── src/                     # Core system modules
│   ├── image_compressor.py  # Handles Pillow-based traditional image format compression
│   ├── video_compressor.py  # Handles FFmpeg-based video compression
│   ├── autoencoder.py       # PyTorch neural autoencoder implementation
│   ├── batch_processor.py   # Multi-threaded batch processor logic
│   ├── metrics.py           # Evaluation algorithms (PSNR, SSIM, MS-SSIM)
│   ├── report.py            # CSV generation and textual report analysis
│   └── ui_helpers.py        # Streamlit layout & Plotly chart builders
├── tests/                   # Pytest suite
│   ├── conftest.py          # Pytest fixtures and mock generation
│   ├── test_batch_processor.py
│   ├── test_image_compressor.py
│   ├── test_metrics.py
│   ├── test_report.py
│   └── test_video_compressor.py
```

## 4. Module Breakdown

### 4.1 Core Processing (`src/image_compressor.py`, `src/video_compressor.py`)
- **Image Compression:** Uses the `Pillow` library to save images in specified formats (JPEG, WebP, AVIF). It includes a "size guard" feature which progressively drops the target quality if the processed file turns out larger than the original.
- **Video Compression:** Uses `subprocess` to trigger `FFmpeg` shell commands. It reliably tests for the existence of `ffmpeg.exe` via `_find_ffmpeg()` and adjusts the command based on user parameters for codec, CRF, framerate, and resolution.

### 4.2 Neural Autoencoder (`src/autoencoder.py`)
Provides both `standard` (3-layer) and `deep` (5-layer) PyTorch convolutional neural networks.
- Uses `CombinedLoss` incorporating both Mean Absolute Error (L1) and MS-SSIM calculations to optimize for perceptual quality rather than just pixel-perfect reconstruction.
- Employs a Cosine Annealing Learning Rate schedule (`CosineAnnealingLR`) to ease convergence.
- Includes utility functions to estimate latents quantization sizes to simulate authentic "compressed sizes."

### 4.3 Metrics Calculation (`src/metrics.py`)
A heavy-lifting mathematical module using `OpenCV`, `scikit-image`, and `NumPy`:
- **PSNR:** Calculates error tracking as logarithmic decibels limit.
- **SSIM/MS-SSIM:** Utilizes structural similarity equations over varying gaussian blurs (Scale resolutions) for highly accurate human-perception approximations.
- Video metrics sample up to 30 evenly-spaced frames from videos to compute average continuous qualities without requiring complete sub-frame decoding, maximizing speed.

### 4.4 Web User Interface (`app.py` & `src/ui_helpers.py`)
The web experience builds upon a single-page Streamlit application utilizing a dark theme with custom CSS and interactive features:
- **Plotly Integration:** Constructs bar charts and scatter plots tracking SSIM vs Ratio, speeds by formats, etc.
- **HTML/JS Interaction:** Implements a native DOM slider utilizing injected HTML for the visual "Before/After" tool.

## 5. Testing & Quality Assurance

The system is highly stable, accompanied by an extensive Pytest suite containing 40+ unit tests across the components.

- **Mock Media Generation:** Uses Pillow to draw geometric configurations generating real valid images dynamically via fixtures in `conftest.py` avoiding hard dependency on external files for simple logic assertions.
- **Metric Verification:** Tests bounds of metrics (e.g., identical images asserting an infinite PSNR and 1.0 SSIM).
- **Size Fallbacks:** Tests that size guards adequately reject bloating files.

Let's look at the codebase breakdown:

| File Context | Approx LOC |
|--------------|------------|
| App/UI       | ~1,376 |
| Core Source  | ~1,072 |
| Testing      | ~475 |
| Utilities    | ~266 |
| **Total**    | **~3,189** |

## 6. Recommendations from Benchmarks

Running the system via `benchmark.py` against the Kodak Lossless True Color Image dataset yields standard conclusions integrated into the product's design:
- **AVIF** delivers the maximum compression efficiency and space reduction for cutting-edge deployments.
- **WebP** acts as an optimal middle-ground for universal web delivery.
- **H.264 (CRF 23)** remains the undisputed balance for video processing, while **H.265** shines where encoding time is irrelevant compared to definitive storage savings.

## 7. Future Enhancements

While feature-complete, future adaptations could include:
1. **Model Exporting:** Exporting the trained autoencoder to ONNX or standard TorchScript models.
2. **Audio Pass-Through:** Deeper integrations in the Video module for advanced Audio Codec (AAC, Opus) resampling.
3. **Hardware Acceleration:** Expanding ffmpeg calls to utilize NVENC (NVIDIA) or QSV (Intel) for dramatically faster video encoding tasks.
