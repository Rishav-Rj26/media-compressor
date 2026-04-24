"""
video_compressor.py – FFmpeg-based Video Compression
Supports H.264 and H.265 codecs with configurable CRF, resolution, and FPS.
Includes input validation and compression speed tracking.
"""

import os
import json
import shutil
import subprocess
import time


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Default encoding parameters
PRESET = "medium"


def is_video(filepath: str) -> bool:
    """Check if file has a supported video extension."""
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTENSIONS


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary – check local project copy first, then PATH."""
    # Check for a local ffmpeg.exe next to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(project_root, "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    # Check for ffmpeg extracted folder
    local_dir = os.path.join(project_root, "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.isfile(local_dir):
        return local_dir
    # Fall back to system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError(
        "FFmpeg not found. Place ffmpeg.exe in the project root or install it system-wide."
    )


def _find_ffprobe() -> str:
    """Locate ffprobe binary – mirrors ffmpeg search logic."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in [
        os.path.join(project_root, "ffprobe.exe"),
        os.path.join(project_root, "ffmpeg", "bin", "ffprobe.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("ffprobe")
    if found:
        return found
    return ""   # ffprobe is optional; we degrade gracefully


def validate_video(filepath: str) -> dict | None:
    """
    Probe a video file with ffprobe and return basic info.
    Returns None if ffprobe is unavailable or the file is invalid.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                filepath,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        info = json.loads(result.stdout)
        # Extract useful metadata
        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            return None
        return {
            "codec": video_stream.get("codec_name"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "duration": float(info.get("format", {}).get("duration", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in str(video_stream.get("r_frame_rate", "")) else float(video_stream.get("r_frame_rate", 0)),
        }
    except Exception:
        return None


def compress_video(
    input_path: str,
    output_path: str,
    crf: int = 23,
    codec: str = "libx264",
    resolution: str | None = None,
    fps: int | None = None,
) -> dict:
    """
    Compress a video, varying CRF and optionally resolution/fps.

    Parameters
    ----------
    input_path  : path to original video
    output_path : path for compressed output (.mp4)
    crf         : Constant Rate Factor (lower = better quality)
    codec       : 'libx264' (H.264) or 'libx265' (H.265/HEVC)
    resolution  : e.g. '1280x720'. None = preserve original resolution
    fps         : target framerate. None = preserve original framerate

    Returns
    -------
    dict with keys: output_path, elapsed_seconds
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    ffmpeg = _find_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    start = time.perf_counter()

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
    ]

    # Only add scale filter if resolution is explicitly requested
    if resolution:
        cmd += ["-vf", f"scale={resolution}"]

    # Only change FPS if explicitly requested
    if fps:
        cmd += ["-r", str(fps)]

    cmd += [
        "-c:v", codec,
        "-preset", PRESET,
        "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        # Provide a clear error with the last portion of stderr
        stderr_tail = result.stderr[-2000:] if result.stderr else "No error output"
        raise RuntimeError(
            f"FFmpeg failed (exit code {result.returncode}) "
            f"compressing '{os.path.basename(input_path)}' with {codec} CRF={crf}:\n"
            f"{stderr_tail}"
        )

    elapsed = time.perf_counter() - start

    return {
        "output_path": output_path,
        "elapsed_seconds": round(elapsed, 4),
    }
