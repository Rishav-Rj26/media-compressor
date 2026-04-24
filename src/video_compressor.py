"""
video_compressor.py – FFmpeg-based Video Compression
Fixed pipeline: 1280×720, 30 fps, medium preset.  Only CRF varies.
"""

import os
import shutil
import subprocess


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# Fixed encoding parameters (per spec)
RESOLUTION = "1280x720"
FPS = 30
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


def compress_video(input_path: str, output_path: str,
                   crf: int = 23, codec: str = "libx264") -> str:
    """
    Compress a video with a fixed pipeline, varying only CRF.

    Parameters
    ----------
    input_path  : path to original video
    output_path : path for compressed output (.mp4)
    crf         : Constant Rate Factor (lower = better quality)
    codec       : 'libx264' (H.264) or 'libx265' (H.265/HEVC)

    Returns
    -------
    output_path on success
    """
    ffmpeg = _find_ffmpeg()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vf", f"scale={RESOLUTION}",
        "-r", str(FPS),
        "-c:v", codec,
        "-preset", PRESET,
        "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-2000:]}")
    return output_path
