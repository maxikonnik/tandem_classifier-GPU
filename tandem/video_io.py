"""Frames out of a recording, fast: GPU decode (NVDEC) and keyframe-only sampling.

Every camera in the archive (GoPro H.264 / HEVC, DJI HEVC) writes a keyframe exactly
once a second, and the 1 fps analyses only need one frame a second — so decoding just
the keyframes (``-skip_frame nokey``) does 1/50-1/60 of the work of decoding the stream
and then dropping frames. Denser sampling (exit refinement at 5 fps, the free-fall
analysis at 2 fps) decodes the span on the GPU and scales there (``scale_cuda``) before
the frames are copied out. Frames come back as uint8 RGB arrays through a pipe — no
temporary image files. Without CUDA everything falls back to the same calls on the CPU.
"""
from __future__ import annotations

import functools
import re
import subprocess

import numpy as np


@functools.lru_cache(maxsize=1)
def gpu_decode_available() -> bool:
    """True when ffmpeg lists CUDA decoding and an NVIDIA GPU answers (checked once).
    A decode that still fails at run time falls back to the CPU in read_frames."""
    try:
        acc = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=30).stdout
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return "cuda" in acc.split() and smi.returncode == 0 and "GPU" in smi.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True when ffmpeg can actually open h264_nvenc here — listed encoders are not
    enough: this ffmpeg build needs NVENC API 13.1 (driver >= 610), the installed
    driver offers 13.0, so the encoder fails to open. Checked once by encoding a frame."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def keyframe_times(path: str, lo: float, hi: float) -> list[float]:
    """Presentation times of the video keyframes in [lo, hi) — from the packet index,
    no decoding."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-read_intervals", f"{max(0.0, lo - 2):.3f}%{hi + 2:.3f}",
                        "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1] and parts[0] not in ("", "N/A"):
            t = float(parts[0])
            if lo <= t < hi:
                out.append(t)
    return sorted(out)


@functools.lru_cache(maxsize=256)
def rotation(path: str) -> int:
    """Clockwise display rotation ffmpeg's autorotate applies (0, 90, 180, 270): the
    camera mounted upside down / sideways stores it as display-matrix side data. NVDEC
    frames skip autorotate, so the GPU path applies it itself."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream_side_data=rotation", "-of", "csv=p=0", path], capture_output=True, text=True)
    vals = [v for v in r.stdout.replace(",", " ").split() if v.lstrip("-").isdigit()]
    return (-int(vals[0])) % 360 if vals else 0


_ROTATE = {90: "transpose=clock", 180: "hflip,vflip", 270: "transpose=cclock"}


def _scale_chain(width: int, height: int, gpu: bool, fps: float | None, rot: int = 0) -> str:
    rate = f"fps={fps}," if fps else ""
    if gpu:   # format=nv12 also takes 10-bit sources (DJI HEVC Main10) down to 8 bit on the GPU
        sw, sh = (height, width) if rot in (90, 270) else (width, height)   # scale before rotating
        turn = f",{_ROTATE[rot]}" if rot in _ROTATE else ""
        return (f"{rate}scale_cuda={sw}:{sh}:interp_algo=bicubic:format=nv12,"
                f"hwdownload,format=nv12{turn},format=rgb24,showinfo")
    return f"{rate}scale={width}:{height}:flags=bicubic,format=rgb24,showinfo"   # CPU decode autorotates


_PTS = re.compile(r"pts_time:\s*(-?[0-9.]+)")
_gpu_failed: set = set()          # stream layouts NVDEC could not decode here


@functools.lru_cache(maxsize=256)
def _layout(path: str) -> tuple:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt,width,height", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return tuple(r.stdout.strip().split(","))


def _read(path: str, lo: float, hi: float, width: int, height: int, fps: float | None,
          keyframes: bool, gpu: bool) -> tuple[np.ndarray, np.ndarray]:
    """(times, frames) as ffmpeg delivers them; times from showinfo (relative to the
    seek point, so absolute = lo + pts_time)."""
    args = ["ffmpeg", "-v", "info", "-nostats", "-hide_banner"]
    if keyframes:
        args += ["-skip_frame", "nokey"]
    if gpu:
        args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    args += ["-ss", f"{lo:.3f}", "-to", f"{hi:.3f}", "-i", path, "-an",
             "-vf", _scale_chain(width, height, gpu, fps, rotation(path) if gpu else 0)]
    if keyframes:
        args += ["-fps_mode", "passthrough"]
    args += ["-f", "rawvideo", "-"]
    r = subprocess.run(args, capture_output=True)
    n = len(r.stdout) // (width * height * 3)
    frames = np.frombuffer(bytearray(r.stdout[:n * width * height * 3]), np.uint8).reshape(n, height, width, 3)
    pts = [float(m) for m in _PTS.findall(r.stderr.decode("utf-8", "replace"))]
    if len(pts) == n:
        times = np.array([lo + p for p in pts], np.float64)
    else:                                   # no timing info: nominal times
        step = 1.0 / fps if fps else 1.0
        times = np.array([lo + i * step for i in range(n)], np.float64)
    return times, frames


def read_frames(path: str, lo: float, hi: float, width: int, height: int, *,
                fps: float | None = None, keyframes: bool = False,
                gpu: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(times, frames[N, height, width, 3] uint8 RGB)`` for [lo, hi).

    ``keyframes=True``: only the keyframes (one a second in this archive) at their exact
    times. Otherwise frames at ``fps`` from ``lo``. ``gpu=None`` uses NVDEC when available;
    a layout the GPU cannot decode falls back to the CPU and is not tried again."""
    use_gpu = gpu_decode_available() if gpu is None else gpu
    if use_gpu and _layout(path) in _gpu_failed:
        use_gpu = False
    rate = None if keyframes else fps
    times, frames = _read(path, lo, hi, width, height, rate, keyframes, use_gpu)
    if len(frames) == 0 and use_gpu:
        _gpu_failed.add(_layout(path))
        times, frames = _read(path, lo, hi, width, height, rate, keyframes, False)
    keep = (times >= lo - 1e-3) & (times < hi)          # drop a frame from before the seek point
    return times[keep].astype(np.float32), frames[keep]


@functools.lru_cache(maxsize=256)
def video_size(path: str) -> tuple[int, int]:
    """(width, height) of the first video stream, rotation applied."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "json", path],
                       capture_output=True, text=True)
    import json
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    w, h = int(st.get("width", 1920)), int(st.get("height", 1080))
    rot = next((abs(int(sd.get("rotation", 0))) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    return (h, w) if rot in (90, 270) else (w, h)


def fit_width(path: str, width: int) -> tuple[int, int]:
    """(width, height) scaled to ``width`` with the source aspect, height even."""
    w, h = video_size(path)
    return width, max(2, int(round(width * h / w / 2)) * 2)
