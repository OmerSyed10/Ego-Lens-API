"""
egolens/preprocessor.py — VideoPreprocessor

Handles stereo side-by-side detection, eye crop, and height downsampling.
Non-stereo videos are passed through completely unchanged.

Stereo rule: if video width ≥ 2 × height → side-by-side stereo.
             Crop the configured eye (left | right), then downscale
             so height = stereo_target_height (e.g. 1080p).
             If the crop is already ≤ target height, no resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import VideoConfig


@dataclass
class VideoMeta:
    """Metadata about the video and how it will be processed."""
    native_w: int
    native_h: int
    fps: float
    total_frames: int
    duration_s: float
    is_stereo: bool
    stereo_eye: str             # "left" | "right" | "n/a"
    crop_x: int                 # left pixel of eye crop (0 if not stereo)
    crop_w: int                 # width after eye crop
    effective_w: int            # final width after any downscale
    effective_h: int            # final height after any downscale
    resize_scale: float         # 1.0 = no resize


class VideoPreprocessor:
    """
    Encapsulates all video-level preprocessing decisions.

    Usage
    -----
    pre = VideoPreprocessor(cfg.video)
    meta = pre.open("video.mp4")          # inspects video, decides crop/scale
    cap  = cv2.VideoCapture("video.mp4")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = pre.process(frame)        # apply crop + scale
        # frame is now always meta.effective_w × meta.effective_h
    """

    def __init__(self, cfg: VideoConfig) -> None:
        self.cfg = cfg
        self._meta: Optional[VideoMeta] = None

    # ── Inspection ───────────────────────────────────────────────────────────

    def inspect(self, cap: cv2.VideoCapture) -> VideoMeta:
        """
        Read video properties and decide crop/scale parameters.
        Does NOT read any frames.
        """
        native_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s   = total_frames / fps

        # Stereo detection: width ≥ 2× height
        is_stereo = self.cfg.stereo_detection and (native_w >= 2 * native_h)

        if is_stereo:
            eye       = self.cfg.stereo_eye        # "left" | "right"
            crop_w    = native_w // 2
            crop_x    = crop_w if eye == "right" else 0
            crop_h    = native_h

            # Downscale so height = stereo_target_height (only if larger)
            target_h  = self.cfg.stereo_target_height
            if crop_h > target_h:
                scale = target_h / crop_h
            else:
                scale = 1.0

            eff_w = int(round(crop_w * scale))
            eff_h = int(round(crop_h * scale))
        else:
            eye    = "n/a"
            crop_x = 0
            crop_w = native_w
            scale  = 1.0
            eff_w  = native_w
            eff_h  = native_h

        meta = VideoMeta(
            native_w=native_w,
            native_h=native_h,
            fps=fps,
            total_frames=total_frames,
            duration_s=duration_s,
            is_stereo=is_stereo,
            stereo_eye=eye,
            crop_x=crop_x,
            crop_w=crop_w,
            effective_w=eff_w,
            effective_h=eff_h,
            resize_scale=scale,
        )
        self._meta = meta
        return meta

    # ── Per-frame processing ─────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply crop (stereo) and scale to a single frame.
        Must call inspect() first.
        """
        if self._meta is None:
            raise RuntimeError("Call inspect() before process()")

        m = self._meta

        # 1. Stereo crop → right or left eye
        if m.is_stereo:
            frame = frame[:, m.crop_x : m.crop_x + m.crop_w, :]

        # 2. Downscale if needed
        if m.resize_scale < 1.0:
            frame = cv2.resize(
                frame,
                (m.effective_w, m.effective_h),
                interpolation=cv2.INTER_AREA,
            )

        return frame

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            raise RuntimeError("Call inspect() first")
        return self._meta

    def describe(self) -> str:
        """Human-readable one-liner describing what preprocessing will happen."""
        m = self._meta
        if m is None:
            return "(not yet inspected)"
        if m.is_stereo:
            return (
                f"Stereo {m.native_w}×{m.native_h} → {m.stereo_eye} eye "
                f"{m.crop_w}×{m.native_h}"
                + (f" → scaled {m.effective_w}×{m.effective_h}" if m.resize_scale < 1.0 else "")
            )
        return f"Non-stereo {m.native_w}×{m.native_h} (unchanged)"
