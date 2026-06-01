"""
Image Sharpness — QUALITY dimension.

Uses the variance of the Laplacian of the grayscale frame.
High variance → many edges → sharp image.
Low variance → smooth / blurry → fails quality gate.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricResult:
    raw: float
    normalised: float
    passed: bool
    reason: Optional[str]


class SharpnessMetric:
    name = "sharpness"

    def compute(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def evaluate(
        self,
        frame: np.ndarray,
        normalised: float,
        gate: float,
        timestamp_s: float,
        raw: float,
    ) -> MetricResult:
        passed = normalised >= gate
        reason = None
        if not passed:
            ts = _fmt(timestamp_s)
            reason = (
                f"Frame blurry (sharpness={normalised:.2f} < gate={gate:.2f}). "
                f"Likely cause: camera motion, out-of-focus lens, or poor lighting. "
                f"Timestamp: {ts}"
            )
        return MetricResult(raw=raw, normalised=normalised, passed=passed, reason=reason)


def _fmt(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"
