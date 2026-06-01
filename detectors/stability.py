"""
Camera Motion / Stability — STABILITY dimension.

Dense optical flow (Farneback) measures pixel displacement between consecutive
frames. Mean flow magnitude = camera shakiness. This metric is *inverted*:
low raw value (stable camera) → high normalised score.
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


class StabilityMetric:
    name = "stability"

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None

    def compute(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0  # first frame: assume stable

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        self._prev_gray = gray
        return float(mag)

    def reset(self) -> None:
        self._prev_gray = None

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
                f"Camera too unstable (stability={normalised:.2f} < gate={gate:.2f}, "
                f"mean_flow={raw:.1f} px/frame). "
                f"Likely cause: walking, accidental bump, or rapid head turn. "
                f"Timestamp: {ts}"
            )
        return MetricResult(raw=raw, normalised=normalised, passed=passed, reason=reason)


def _fmt(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"
