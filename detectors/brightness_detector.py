"""
egolens/detectors/brightness_detector.py  — T3

Measures mean luminance of the Y channel (YCrCb colour space).

Raw score   : mean Y value in [0, 255]
Norm score  : raw / 255  (linear, 0.0 = pitch black, 1.0 = pure white)
Gate (cfg)  : 0.30 → reject frames where mean Y < 76.5 / 255

Violation rule (DurationViolationTracker):
  Only flagged when darkness persists ≥ max_violation_s seconds
  (default 1 s).  A single dark frame from motion blur is ignored.

NOTE: This is NOT the same as sharpness.  Sharpness (Laplacian variance)
measures texture contrast; brightness measures overall luminance.
A blurry well-lit frame can pass brightness but fail sharpness, and
vice-versa (a dark scene can have edges but still be too dim to use).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..duration_tracker import DurationViolationTracker, ViolationSummary


class BrightnessDetector:
    """
    Per-frame brightness check using YCrCb Y channel.

    Parameters
    ----------
    max_violation_s : float
        Minimum continuous seconds of darkness before the violation is flagged.
    """

    def __init__(self, max_violation_s: float = 1.0) -> None:
        self._tracker = DurationViolationTracker(
            max_violation_s=max_violation_s,
            name="brightness",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        gate: float = 0.30,
    ) -> Tuple[float, bool]:
        """
        Analyse one frame.

        Parameters
        ----------
        frame_bgr   : H×W×3 BGR uint8 frame
        timestamp_s : video timestamp (seconds)
        gate        : norm threshold below which frame is "bad"

        Returns
        -------
        norm   : float in [0, 1]   (brightness normalised)
        flagged: bool               (True = violation active)
        """
        raw  = self._mean_luminance(frame_bgr)
        norm = float(np.clip(raw / 255.0, 0.0, 1.0))

        is_bad = norm < gate
        flagged = self._tracker.update(is_bad, timestamp_s)

        return norm, flagged

    def finalize(self) -> ViolationSummary:
        return self._tracker.finalize()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_luminance(frame_bgr: np.ndarray) -> float:
        """
        Convert BGR → YCrCb and return mean Y value [0, 255].

        YCrCb Y channel is the standard perceptual luminance used in JPEG,
        video codecs, and display standards (BT.601).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        return float(np.mean(ycrcb[:, :, 0]))
