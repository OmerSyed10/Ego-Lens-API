"""
egolens/detectors/hand_speed_detector.py — HandSpeedDetector

Measures mean Euclidean displacement of hand landmarks between consecutive
analysed frames (pixels/frame).  Too low = demonstrator is idle (no useful
training signal in the clip).

Violation: hands stationary for ≥ max_violation_s seconds continuously.

Normalisation: raw speed is mapped to [0, 1] via a soft sigmoid so the
gate (0.10) maps to a raw speed of ~5 px/frame — very slight movement counts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..duration_tracker import DurationViolationTracker, ViolationSummary

# Speed at which score = 0.5 (half-sigmoid inflection).  Calibrated so
# 5 px/frame (~slow deliberate motion) gives score ~0.10.
_SPEED_SCALE = 80.0   # px/frame for score ≈ 0.63


class HandSpeedDetector:
    """
    Parameters
    ----------
    frame_w, frame_h : int
        Frame dimensions (for converting normalised landmarks to pixels).
    max_violation_s  : float
        Seconds of continuous stillness before flagging.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        max_violation_s: float = 1.0,
    ) -> None:
        self.W = frame_w
        self.H = frame_h
        self._prev_pts: Optional[np.ndarray] = None
        self._dur = DurationViolationTracker(max_violation_s=max_violation_s,
                                              name="hand_speed")

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(
        self,
        landmarks,            # List of NormalizedLandmarkList | None
        timestamp_s: float,
    ) -> Tuple[float, float, bool]:
        """
        Parameters
        ----------
        landmarks   : output from HandTracker / MediaPipe (list of hand landmark lists)
        timestamp_s : float

        Returns
        -------
        raw_speed : float  — px/frame  (0.0 if no landmarks)
        score     : float  — normalised [0, 1]
        flagged   : bool   — True when violation threshold crossed
        """
        raw_speed = self._compute(landmarks)
        score     = self._normalise(raw_speed)

        is_violation = score < 0.10   # gate value from config default
        flagged       = self._dur.update(is_violation, timestamp_s)

        return raw_speed, score, flagged

    def finalize(self) -> ViolationSummary:
        return self._dur.finalize()

    def reset(self) -> None:
        self._prev_pts = None
        self._dur.reset()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, landmarks) -> float:
        if landmarks is None or len(landmarks) == 0:
            self._prev_pts = None
            return 0.0

        pts = np.array(
            [[lm.x * self.W, lm.y * self.H]
             for hand in landmarks
             for lm in hand.landmark],
            dtype=np.float32,
        )

        if self._prev_pts is None or self._prev_pts.shape != pts.shape:
            self._prev_pts = pts
            return 0.0

        speed = float(np.linalg.norm(pts - self._prev_pts, axis=1).mean())
        self._prev_pts = pts
        return speed

    @staticmethod
    def _normalise(raw: float) -> float:
        """Soft-clip via tanh: 0 → 0, _SPEED_SCALE → ~0.76."""
        return float(np.tanh(raw / _SPEED_SCALE))
