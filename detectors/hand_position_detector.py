"""
egolens/detectors/hand_position_detector.py — HandPositionDetector

Checks what fraction of hand landmarks fall inside the valid workspace
rectangle defined in config (default: x ∈ [10%, 90%], y ∈ [10%, 90%]).

Violation: hand landmarks outside the workspace for ≥ max_violation_s seconds.
"""

from __future__ import annotations

from typing import Tuple

from ..duration_tracker import DurationViolationTracker, ViolationSummary


class HandPositionDetector:
    """
    Parameters
    ----------
    x_lo, x_hi, y_lo, y_hi : float
        Workspace bounds in normalised [0, 1] coordinates.
    max_violation_s : float
        Seconds outside workspace before flagging.
    """

    def __init__(
        self,
        x_lo: float = 0.10,
        x_hi: float = 0.90,
        y_lo: float = 0.10,
        y_hi: float = 0.90,
        max_violation_s: float = 1.0,
    ) -> None:
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.y_lo = y_lo
        self.y_hi = y_hi
        self._dur = DurationViolationTracker(max_violation_s=max_violation_s,
                                              name="hand_position")

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(
        self,
        landmarks,            # List of NormalizedLandmarkList | None
        timestamp_s: float,
    ) -> Tuple[float, bool]:
        """
        Returns
        -------
        score   : float — fraction of landmarks inside workspace [0, 1]
        flagged : bool  — True when violation threshold crossed
        """
        score = self._compute(landmarks)

        is_violation = score < 0.50   # default gate from config
        flagged       = self._dur.update(is_violation, timestamp_s)

        return score, flagged

    def finalize(self) -> ViolationSummary:
        return self._dur.finalize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, landmarks) -> float:
        if landmarks is None:
            return 0.0

        all_lm = [lm for hand in landmarks for lm in hand.landmark]
        if not all_lm:
            return 0.0

        inside = sum(
            1 for lm in all_lm
            if self.x_lo <= lm.x <= self.x_hi and self.y_lo <= lm.y <= self.y_hi
        )
        return float(inside / len(all_lm))
