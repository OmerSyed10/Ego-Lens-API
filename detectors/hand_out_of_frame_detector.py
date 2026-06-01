"""
egolens/detectors/hand_out_of_frame_detector.py — HandOutOfFrameDetector

Checks whether wrist keypoints are dangerously close to any frame edge.

Rule: wrist within `margin` (default 5%) of any edge = "hand out of frame".
Score = fraction of detected wrists that are WELL INSIDE the frame.

Secondary check (T10): velocity-direction confirmation.
  If a wrist is near the edge AND moving toward it (velocity points outward),
  confidence in the "out of frame" reading is boosted to avoid false positives
  from hands that momentarily brush the edge but aren't actually leaving.

Violation: wrist near edge for ≥ max_violation_s seconds continuously.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..duration_tracker import DurationViolationTracker, ViolationSummary

# MediaPipe Hands landmark index for wrist
_WRIST_IDX = 0

# Velocity history window (frames) for direction check
_VEL_WINDOW = 3


class HandOutOfFrameDetector:
    """
    Parameters
    ----------
    margin          : float  — normalised distance from edge to be "too close"
    max_violation_s : float  — seconds before flagging
    """

    def __init__(
        self,
        margin: float = 0.05,
        max_violation_s: float = 1.0,
    ) -> None:
        self.margin = margin
        self._dur = DurationViolationTracker(max_violation_s=max_violation_s,
                                              name="hand_out_of_frame")
        # Per-side wrist position history for velocity direction check
        # key: hand index (0 or 1), value: list of (x, y) normalised positions
        self._history: Dict[int, List[Tuple[float, float]]] = {}

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(
        self,
        landmarks,            # List of NormalizedLandmarkList | None
        timestamp_s: float,
    ) -> Tuple[float, bool]:
        """
        Returns
        -------
        score   : float — fraction of wrists well inside frame [0, 1]
                          (1.0 if no hands detected)
        flagged : bool  — True when violation threshold crossed
        """
        score = self._compute(landmarks)

        # gate=0.90 from config: at least 90% of wrist positions must be safe
        is_violation = score < 0.90
        flagged       = self._dur.update(is_violation, timestamp_s)

        return score, flagged

    def finalize(self) -> ViolationSummary:
        return self._dur.finalize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, landmarks) -> float:
        if landmarks is None or len(landmarks) == 0:
            # Clear history when hands disappear
            self._history.clear()
            return 1.0   # no hands → not out of frame

        m   = self.margin
        ok  = 0
        total = 0

        for hand_idx, hand in enumerate(landmarks):
            wrist = hand.landmark[_WRIST_IDX]
            wx, wy = wrist.x, wrist.y
            total += 1

            # Update position history for this hand
            hist = self._history.setdefault(hand_idx, [])
            hist.append((wx, wy))
            if len(hist) > _VEL_WINDOW:
                hist.pop(0)

            near_edge = not (wx >= m and wx <= 1.0 - m and
                             wy >= m and wy <= 1.0 - m)

            if not near_edge:
                ok += 1   # well inside → safe
                continue

            # Wrist is near the edge.
            # Secondary check: is it moving TOWARD the edge (escaping)?
            if len(hist) >= 2 and self._moving_toward_edge(hist, m):
                # Confirmed: wrist near edge AND moving outward → bad (don't add to ok)
                pass
            else:
                # Near edge but stationary / moving inward → borderline, be lenient
                ok += 0.5   # partial credit

        if total == 0:
            return 1.0
        return float(ok / total)

    def _moving_toward_edge(
        self, history: List[Tuple[float, float]], margin: float
    ) -> bool:
        """
        Return True if velocity vector points toward the nearest edge.

        Uses the last two positions to compute velocity direction.
        """
        (x0, y0), (x1, y1) = history[-2], history[-1]
        vx, vy = x1 - x0, y1 - y0

        # Distance to each edge (0 = at edge)
        dist_left   = x1
        dist_right  = 1.0 - x1
        dist_top    = y1
        dist_bottom = 1.0 - y1

        # Find which edge wrist is closest to
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # velocity component toward that edge
        if min_dist == dist_left   and vx < -0.002:
            return True
        if min_dist == dist_right  and vx >  0.002:
            return True
        if min_dist == dist_top    and vy < -0.002:
            return True
        if min_dist == dist_bottom and vy >  0.002:
            return True

        return False
