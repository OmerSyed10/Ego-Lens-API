"""
egolens/detectors/person_detector.py — PersonDetector  (v2)

Privacy/scene check: detects whether another person's body is visible in frame.

Two-stage rule:
  1. MediaPipe Pose: detects upper-body keypoints (shoulders/hips)
  2. If primary flags another person → YOLOv8n (class=person) secondary check
  3. Violation only when BOTH agree AND the bounding box isn't the worker's own body

A brief appearance (< max_violation_s) is tolerated.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

from ..duration_tracker import DurationViolationTracker, ViolationSummary
from .person_verifier import PersonVerifier

# Keypoint indices in MediaPipe Pose (33-point model)
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24

# Minimum visibility for a keypoint to count
_VIS_THRESHOLD  = 0.5
# Need this many upper-body keypoints visible to declare "another person"
_MIN_BODY_KPS   = 2


class PersonDetector:
    """
    Parameters
    ----------
    max_violation_s : float
        Person must be continuously visible this long before flagging.
    """

    def __init__(
        self,
        max_violation_s: float = 1.0,
        min_detection_confidence: float = 0.5,
        use_secondary_verify: bool = True,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )
        self._dur = DurationViolationTracker(max_violation_s=max_violation_s,
                                              name="person_detection")
        self._verifier: PersonVerifier = PersonVerifier() if use_secondary_verify else None
        self._use_secondary = use_secondary_verify

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, timestamp_s: float) -> Tuple[float, bool]:
        """
        Returns
        -------
        score   : float — 1.0 = no other person (pass), 0.0 = person detected (fail)
        flagged : bool  — True if violation threshold crossed
        """
        raw = self._compute(frame)

        # Two-stage: primary says person → secondary confirmation
        bbox = None
        if raw < 0.5 and self._use_secondary and self._verifier is not None:
            confirmed, bbox = self._verifier.verify(frame)
            if not confirmed:
                raw  = 1.0   # secondary disagrees → reduce false positive
                bbox = None

        is_violation = raw < 0.5
        flagged       = self._dur.update(
            is_violation, timestamp_s,
            bbox=bbox if is_violation else None,
        )

        return raw, flagged

    def finalize(self) -> ViolationSummary:
        return self._dur.finalize()

    def close(self) -> None:
        self._pose.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, frame: np.ndarray) -> float:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return 1.0   # no pose → no other person

        lm = result.pose_landmarks.landmark
        body_kps = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
        visible  = sum(1 for idx in body_kps if lm[idx].visibility >= _VIS_THRESHOLD)

        if visible >= _MIN_BODY_KPS:
            return 0.0   # another person present

        return 1.0   # only partial/arms-only pose — camera wearer's own arms
