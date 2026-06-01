"""
egolens/detectors/face_detector.py — FaceDetector  (v2)

Privacy check: detects identifiable faces using MediaPipe FaceDetection
+ secondary confirmation by YOLOv8n-face to reduce false positives.

Two-stage rule:
  1. MediaPipe FaceDetection: fast primary scan
  2. If primary flags a face → YOLOv8n-face secondary check
  3. Violation only when BOTH agree (face present AND secondary confirms)

A face appearing briefly (< max_violation_s seconds) is tolerated.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

from ..duration_tracker import DurationViolationTracker, ViolationSummary
from .face_verifier import FaceVerifier

DEFAULT_EYE_BLUR_THRESHOLD = 40.0
MIN_INTER_EYE_PX           = 10.0


class FaceDetector:
    """
    Parameters
    ----------
    max_violation_s      : float  — face must be continuously visible this long
    min_confidence       : float  — MediaPipe detection threshold
    use_secondary_verify : bool   — enable YOLOv8n-face double-check (reduces FP)
    """

    def __init__(
        self,
        max_violation_s: float = 1.0,
        min_confidence: float = 0.3,
        eye_blur_threshold: float = DEFAULT_EYE_BLUR_THRESHOLD,
        use_secondary_verify: bool = True,
    ) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=1,
        )
        self._eye_blur_threshold = eye_blur_threshold
        self._dur = DurationViolationTracker(max_violation_s=max_violation_s,
                                              name="face_detection")
        # Secondary verifier — lazy-loaded on first face detection
        self._verifier: FaceVerifier = FaceVerifier() if use_secondary_verify else None
        self._use_secondary = use_secondary_verify

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, timestamp_s: float) -> Tuple[float, bool]:
        """
        Returns
        -------
        score   : float — 1.0 = no identifiable face (pass), 0.0 = face visible (fail)
        flagged : bool  — True if violation threshold crossed
        """
        raw = self._compute(frame)

        # Two-stage: primary says face present → run secondary verification
        bbox = None
        if raw < 0.5 and self._use_secondary and self._verifier is not None:
            confirmed, bbox = self._verifier.verify(frame)
            if not confirmed:
                raw  = 1.0   # secondary disagrees → treat as no face (reduce FP)
                bbox = None

        # Violation = identifiable face present (raw = 0.0)
        is_violation = raw < 0.5
        flagged       = self._dur.update(
            is_violation, timestamp_s,
            bbox=bbox if is_violation else None,
        )

        return raw, flagged

    def finalize(self) -> ViolationSummary:
        return self._dur.finalize()

    def close(self) -> None:
        self._detector.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, frame: np.ndarray) -> float:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.detections:
            return 1.0

        h, w = frame.shape[:2]
        for det in result.detections:
            eye_sharp = self._eye_region_sharpness(frame, det, w, h)
            if eye_sharp >= self._eye_blur_threshold:
                return 0.0   # identifiable face

        return 1.0   # detections present but all blurry / too small

    def _eye_region_sharpness(self, frame, detection, w, h) -> float:
        kps = detection.location_data.relative_keypoints
        if len(kps) < 2:
            return self._face_crop_sharpness(frame, detection, w, h)

        re_x, re_y = kps[0].x * w, kps[0].y * h
        le_x, le_y = kps[1].x * w, kps[1].y * h
        inter_eye  = float(np.hypot(re_x - le_x, re_y - le_y))
        if inter_eye < MIN_INTER_EYE_PX:
            return 0.0

        cx     = int((re_x + le_x) / 2)
        cy     = int((re_y + le_y) / 2)
        half_w = int(inter_eye * 1.1)
        half_h = int(inter_eye * 0.55)
        x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
        x2, y2 = min(w, cx + half_w), min(h, cy + half_h)
        if x2 - x1 < 8 or y2 - y1 < 4:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _face_crop_sharpness(self, frame, detection, w, h) -> float:
        bbox = detection.location_data.relative_bounding_box
        x  = max(0, int(bbox.xmin * w))
        y  = max(0, int(bbox.ymin * h))
        bw = max(0, min(w - x, int(bbox.width * w)))
        bh = max(0, min(h - y, int(bbox.height * h)))
        if bw < 16 or bh < 16:
            return 0.0
        crop = frame[y:y + bh, x:x + bw]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
