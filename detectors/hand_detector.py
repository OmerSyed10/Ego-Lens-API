"""
egolens/detectors/hand_detector.py — HandDetector

5-tier cascade hand detector backed by DurationViolationTracker.

Tier 1  MediaPipe Hands (static_image_mode, min_conf=0.1, flip+CLAHE retries)
Tier 2  100DOH ResNet-101 (palm/wrist from Faster-RCNN; ~90% AP egocentric)
Tier 3  MediaPipe Pose wrist (arm geometry; works on fists)
Tier 4  YCbCr skin blob (appearance fallback)
Tier 5  Phantom coast (velocity extrapolation through brief occlusion)

The tracker only flags "hand absent" when the bad condition has been
continuously present for ≥ max_violation_s seconds (from DurationViolationTracker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .hand_tracker import HandTracker, HandsCandidate
from ..duration_tracker import DurationViolationTracker, ViolationSummary
from ..tracking_loader import TrackingData

RETRY_THRESHOLD    = 0.5   # MP-Hands score below this triggers flip/CLAHE retries
POSE_TRIGGER_SCORE = 0.7   # MP-Hands score below this triggers a Pose pass


class HandDetector:
    """
    Per-video hand detector with duration-based violation tracking.

    Parameters
    ----------
    frame_w, frame_h : int
        Effective frame dimensions (after any stereo crop / downscale).
    max_violation_s : float
        Seconds a hand must be continuously absent before counting as flagged.
    doh_detector : optional
        Pre-constructed DOHDetector instance (Tier 2). Pass None to skip DOH.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        max_violation_s: float = 1.0,
        doh_detector=None,
        tracking_data: Optional["TrackingData"] = None,
    ) -> None:
        self.W = frame_w
        self.H = frame_h
        self._tracking = tracking_data   # Tier-0: XR hand tracking data

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.1,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.3,
        )
        self._clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._tracker = HandTracker(frame_w, frame_h, doh_detector=doh_detector)
        self._dur     = DurationViolationTracker(max_violation_s=max_violation_s,
                                                  name="hand_detection")

    # ── Per-frame call ────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, timestamp_s: float) -> Tuple[float, bool, dict, object]:
        """
        Analyse one frame.

        Returns
        -------
        score     : float  — tracker confidence in [0, 1]
        flagged   : bool   — True if violation threshold crossed
        sources   : dict   — {"left": tag, "right": tag}
        landmarks : list | None — NormalizedLandmarkList per detected hand
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tier 0: XR tracking data — if wrist world-position is available,
        # synthesise a phantom candidate so downstream tiers get a head start.
        tracking_boost = False
        if self._tracking and bool(self._tracking):
            left_xyz, right_xyz = self._tracking.at(timestamp_s)
            tracking_boost = (left_xyz is not None or right_xyz is not None)

        # Tier 1: MP Hands + augmentation retries
        score, candidates = self._run_hands(rgb)

        if score < RETRY_THRESHOLD:
            flipped = cv2.flip(rgb, 1)
            f_score, f_cands = self._run_hands(flipped)
            if f_score > score:
                _unflip_candidates(f_cands, self.W)
                score, candidates = f_score, f_cands

        if score < RETRY_THRESHOLD:
            enhanced = self._clahe_enhance(rgb)
            c_score, c_cands = self._run_hands(enhanced)
            if c_score > score:
                score, candidates = c_score, c_cands

        # Tier 3 (Pose): only when Hands is weak or only one hand found
        need_pose   = (score < POSE_TRIGGER_SCORE) or (len(candidates) < 2)
        pose_wrists = self._run_pose(rgb) if need_pose else []

        # Tier 2–5 via HandTracker
        out = self._tracker.update(
            hands_candidates=candidates,
            pose_wrist_candidates=pose_wrists,
            frame_bgr=frame,
        )

        # Violation = tracker confidence below 0.5 (hand not detected).
        # Exception: if Tier-0 XR tracking confirms a wrist is present,
        # treat the frame as passing even if vision detectors missed it.
        effective_score = max(out.score, 0.6) if tracking_boost else out.score
        is_absent = effective_score < 0.5
        flagged   = self._dur.update(is_absent, timestamp_s)

        return effective_score, flagged, (out.sources or {}), out.landmarks

    def finalize(self) -> ViolationSummary:
        """Call after all frames to close open segments and get the summary."""
        return self._dur.finalize()

    def close(self) -> None:
        self._hands.close()
        self._pose.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_hands(self, rgb: np.ndarray) -> Tuple[float, List[HandsCandidate]]:
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return 0.0, []
        handedness = result.multi_handedness or []
        candidates: List[HandsCandidate] = []
        for i, hand_lm in enumerate(result.multi_hand_landmarks):
            wrist    = hand_lm.landmark[0]
            wrist_xy = np.array([wrist.x * self.W, wrist.y * self.H], dtype=np.float32)
            conf = float(handedness[i].classification[0].score) if i < len(handedness) else 1.0
            candidates.append(HandsCandidate(
                wrist_xy=wrist_xy, score=conf, landmarks=hand_lm,
            ))
        return (max(c.score for c in candidates) if candidates else 0.0), candidates

    def _run_pose(self, rgb: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return []
        lm = result.pose_landmarks.landmark
        wrists = []
        for idx in (15, 16):   # LEFT_WRIST=15, RIGHT_WRIST=16
            kp = lm[idx]
            xy = np.array([kp.x * self.W, kp.y * self.H], dtype=np.float32)
            wrists.append((xy, float(kp.visibility)))
        return wrists

    def _clahe_enhance(self, rgb: np.ndarray) -> np.ndarray:
        lab      = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b  = cv2.split(lab)
        l        = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# ── Module-level helper ───────────────────────────────────────────────────────

def _unflip_candidates(candidates: List[HandsCandidate], frame_w: int) -> None:
    for cand in candidates:
        cand.wrist_xy[0] = float(frame_w) - cand.wrist_xy[0]
        for lm in cand.landmarks.landmark:
            lm.x = 1.0 - lm.x
