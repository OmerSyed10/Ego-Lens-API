"""
egolens/detectors/person_verifier.py — YOLOv8n secondary person verifier

Confirms whether another SEPARATE person (not the camera wearer) is visible.

────────────────────────────────────────────────────────────────────────
LAYERED ANTI-FALSE-POSITIVE STRATEGY
────────────────────────────────────────────────────────────────────────

Hard gates (all must pass — instant reject if any fails):
  1. Confidence ≥ 0.60
       Own-body detections (hands/arms) score 0.40-0.55 under YOLOv8n.
       Raising threshold eliminates the bulk of them.

  2. Aspect ratio h/w ≥ 0.40
       Worker's hands on a table produce a wide flat box (h << w).
       A real upright person has h ≥ 0.40 × w.

  3. Box area ≤ 40 % of frame
       Own torso close to the egocentric lens fills the frame.
       A real nearby person at 0.5+ m never exceeds 40 % of area.

  4. Entire box below 65 % of frame height
       If BOTH y1 AND y2 are below 0.65h the object is sitting in the
       bottom-third of the frame where own hands live.
       Camera-tilt robustness: one above-midline corner saves a real person.

Evidence scoring (soft signals — combined gate):
  A. Position score = 1 - (centroid_y / h)   [0..1, higher = better]
       Own hands: centroid at 0.75h → score 0.25
       Real person: centroid at 0.40h → score 0.60
       Softens the fixed-threshold problem caused by camera tilt.

  B. Hand overlap veto  [0 or -1.5]
       Run a lightweight MediaPipe Hands check on the frame.
       If the candidate person bbox overlaps ≥ 40 % of the hand-region bbox
       AND the person bbox is < 5× the hand bbox area
       → strong negative signal (worker's own hands mis-classified).
       Handshake guard: skip veto when person bbox is 5× larger (incidental
       overlap from co-worker close to the worker's hands).

  C. Face-in-crop bonus  [+1.5 if face detected]
       Crop the person bbox and run RetinaFace on the crop.
       Face found → very strong confirmation (own hands never have a face).
       No face → NEUTRAL (person may have back turned, mask, helmet).
       Never used as a reject signal — only as a bonus.

Final decision:
  score = position_score + hand_overlap_score + face_bonus
  Accept if: score ≥ 1.0   OR   face_bonus > 0   (face always wins)

Graceful degradation:
  - ultralytics not installed → (True, None)  [trust primary]
  - model fails to load      → (True, None)
  - mediapipe unavailable    → hand check silently skipped
  - insightface unavailable  → face check silently skipped
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

_MODEL_CACHE       = os.path.join(os.path.expanduser("~"), ".cache", "egolens")
_PERSON_MODEL_NAME = "yolov8n.pt"
_PERSON_CLASS_ID   = 0   # COCO class 0 = person

# Scoring thresholds
_SCORE_ACCEPT          = 1.0   # combined score gate
_HAND_OVERLAP_THRESH   = 0.40  # IoU fraction above which hand overlap is a veto
_HAND_SIZE_RATIO_GUARD = 5.0   # person bbox must be < N× hand bbox to trigger veto
_FACE_CROP_MIN_PX      = 80    # skip face check on crops smaller than this
_FACE_BONUS            = 1.5
_HAND_PENALTY          = -1.5


class PersonVerifier:
    """
    Parameters
    ----------
    conf             : float — YOLO confidence threshold (default 0.60)
    iou              : float — NMS IoU threshold
    max_bbox_frac    : float — hard-reject boxes > this fraction of frame area
    min_aspect_ratio : float — hard-reject boxes with h/w < this (flat = hands)
    face_verifier    : FaceVerifier | None
                       Pass the pipeline's shared FaceVerifier instance to avoid
                       loading InsightFace a second time.  If None and insightface
                       is installed, a private instance is created.
    """

    def __init__(
        self,
        conf: float                = 0.60,
        iou:  float                = 0.45,
        max_bbox_frac: float       = 0.40,
        min_aspect_ratio: float    = 0.40,
        face_verifier              = None,   # type: Optional[FaceVerifier]
    ) -> None:
        self._conf             = conf
        self._iou              = iou
        self._max_bbox_frac    = max_bbox_frac
        self._min_aspect_ratio = min_aspect_ratio
        self._model            = None
        self._available        = _YOLO_AVAILABLE

        # MediaPipe Hands for own-body overlap check
        self._mp_hands = None
        self._init_mp_hands()

        # RetinaFace for face-in-crop bonus
        self._face_verifier = face_verifier
        if self._face_verifier is None:
            self._init_face_verifier()

        self._load()

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _YOLO_AVAILABLE:
            return
        os.makedirs(_MODEL_CACHE, exist_ok=True)
        model_path = os.path.join(_MODEL_CACHE, _PERSON_MODEL_NAME)
        try:
            self._model = _YOLO(
                model_path if os.path.exists(model_path) else _PERSON_MODEL_NAME
            )
        except Exception as exc:
            print(f"[PersonVerifier] Could not load YOLOv8n: {exc} — disabling")
            self._model     = None
            self._available = False

    def _init_mp_hands(self) -> None:
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.4,
            )
        except Exception:
            self._mp_hands = None   # silently degrade

    def _init_face_verifier(self) -> None:
        try:
            from .face_verifier import FaceVerifier
            self._face_verifier = FaceVerifier()
        except Exception:
            self._face_verifier = None   # silently degrade

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Optional[List[float]]]:
        """
        Run all anti-false-positive filters on one frame.

        Returns
        -------
        (confirmed, bbox_norm)
          confirmed  : True = a separate (non-self) person found
          bbox_norm  : [x1n,y1n,x2n,y2n] in [0,1], or None
        """
        if not self._available or self._model is None:
            return True, None

        try:
            import cv2
            h, w = frame_bgr.shape[:2]
            frame_area = h * w

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._model.predict(
                frame_rgb,
                conf=self._conf,
                iou=self._iou,
                classes=[_PERSON_CLASS_ID],
                verbose=False,
                stream=False,
            )

            # Pre-compute hand bbox once for all candidates (lazy, done once)
            hand_bbox_abs = self._detect_hand_bbox(frame_rgb)

            for r in results:
                if r.boxes is None:
                    continue

                # Sort by confidence descending — evaluate best candidate first
                boxes = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bw = x2 - x1
                    bh = y2 - y1
                    bbox_area = bw * bh

                    # ── Hard gates ────────────────────────────────────────────

                    # Gate 1: area (own torso fills lens)
                    if bbox_area / frame_area > self._max_bbox_frac:
                        continue

                    # Gate 2: aspect ratio (hands on table → flat box)
                    if bw > 0 and (bh / bw) < self._min_aspect_ratio:
                        continue

                    # Gate 3: entire box below 65 % frame height
                    # Both top AND bottom of box below the 65 % line → own body zone
                    # Mitigation: if only y2 is below (tall box spanning midline) → OK
                    if y1 > 0.65 * h and y2 > 0.65 * h:
                        continue

                    # ── Evidence scoring ──────────────────────────────────────

                    score = 0.0

                    # A. Position score: centroid_y relative to frame height
                    #    Own hands: centroid ~0.75h → score 0.25
                    #    Real person: centroid ~0.40h → score 0.60
                    centroid_y = (y1 + y2) / 2.0
                    score += max(0.0, 1.0 - (centroid_y / h))

                    # B. Hand overlap veto (skip if no hand detector)
                    if hand_bbox_abs is not None:
                        overlap = self._iou_overlap(
                            (x1, y1, x2, y2), hand_bbox_abs
                        )
                        # Size guard: if person bbox >> hand bbox, overlap is incidental
                        hx1, hy1, hx2, hy2 = hand_bbox_abs
                        hand_area = (hx2 - hx1) * (hy2 - hy1)
                        size_ratio = bbox_area / max(hand_area, 1.0)
                        if overlap >= _HAND_OVERLAP_THRESH and size_ratio < _HAND_SIZE_RATIO_GUARD:
                            score += _HAND_PENALTY   # -1.5

                    # C. Face-in-crop bonus (face found → strong confirmation)
                    face_bonus = 0.0
                    if self._face_verifier is not None:
                        crop_w = int(x2 - x1)
                        crop_h = int(y2 - y1)
                        if crop_w >= _FACE_CROP_MIN_PX and crop_h >= _FACE_CROP_MIN_PX:
                            ix1 = max(0, int(x1))
                            iy1 = max(0, int(y1))
                            ix2 = min(w, int(x2))
                            iy2 = min(h, int(y2))
                            crop = frame_bgr[iy1:iy2, ix1:ix2]
                            if crop.size > 0:
                                face_confirmed, _ = self._face_verifier.verify(crop)
                                if face_confirmed:
                                    face_bonus = _FACE_BONUS   # +1.5

                    score += face_bonus

                    # ── Decision ──────────────────────────────────────────────
                    # Accept if combined score clears threshold
                    # OR if face was confirmed (face always wins regardless of score)
                    if score >= _SCORE_ACCEPT or face_bonus > 0:
                        bbox_norm = [
                            round(max(0.0, x1) / w, 4),
                            round(max(0.0, y1) / h, 4),
                            round(min(1.0, x2 / w), 4),
                            round(min(1.0, y2 / h), 4),
                        ]
                        return True, bbox_norm

            return False, None

        except Exception:
            return True, None   # on error, trust primary

    @property
    def available(self) -> bool:
        return self._available and self._model is not None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_hand_bbox(
        self, frame_rgb: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Run MediaPipe Hands and return a single bounding box covering all
        detected hand landmarks (absolute pixel coords), or None.
        """
        if self._mp_hands is None:
            return None
        try:
            h, w = frame_rgb.shape[:2]
            result = self._mp_hands.process(frame_rgb)
            if not result.multi_hand_landmarks:
                return None

            xs, ys = [], []
            for hand_lm in result.multi_hand_landmarks:
                for lm in hand_lm.landmark:
                    xs.append(lm.x * w)
                    ys.append(lm.y * h)

            if not xs:
                return None

            # Pad 10 % around the landmark cloud
            pad_x = (max(xs) - min(xs)) * 0.10
            pad_y = (max(ys) - min(ys)) * 0.10
            return (
                max(0.0, min(xs) - pad_x),
                max(0.0, min(ys) - pad_y),
                min(float(w), max(xs) + pad_x),
                min(float(h), max(ys) + pad_y),
            )
        except Exception:
            return None

    @staticmethod
    def _iou_overlap(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """
        Intersection-over-union between two (x1,y1,x2,y2) boxes.
        Returns fraction of box_a covered by box_b (not symmetric IoU).
        Using intersection / area_a so that a small hand inside a large
        person-box doesn't get a falsely high score.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1.0)
        return inter / area_a
