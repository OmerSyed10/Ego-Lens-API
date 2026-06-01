"""
YCbCr skin detector — Stage 4 of the hand-detection cascade.

Appearance-based fallback: when MediaPipe Hands, 100DOH, and MP Pose all miss
(very dark skin, gloves excluded by the cascade, extreme lighting), a skin-colour
blob gives a rough wrist centroid that keeps tracking alive.

Algorithm
---------
1. Convert BGR → YCbCr
2. Threshold: Cb ∈ [77, 127], Cr ∈ [133, 173]  (Kovac et al. 2003 — robust across
   a wide range of skin tones under varied lighting)
3. Morphological open/close to remove noise
4. Find contours → centroids of blobs above min_area
5. Optionally filter by proximity to previous hand bbox
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── YCbCr thresholds (Kovac et al. 2003) ────────────────────────────────────
_CB_LO, _CB_HI = 77, 127
_CR_LO, _CR_HI = 133, 173

# Minimum blob area to be considered a hand (px²)
DEFAULT_MIN_AREA = 1500

# Morphological kernel size
_MORPH_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


@dataclass
class SkinBlob:
    """One skin-colour blob."""
    cx: float    # centroid x in pixels
    cy: float    # centroid y in pixels
    area: float  # blob area in pixels²
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=np.float32)


def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) of skin-coloured pixels.

    Parameters
    ----------
    frame_bgr : (H, W, 3) uint8 BGR frame

    Returns
    -------
    mask : (H, W) uint8  — 255 = skin, 0 = background
    """
    ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    # OpenCV YCrCb channel order: Y, Cr, Cb
    y_ch, cr_ch, cb_ch = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    mask = (
        (cb_ch >= _CB_LO) & (cb_ch <= _CB_HI) &
        (cr_ch >= _CR_LO) & (cr_ch <= _CR_HI)
    ).astype(np.uint8) * 255

    # Morphological clean-up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_K)
    return mask


def skin_blobs(
    mask: np.ndarray,
    min_area: float = DEFAULT_MIN_AREA,
) -> List[SkinBlob]:
    """
    Find contours in a skin mask and return blobs above `min_area`.

    Returns list sorted by area descending (largest blob first).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[SkinBlob] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(cnt)
        blobs.append(SkinBlob(
            cx=cx, cy=cy, area=area,
            bbox=(float(x), float(y), float(x + w), float(y + h)),
        ))
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


def filter_by_prev_bbox(
    blobs: List[SkinBlob],
    prev_bboxes: List[Tuple[float, float, float, float]],
    slack: float = 80.0,
) -> List[SkinBlob]:
    """
    Keep only blobs whose centroid falls within any previous hand bbox + slack.

    If `prev_bboxes` is empty, all blobs are returned (no prior track to anchor
    against — first-frame fallback).
    """
    if not prev_bboxes:
        return blobs
    kept: List[SkinBlob] = []
    for blob in blobs:
        for (x1, y1, x2, y2) in prev_bboxes:
            if (x1 - slack <= blob.cx <= x2 + slack and
                    y1 - slack <= blob.cy <= y2 + slack):
                kept.append(blob)
                break
    return kept


class SkinDetector:
    """
    Stateless convenience wrapper for the YCbCr skin cascade.

    Usage
    -----
    sd = SkinDetector(min_area=1500)
    blobs = sd.detect(frame_bgr, prev_bboxes=[(x1,y1,x2,y2), ...])
    # blobs[i].xy  → (cx, cy) wrist proxy in pixels
    """

    def __init__(self, min_area: float = DEFAULT_MIN_AREA) -> None:
        self.min_area = min_area

    def detect(
        self,
        frame_bgr: np.ndarray,
        prev_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[SkinBlob]:
        """
        Detect skin blobs in `frame_bgr`.

        If `prev_bboxes` is given (previous frame's hand bboxes), only blobs
        near those regions are returned — greatly reduces face/arm false positives.
        """
        mask = skin_mask(frame_bgr)
        blobs = skin_blobs(mask, self.min_area)
        if prev_bboxes is not None:
            blobs = filter_by_prev_bbox(blobs, prev_bboxes, slack=80.0)
        return blobs
