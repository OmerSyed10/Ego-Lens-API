"""
egolens/detectors/face_verifier.py — RetinaFace secondary face verifier

Uses InsightFace's RetinaFace (det_500m ONNX model, ~14 MB) to confirm
whether a face is actually present when MediaPipe FaceDetection flags one.

Two-stage strategy:
  1. MediaPipe flags a face → run RetinaFace
  2. Violation only if RetinaFace ALSO confirms the face

Returns:
  confirmed : bool            — True = face confirmed
  bbox_norm : [x1,y1,x2,y2] — normalised [0,1] coords, or None

Graceful degradation:
  - If insightface is not installed → returns (True, None) [trust primary]
  - If model download fails        → returns (True, None) [trust primary]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False

_MODEL_NAME = "buffalo_sc"   # lightweight det-only model (~14 MB download)
_DET_SIZE   = (640, 640)


class FaceVerifier:
    """
    RetinaFace (via InsightFace buffalo_sc) secondary face verifier.

    Parameters
    ----------
    conf : float  — minimum detection confidence (InsightFace det_thresh)
    """

    def __init__(self, conf: float = 0.5) -> None:
        self._conf      = conf
        self._app       = None
        self._available = _INSIGHTFACE_AVAILABLE
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _INSIGHTFACE_AVAILABLE:
            print("[FaceVerifier] insightface not installed — "
                  "pip install insightface onnxruntime")
            return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                app = _FaceAnalysis(
                    name=_MODEL_NAME,
                    allowed_modules=["detection"],
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=-1, det_size=_DET_SIZE, det_thresh=self._conf)
            self._app = app
            print("[FaceVerifier] RetinaFace (InsightFace buffalo_sc) loaded ✓")
        except Exception as exc:
            print(f"[FaceVerifier] Load failed: {exc} — disabling")
            self._app       = None
            self._available = False

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Optional[List[float]]]:
        """
        Run RetinaFace on one frame.

        Parameters
        ----------
        frame_bgr : H×W×3 BGR uint8 array

        Returns
        -------
        (confirmed, bbox_norm)
          confirmed  : True = face detected by RetinaFace
          bbox_norm  : [x1n, y1n, x2n, y2n] in [0,1] coords, or None
        """
        if not self._available or self._app is None:
            return True, None   # trust primary if verifier unavailable

        try:
            import cv2
            # InsightFace expects BGR (same as OpenCV) — pass directly
            faces = self._app.get(frame_bgr)
            if not faces:
                return False, None

            h, w = frame_bgr.shape[:2]
            # Pick highest-confidence detection
            best = max(faces, key=lambda f: float(f.det_score))
            x1, y1, x2, y2 = best.bbox.tolist()
            bbox_norm = [
                round(max(0.0, x1) / w, 4),
                round(max(0.0, y1) / h, 4),
                round(min(1.0, x2 / w), 4),
                round(min(1.0, y2 / h), 4),
            ]
            return True, bbox_norm

        except Exception:
            return True, None   # on error, trust primary

    @property
    def available(self) -> bool:
        return self._available and self._app is not None
