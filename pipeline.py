"""
egolens/pipeline.py — EgoLensPipeline

Orchestrates all detectors on a single video file:

  1. Open video, run VideoPreprocessor.inspect()
  2. Compute frame stride for target sampling fps
  3. Loop over sampled frames with tqdm progress bar
  4. Per sampled frame: run all ENABLED detectors
  5. After all frames: finalize each detector → ViolationSummary
  6. Return AnalysisResult (raw scores + violation summaries per detector)

The pipeline is stateless between videos — create a new instance per video,
or call reset() before reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .config import EgoLensConfig
from .preprocessor import VideoPreprocessor, VideoMeta
from .duration_tracker import ViolationSummary
from .detectors.brightness_detector import BrightnessDetector
from .detectors.sharpness  import SharpnessMetric
from .detectors.stability  import StabilityMetric
from .detectors.hand_detector import HandDetector
from .detectors.face_detector import FaceDetector
from .detectors.person_detector import PersonDetector
from .detectors.hand_speed_detector import HandSpeedDetector
from .detectors.hand_position_detector import HandPositionDetector
from .detectors.hand_out_of_frame_detector import HandOutOfFrameDetector
from .tracking_loader import TrackingData


# ── Normalisation helpers ────────────────────────────────────────────────────
#
# Fixed-calibration curves derived from real egocentric footage so that scores
# are interpretable across different videos (not just relative within a single video).
#
# Sharpness  — sigmoid:    norm = sigmoid((raw - 27) / 10.5)
#   Calibrated so 16 px²→0.27 (soft fail), 41 px²→0.80 (clear pass), 173 px²→0.98
#
# Stability  — exponential: norm = exp(-raw / 35)
#   Calibrated so 2.95 px/frame→0.92 (stable), 12 px/frame→0.71 (borderline),
#   20 px/frame→0.56 (shaky)
#
# Hand speed — tanh: norm = tanh(raw / 80)
#   0 px/frame→0.0, 65 px/frame→0.55, 140 px/frame→0.84

_SHARP_MID   = 27.0    # raw sharpness at norm=0.5
_SHARP_SCALE = 10.5    # sigmoid steepness

_STAB_DECAY  = 35.0    # exponential half-life for flow magnitude (px/frame)

_SPEED_SCALE = 80.0    # tanh scale for hand speed (px/frame)


def _norm_sharpness(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    return np.clip(1.0 / (1.0 + np.exp(-(arr - _SHARP_MID) / _SHARP_SCALE)), 0.0, 1.0).tolist()


def _norm_stability(values: List[float]) -> List[float]:
    """Lower flow = more stable = higher score."""
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    return np.clip(np.exp(-arr / _STAB_DECAY), 0.0, 1.0).tolist()


def _norm_speed(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    return np.clip(np.tanh(arr / _SPEED_SCALE), 0.0, 1.0).tolist()


# ── Per-frame results ────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    frame_idx:   int
    timestamp_s: float
    # Raw per-detector values (before normalization)
    brightness_norm: float = 0.0   # normalised [0,1] luminance
    sharpness_raw:   float = 0.0
    stability_raw:   float = 0.0
    hand_score:      float = 0.0
    hand_sources:    dict  = field(default_factory=dict)
    face_score:      float = 1.0
    person_score:    float = 1.0
    hand_speed_raw:  float = 0.0
    hand_position:   float = 0.0
    hand_oof:        float = 1.0   # out-of-frame: 1.0 = OK
    # landmarks from hand detector (for downstream use by speed/position/oof)
    landmarks:       object = None


# ── Main result object ───────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    """Per-detector aggregated result."""
    name:             str
    enabled:          bool
    raw_values:       List[float] = field(default_factory=list)
    norm_values:      List[float] = field(default_factory=list)
    pass_count:       int = 0
    total_count:      int = 0
    violation:        Optional[ViolationSummary] = None
    gate:             float = 0.5
    weight:           float = 1.0

    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.pass_count / self.total_count

    @property
    def mean_norm(self) -> float:
        if not self.norm_values:
            return 0.0
        return float(np.mean(self.norm_values))


@dataclass
class AnalysisResult:
    video_path:    str
    meta:          VideoMeta
    frames:        List[FrameRecord]
    detectors:     Dict[str, DetectorResult]
    overall_score: float = 0.0          # 0–100, computed by scorer
    passed:        bool  = False


# ── Pipeline ─────────────────────────────────────────────────────────────────

class EgoLensPipeline:
    """
    Usage
    -----
    pipeline = EgoLensPipeline(cfg)
    result   = pipeline.run("video.mp4")
    """

    def __init__(self, cfg: EgoLensConfig) -> None:
        self.cfg = cfg

    def run(
        self,
        video_path: str,
        verbose: bool = True,
        tracking_data: Optional[TrackingData] = None,
    ) -> AnalysisResult:
        """Analyse a single video. Returns AnalysisResult."""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        try:
            return self._run(cap, video_path, verbose, tracking_data=tracking_data)
        finally:
            cap.release()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        verbose: bool,
        tracking_data: Optional[TrackingData] = None,
    ) -> AnalysisResult:
        cfg = self.cfg

        # 1. Inspect video (stereo detection, crop/scale decisions)
        pre  = VideoPreprocessor(cfg.video)
        meta = pre.inspect(cap)

        # 2. Compute frame stride
        stride = max(1, round(meta.fps / cfg.sampling.fps))
        n_frames = int(math.ceil(meta.total_frames / stride))

        if verbose:
            print(f"\n  → Analysing {video_path.split('/')[-1]} …")
            print(f"    {pre.describe()}")
            print(f"    {meta.fps:.1f} fps → stride {stride} → "
                  f"{meta.fps/stride:.1f} fps effective  |  ~{n_frames} frames")

        # 3. Initialise detectors
        det_bright = BrightnessDetector(
            max_violation_s=cfg.violation_s("brightness"),
        ) if cfg.is_enabled("brightness") else None

        det_hand  = HandDetector(
            frame_w=meta.effective_w,
            frame_h=meta.effective_h,
            max_violation_s=cfg.violation_s("hand_detection"),
            tracking_data=tracking_data,
        ) if cfg.is_enabled("hand_detection") else None

        det_face   = FaceDetector(
            max_violation_s=cfg.violation_s("face_detection"),
        ) if cfg.is_enabled("face_detection") else None

        det_person = PersonDetector(
            max_violation_s=cfg.violation_s("person_detection"),
        ) if cfg.is_enabled("person_detection") else None

        det_speed  = HandSpeedDetector(
            frame_w=meta.effective_w,
            frame_h=meta.effective_h,
            max_violation_s=cfg.violation_s("hand_speed"),
        ) if cfg.is_enabled("hand_speed") else None

        det_pos    = HandPositionDetector(
            x_lo=cfg.workspace.x_lo,
            x_hi=cfg.workspace.x_hi,
            y_lo=cfg.workspace.y_lo,
            y_hi=cfg.workspace.y_hi,
            max_violation_s=cfg.violation_s("hand_position"),
        ) if cfg.is_enabled("hand_position") else None

        det_oof    = HandOutOfFrameDetector(
            margin=cfg.frame_edge.margin,
            max_violation_s=cfg.violation_s("hand_out_of_frame"),
        ) if cfg.is_enabled("hand_out_of_frame") else None

        sharp_m  = SharpnessMetric()
        stab_m   = StabilityMetric()

        # Helper: get gate value for a detector from config
        def _get_gate_now(name: str, _cfg, default: float) -> float:
            d = _cfg.detectors.get(name)
            return d.gate if d else default

        # 4. Frame loop
        records: List[FrameRecord] = []
        frame_idx  = -1
        sample_idx = -1

        bar = _make_bar(n_frames, "  Frames", verbose)

        while True:
            ret, frame_raw = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            sample_idx += 1

            frame = pre.process(frame_raw)
            ts    = frame_idx / meta.fps

            rec = FrameRecord(frame_idx=frame_idx, timestamp_s=ts)

            # Brightness
            if det_bright is not None:
                bright_gate = _get_gate_now("brightness", cfg, 0.30)
                rec.brightness_norm, _ = det_bright.update(frame, ts, gate=bright_gate)

            # Sharpness
            if cfg.is_enabled("sharpness"):
                rec.sharpness_raw = sharp_m.compute(frame)

            # Stability
            if cfg.is_enabled("stability"):
                rec.stability_raw = stab_m.compute(frame)

            # Hand detection (also provides landmarks for downstream)
            if det_hand is not None:
                rec.hand_score, _, rec.hand_sources, rec.landmarks = det_hand.update(frame, ts)

            # Face
            if det_face is not None:
                rec.face_score, _ = det_face.update(frame, ts)

            # Person
            if det_person is not None:
                rec.person_score, _ = det_person.update(frame, ts)

            # Hand speed (needs landmarks from hand detection)
            if det_speed is not None:
                rec.hand_speed_raw, _, _ = det_speed.update(rec.landmarks, ts)

            # Hand position (needs landmarks)
            if det_pos is not None:
                rec.hand_position, _ = det_pos.update(rec.landmarks, ts)

            # Hand out-of-frame (needs landmarks)
            if det_oof is not None:
                rec.hand_oof, _ = det_oof.update(rec.landmarks, ts)

            records.append(rec)
            if bar is not None:
                bar.update(1)

        if bar is not None:
            bar.close()

        # 5. Finalise detectors → violation summaries
        v_bright = det_bright.finalize() if det_bright else None
        v_hand   = det_hand.finalize()   if det_hand   else None
        v_face   = det_face.finalize()   if det_face   else None
        v_person = det_person.finalize() if det_person else None
        v_speed  = det_speed.finalize()  if det_speed  else None
        v_pos    = det_pos.finalize()    if det_pos    else None
        v_oof    = det_oof.finalize()    if det_oof    else None

        # Close MediaPipe handles
        for d in [det_hand, det_face, det_person]:
            if d is not None:
                d.close()

        # 6. Build DetectorResult objects with normalization
        bright_raws = [r.brightness_norm for r in records]   # already normalised
        sharp_raws  = [r.sharpness_raw   for r in records]
        stab_raws   = [r.stability_raw   for r in records]
        hand_raws   = [r.hand_score      for r in records]
        face_raws   = [r.face_score      for r in records]
        person_raws = [r.person_score    for r in records]
        speed_raws  = [r.hand_speed_raw  for r in records]
        pos_raws    = [r.hand_position   for r in records]
        oof_raws    = [r.hand_oof        for r in records]

        # Normalise to [0, 1] using fixed-calibration curves
        # Brightness is already [0,1] (raw/255), store raw=norm for consistency
        bright_norm = bright_raws
        sharp_norm  = _norm_sharpness(sharp_raws)
        stab_norm   = _norm_stability(stab_raws)
        # Hand/face/person already in [0, 1]
        hand_norm   = hand_raws
        face_norm   = face_raws
        person_norm = person_raws
        speed_norm  = _norm_speed(speed_raws)
        pos_norm    = pos_raws
        oof_norm    = oof_raws

        def _gate(name: str) -> float:
            d = cfg.detectors.get(name)
            return d.gate if d else 0.5

        def _weight(name: str) -> float:
            d = cfg.detectors.get(name)
            return d.weight if d else 1.0

        def _build(
            name: str,
            enabled: bool,
            raws: List[float],
            norms: List[float],
            violation: Optional[ViolationSummary],
        ) -> DetectorResult:
            gate = _gate(name)
            passes = sum(1 for n in norms if n >= gate)
            return DetectorResult(
                name=name,
                enabled=enabled,
                raw_values=raws,
                norm_values=norms,
                pass_count=passes,
                total_count=len(norms),
                violation=violation,
                gate=gate,
                weight=_weight(name),
            )

        N = len(records)
        detectors_map: Dict[str, DetectorResult] = {
            "brightness":      _build("brightness",       cfg.is_enabled("brightness"),      bright_raws, bright_norm, v_bright),
            "sharpness":       _build("sharpness",        cfg.is_enabled("sharpness"),       sharp_raws,  sharp_norm,  None),
            "stability":       _build("stability",        cfg.is_enabled("stability"),       stab_raws,   stab_norm,   None),
            "hand_detection":  _build("hand_detection",   cfg.is_enabled("hand_detection"),  hand_raws,   hand_norm,   v_hand),
            "face_detection":  _build("face_detection",   cfg.is_enabled("face_detection"),  face_raws,   face_norm,   v_face),
            "person_detection":_build("person_detection", cfg.is_enabled("person_detection"),person_raws, person_norm, v_person),
            "hand_speed":      _build("hand_speed",       cfg.is_enabled("hand_speed"),      speed_raws,  speed_norm,  v_speed),
            "hand_position":   _build("hand_position",    cfg.is_enabled("hand_position"),   pos_raws,    pos_norm,    v_pos),
            "hand_out_of_frame":_build("hand_out_of_frame",cfg.is_enabled("hand_out_of_frame"),oof_raws,  oof_norm,    v_oof),
        }

        result = AnalysisResult(
            video_path=video_path,
            meta=meta,
            frames=records,
            detectors=detectors_map,
        )
        return result


# ── Progress bar ─────────────────────────────────────────────────────────────

def _make_bar(total: int, desc: str, verbose: bool):
    if not verbose:
        return None
    if _TQDM_AVAILABLE:
        return tqdm(total=total, desc=desc, unit="fr", ncols=72, leave=False)
    return None
