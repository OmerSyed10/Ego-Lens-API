"""
Hand Tracker — 5-tier cascade with per-hand position memory.

For each analysed frame the tracker runs a cascade over up to two hands
(left, right) tracked independently:

  Tier 1 — MP Hands wrist:        primary detector, gives full landmarks
  Tier 2 — 100DOH (hand+obj det): fires on palm/wrist even when fingers hidden
                                   (gripping, tool use, fist — 90% AP egocentric)
  Tier 3 — MP Pose wrist:         fallback from arm geometry (works on fists)
  Tier 4 — YCbCr skin blob:       appearance fallback (lighting/dark skin)
  Tier 5 — Phantom coast:         short-lived velocity extrapolation through
                                  brief occlusion (declines linearly, dies in
                                  MAX_STALE_FRAMES)

Every tier validates the new position against the track's plausible-motion
radius — so MP Hands "teleports" (left/right confusion, spurious blobs) and
hallucinations from any tier are rejected. Sides are assigned to tracks by
proximity, not by detector-reported handedness.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterable

# mediapipe.framework.formats.landmark_pb2 was removed in mediapipe 0.10+.
# We define lightweight replacements so synthetic landmarks keep the same API
# (.landmark list with .x / .y / .z attributes) that downstream detectors use.
from dataclasses import dataclass as _dc, field as _field
from typing import List as _List

@_dc
class _NLM:
    """Normalised landmark point — mimics protobuf NormalizedLandmark."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@_dc
class _NLMList:
    """Normalised landmark list — mimics protobuf NormalizedLandmarkList."""
    landmark: _List[_NLM] = _field(default_factory=list)


# ── Tuning constants ─────────────────────────────────────────────────────────
POSE_VIS_THRESHOLD = 0.4     # min keypoint visibility to trust a Pose wrist
MAX_STALE_FRAMES   = 5       # phantom budget before declaring track lost
VEL_ALPHA          = 0.5     # EMA factor for velocity smoothing
RADIUS_FRAC        = 0.15    # per-frame plausible-motion radius (frac of min(W,H))
RADIUS_SLACK_PX    = 30      # absolute slack added to plausible radius
SYNTH_LANDMARK_COUNT = 21    # MediaPipe Hands' standard landmark count

# Source tags surfaced for debugging / future per-frame attribution
SOURCE_NONE    = "none"
SOURCE_HANDS   = "hands"
SOURCE_DOH     = "doh"
SOURCE_POSE    = "pose"
SOURCE_SKIN    = "skin"
SOURCE_PHANTOM = "phantom"


@dataclass
class HandsCandidate:
    """A MediaPipe Hands detection ready for tracker consumption."""
    wrist_xy:  np.ndarray   # (x, y) in pixels
    score:     float        # MP-Hands handedness confidence
    landmarks: object       # full NormalizedLandmarkList — passed through


@dataclass
class HandTrack:
    """Per-hand state. left and right are tracked independently."""
    last_xy:         Optional[np.ndarray] = None
    velocity:        np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    last_frame_idx:  int   = -1
    last_confidence: float = 0.0
    staleness:       int   = 0
    active:          bool  = False
    last_bbox:       Optional[Tuple[float, float, float, float]] = None  # for skin filter

    def confirm(self, xy: np.ndarray, frame_idx: int, confidence: float,
                bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        if self.active and self.last_xy is not None:
            elapsed = max(1, frame_idx - self.last_frame_idx)
            new_vel = (xy - self.last_xy) / elapsed
            self.velocity = VEL_ALPHA * new_vel + (1.0 - VEL_ALPHA) * self.velocity
        self.last_xy         = xy.astype(np.float32)
        self.last_frame_idx  = frame_idx
        self.last_confidence = float(confidence)
        self.staleness       = 0
        self.active          = True
        if bbox is not None:
            self.last_bbox = bbox

    def reset(self) -> None:
        self.last_xy         = None
        self.velocity        = np.zeros(2, dtype=np.float32)
        self.last_frame_idx  = -1
        self.last_confidence = 0.0
        self.staleness       = 0
        self.active          = False
        self.last_bbox       = None


@dataclass
class TrackerFrame:
    """Per-frame tracker output."""
    score:         float              # max detection confidence across hands
    landmarks:     Optional[List]     # MP landmarks from Tier-1 hits only
    sources:       Dict[str, str]     # {"left": tag, "right": tag}
    contact_state: Optional[int] = None  # 100DOH contact state (Tier-2 hit)


class HandTracker:
    """Manages two HandTrack instances (left, right) and runs the 5-tier cascade."""

    SIDES = ("left", "right")

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        doh_detector=None,     # Optional[DOHDetector]
    ) -> None:
        self.W = int(frame_w)
        self.H = int(frame_h)
        self._radius_per_frame = RADIUS_FRAC * min(frame_w, frame_h)
        self._tracks: Dict[str, HandTrack] = {s: HandTrack() for s in self.SIDES}
        self._frame_idx = 0
        self._doh = doh_detector

        # Skin detector — always available (pure numpy/opencv)
        from .skin_detector import SkinDetector
        self._skin = SkinDetector(min_area=1500)

    # ── Synthetic landmarks ──────────────────────────────────────────────────
    def _synthesize_wrist_landmarks(self, wrist_xy_px: np.ndarray) -> object:
        """Build a 21-point NormalizedLandmarkList stacked at the wrist position."""
        x_norm = float(np.clip(wrist_xy_px[0] / self.W, 0.0, 1.0))
        y_norm = float(np.clip(wrist_xy_px[1] / self.H, 0.0, 1.0))
        lm_list = _NLMList(landmark=[
            _NLM(x=x_norm, y=y_norm, z=0.0)
            for _ in range(SYNTH_LANDMARK_COUNT)
        ])
        return lm_list

    # ── Validation ───────────────────────────────────────────────────────────
    def _within_radius(self, track: HandTrack, xy: np.ndarray, frame_idx: int) -> bool:
        """Is `xy` close enough to where this track *should* be by now?"""
        if not track.active:
            return True  # first detection — accept anywhere
        elapsed = max(1, frame_idx - track.last_frame_idx)
        predicted = track.last_xy + track.velocity * elapsed
        radius = self._radius_per_frame * elapsed + RADIUS_SLACK_PX
        return float(np.linalg.norm(xy - predicted)) <= radius

    # ── Side assignment ──────────────────────────────────────────────────────
    def _assign_by_proximity(
        self,
        positions: List[np.ndarray],
        available_sides: Iterable[str],
    ) -> Dict[str, int]:
        """Greedy nearest-neighbour assignment. Leftovers fill free sides by x-order."""
        available = list(available_sides)
        if not positions or not available:
            return {}

        assigned: Dict[str, int] = {}
        remaining = list(range(len(positions)))

        # Pull each active track toward its nearest candidate
        for side in available:
            track = self._tracks[side]
            if not track.active or not remaining:
                continue
            best_ci, best_d = None, float("inf")
            for ci in remaining:
                d = float(np.linalg.norm(positions[ci] - track.last_xy))
                if d < best_d:
                    best_d = d
                    best_ci = ci
            if best_ci is not None:
                assigned[side] = best_ci
                remaining.remove(best_ci)

        # Leftover candidates fill free sides sorted by x-position
        free = [s for s in available if s not in assigned]
        remaining_sorted = sorted(remaining, key=lambda ci: float(positions[ci][0]))
        if len(remaining_sorted) == 1 and len(free) == 2:
            ci = remaining_sorted[0]
            side = "left" if float(positions[ci][0]) < self.W / 2 else "right"
            assigned[side if side in free else free[0]] = ci
        else:
            for ci, side in zip(remaining_sorted, free):
                assigned[side] = ci
        return assigned

    # ── Public API ───────────────────────────────────────────────────────────
    def update(
        self,
        hands_candidates: List[HandsCandidate],
        pose_wrist_candidates: List[Tuple[np.ndarray, float]],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> TrackerFrame:
        """
        Run the 5-tier cascade for one frame.

        hands_candidates:        list of HandsCandidate from MP Hands
        pose_wrist_candidates:   list of (wrist_xy_px, visibility) from MP Pose
                                 (pass [] when Pose wasn't run)
        frame_bgr:               raw BGR frame for DOH and skin inference
                                 (pass None to skip those tiers)
        """
        self._frame_idx += 1
        frame_idx = self._frame_idx

        # (confidence, source_tag, landmarks_or_None)
        confirmed: Dict[str, Tuple[float, str, Optional[object]]] = {}
        contact_state: Optional[int] = None

        # ── Tier 1: MP Hands ─────────────────────────────────────────────────
        if hands_candidates:
            positions  = [c.wrist_xy for c in hands_candidates]
            assignment = self._assign_by_proximity(positions, self.SIDES)
            for side, ci in assignment.items():
                cand  = hands_candidates[ci]
                track = self._tracks[side]
                if self._within_radius(track, cand.wrist_xy, frame_idx):
                    track.confirm(cand.wrist_xy, frame_idx, cand.score)
                    confirmed[side] = (cand.score, SOURCE_HANDS, cand.landmarks)

        # ── Tier 2: 100DOH ───────────────────────────────────────────────────
        if frame_bgr is not None and self._doh is not None:
            free_sides = [s for s in self.SIDES if s not in confirmed]
            if free_sides:
                try:
                    doh_results = self._doh.detect(frame_bgr)
                except Exception:
                    doh_results = []
                if doh_results:
                    positions = [r.wrist_xy for r in doh_results]
                    assignment = self._assign_by_proximity(positions, free_sides)
                    for side, ci in assignment.items():
                        r     = doh_results[ci]
                        track = self._tracks[side]
                        if self._within_radius(track, r.wrist_xy, frame_idx):
                            track.confirm(r.wrist_xy, frame_idx, r.score,
                                          bbox=(r.x1, r.y1, r.x2, r.y2))
                            synth_lm = self._synthesize_wrist_landmarks(r.wrist_xy)
                            confirmed[side] = (r.score, SOURCE_DOH, synth_lm)
                            if contact_state is None:
                                contact_state = r.contact_state

        # ── Tier 3: MP Pose wrist ────────────────────────────────────────────
        if pose_wrist_candidates:
            valid     = [(xy, vis) for xy, vis in pose_wrist_candidates
                         if vis >= POSE_VIS_THRESHOLD]
            free_sides = [s for s in self.SIDES if s not in confirmed]
            if valid and free_sides:
                positions  = [v[0] for v in valid]
                assignment = self._assign_by_proximity(positions, free_sides)
                for side, ci in assignment.items():
                    xy, vis = valid[ci]
                    track   = self._tracks[side]
                    if self._within_radius(track, xy, frame_idx):
                        conf = float(0.5 + 0.5 * vis)
                        track.confirm(xy, frame_idx, conf)
                        synth_lm = self._synthesize_wrist_landmarks(xy)
                        confirmed[side] = (conf, SOURCE_POSE, synth_lm)

        # ── Tier 4: YCbCr skin blobs ─────────────────────────────────────────
        # Skin is appearance-only and noisy (arm/face blobs common). We provide
        # a landmark and detection signal but deliberately do NOT call
        # track.confirm() — this prevents a wrongly-placed blob from anchoring
        # the track's position and later rejecting true MP Hands/DOH hits via
        # the radius check.
        if frame_bgr is not None:
            free_sides = [s for s in self.SIDES if s not in confirmed]
            if free_sides:
                # Anchor skin search to last-known bbox to reduce false positives.
                # If no previous bbox exists, skip skin tier — without an anchor
                # the skin detector fires on arms/faces and can poison fresh tracks.
                prev_bboxes = [
                    t.last_bbox for t in self._tracks.values()
                    if t.last_bbox is not None
                ]
                if not prev_bboxes:
                    blobs = []  # no anchor — skip skin to avoid false track init
                else:
                    try:
                        blobs = self._skin.detect(frame_bgr, prev_bboxes)
                    except Exception:
                        blobs = []
                if blobs:
                    positions  = [b.xy for b in blobs]
                    assignment = self._assign_by_proximity(positions, free_sides)
                    for side, ci in assignment.items():
                        blob  = blobs[ci]
                        track = self._tracks[side]
                        if self._within_radius(track, blob.xy, frame_idx):
                            conf = 0.45  # just above gate; appearance-only
                            # Do NOT call track.confirm() — preserve existing
                            # position estimate so future precise detections
                            # are not blocked by a noisy skin blob anchor.
                            synth_lm = self._synthesize_wrist_landmarks(blob.xy)
                            confirmed[side] = (conf, SOURCE_SKIN, synth_lm)

        # ── Tier 5: Phantom coast ─────────────────────────────────────────────
        for side in self.SIDES:
            if side in confirmed:
                continue
            track = self._tracks[side]
            if not track.active:
                continue
            if track.staleness >= MAX_STALE_FRAMES:
                track.reset()
                continue
            decay       = 1.0 - (track.staleness / MAX_STALE_FRAMES)
            conf        = float(track.last_confidence * decay)
            elapsed     = max(1, frame_idx - track.last_frame_idx)
            predicted   = track.last_xy + track.velocity * elapsed
            track.staleness += 1
            synth_lm = self._synthesize_wrist_landmarks(predicted)
            confirmed[side] = (conf, SOURCE_PHANTOM, synth_lm)

        # ── Aggregate ─────────────────────────────────────────────────────────
        sources: Dict[str, str] = {s: SOURCE_NONE for s in self.SIDES}
        landmarks_list: List[object] = []
        max_score = 0.0
        for side, (conf, src, lm) in confirmed.items():
            sources[side] = src
            if lm is not None:
                landmarks_list.append(lm)
            if conf > max_score:
                max_score = conf

        return TrackerFrame(
            score=max_score,
            landmarks=landmarks_list if landmarks_list else None,
            sources=sources,
            contact_state=contact_state,
        )
