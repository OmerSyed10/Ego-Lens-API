"""
egolens/duration_tracker.py

DurationViolationTracker — only flags a problem when the bad condition
has been continuously present for ≥ max_violation_s seconds.

Violation segments are stored as dicts so they can carry arbitrary
per-segment metadata (e.g. bounding boxes for visual debugging).

Segment format:
    {
        "start_s":  float,
        "end_s":    float,
        "bboxes":   [{"frame_s": float, "bbox": [x1n, y1n, x2n, y2n]}, ...]
                    # bbox coords normalised to [0,1] relative to frame size
                    # only present for detectors that return bbox data
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ViolationSummary:
    """Result from DurationViolationTracker.finalize()."""
    # List of segment dicts: {start_s, end_s, bboxes:[{frame_s, bbox}]}
    violation_segments: List[Dict[str, Any]] = field(default_factory=list)
    # Frames that were individually flagged (violation crossed duration threshold)
    flagged_frame_count: int = 0
    # Total frames evaluated
    total_frames: int = 0
    # Detector name — set automatically by DurationViolationTracker.finalize()
    detector_name: str = ""

    @property
    def total_violation_s(self) -> float:
        return sum(s["end_s"] - s["start_s"] for s in self.violation_segments)

    @property
    def violation_count(self) -> int:
        return len(self.violation_segments)

    @property
    def pass_rate(self) -> float:
        if self.total_frames == 0:
            return 1.0
        return 1.0 - (self.flagged_frame_count / self.total_frames)

    def __str__(self) -> str:
        segs = ", ".join(
            f"{s['start_s']:.1f}s–{s['end_s']:.1f}s"
            for s in self.violation_segments
        )
        return (f"violations={self.violation_count}  "
                f"total={self.total_violation_s:.1f}s  "
                f"segments=[{segs}]")


class DurationViolationTracker:
    """
    Tracks whether a bad condition has been continuously present long enough
    to count as a real violation.

    Parameters
    ----------
    max_violation_s : float
        Minimum continuous seconds of a bad condition before it is flagged.
        0.0 = flag every bad frame immediately.
    name : str
        Optional label for debugging / segment metadata.
    """

    def __init__(self, max_violation_s: float = 1.0, name: str = "") -> None:
        self.max_violation_s = max(0.0, max_violation_s)
        self.name = name
        self._reset_state()

    def _reset_state(self) -> None:
        self._violation_start: Optional[float] = None
        self._segments: List[Dict[str, Any]] = []
        self._current_bboxes: List[Dict[str, Any]] = []
        self._flagged_frames: int = 0
        self._total_frames: int = 0
        self._last_ts: float = 0.0

    def update(
        self,
        is_violation: bool,
        timestamp_s: float,
        bbox: Optional[List[float]] = None,
    ) -> bool:
        """
        Call once per sampled frame.

        Parameters
        ----------
        is_violation : bool   True = the bad condition is present this frame
        timestamp_s  : float  Timestamp of this frame in the video (seconds)
        bbox         : list   Optional [x1n,y1n,x2n,y2n] normalised bbox
                              (only relevant for face / person detectors)

        Returns
        -------
        bool : True if this frame is considered FLAGGED
               (i.e. the violation has lasted ≥ max_violation_s)
        """
        self._total_frames += 1
        self._last_ts = timestamp_s
        flagged = False

        if is_violation:
            if self._violation_start is None:
                self._violation_start = timestamp_s
                self._current_bboxes = []

            # Accumulate up to 10 bboxes per segment (spread across the segment)
            if bbox is not None and len(self._current_bboxes) < 10:
                self._current_bboxes.append(
                    {"frame_s": round(timestamp_s, 3), "bbox": bbox}
                )

            continuous_s = timestamp_s - self._violation_start
            if continuous_s >= self.max_violation_s:
                flagged = True
                self._flagged_frames += 1
        else:
            if self._violation_start is not None:
                self._segments.append({
                    "start_s": round(self._violation_start, 3),
                    "end_s":   round(timestamp_s, 3),
                    "bboxes":  list(self._current_bboxes),
                })
                self._violation_start = None
                self._current_bboxes = []

        return flagged

    def finalize(self) -> ViolationSummary:
        """
        Call after all frames are processed.
        Closes any open segment and returns the full summary.
        """
        if self._violation_start is not None:
            self._segments.append({
                "start_s": round(self._violation_start, 3),
                "end_s":   round(self._last_ts, 3),
                "bboxes":  list(self._current_bboxes),
            })
            self._violation_start = None
            self._current_bboxes = []

        return ViolationSummary(
            violation_segments=list(self._segments),
            flagged_frame_count=self._flagged_frames,
            total_frames=self._total_frames,
            detector_name=self.name,
        )

    def reset(self) -> None:
        """Reset all state (reuse tracker for a new video)."""
        self._reset_state()
