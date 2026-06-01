"""
egolens/config.py — ConfigLoader

Reads config.yaml, validates all fields, and exposes typed dataclasses
so the rest of the pipeline never touches raw dicts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

# Default config path (same directory as this file)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


# ── Typed dataclasses ────────────────────────────────────────────────────────

@dataclass
class VideoConfig:
    stereo_detection: bool = True
    stereo_eye: str = "right"           # "left" | "right"
    stereo_target_height: int = 1080


@dataclass
class SamplingConfig:
    fps: float = 3.0


@dataclass
class DurationConfig:
    default_max_violation_s: float = 1.0


@dataclass
class DetectorConfig:
    enabled: bool = True
    gate: float = 0.50
    max_violation_s: Optional[float] = None   # None → use DurationConfig.default
    weight: float = 1.0
    description: str = ""


@dataclass
class WorkspaceConfig:
    x_lo: float = 0.10
    x_hi: float = 0.90
    y_lo: float = 0.10
    y_hi: float = 0.90


@dataclass
class FrameEdgeConfig:
    margin: float = 0.05


@dataclass
class ScoringConfig:
    pass_threshold: float = 70.0       # 0–100


@dataclass
class EgoLensConfig:
    video:      VideoConfig    = field(default_factory=VideoConfig)
    sampling:   SamplingConfig = field(default_factory=SamplingConfig)
    duration:   DurationConfig = field(default_factory=DurationConfig)
    detectors:  Dict[str, DetectorConfig] = field(default_factory=dict)
    workspace:  WorkspaceConfig  = field(default_factory=WorkspaceConfig)
    frame_edge: FrameEdgeConfig  = field(default_factory=FrameEdgeConfig)
    scoring:    ScoringConfig    = field(default_factory=ScoringConfig)

    # Convenience: effective max_violation_s for a detector
    def violation_s(self, name: str) -> float:
        det = self.detectors.get(name)
        if det is None:
            return self.duration.default_max_violation_s
        return det.max_violation_s if det.max_violation_s is not None \
               else self.duration.default_max_violation_s

    def is_enabled(self, name: str) -> bool:
        det = self.detectors.get(name)
        return det.enabled if det else False


# ── Known detector names (used for validation) ───────────────────────────────
KNOWN_DETECTORS = {
    "brightness",
    "sharpness",
    "stability",
    "hand_detection",
    "hand_speed",
    "hand_position",
    "hand_out_of_frame",
    "face_detection",
    "person_detection",
}


# ── Loader ───────────────────────────────────────────────────────────────────

class ConfigLoader:
    """Load and validate egolens config.yaml."""

    def __init__(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        self.path = path

    def load(self) -> EgoLensConfig:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Config not found: {self.path}")

        with open(self.path, "r") as f:
            raw: dict = yaml.safe_load(f) or {}

        cfg = EgoLensConfig()

        # ── video ────────────────────────────────────────────────────
        v = raw.get("video", {})
        cfg.video = VideoConfig(
            stereo_detection=bool(v.get("stereo_detection", True)),
            stereo_eye=str(v.get("stereo_eye", "right")).lower(),
            stereo_target_height=int(v.get("stereo_target_height", 1080)),
        )
        if cfg.video.stereo_eye not in ("left", "right"):
            raise ValueError(f"video.stereo_eye must be 'left' or 'right', got '{cfg.video.stereo_eye}'")

        # ── sampling ─────────────────────────────────────────────────
        s = raw.get("sampling", {})
        cfg.sampling = SamplingConfig(
            fps=float(s.get("fps", 3.0)),
        )
        if cfg.sampling.fps <= 0:
            raise ValueError("sampling.fps must be > 0")

        # ── duration ─────────────────────────────────────────────────
        d = raw.get("duration", {})
        cfg.duration = DurationConfig(
            default_max_violation_s=float(d.get("default_max_violation_s", 1.0)),
        )

        # ── detectors ────────────────────────────────────────────────
        raw_dets = raw.get("detectors", {})
        for name, det_raw in raw_dets.items():
            if name not in KNOWN_DETECTORS:
                raise ValueError(f"Unknown detector '{name}' in config. "
                                 f"Valid: {sorted(KNOWN_DETECTORS)}")
            mv = det_raw.get("max_violation_s", None)
            cfg.detectors[name] = DetectorConfig(
                enabled=bool(det_raw.get("enabled", True)),
                gate=float(det_raw.get("gate", 0.50)),
                max_violation_s=float(mv) if mv is not None else None,
                weight=float(det_raw.get("weight", 1.0)),
                description=str(det_raw.get("description", "")),
            )

        # Fill missing detectors with defaults (all enabled)
        for name in KNOWN_DETECTORS:
            if name not in cfg.detectors:
                cfg.detectors[name] = DetectorConfig()

        # ── workspace ────────────────────────────────────────────────
        w = raw.get("workspace", {})
        cfg.workspace = WorkspaceConfig(
            x_lo=float(w.get("x_lo", 0.10)),
            x_hi=float(w.get("x_hi", 0.90)),
            y_lo=float(w.get("y_lo", 0.10)),
            y_hi=float(w.get("y_hi", 0.90)),
        )

        # ── frame_edge ───────────────────────────────────────────────
        fe = raw.get("frame_edge", {})
        cfg.frame_edge = FrameEdgeConfig(
            margin=float(fe.get("margin", 0.05)),
        )

        # ── scoring ──────────────────────────────────────────────────
        sc = raw.get("scoring", {})
        cfg.scoring = ScoringConfig(
            pass_threshold=float(sc.get("pass_threshold", 70.0)),
        )

        return cfg


def load_config(path: str = DEFAULT_CONFIG_PATH) -> EgoLensConfig:
    """Convenience one-liner: load and return EgoLensConfig."""
    return ConfigLoader(path).load()
