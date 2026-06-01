"""
egolens/scorer.py — EgoLensScorer  (v2)

Overall pass rule (strict AND):
  1. Weighted average score ≥ pass_threshold (e.g. 70/100)
  2. AND every enabled detector individually passes its gate

If ANY detector's mean_norm < its gate → el_overall_pass = False,
regardless of how high the weighted average is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import EgoLensConfig
from .pipeline import AnalysisResult, DetectorResult


@dataclass
class ScoredResult:
    """Scores attached to an AnalysisResult."""
    per_detector:     Dict[str, float] = field(default_factory=dict)   # 0–100
    overall_score:    float = 0.0                                       # 0–100
    passed:           bool  = False
    failed_detectors: List[str] = field(default_factory=list)          # detectors that individually failed
    fail_reason_text: str  = ""                                         # always set when any failure


def score(result: AnalysisResult, cfg: EgoLensConfig) -> ScoredResult:
    """
    Compute per-detector and overall weighted score.

    Pass logic (strict AND):
      - weighted avg ≥ pass_threshold
      - AND no individual detector below its gate

    Parameters
    ----------
    result : AnalysisResult from EgoLensPipeline.run()
    cfg    : EgoLensConfig (for weights and pass threshold)
    """
    per_detector:     Dict[str, float] = {}
    failed_detectors: List[str]        = []
    total_weight  = 0.0
    weighted_sum  = 0.0

    for name, det in result.detectors.items():
        if not det.enabled:
            per_detector[name] = 0.0
            continue
        if det.total_count == 0:
            per_detector[name] = 0.0
            continue

        det_score = det.mean_norm * 100.0
        per_detector[name] = round(det_score, 2)

        weighted_sum += det_score * det.weight
        total_weight += det.weight

        # Individual gate check
        if det.mean_norm < det.gate:
            failed_detectors.append(name)

    overall = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # STRICT: both conditions must hold
    passed = (overall >= cfg.scoring.pass_threshold) and (len(failed_detectors) == 0)

    # Always build reason text when anything fails
    if failed_detectors:
        parts = []
        for name in failed_detectors:
            s = per_detector.get(name, 0.0)
            d = result.detectors.get(name)
            gate_pct = (d.gate * 100) if d else 0
            parts.append(f"{name} ({s:.1f}/100, gate={gate_pct:.0f})")
        fail_reason = "FAIL: " + ", ".join(parts)
    elif not passed:
        fail_reason = f"FAIL: overall score {overall:.1f} below threshold {cfg.scoring.pass_threshold}"
    else:
        fail_reason = ""

    # Write back into AnalysisResult
    result.overall_score = overall
    result.passed        = passed

    return ScoredResult(
        per_detector=per_detector,
        overall_score=round(overall, 2),
        passed=passed,
        failed_detectors=failed_detectors,
        fail_reason_text=fail_reason,
    )
