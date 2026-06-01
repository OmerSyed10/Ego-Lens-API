"""
egolens/reporter.py — EgoLensReporter

Formats and prints the final analysis report to stdout.

Output format
-------------
════════════════════════════════════════════════════════════
  [PASSED / FAILED]  <filename>
  <fps> fps → stride <N> → <eff_fps> fps effective | <N> frames
  Stereo: <description>
════════════════════════════════════════════════════════════
  ✅/❌ Sharpness        norm=<X>  pass_rate=<Y>% <violations>
  ✅/❌ Stability        norm=<X>  pass_rate=<Y>% <violations>
  ✅/❌ Hand Detection   norm=<X>  pass_rate=<Y>% <violations>
  ...
  ──────────────────────────────────────────────────────────
  Overall score : <S>/100   threshold=<T>
"""

from __future__ import annotations

import os
from typing import Dict

from .pipeline import AnalysisResult
from .scorer   import ScoredResult

_SEP  = "═" * 66
_DASH = "─" * 66

_LABELS: Dict[str, str] = {
    "sharpness":         "Sharpness       ",
    "stability":         "Stability       ",
    "hand_detection":    "Hand Detection  ",
    "face_detection":    "Face (Privacy)  ",
    "person_detection":  "Person Detect.  ",
    "hand_speed":        "Hand Speed      ",
    "hand_position":     "Hand Position   ",
    "hand_out_of_frame": "Hand OOF        ",
}


def print_report(result: AnalysisResult, scored: ScoredResult, pass_threshold: float = 70.0) -> None:
    """Print the full analysis report to stdout."""
    meta = result.meta
    fname = os.path.basename(result.video_path)
    label = "[PASSED]" if scored.passed else "[FAILED]"
    icon  = "✅" if scored.passed else "❌"

    n_frames = len(result.frames)

    print()
    print(_SEP)
    print(f"  {label} {fname}")

    # Stereo info
    if meta.is_stereo:
        print(f"  Stereo {meta.native_w}×{meta.native_h} → {meta.stereo_eye} eye "
              f"{meta.crop_w}×{meta.native_h}"
              + (f" → scaled {meta.effective_w}×{meta.effective_h}"
                 if meta.resize_scale < 1.0 else ""))
    else:
        print(f"  {meta.effective_w}×{meta.effective_h}  non-stereo")

    # Frame rate info
    if n_frames > 0:
        stride = max(1, round(meta.fps / (n_frames / (meta.duration_s or 1))))
        eff_fps = meta.fps / stride if stride > 0 else meta.fps
        print(f"  {meta.fps:.1f} fps → stride {stride} → "
              f"{eff_fps:.1f} fps effective  |  {n_frames} frames")

    print(_SEP)

    # Per-detector rows
    for det_name, det in result.detectors.items():
        if not det.enabled:
            continue

        det_score   = scored.per_detector.get(det_name, 0.0)
        label_str   = _LABELS.get(det_name, det_name.ljust(16))
        passed_row  = det.mean_norm >= det.gate
        row_icon    = "✅" if passed_row else "❌"

        pct  = f"{det.pass_count}/{det.total_count}"
        line = (f"  {row_icon} {label_str}  "
                f"score={det_score:5.1f}  norm={det.mean_norm:.3f}  "
                f"({pct} frames pass)")
        print(line)

        # Violation segments
        if det.violation and det.violation.violation_count > 0:
            segs = det.violation.violation_segments
            seg_str = ", ".join(f"{s:.1f}s–{e:.1f}s" for s, e in segs[:5])
            if len(segs) > 5:
                seg_str += f" … (+{len(segs)-5} more)"
            print(f"       violations: {seg_str}  "
                  f"[total {det.violation.total_violation_s:.1f}s]")

        # Hand sources breakdown
        if det_name == "hand_detection" and result.frames:
            _print_source_breakdown(result)

    print("  " + _DASH)
    overall_icon = "✅" if scored.passed else "❌"
    print(f"  {overall_icon} Overall score : {scored.overall_score:.1f}/100  "
          f"(threshold={pass_threshold:.0f}  "
          f"{'PASS' if scored.passed else 'FAIL'})")
    print()


def _print_source_breakdown(result: AnalysisResult) -> None:
    """Print % of hand-side detections credited to each cascade tier."""
    source_counts: Dict[str, int] = {}
    total_sides = 0
    for rec in result.frames:
        for side, tag in (rec.hand_sources or {}).items():
            source_counts[tag] = source_counts.get(tag, 0) + 1
            total_sides += 1

    if total_sides == 0:
        return

    tag_pct = {tag: round(cnt / total_sides * 100) for tag, cnt in source_counts.items()}
    breakdown = "  ".join(f"{t}={p}%" for t, p in sorted(tag_pct.items()))
    if breakdown:
        print(f"       sources: {breakdown}")
