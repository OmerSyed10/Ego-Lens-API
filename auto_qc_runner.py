"""
egolens/auto_qc_runner.py — AutoQC Orchestrator

Full pipeline:
  1. Query lightwheel.robotics_qc_assets (via db.py) → list of assets
  2. Filter out gopro (already done in SQL)
  3. INSERT pending rows into el_auto_qc
  4. For each pending asset:
       a. mark_running()
       b. Download video from S3  (s3_downloader.py)
       c. Run EgoLensPipeline     (pipeline.py)
       d. Build scored_row dict
       e. write_result()          → update el_auto_qc
       f. Delete temp video file
  5. Print per-asset summary + final batch summary

On any unhandled exception for an asset:
    → mark_error() is called so the row is flagged and can be retried.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH, override=False)

from .config         import load_config
from .pipeline       import EgoLensPipeline
from .scorer         import score as el_score
from . import db
from .s3_downloader  import S3Downloader
from .tracking_loader import TrackingData


# ── Score → DB row mapper ────────────────────────────────────────────────────

def _build_violation_segments(result) -> Optional[str]:
    """
    Build a JSONB-ready JSON string of all detector violation segments.

    Format:
      [
        {
          "detector":   "face_detection",
          "start_s":    3.2,
          "end_s":      4.5,
          "duration_s": 1.3,
          "bboxes":     [{"frame_s": 3.4, "bbox": [x1n, y1n, x2n, y2n]}, ...]
        },
        ...
      ]
    Returns None if no violations exist across any detector.
    """
    segments = []
    for det_name, det in result.detectors.items():
        if not det.enabled or det.violation is None:
            continue
        for seg in det.violation.violation_segments:
            # seg is a dict: {start_s, end_s, bboxes:[{frame_s, bbox}]}
            start_s = seg["start_s"]
            end_s   = seg["end_s"]
            entry = {
                "detector":   det_name,
                "start_s":    round(start_s, 3),
                "end_s":      round(end_s, 3),
                "duration_s": round(end_s - start_s, 3),
            }
            # Include bboxes if any were collected for this segment
            bboxes = seg.get("bboxes", [])
            if bboxes:
                entry["bboxes"] = bboxes
            segments.append(entry)
    if not segments:
        return None
    # Sort by start time for readability
    segments.sort(key=lambda x: (x["start_s"], x["detector"]))
    return json.dumps(segments)


def _build_scored_row(result, scored) -> Dict[str, Any]:
    """
    Map EgoLens AnalysisResult + ScoredResult into a flat dict
    matching the el_auto_qc column names.
    """
    meta = result.meta
    dets = result.detectors

    def _d(name: str) -> Any:
        """Get DetectorResult for a detector name (or None if disabled)."""
        return dets.get(name)

    def _score(name: str) -> Optional[float]:
        d = _d(name)
        return round(d.mean_norm * 100, 2) if d and d.enabled else None

    def _norm(name: str) -> Optional[float]:
        d = _d(name)
        return round(d.mean_norm, 4) if d and d.enabled else None

    def _pass(name: str) -> Optional[bool]:
        d = _d(name)
        if d is None or not d.enabled:
            return None
        return d.mean_norm >= d.gate

    # ── Cascade tier breakdown for hand detection ─────────────────────────
    hand_src: Dict[str, float] = {}
    if result.frames:
        total_sides = 0
        tag_counts: Dict[str, int] = {}
        for rec in result.frames:
            for side, tag in (rec.hand_sources or {}).items():
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                total_sides += 1
        if total_sides > 0:
            hand_src = {t: round(c / total_sides * 100, 1)
                        for t, c in tag_counts.items()}

    # ── Use scorer-computed fail info (strict AND logic already applied) ───
    failed_detectors = scored.failed_detectors or None
    fail_reason      = scored.fail_reason_text or None

    # ── Violation segments JSONB ───────────────────────────────────────────
    violation_segments_json = _build_violation_segments(result)

    return {
        # Video metadata
        "el_video_native_w":        meta.native_w,
        "el_video_native_h":        meta.native_h,
        "el_video_fps":             round(meta.fps, 2),
        "el_video_duration_s":      round(meta.duration_s, 2),
        "el_video_frames_sampled":  len(result.frames),
        "el_is_stereo":             meta.is_stereo,
        "el_effective_w":           meta.effective_w,
        "el_effective_h":           meta.effective_h,

        # Brightness (new)
        "el_brightness_score":      _score("brightness"),
        "el_brightness_norm":       _norm("brightness"),
        "el_brightness_pass":       _pass("brightness"),

        # Sharpness
        "el_sharpness_score":       _score("sharpness"),
        "el_sharpness_norm":        _norm("sharpness"),
        "el_sharpness_pass":        _pass("sharpness"),

        # Stability
        "el_stability_score":       _score("stability"),
        "el_stability_norm":        _norm("stability"),
        "el_stability_pass":        _pass("stability"),

        # Hand detection
        "el_hand_present_score":    _score("hand_detection"),
        "el_hand_present_norm":     _norm("hand_detection"),
        "el_hand_present_pass":     _pass("hand_detection"),
        "el_hand_src_hands_pct":    hand_src.get("hands", 0.0),
        "el_hand_src_pose_pct":     hand_src.get("pose", 0.0),
        "el_hand_src_phantom_pct":  hand_src.get("phantom", 0.0),
        "el_hand_src_none_pct":     hand_src.get("none", 0.0),

        # Hand speed
        "el_hand_speed_score":      _score("hand_speed"),
        "el_hand_speed_norm":       _norm("hand_speed"),
        "el_hand_speed_pass":       _pass("hand_speed"),

        # Hand position
        "el_hand_position_score":   _score("hand_position"),
        "el_hand_position_norm":    _norm("hand_position"),
        "el_hand_position_pass":    _pass("hand_position"),

        # Hand out-of-frame
        "el_hand_oof_score":        _score("hand_out_of_frame"),
        "el_hand_oof_norm":         _norm("hand_out_of_frame"),
        "el_hand_oof_pass":         _pass("hand_out_of_frame"),

        # Face
        "el_face_score":            _score("face_detection"),
        "el_face_norm":             _norm("face_detection"),
        "el_face_pass":             _pass("face_detection"),

        # Person
        "el_person_score":          _score("person_detection"),
        "el_person_norm":           _norm("person_detection"),
        "el_person_pass":           _pass("person_detection"),

        # Overall
        "el_overall_score":         round(scored.overall_score, 2),
        "el_overall_pass":          scored.passed,
        "el_failed_detectors":      failed_detectors,
        "el_fail_reason_text":      fail_reason,

        # Violation segments JSONB
        "el_violation_segments":    violation_segments_json,
    }


# ── Per-asset processor ────────────────────────────────────────────────────────

def _process_one(
    asset_id: str,
    s3_link: str,
    pipeline: EgoLensPipeline,
    cfg,
    downloader: S3Downloader,
    verbose: bool,
) -> bool:
    """
    Download → QC → write result for a single asset.
    Returns True if the asset passed EgoLens QC, False otherwise.
    Raises on unrecoverable errors (caller handles mark_error).
    """
    local_path = None
    try:
        db.mark_running(asset_id)

        # 1. Download from S3 (video + optional tracking file)
        t0 = time.time()
        local_path = downloader.download(s3_link)
        dl_s = time.time() - t0
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"    ✓  Downloaded in {dl_s:.1f}s  ({size_mb:.1f} MB)")

        # 1b. Try to fetch companion XR tracking file
        tracking_local = downloader.download_tracking_file(s3_link)
        tracking_data = None
        if tracking_local:
            tracking_data = TrackingData.from_file(tracking_local)
            print(f"    🎯  Tracking data loaded: {len(tracking_data)} frames")
        else:
            print(f"    ℹ   No tracking file found — hand Tier-0 disabled")

        # 2. Run EgoLens pipeline
        t1 = time.time()
        result = pipeline.run(local_path, verbose=verbose, tracking_data=tracking_data)
        scored = el_score(result, cfg)
        qc_s   = time.time() - t1

        verdict = "✅ PASS" if scored.passed else "❌ FAIL"
        print(f"    {verdict}  score={scored.overall_score:.1f}/100  "
              f"({qc_s:.0f}s)")

        # 3. Build DB row and write
        scored_row = _build_scored_row(result, scored)
        db.write_result(asset_id, scored_row)

        # 4. Cleanup temp file
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

        return scored.passed

    except Exception as exc:
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        raise   # let the caller write mark_error


# ── Batch runners ─────────────────────────────────────────────────────────────

def run_failed(
    limit: int = 0,
    verbose: bool = True,
    config_path: Optional[str] = None,
) -> None:
    """
    Re-run QC only on assets that previously FAILED (overall or any detector).

    This resets their status to 'pending' and reprocesses them with the
    current pipeline (all v2 improvements: brightness, YOLO verification,
    tracking data, violation segments, strict AND pass logic).

    Assets that cleanly passed before are NOT touched.
    """
    from .config import DEFAULT_CONFIG_PATH
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg      = load_config(cfg_path)
    pipeline = EgoLensPipeline(cfg)

    print("═" * 66)
    print("  EgoLens Auto-QC — Rerun Failed Assets Only")
    print(f"  Config : {cfg_path}")
    print(f"  Limit  : {'all failed' if limit == 0 else limit}")
    print("═" * 66)

    # Reset failed/errored assets to pending
    print("\n  → Resetting failed assets to pending …")
    n_reset = db.reset_failed_for_rerun()
    print(f"  Reset {n_reset} asset(s) to pending.")

    if n_reset == 0:
        print("  Nothing to reprocess.")
        return

    # Fetch the just-reset assets
    assets = db.fetch_failed_assets(limit=limit)
    n = len(assets)
    print(f"  Found {n} asset(s) to reprocess.\n")

    n_pass = n_fail = n_err = 0

    with S3Downloader() as downloader:
        for idx, asset in enumerate(assets, start=1):
            asset_id = asset["asset_id"]
            s3_link  = asset["s3_link"]

            print(f"\n[{idx}/{n}]  {asset_id}")
            print(f"       {s3_link}")

            try:
                passed = _process_one(
                    asset_id, s3_link, pipeline, cfg, downloader, verbose
                )
                if passed:
                    n_pass += 1
                else:
                    n_fail += 1

            except Exception as exc:
                print(f"    💥 ERROR: {exc}")
                db.mark_error(asset_id, exc)
                n_err += 1

    print("\n" + "═" * 66)
    print(f"  Rerun complete — processed {n} failed asset(s)")
    print(f"  ✅ Now pass : {n_pass}")
    print(f"  ❌ Still fail: {n_fail}")
    print(f"  💥 Errors   : {n_err}")
    print("═" * 66 + "\n")


def run_batch(
    limit: int = 0,
    verbose: bool = True,
    config_path: Optional[str] = None,
) -> None:
    """
    Main entry point.  Fetches pending assets → runs QC → writes results.

    Parameters
    ----------
    limit      : int   — max videos to process (0 = all)
    verbose    : bool  — show tqdm progress bar per video
    config_path: str   — path to config.yaml (None = default)
    """
    from .config import DEFAULT_CONFIG_PATH
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    cfg      = load_config(cfg_path)
    pipeline = EgoLensPipeline(cfg)

    print("═" * 66)
    print("  EgoLens Auto-QC Batch Runner")
    print(f"  Config : {cfg_path}")
    print(f"  Limit  : {'all' if limit == 0 else limit}")
    print("═" * 66)

    # 1. Fetch assets from source view
    print("\n  → Fetching pending assets from DB …")
    assets = db.fetch_pending_assets(limit=limit)
    n = len(assets)
    if n == 0:
        print("  No new assets to process. All done.")
        return

    print(f"  Found {n} asset(s) to process.\n")

    # 2. Pre-register all as 'pending' (idempotent)
    for a in assets:
        db.upsert_pending(a["asset_id"], a["s3_link"])

    # 3. Process each asset
    n_pass = n_fail = n_err = 0

    with S3Downloader() as downloader:
        for idx, asset in enumerate(assets, start=1):
            asset_id = asset["asset_id"]
            s3_link  = asset["s3_link"]

            print(f"\n[{idx}/{n}]  {asset_id}")
            print(f"       {s3_link}")

            try:
                passed = _process_one(
                    asset_id, s3_link, pipeline, cfg, downloader, verbose
                )
                if passed:
                    n_pass += 1
                else:
                    n_fail += 1

            except Exception as exc:
                print(f"    💥 ERROR: {exc}")
                db.mark_error(asset_id, exc)
                n_err += 1

    # 4. Summary
    print("\n" + "═" * 66)
    print(f"  Batch complete — processed {n} asset(s)")
    print(f"  ✅ Passed : {n_pass}")
    print(f"  ❌ Failed : {n_fail}")
    print(f"  💥 Errors : {n_err}")
    print("═" * 66 + "\n")
