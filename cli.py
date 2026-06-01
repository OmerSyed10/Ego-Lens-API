"""
egolens/cli.py — Command-line entry point.

Usage
-----
    python -m egolens VIDEO [VIDEO ...] [--config CONFIG_YAML]

Examples
--------
    python -m egolens testdata/passed/ego_view.mp4
    python -m egolens testdata/failed/*.mp4 --config my_config.yaml
    python -m egolens video.mp4 --no-doh          # skip 100DOH (faster)
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure the package root is on sys.path when running as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from egolens.config   import load_config, DEFAULT_CONFIG_PATH
from egolens.pipeline import EgoLensPipeline
from egolens.scorer   import score
from egolens.reporter import print_report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m egolens",
        description="EgoLens video QC pipeline",
    )
    parser.add_argument(
        "videos",
        nargs="+",
        metavar="VIDEO",
        help="Path(s) to video file(s) to analyse",
    )
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        metavar="CONFIG_YAML",
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--no-doh",
        action="store_true",
        help="Skip 100DOH detector (faster, but weaker hand detection on fists/tools)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-frame progress bar",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"[egolens] Config error: {e}", file=sys.stderr)
        return 1

    # ── Optional: pre-load 100DOH ─────────────────────────────────────────────
    doh_detector = None
    if not args.no_doh:
        try:
            from egolens.detectors.doh_detector import DOHDetector, DOH_AVAILABLE
            if DOH_AVAILABLE:
                doh_detector = DOHDetector()
                print(f"[egolens] 100DOH loaded (CPU mode)")
            else:
                print(f"[egolens] 100DOH not available — skipping Tier 2")
        except Exception as exc:
            print(f"[egolens] Warning: 100DOH failed to load ({exc}) — skipping Tier 2",
                  file=sys.stderr)

    # ── Run pipeline on each video ─────────────────────────────────────────────
    pipeline  = EgoLensPipeline(cfg)
    n_pass    = 0
    n_fail    = 0
    n_total   = len(args.videos)

    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"[egolens] File not found: {video_path}", file=sys.stderr)
            n_fail += 1
            continue

        try:
            result = pipeline.run(video_path, verbose=not args.quiet)
            scored = score(result, cfg)
            print_report(result, scored, pass_threshold=cfg.scoring.pass_threshold)

            if scored.passed:
                n_pass += 1
            else:
                n_fail += 1

        except Exception as exc:
            print(f"[egolens] ERROR processing {video_path}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            n_fail += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if n_total > 1:
        print(f"\n  Summary: {n_pass}/{n_total} passed, "
              f"{n_fail}/{n_total} failed\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
