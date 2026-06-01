"""
egolens/run_auto_qc.py — CLI entry point for the Auto-QC batch pipeline.

Usage
-----
    # From the project root (deepreech/):
    python -m egolens.run_auto_qc

    # ONLY re-run videos that previously failed (recommended for re-runs)
    python -m egolens.run_auto_qc --failed-only

    # Process at most 10 videos
    python -m egolens.run_auto_qc --limit 10

    # Use a custom config
    python -m egolens.run_auto_qc --config egolens/strict.yaml --limit 5

    # Quiet mode (no per-frame progress bars)
    python -m egolens.run_auto_qc --quiet

Steps executed
--------------
  1. Load .env  (egolens/.env)
  2. Connect to PostgreSQL
  3. Query lightwheel.robotics_qc_assets → filter gopro → list of assets
  4. For each asset:
       → Download video from S3
       → Run EgoLens pipeline
       → Write all detector scores + overall verdict to lightwheel.el_auto_qc
  5. Print summary

Prerequisites
-------------
    pip install psycopg2-binary boto3 python-dotenv
    Fill in egolens/.env with real credentials before running.
"""

from __future__ import annotations

import argparse
import sys
import os

# Make sure the package root is on sys.path when running as a script
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m egolens.run_auto_qc",
        description=(
            "EgoLens Auto-QC: fetch videos from S3, "
            "run quality checks, write results to DB."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m egolens.run_auto_qc                          # new assets only
  python -m egolens.run_auto_qc --failed-only            # re-run previous failures
  python -m egolens.run_auto_qc --failed-only --limit 5  # re-run at most 5 failures
  python -m egolens.run_auto_qc --limit 10               # up to 10 new assets
  python -m egolens.run_auto_qc --config strict.yaml --quiet
        """,
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Max number of videos to process in this run (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="CONFIG_YAML",
        help="Path to egolens config YAML (default: egolens/config.yaml)",
    )
    parser.add_argument(
        "--failed-only", "-f",
        action="store_true",
        help=(
            "Re-run only assets that previously FAILED "
            "(el_overall_pass=false OR any detector failed). "
            "Passed assets are left untouched. "
            "Ideal for re-runs after a pipeline fix."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-frame progress bars (useful in CI / cron jobs)",
    )
    args = parser.parse_args()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    _check_imports()
    _check_env()

    # ── Run ───────────────────────────────────────────────────────────────────
    from egolens.auto_qc_runner import run_batch, run_failed

    try:
        if args.failed_only:
            run_failed(
                limit       = args.limit,
                verbose     = not args.quiet,
                config_path = args.config,
            )
        else:
            run_batch(
                limit       = args.limit,
                verbose     = not args.quiet,
                config_path = args.config,
            )
        return 0
    except KeyboardInterrupt:
        print("\n[auto_qc] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[auto_qc] Fatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _check_imports():
    """Fail fast with a helpful message if required packages are missing."""
    missing = []
    for pkg, install in [
        ("psycopg2",   "psycopg2-binary"),
        ("boto3",      "boto3"),
        ("dotenv",     "python-dotenv"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install)

    if missing:
        print(
            f"[auto_qc] Missing packages: {', '.join(missing)}\n"
            f"  Install with:  pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_env():
    """Warn if required .env keys are empty."""
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path, override=False)

    required = [
        "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(
            f"[auto_qc] Missing environment variables: {', '.join(missing)}\n"
            f"  Fill in egolens/.env before running.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
