"""
egolens/cron_sync_runner.py — Hourly cron job entry point

Runs the full sync + process cycle:

  PHASE 1  Sync     — find new assets in robotics_qc_assets that are not yet
                      in el_auto_qc, insert them with el_status='pending'
  PHASE 2  Process  — fetch all el_status='pending' rows from el_auto_qc,
                      run the EgoLens pipeline on each, write results
  PHASE 3  Report   — log a summary line with counts and duration

Safety
------
- Lock file at /tmp/egolens_cron.lock prevents two simultaneous runs.
  If a previous run is still going, the new one exits immediately.
- Per-asset try/except — one bad video does not block the rest.
- All output written to stdout (capture in cron with `>> /var/log/egolens.log`).

Usage
-----
  # Manual one-off
  python -m egolens.cron_sync_runner

  # Crontab entry (every hour, top of the hour)
  0 * * * * cd /path/to/project && \\
            /path/to/.venv/bin/python -m egolens.cron_sync_runner \\
            >> /var/log/egolens.log 2>&1
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import db, sync_assets
from .auto_qc_runner import run_batch


# ── Configuration ────────────────────────────────────────────────────────────

LOCK_FILE   = Path(os.getenv("EGOLENS_CRON_LOCK", "/tmp/egolens_cron.lock"))
LOG_PREFIX  = "[egolens-cron]"


# ── Lock file (prevents overlapping runs) ────────────────────────────────────

def _acquire_lock() -> bool:
    """
    Create the lock file with the current PID.

    Returns
    -------
    True  — lock acquired
    False — another process holds the lock and is still alive
    """
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            # Check if the old process is still running
            try:
                os.kill(old_pid, 0)
                # Process is alive → lock is valid
                print(f"{LOG_PREFIX} ⚠  Previous run (PID {old_pid}) still active. Skipping.")
                return False
            except OSError:
                # PID does not exist → stale lock, take over
                print(f"{LOG_PREFIX} Stale lock from PID {old_pid} — removing.")
                LOCK_FILE.unlink(missing_ok=True)
        except (ValueError, OSError):
            # Lock file unreadable — remove it
            LOCK_FILE.unlink(missing_ok=True)

    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    """Remove the lock file (only if it belongs to us)."""
    try:
        if LOCK_FILE.exists():
            pid_in_file = int(LOCK_FILE.read_text().strip())
            if pid_in_file == os.getpid():
                LOCK_FILE.unlink()
    except Exception:
        pass


# ── Counts helper ────────────────────────────────────────────────────────────

def _count_results() -> dict:
    """Return latest pass/fail/error counts for the log line."""
    schema = os.getenv("DB_SCHEMA", "lightwheel")
    sql = f"""
        SELECT
          COUNT(*) FILTER (WHERE el_status = 'done' AND el_overall_pass = true)  AS n_pass,
          COUNT(*) FILTER (WHERE el_status = 'done' AND el_overall_pass = false) AS n_fail,
          COUNT(*) FILTER (WHERE el_status = 'error')                            AS n_error,
          COUNT(*) FILTER (WHERE el_status = 'pending')                          AS n_pending,
          COUNT(*)                                                               AS n_total
        FROM {schema}.el_auto_qc
    """
    with db.get_cursor() as cur:
        cur.execute(sql)
        return dict(cur.fetchone())


# ── Main entry ───────────────────────────────────────────────────────────────

def main() -> int:
    """
    Returns process exit code (0 = OK, 1 = lock contention or fatal error).
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.time()

    print()
    print("═" * 70)
    print(f"{LOG_PREFIX} Run started at {started_at.isoformat()}")
    print("═" * 70)

    # ── Lock ──────────────────────────────────────────────────────────────
    if not _acquire_lock():
        return 1

    try:
        # ── PHASE 1 — SYNC ────────────────────────────────────────────────
        print(f"\n{LOG_PREFIX} PHASE 1 — Syncing robotics_qc_assets → el_auto_qc")
        try:
            sync_result = sync_assets.sync()
            print(f"{LOG_PREFIX}   Found    : {sync_result['found']}  new in source view")
            print(f"{LOG_PREFIX}   Inserted : {sync_result['inserted']}  into el_auto_qc (pending)")
        except Exception as exc:
            print(f"{LOG_PREFIX} 💥 Sync failed: {exc}")
            traceback.print_exc()
            sync_result = {"found": 0, "inserted": 0}

        # ── PHASE 2 — PROCESS ─────────────────────────────────────────────
        print(f"\n{LOG_PREFIX} PHASE 2 — Processing all pending assets")
        try:
            # run_batch fetches el_status='pending' assets directly
            run_batch(limit=0, verbose=False)
        except Exception as exc:
            print(f"{LOG_PREFIX} 💥 Batch processing failed: {exc}")
            traceback.print_exc()

        # ── PHASE 3 — REPORT ──────────────────────────────────────────────
        elapsed = time.time() - t0
        try:
            counts = _count_results()
        except Exception as exc:
            print(f"{LOG_PREFIX} ⚠  Could not fetch summary counts: {exc}")
            counts = {}

        print()
        print("═" * 70)
        print(f"{LOG_PREFIX} SUMMARY")
        print(f"{LOG_PREFIX}   Run started  : {started_at.isoformat()}")
        print(f"{LOG_PREFIX}   Duration     : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
        print(f"{LOG_PREFIX}   New synced   : {sync_result.get('inserted', 0)}")
        if counts:
            print(f"{LOG_PREFIX}   ─ DB Totals ─")
            print(f"{LOG_PREFIX}   ✅ Pass     : {counts.get('n_pass', 0)}")
            print(f"{LOG_PREFIX}   ❌ Fail     : {counts.get('n_fail', 0)}")
            print(f"{LOG_PREFIX}   💥 Errors   : {counts.get('n_error', 0)}")
            print(f"{LOG_PREFIX}   ⏳ Pending  : {counts.get('n_pending', 0)}")
            print(f"{LOG_PREFIX}   📦 Total    : {counts.get('n_total', 0)}")
        print("═" * 70)

        return 0

    finally:
        _release_lock()


if __name__ == "__main__":
    sys.exit(main())
