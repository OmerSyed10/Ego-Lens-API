"""
egolens/sync_assets.py — Sync new assets from robotics_qc_assets → el_auto_qc

Phase 1 of the cron job.

Responsibilities
----------------
1. find_new_assets()  — assets in robotics_qc_assets but NOT in el_auto_qc
2. insert_pending()   — bulk-insert them with status='pending'

The actual processing (running EgoLens on the pending rows) is handled
separately by auto_qc_runner.run_batch() — see cron_sync_runner.py.

Filters applied (same as auto_qc_runner.fetch_pending_assets):
  - expected_slot = 'main'
  - present       = true
  - bucket_key does NOT contain 'gopro' (case-insensitive)

Idempotent: ON CONFLICT (asset_id) DO NOTHING — safe to re-run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from . import db


def find_new_assets(limit: int = 0) -> List[Dict[str, Any]]:
    """
    Return assets present in robotics_qc_assets but missing from el_auto_qc.

    Same SQL as db.fetch_pending_assets — kept here as a thin wrapper
    so the cron job has a self-documenting entry point.

    Returns
    -------
    List of {"asset_id": str, "s3_link": str}.
    """
    return db.fetch_pending_assets(limit=limit)


def insert_pending(assets: List[Dict[str, Any]]) -> int:
    """
    Bulk-insert new assets into el_auto_qc with el_status='pending'.

    Uses ON CONFLICT (asset_id) DO NOTHING so this is safe to call even
    if some rows already exist.

    Parameters
    ----------
    assets : list of {"asset_id", "s3_link"} dicts (output of find_new_assets)

    Returns
    -------
    int — number of rows actually inserted
    """
    if not assets:
        return 0

    schema = os.getenv("DB_SCHEMA", "lightwheel")
    table  = f"{schema}.el_auto_qc"

    sql = f"""
        INSERT INTO {table} (asset_id, s3_link, el_status, el_created_at)
        VALUES (%s, %s, 'pending', NOW())
        ON CONFLICT (asset_id) DO NOTHING
        RETURNING asset_id
    """

    inserted = 0
    with db.get_cursor() as cur:
        for a in assets:
            cur.execute(sql, (a["asset_id"], a["s3_link"]))
            if cur.fetchone() is not None:
                inserted += 1
    return inserted


def sync() -> Dict[str, int]:
    """
    Top-level convenience: find new assets and insert them as 'pending'.

    Returns
    -------
    {"found": N, "inserted": M}
    """
    new_assets = find_new_assets()
    n_inserted = insert_pending(new_assets)
    return {
        "found":    len(new_assets),
        "inserted": n_inserted,
    }


# ── CLI entry point for manual testing ───────────────────────────────────────

if __name__ == "__main__":
    result = sync()
    print(f"Found    : {result['found']} new asset(s) in robotics_qc_assets")
    print(f"Inserted : {result['inserted']} into el_auto_qc as pending")
