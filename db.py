"""
egolens/db.py — PostgreSQL client for EgoLens Auto-QC

Responsibilities
----------------
1. connect()           — open a psycopg2 connection from .env
2. fetch_pending_assets() — query the source view, return assets to process
3. upsert_pending()    — INSERT a row with status='pending' (idempotent)
4. mark_running()      — set status='running' before download starts
5. write_result()      — write all detector scores + overall verdict
6. mark_error()        — write error message + traceback if pipeline crashes

Column naming: all EgoLens columns are prefixed with "el_" so other QC
systems can coexist in the same table later without conflicts.
"""

from __future__ import annotations

import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

from dotenv import load_dotenv

# Load .env from the egolens/ directory
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH, override=False)

_SCHEMA = os.getenv("DB_SCHEMA", "lightwheel")
_TABLE  = f"{_SCHEMA}.el_auto_qc"


# ── Connection ────────────────────────────────────────────────────────────────

def _get_conn():
    """Open a new psycopg2 connection using .env credentials."""
    if not _PG_AVAILABLE:
        raise ImportError(
            "psycopg2 is required: pip install psycopg2-binary"
        )

    # Guard against empty values (template .env not filled in)
    required = {"DB_HOST": os.getenv("DB_HOST", ""), "DB_NAME": os.getenv("DB_NAME", ""),
                "DB_USER": os.getenv("DB_USER", ""), "DB_PASSWORD": os.getenv("DB_PASSWORD", "")}
    missing = [k for k, v in required.items() if not v.strip()]
    if missing:
        raise EnvironmentError(
            f"Missing or empty .env values: {', '.join(missing)}\n"
            f"  → Fill in egolens/.env then re-run."
        )

    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


@contextmanager
def get_cursor() -> Generator:
    """
    Context manager that yields a psycopg2 DictCursor and commits on exit.

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT ...")
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
    finally:
        conn.close()


# ── Fetch source assets ────────────────────────────────────────────────────────

def fetch_pending_assets(limit: int = 0) -> List[Dict[str, Any]]:
    """
    Query lightwheel.robotics_qc_assets and return assets to process.

    The view has a JSON column called 'views' — an array of objects like:
        {"expected_slot": "main", "present": true, "bucket_key": "DATA/...", ...}

    We pick only rows where expected_slot='main' AND present=true, then
    build the S3 URL as:  s3://<AWS_S3_BUCKET>/<bucket_key>

    Filters applied:
      - expected_slot = 'main'
      - present = true
      - bucket_key does NOT contain 'gopro' (case-insensitive)
      - asset not already in el_auto_qc with status pending/running/done/skipped

    Returns a list of dicts:
        {"asset_id": "...", "s3_link": "s3://project-parallax/DATA/.../video.mp4"}
    """
    view     = os.getenv("ASSETS_VIEW", "robotics_qc_assets")
    col_id   = os.getenv("ASSETS_COL_ASSET_ID", "asset_id")
    schema   = os.getenv("DB_SCHEMA", "lightwheel")
    bucket   = os.getenv("AWS_S3_BUCKET", "project-parallax")

    full_view    = f"{schema}.{view}"
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    sql = f"""
        SELECT
            src."{col_id}"                                   AS asset_id,
            's3://' || %s || '/' || (v->>'bucket_key')      AS s3_link
        FROM {full_view} src,
             jsonb_array_elements(src.views::jsonb) v
        WHERE v->>'expected_slot' = 'main'
          AND (v->>'present')::boolean = true
          -- Exclude GoPro cameras
          AND LOWER(v->>'bucket_key') NOT LIKE '%%gopro%%'
          -- Only assets not yet processed (or previously errored → retry)
          AND NOT EXISTS (
              SELECT 1 FROM {_TABLE} qc
              WHERE qc.asset_id = src."{col_id}"
                AND qc.el_status IN ('pending', 'running', 'done', 'skipped')
          )
        ORDER BY src."{col_id}"
        {limit_clause}
    """

    with get_cursor() as cur:
        cur.execute(sql, (bucket,))
        return [dict(row) for row in cur.fetchall()]


# ── Write lifecycle rows ────────────────────────────────────────────────────────

def upsert_pending(asset_id: str, s3_link: str) -> None:
    """
    INSERT a row with status='pending'.
    Ignored if the asset already exists (ON CONFLICT DO NOTHING).
    """
    sql = f"""
        INSERT INTO {_TABLE} (asset_id, s3_link, el_status)
        VALUES (%s, %s, 'pending')
        ON CONFLICT (asset_id) DO NOTHING
    """
    with get_cursor() as cur:
        cur.execute(sql, (asset_id, s3_link))


def mark_running(asset_id: str) -> None:
    """Set status='running' and clear any previous error."""
    sql = f"""
        UPDATE {_TABLE}
        SET    el_status        = 'running',
               el_error_message = NULL,
               el_error_traceback = NULL
        WHERE  asset_id = %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (asset_id,))


def write_result(asset_id: str, scored_row: Dict[str, Any]) -> None:
    """
    Write the full EgoLens result for one asset.

    scored_row keys (all optional, set to NULL if missing):
        el_video_native_w, el_video_native_h, el_video_fps,
        el_video_duration_s, el_video_frames_sampled,
        el_is_stereo, el_effective_w, el_effective_h,
        el_sharpness_score, el_sharpness_norm, el_sharpness_pass,
        el_stability_score, el_stability_norm, el_stability_pass,
        el_hand_present_score, el_hand_present_norm, el_hand_present_pass,
        el_hand_src_hands_pct, el_hand_src_pose_pct,
        el_hand_src_phantom_pct, el_hand_src_none_pct,
        el_hand_speed_score, el_hand_speed_norm, el_hand_speed_pass,
        el_hand_position_score, el_hand_position_norm, el_hand_position_pass,
        el_hand_oof_score, el_hand_oof_norm, el_hand_oof_pass,
        el_face_score, el_face_norm, el_face_pass,
        el_person_score, el_person_norm, el_person_pass,
        el_overall_score, el_overall_pass,
        el_failed_detectors, el_fail_reason_text
    """
    _RESULT_COLS = [
        "el_video_native_w", "el_video_native_h", "el_video_fps",
        "el_video_duration_s", "el_video_frames_sampled",
        "el_is_stereo", "el_effective_w", "el_effective_h",
        "el_brightness_score", "el_brightness_norm", "el_brightness_pass",
        "el_sharpness_score", "el_sharpness_norm", "el_sharpness_pass",
        "el_stability_score", "el_stability_norm", "el_stability_pass",
        "el_hand_present_score", "el_hand_present_norm", "el_hand_present_pass",
        "el_hand_src_hands_pct", "el_hand_src_pose_pct",
        "el_hand_src_phantom_pct", "el_hand_src_none_pct",
        "el_hand_speed_score", "el_hand_speed_norm", "el_hand_speed_pass",
        "el_hand_position_score", "el_hand_position_norm", "el_hand_position_pass",
        "el_hand_oof_score", "el_hand_oof_norm", "el_hand_oof_pass",
        "el_face_score", "el_face_norm", "el_face_pass",
        "el_person_score", "el_person_norm", "el_person_pass",
        "el_overall_score", "el_overall_pass",
        "el_failed_detectors", "el_fail_reason_text",
        "el_violation_segments",
    ]

    # Columns that must be cast to JSONB explicitly (text[] handled natively)
    _JSONB_COLS = {"el_violation_segments"}

    set_parts = []
    values    = []
    for col in _RESULT_COLS:
        raw = scored_row.get(col)
        if col in _JSONB_COLS:
            if raw is None:
                set_parts.append(f"{col} = NULL")
            elif isinstance(raw, str):
                # Already a JSON string — cast in SQL
                set_parts.append(f"{col} = %s::jsonb")
                values.append(raw)
            else:
                # Python list/dict — serialise + cast
                import json as _json
                set_parts.append(f"{col} = %s::jsonb")
                values.append(_json.dumps(raw))
        else:
            set_parts.append(f"{col} = %s")
            values.append(raw)

    set_parts += ["el_status = 'done'", "el_processed_at = NOW()"]

    sql = f"""
        UPDATE {_TABLE}
        SET    {', '.join(set_parts)}
        WHERE  asset_id = %s
    """
    with get_cursor() as cur:
        cur.execute(sql, values + [asset_id])


def reset_failed_for_rerun() -> int:
    """
    Reset only assets that previously FAILED — either:
      - el_overall_pass = false, OR
      - el_failed_detectors is not null/empty (individual detector failure), OR
      - el_status = 'error' (pipeline crash on first attempt)

    Assets that cleanly PASSED are left untouched.

    Returns the number of rows reset.
    """
    sql = f"""
        UPDATE {_TABLE}
        SET    el_status             = 'pending',
               el_error_message     = NULL,
               el_error_traceback   = NULL,
               el_processed_at      = NULL
        WHERE  el_status IN ('done', 'error')
          AND (
                el_overall_pass = false
             OR el_failed_detectors IS NOT NULL
             OR el_status = 'error'
          )
        RETURNING asset_id
    """
    with get_cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        return len(rows)


def fetch_failed_assets(limit: int = 0) -> List[Dict[str, Any]]:
    """
    Return assets that are currently in 'pending' status AND previously had
    a failure (el_overall_pass = false OR el_failed_detectors set).

    Used after reset_failed_for_rerun() to get the list to process.
    """
    view   = os.getenv("ASSETS_VIEW", "robotics_qc_assets")
    col_id = os.getenv("ASSETS_COL_ASSET_ID", "asset_id")
    schema = os.getenv("DB_SCHEMA", "lightwheel")
    bucket = os.getenv("AWS_S3_BUCKET", "project-parallax")

    full_view    = f"{schema}.{view}"
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    sql = f"""
        SELECT
            qc.asset_id,
            qc.s3_link
        FROM {_TABLE} qc
        WHERE qc.el_status = 'pending'
        ORDER BY qc.asset_id
        {limit_clause}
    """
    with get_cursor() as cur:
        cur.execute(sql)
        return [dict(row) for row in cur.fetchall()]


def mark_error(asset_id: str, exc: Exception) -> None:
    """Record an exception for a failed asset."""
    sql = f"""
        UPDATE {_TABLE}
        SET    el_status           = 'error',
               el_error_message   = %s,
               el_error_traceback = %s,
               el_processed_at    = NOW()
        WHERE  asset_id = %s
    """
    tb = traceback.format_exc()
    with get_cursor() as cur:
        cur.execute(sql, (str(exc), tb, asset_id))
