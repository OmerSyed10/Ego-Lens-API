"""
egolens/test_connections.py — Pre-flight connection tester

Run this BEFORE run_auto_qc.py to verify:
  1. DB credentials are set
  2. DB connection works
  3. Source view exists and has the expected columns
  4. el_auto_qc table exists (or can be created)
  5. AWS S3 credentials are valid
  6. Print sample rows from the source view (non-gopro)

Usage:
    python egolens/test_connections.py
    python egolens/test_connections.py --create-table   # also runs SQL migration
    python egolens/test_connections.py --sample 5       # show N sample assets
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)


def sep(title: str = "") -> None:
    if title:
        print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    else:
        print("─" * 64)


def _require_env(*keys: str) -> bool:
    """Return False and print which keys are missing or empty."""
    missing = [k for k in keys if not os.getenv(k, "").strip()]
    if missing:
        for k in missing:
            print(f"  ❌  {k} is not set in egolens/.env")
        return False
    return True


def _pg_conn():
    """Open a psycopg2 connection (raises on failure)."""
    import psycopg2
    return psycopg2.connect(
        host    =os.environ["DB_HOST"],
        port    =int(os.getenv("DB_PORT", "5432")),
        dbname  =os.environ["DB_NAME"],
        user    =os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


# ── Test 1: DB credentials present ───────────────────────────────────────────

def test_db_env() -> bool:
    sep("1. DB credentials in .env")
    ok = _require_env("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
    if ok:
        print(f"  ✅  DB_HOST={os.getenv('DB_HOST')}  DB_NAME={os.getenv('DB_NAME')}  DB_USER={os.getenv('DB_USER')}")
    return ok


# ── Test 2: DB connection live ────────────────────────────────────────────────

def test_db_connection() -> bool:
    sep("2. PostgreSQL live connection")
    if not _require_env("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        print("  ⏭   Skipped (credentials missing)")
        return False
    try:
        conn = _pg_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
        conn.close()
        print(f"  ✅  Connected  →  {ver[:70]}")
        return True
    except Exception as e:
        print(f"  ❌  {e}")
    return False


# ── Test 3: Source view exists + inspect columns ──────────────────────────────

def test_source_view(sample_n: int = 3) -> bool:
    sep("3. Source view / table (JSON views column)")
    if not _require_env("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        print("  ⏭   Skipped (credentials missing)")
        return False

    schema = os.getenv("DB_SCHEMA", "lightwheel")
    view   = os.getenv("ASSETS_VIEW", "robotics_qc_assets")
    bucket = os.getenv("AWS_S3_BUCKET", "project-parallax")
    col_id = os.getenv("ASSETS_COL_ASSET_ID", "asset_id")

    try:
        import psycopg2.extras
        conn = _pg_conn()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # Check existence
            cur.execute("""
                SELECT table_type
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            """, (schema, view))
            row = cur.fetchone()
            if not row:
                print(f"  ❌  {schema}.{view} does not exist")
                print(f"       Check ASSETS_VIEW in .env. Available tables:")
                cur.execute("""
                    SELECT table_schema || '.' || table_name AS full_name
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('pg_catalog','information_schema')
                    ORDER BY full_name LIMIT 20
                """)
                for r in cur.fetchall():
                    print(f"         {r['full_name']}")
                conn.close()
                return False
            print(f"  ✅  {schema}.{view} exists  (type={row['table_type']})")

            # Total row count
            cur.execute(f'SELECT COUNT(*) AS n FROM "{schema}"."{view}"')
            total = cur.fetchone()["n"]
            print(f"  Total asset rows : {total:,}")

            # Count main-slot, present=true
            cur.execute(f"""
                SELECT COUNT(*) AS n
                FROM "{schema}"."{view}",
                     jsonb_array_elements(views::jsonb) v
                WHERE v->>'expected_slot' = 'main'
                  AND (v->>'present')::boolean = true
            """)
            main_total = cur.fetchone()["n"]

            # Count non-gopro
            cur.execute(f"""
                SELECT COUNT(*) AS n
                FROM "{schema}"."{view}",
                     jsonb_array_elements(views::jsonb) v
                WHERE v->>'expected_slot' = 'main'
                  AND (v->>'present')::boolean = true
                  AND LOWER(v->>'bucket_key') NOT LIKE '%%gopro%%'
            """)
            non_gopro = cur.fetchone()["n"]
            gopro_n   = main_total - non_gopro

            print(f"  Main-slot present: {main_total:,}")
            print(f"  Non-gopro        : {non_gopro:,}  (will be processed)")
            print(f"  Gopro            : {gopro_n:,}  (will be skipped)")
            print(f"  S3 bucket        : {bucket}")

            # Sample rows
            if sample_n > 0 and non_gopro > 0:
                cur.execute(f"""
                    SELECT
                        "{col_id}"                                AS asset_id,
                        's3://' || %s || '/' || (v->>'bucket_key') AS s3_link,
                        (v->>'width')::int                        AS width,
                        (v->>'height')::int                       AS height,
                        (v->>'duration_sec')::numeric             AS duration_sec
                    FROM "{schema}"."{view}",
                         jsonb_array_elements(views::jsonb) v
                    WHERE v->>'expected_slot' = 'main'
                      AND (v->>'present')::boolean = true
                      AND LOWER(v->>'bucket_key') NOT LIKE '%%gopro%%'
                    ORDER BY "{col_id}"
                    LIMIT %s
                """, (bucket, sample_n))
                rows = cur.fetchall()
                print(f"\n  Sample (non-gopro main-view assets):")
                for r in rows:
                    stereo = (r['width'] or 0) >= 2 * (r['height'] or 1)
                    print(f"    asset_id = {r['asset_id']}")
                    print(f"    s3_link  = {r['s3_link']}")
                    print(f"    size     = {r['width']}×{r['height']}  "
                          f"dur={r['duration_sec']:.1f}s  "
                          f"stereo={'yes' if stereo else 'no'}")
                    print()

        conn.close()
        return True

    except Exception as e:
        print(f"  ❌  {e}")
        import traceback; traceback.print_exc()
    return False


# ── Test 4: el_auto_qc table ──────────────────────────────────────────────────

def test_qc_table(create: bool = False) -> bool:
    sep("4. lightwheel.el_auto_qc table")
    if not _require_env("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        print("  ⏭   Skipped (credentials missing)")
        return False

    schema = os.getenv("DB_SCHEMA", "lightwheel")

    try:
        import psycopg2.extras
        conn = _pg_conn()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT COUNT(*) AS n
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'el_auto_qc'
            """, (schema,))
            exists = cur.fetchone()["n"] > 0

            if exists:
                cur.execute(f'SELECT COUNT(*) AS n FROM "{schema}".el_auto_qc')
                n_rows = cur.fetchone()["n"]
                print(f"  ✅  {schema}.el_auto_qc exists  ({n_rows:,} rows)")

                if n_rows > 0:
                    cur.execute(f"""
                        SELECT el_status, COUNT(*) AS n
                        FROM "{schema}".el_auto_qc
                        GROUP BY el_status ORDER BY el_status
                    """)
                    for r in cur.fetchall():
                        print(f"       el_status={r['el_status']:10s}  rows={r['n']}")

            elif create:
                sql_path = os.path.join(
                    os.path.dirname(__file__),
                    "migrations", "001_create_el_auto_qc.sql"
                )
                if not os.path.exists(sql_path):
                    print(f"  ❌  Migration file not found: {sql_path}")
                    conn.close()
                    return False

                print(f"  ℹ️   Table not found — running migration …")
                with open(sql_path) as f:
                    ddl = f.read()
                # Execute DDL outside the cursor context manager so we can commit cleanly
                conn2 = _pg_conn()
                try:
                    conn2.autocommit = True
                    with conn2.cursor() as c2:
                        c2.execute(ddl)
                    print(f"  ✅  {schema}.el_auto_qc created successfully")
                finally:
                    conn2.close()
            else:
                print(f"  ⚠️   {schema}.el_auto_qc does not exist yet.")
                print(f"       Run:  python egolens/test_connections.py --create-table")
                print(f"       OR:   psql ... -f egolens/migrations/001_create_el_auto_qc.sql")

        conn.close()
        return True
    except Exception as e:
        print(f"  ❌  {e}")
        import traceback; traceback.print_exc()
    return False


# ── Test 5: AWS S3 credentials ───────────────────────────────────────────────

def test_s3() -> bool:
    sep("5. AWS S3 credentials")
    if not _require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        return False
    try:
        import boto3
        session = boto3.Session(
            aws_access_key_id    =os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name          =os.getenv("AWS_REGION", "us-east-1"),
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        print(f"  ✅  AWS identity : {identity.get('Arn', 'unknown')}")
        print(f"       Account      : {identity.get('Account', '')}")
        return True
    except Exception as e:
        print(f"  ❌  {e}")
    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python egolens/test_connections.py",
        description="EgoLens pre-flight connection tests"
    )
    parser.add_argument("--create-table", action="store_true",
                        help="Run SQL migration to create el_auto_qc if missing")
    parser.add_argument("--sample", type=int, default=3, metavar="N",
                        help="Number of sample rows from source view (default: 3)")
    args = parser.parse_args()

    print("\n" + "═" * 64)
    print("  EgoLens Auto-QC — Pre-flight Tests")
    print("═" * 64)

    results = {}
    results["1. DB env vars"]     = test_db_env()
    results["2. DB connection"]   = test_db_connection()
    results["3. Source view"]     = test_source_view(sample_n=args.sample)
    results["4. QC table"]        = test_qc_table(create=args.create_table)
    results["5. AWS S3"]          = test_s3()

    sep("Summary")
    all_ok = True
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All checks passed! Ready to run:")
        print("    python -m egolens.run_auto_qc --limit 5")
    else:
        print("  Fix the issues above, then re-run this script.")
    print()


if __name__ == "__main__":
    main()
