"""
test_api.py — Quick test script for the EgoLens QC API.

Usage:
    # 1. Start the server in one terminal:
    #    uvicorn egolens.api:app --host 0.0.0.0 --port 8000

    # 2. In another terminal:
    #    python egolens/test_api.py --url "https://your-video-url" --bearer "your_token"

    # Or test the health check only:
    #    python egolens/test_api.py --health-only
"""

import argparse
import json
import sys
import requests


DEFAULT_BASE = "http://localhost:8000"


def test_health(base: str):
    resp = requests.get(f"{base}/health", timeout=5)
    resp.raise_for_status()
    print("Health check:", resp.json())


def test_qc(url: str, bearer: str, base: str):
    print(f"\nSending QC request for:\n  {url}\n")

    resp = requests.post(
        f"{base}/qc",
        json={"url": url, "bearer": bearer},
        timeout=600,   # large timeout — pipeline can take a few minutes
    )

    if not resp.ok:
        print(f"ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()

    # Pretty-print the full JSON
    print(json.dumps(result, indent=2))

    # Summary
    print("\n" + "=" * 50)
    print(f"Overall score : {result['overall_score']}/100")
    print(f"Overall pass  : {result['overall_pass']}")
    if result.get("failed_detectors"):
        print(f"Failed checks : {', '.join(result['failed_detectors'])}")
    if result.get("fail_reason"):
        print(f"Reason        : {result['fail_reason']}")
    print(f"Processing    : {result['processing_time_s']}s")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test the EgoLens QC API")
    parser.add_argument("--url",          help="Video URL to test")
    parser.add_argument("--bearer",       help="Authorization bearer token")
    parser.add_argument("--base",         default=DEFAULT_BASE, help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--health-only",  action="store_true", help="Only check /health endpoint")
    args = parser.parse_args()

    test_health(args.base)

    if args.health_only:
        return

    if not args.url or not args.bearer:
        parser.error("--url and --bearer are required (unless --health-only)")

    test_qc(args.url, args.bearer, args.base)


if __name__ == "__main__":
    main()
