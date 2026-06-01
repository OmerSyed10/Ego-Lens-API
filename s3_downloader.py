"""
egolens/s3_downloader.py — Stream an S3 video to a local temp file.

Usage
-----
    with S3Downloader() as dl:
        local_path = dl.download("s3://my-bucket/clips/video.mp4")
        # … use local_path …
        # file is deleted automatically when the context manager exits

The downloader reads AWS credentials from the .env file:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION          (default: us-east-1)
    AWS_ENDPOINT_URL    (optional — for non-AWS S3-compatible stores)
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv
from .tracking_loader import find_tracking_file_for_video

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH, override=False)

try:
    import boto3
    from botocore.exceptions import ClientError
    _BOTO_AVAILABLE = True
except ImportError:
    _BOTO_AVAILABLE = False


# ── S3 URL parsing ────────────────────────────────────────────────────────────

def parse_s3_url(s3_url: str):
    """
    Parse  s3://bucket-name/path/to/object.mp4
    into   (bucket, key).
    """
    m = re.match(r"s3://([^/]+)/(.+)", s3_url.strip())
    if not m:
        raise ValueError(f"Not a valid S3 URL: {s3_url!r}")
    return m.group(1), m.group(2)


# ── Downloader ────────────────────────────────────────────────────────────────

class S3Downloader:
    """
    Context manager that downloads S3 objects to local temp files
    and cleans them up automatically on exit.

    Parameters
    ----------
    tmp_dir : str | None
        Directory for temp files. Defaults to TMP_DIR env var or system tmp.
    """

    def __init__(self, tmp_dir: Optional[str] = None) -> None:
        if not _BOTO_AVAILABLE:
            raise ImportError("boto3 is required: pip install boto3")

        self._tmp_dir    = tmp_dir or os.getenv("TMP_DIR") or tempfile.gettempdir()
        self._temp_files: list[str] = []

        # Build boto3 session from env
        session = boto3.Session(
            aws_access_key_id     = os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name           = os.getenv("AWS_REGION", "us-east-1"),
        )

        kwargs = {}
        endpoint = os.getenv("AWS_ENDPOINT_URL", "")
        if endpoint:
            kwargs["endpoint_url"] = endpoint

        self._s3 = session.client("s3", **kwargs)

    def download_tracking_file(self, video_s3_url: str) -> Optional[str]:
        """
        Auto-discover and download the XR tracking file that belongs to this video.

        The tracking file lives in the same S3 folder as the video:
            CameraRecord_YYYYMMDD_HHMMSS_av.mp4  →  trackingData_YYYYMMDD_HHMMSS.txt

        Returns the local file path, or None if:
          - No matching tracking file pattern found
          - The file doesn't exist in S3 (silently skipped)
        """
        bucket, video_key = parse_s3_url(video_s3_url)
        tracking_key = find_tracking_file_for_video(video_key)

        if tracking_key is None:
            return None  # video filename doesn't match expected pattern

        # Check if object exists before downloading
        try:
            self._s3.head_object(Bucket=bucket, Key=tracking_key)
        except Exception:
            return None  # tracking file not present in S3

        filename  = os.path.basename(tracking_key)
        local_path = os.path.join(self._tmp_dir, filename)

        print(f"    ↓  Downloading tracking data …")
        print(f"       s3://{bucket}/{tracking_key}")
        print(f"       → {local_path}")

        try:
            self._s3.download_file(bucket, tracking_key, local_path)
        except Exception as exc:
            print(f"    ⚠  Tracking download failed: {exc} — continuing without it")
            return None

        self._temp_files.append(local_path)
        return local_path

    def download(self, s3_url: str) -> str:
        """
        Download the video at s3_url to a local temp file.

        Returns the local file path.
        The file is registered for automatic cleanup when the context exits.
        """
        bucket, key = parse_s3_url(s3_url)

        # Derive a readable filename from the S3 key
        filename  = os.path.basename(key) or "video.mp4"
        # Ensure .mp4 extension
        if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            filename += ".mp4"

        local_path = os.path.join(self._tmp_dir, filename)

        print(f"    ↓  Downloading  {s3_url}")
        print(f"       → {local_path}")

        try:
            self._s3.download_file(bucket, key, local_path)
        except ClientError as exc:
            raise IOError(
                f"S3 download failed for {s3_url}: {exc}"
            ) from exc

        self._temp_files.append(local_path)
        return local_path

    def cleanup(self) -> None:
        """Delete all temp files registered during this session."""
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"    🗑  Deleted temp file: {path}")
            except OSError:
                pass
        self._temp_files.clear()

    # ── Context manager protocol ──────────────────────────────────────────────

    def __enter__(self) -> "S3Downloader":
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()
