"""
egolens/tracking_loader.py  — T11

Loads XR hand-tracking data from the device's trackingData_*.txt file.

File format: newline-delimited JSON (same as XRoboToolkit pose stream),
each line is a full frame:
  {"Head": {"pose": "x,y,z,qx,qy,qz,qw"},
   "Hand": {"leftHand": {"HandJointLocations": [...]},
            "rightHand": {"HandJointLocations": [...]}},
   "timeStampNs": 1234567890123456789}

Each HandJointLocations entry:
  {"pose": "x,y,z,qx,qy,qz,qw", "radius": 0.01, "isActive": 1}

Joint 0 = Wrist (OpenXR XrHandJointEXT layout)

The loader builds a timestamp-indexed table:
  {timestamp_ns → (left_wrist_xyz, right_wrist_xyz)}

Both wrist positions are in world-space metres.  The caller uses them as
a Tier-0 signal in HandDetector (before MediaPipe) to reduce missed frames.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# OpenXR wrist joint index
_WRIST_IDX = 0

# Type aliases
WristXYZ  = np.ndarray            # [x, y, z] world metres
TrackDict = Dict[int, Tuple[Optional[WristXYZ], Optional[WristXYZ]]]
# timestamp_ns → (left_wrist_xyz or None, right_wrist_xyz or None)


def _parse_pose_str(s: str) -> Optional[np.ndarray]:
    """'x,y,z,qx,qy,qz,qw' → np.array([x,y,z]) or None."""
    try:
        v = [float(x) for x in s.split(",")]
        if len(v) < 3:
            return None
        return np.array(v[:3], dtype=np.float64)
    except (ValueError, AttributeError):
        return None


def _extract_wrist(joint_list: Optional[list]) -> Optional[WristXYZ]:
    """Extract wrist (joint 0) world position from HandJointLocations list."""
    if not joint_list or len(joint_list) == 0:
        return None
    entry = joint_list[0]
    pose_str = entry.get("pose") or entry.get("p")
    if pose_str is None:
        return None
    return _parse_pose_str(pose_str)


def load_tracking_file(path: str) -> TrackDict:
    """
    Parse a trackingData_*.txt file into a timestamp-indexed dict.

    Parameters
    ----------
    path : str  Local path to the tracking text file.

    Returns
    -------
    TrackDict : {timestamp_ns → (left_wrist_xyz, right_wrist_xyz)}
                Values may be None if joint data is missing.
    """
    tracks: TrackDict = {}

    if not os.path.exists(path):
        return tracks

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip non-pose lines (headers, sync events)
            if "notice" in obj or "type" in obj:
                continue

            ts_ns = int(obj.get("timeStampNs", 0))
            if ts_ns == 0:
                continue

            hand_root  = obj.get("Hand") or obj.get("hands") or {}
            left_data  = (hand_root.get("leftHand")  or hand_root.get("left")  or {})
            right_data = (hand_root.get("rightHand") or hand_root.get("right") or {})

            left_joints  = left_data.get("HandJointLocations")
            right_joints = right_data.get("HandJointLocations")

            left_wrist  = _extract_wrist(left_joints)
            right_wrist = _extract_wrist(right_joints)

            if left_wrist is not None or right_wrist is not None:
                tracks[ts_ns] = (left_wrist, right_wrist)

    return tracks


def query_tracking_at(
    tracks: TrackDict,
    video_timestamp_s: float,
    video_start_ns: int,
    tolerance_ns: int = 100_000_000,   # ±100 ms
) -> Tuple[Optional[WristXYZ], Optional[WristXYZ]]:
    """
    Look up the closest tracking frame to a given video timestamp.

    Parameters
    ----------
    tracks           : output of load_tracking_file()
    video_timestamp_s: video frame timestamp in seconds
    video_start_ns   : camera start time in nanoseconds (UTC epoch)
                       If 0, alignment is done by relative offset.
    tolerance_ns     : max allowed time difference in ns (default ±100 ms)

    Returns
    -------
    (left_wrist_xyz, right_wrist_xyz) or (None, None) if no match.
    """
    if not tracks:
        return None, None

    target_ns = video_start_ns + int(video_timestamp_s * 1e9)
    best_key  = None
    best_diff = tolerance_ns + 1

    for ts_ns in tracks:
        diff = abs(ts_ns - target_ns)
        if diff < best_diff:
            best_diff = diff
            best_key  = ts_ns

    if best_key is None or best_diff > tolerance_ns:
        return None, None

    return tracks[best_key]


def find_tracking_file_for_video(video_s3_key: str) -> Optional[str]:
    """
    Derive the expected S3 key for the tracking file next to a video.

    Video key:   DATA/Inhouse_Reviewed/session_20260513_141945/CameraRecord_20260513_141946_av.mp4
    Tracking key: DATA/Inhouse_Reviewed/session_20260513_141945/trackingData_20260513_141946.txt

    Rules:
      - Same folder as the video
      - Stem: replace 'CameraRecord' → 'trackingData', remove '_av' suffix

    Returns None if pattern does not match.
    """
    import re
    folder  = "/".join(video_s3_key.split("/")[:-1])
    base    = video_s3_key.split("/")[-1]

    # CameraRecord_YYYYMMDD_HHMMSS_av.mp4 → trackingData_YYYYMMDD_HHMMSS.txt
    m = re.match(r"CameraRecord_(\d{8}_\d{6})(?:_av)?\.mp4$", base, re.IGNORECASE)
    if not m:
        return None

    date_part    = m.group(1)
    tracking_key = f"{folder}/trackingData_{date_part}.txt"
    return tracking_key


class TrackingData:
    """
    Convenience wrapper around a loaded TrackDict.
    Pass an instance to EgoLensPipeline so HandDetector can use Tier-0 data.
    """

    def __init__(self, tracks: TrackDict, video_start_ns: int = 0) -> None:
        self._tracks         = tracks
        self._video_start_ns = video_start_ns
        self._sorted_keys    = sorted(tracks.keys()) if tracks else []

    @classmethod
    def empty(cls) -> "TrackingData":
        return cls({}, 0)

    @classmethod
    def from_file(cls, path: str, video_start_ns: int = 0) -> "TrackingData":
        tracks = load_tracking_file(path)
        return cls(tracks, video_start_ns)

    def at(
        self,
        video_timestamp_s: float,
        tolerance_ns: int = 100_000_000,
    ) -> Tuple[Optional[WristXYZ], Optional[WristXYZ]]:
        """Return (left_wrist_xyz, right_wrist_xyz) closest to video_timestamp_s."""
        if not self._tracks:
            return None, None

        target_ns = self._video_start_ns + int(video_timestamp_s * 1e9)

        # Binary search for closest key
        import bisect
        idx = bisect.bisect_left(self._sorted_keys, target_ns)

        best_diff = tolerance_ns + 1
        best_key  = None

        for i in (idx - 1, idx):
            if 0 <= i < len(self._sorted_keys):
                k    = self._sorted_keys[i]
                diff = abs(k - target_ns)
                if diff < best_diff:
                    best_diff = diff
                    best_key  = k

        if best_key is None or best_diff > tolerance_ns:
            return None, None

        return self._tracks[best_key]

    def __len__(self) -> int:
        return len(self._tracks)

    def __bool__(self) -> bool:
        return bool(self._tracks)
