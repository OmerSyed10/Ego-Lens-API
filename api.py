"""
egolens/api.py — EgoLens QC REST API

Exposes a single endpoint:

    POST /qc
    Body: { "url": "<video_url>", "bearer": "<auth_token>" }

    Returns a JSON object with all QC scores (mirrors the el_auto_qc DB columns).

Start the server:
    pip install fastapi uvicorn
    uvicorn egolens.api:app --host 0.0.0.0 --port 8000

Or from the project root:
    python -m egolens.api

Example curl:
    curl -X POST http://localhost:8000/qc \\
         -H "Content-Type: application/json" \\
         -d '{"url": "https://...", "bearer": "your_token_here"}'
"""

from __future__ import annotations

import os
import tempfile
import time
import json
import traceback
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── EgoLens imports ───────────────────────────────────────────────────────────
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from egolens.config import load_config, DEFAULT_CONFIG_PATH
from egolens.pipeline import EgoLensPipeline
from egolens.scorer import score as el_score


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EgoLens QC API",
    description="Run automatic quality checks on an egocentric video from a URL.",
    version="1.0.0",
)

# Load config + pipeline once at startup (heavy — reused across requests)
_cfg     = load_config(DEFAULT_CONFIG_PATH)
_pipeline = EgoLensPipeline(_cfg)


# ── Request / Response models ─────────────────────────────────────────────────

class QCRequest(BaseModel):
    url: str
    bearer: str
    config_path: Optional[str] = None   # override default config if needed


class ViolationSegment(BaseModel):
    detector:   str
    start_s:    float
    end_s:      float
    duration_s: float
    bboxes:     Optional[List[Dict[str, Any]]] = None


class DetectorQC(BaseModel):
    score:      Optional[float]   # 0–100
    norm:       Optional[float]   # 0–1
    passed:     Optional[bool]


class QCResponse(BaseModel):
    # Overall
    overall_score:       float
    overall_pass:        bool
    failed_detectors:    Optional[List[str]]
    fail_reason:         Optional[str]

    # Video metadata
    video_fps:           float
    video_duration_s:    float
    video_native_w:      int
    video_native_h:      int
    is_stereo:           bool
    effective_w:         int
    effective_h:         int
    frames_sampled:      int

    # Per-detector results
    brightness:          DetectorQC
    sharpness:           DetectorQC
    stability:           DetectorQC
    hand_detection:      DetectorQC
    hand_speed:          DetectorQC
    hand_position:       DetectorQC
    hand_out_of_frame:   DetectorQC
    face_detection:      DetectorQC
    person_detection:    DetectorQC

    # Hand source breakdown (%)
    hand_src_hands_pct:   float
    hand_src_pose_pct:    float
    hand_src_phantom_pct: float
    hand_src_none_pct:    float

    # Violation segments
    violation_segments:  Optional[List[ViolationSegment]]

    # Timing
    processing_time_s:   float


# ── Video downloader ──────────────────────────────────────────────────────────

def _download_video(url: str, bearer: str, dest_path: str) -> None:
    """
    Download the video at `url` via the luigi proxy endpoint to `dest_path`.

    Uses streaming to avoid loading the entire file into memory.
    Raises requests.HTTPError on non-200 responses.
    """
    proxy_url = "https://luigi.playment.io/api/v1/attachments?url=" + url
    resp = requests.get(
        proxy_url,
        headers={
            "Authorization": bearer,
            "X-Caller-Service": "egolens-api",
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


# ── QC result builder ─────────────────────────────────────────────────────────

def _build_response(result, scored, processing_time_s: float) -> QCResponse:
    """Convert AnalysisResult + ScoredResult into the API response model."""
    meta = result.meta
    dets = result.detectors

    def _det(name: str) -> DetectorQC:
        d = dets.get(name)
        if d is None or not d.enabled:
            return DetectorQC(score=None, norm=None, passed=None)
        norm  = d.mean_norm
        score = round(norm * 100, 2)
        passed = norm >= d.gate
        return DetectorQC(score=score, norm=round(norm, 4), passed=passed)

    # Hand source breakdown
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

    # Violation segments
    violation_segments: List[ViolationSegment] = []
    for det_name, det in dets.items():
        if not det.enabled or det.violation is None:
            continue
        for seg in det.violation.violation_segments:
            vs = ViolationSegment(
                detector=det_name,
                start_s=round(seg["start_s"], 3),
                end_s=round(seg["end_s"], 3),
                duration_s=round(seg["end_s"] - seg["start_s"], 3),
                bboxes=seg.get("bboxes") or None,
            )
            violation_segments.append(vs)
    violation_segments.sort(key=lambda x: (x.start_s, x.detector))

    return QCResponse(
        overall_score=round(scored.overall_score, 2),
        overall_pass=scored.passed,
        failed_detectors=scored.failed_detectors or None,
        fail_reason=scored.fail_reason_text or None,

        video_fps=round(meta.fps, 2),
        video_duration_s=round(meta.duration_s, 2),
        video_native_w=meta.native_w,
        video_native_h=meta.native_h,
        is_stereo=meta.is_stereo,
        effective_w=meta.effective_w,
        effective_h=meta.effective_h,
        frames_sampled=len(result.frames),

        brightness=_det("brightness"),
        sharpness=_det("sharpness"),
        stability=_det("stability"),
        hand_detection=_det("hand_detection"),
        hand_speed=_det("hand_speed"),
        hand_position=_det("hand_position"),
        hand_out_of_frame=_det("hand_out_of_frame"),
        face_detection=_det("face_detection"),
        person_detection=_det("person_detection"),

        hand_src_hands_pct=hand_src.get("hands", 0.0),
        hand_src_pose_pct=hand_src.get("pose", 0.0),
        hand_src_phantom_pct=hand_src.get("phantom", 0.0),
        hand_src_none_pct=hand_src.get("none", 0.0),

        violation_segments=violation_segments or None,
        processing_time_s=round(processing_time_s, 2),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok"}


@app.post("/qc", response_model=QCResponse)
def run_qc(req: QCRequest):
    """
    Run EgoLens QC on the video at `req.url`.

    Steps:
      1. Stream-download video to a temp file via the luigi proxy
      2. Run EgoLensPipeline
      3. Score results
      4. Return JSON with all detector scores + violation segments

    Raises HTTP 400 on bad requests, HTTP 502 on downstream errors,
    HTTP 500 on pipeline failures.
    """
    t_start = time.time()

    # Use per-request config if supplied, otherwise use the preloaded one
    if req.config_path:
        try:
            cfg      = load_config(req.config_path)
            pipeline = EgoLensPipeline(cfg)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Bad config: {exc}")
    else:
        cfg      = _cfg
        pipeline = _pipeline

    # Determine file extension from URL (default .mp4)
    url_path = req.url.split("?")[0]
    ext = os.path.splitext(url_path)[-1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv"):
        ext = ".mp4"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="egolens_")
    os.close(tmp_fd)

    try:
        # 1. Download
        try:
            _download_video(req.url, req.bearer, tmp_path)
        except requests.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to download video: {exc.response.status_code} {exc.response.reason}",
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Download error: {exc}")

        # 2. Run pipeline
        try:
            result = pipeline.run(tmp_path, verbose=False)
        except IOError as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline IO error: {exc}")
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

        # 3. Score
        scored = el_score(result, cfg)

        # 4. Build response
        processing_time_s = time.time() - t_start
        return _build_response(result, scored, processing_time_s)

    finally:
        # Always clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("egolens.api:app", host="0.0.0.0", port=8000, reload=False)
