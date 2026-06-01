# EgoLens — Egocentric Video Quality Control Pipeline

> Automated, config-driven QC for head-worn camera training datasets.  
> Nine quality detectors · PostgreSQL results store · Interactive HTML dashboard · Bounding-box overlays on video.

---

## Table of Contents

1. [What It Checks](#1-what-it-checks)
2. [What's New — v2 (Recent Changes)](#2-whats-new--v2-recent-changes)
3. [Architecture](#3-architecture)
4. [Project Layout](#4-project-layout)
5. [Requirements & Installation](#5-requirements--installation)
6. [Quick Start — Batch Auto-QC](#6-quick-start--batch-auto-qc)
7. [HTML Dashboard](#7-html-dashboard)
8. [CLI Reference](#8-cli-reference)
9. [Anti-False-Positive Design (Privacy Detectors)](#9-anti-false-positive-design-privacy-detectors)
10. [config.yaml — Field-by-Field Guide](#10-configyaml--field-by-field-guide)
11. [Stereo Video Handling](#11-stereo-video-handling)
12. [Duration-Based Violations Explained](#12-duration-based-violations-explained)
13. [Scoring Formula](#13-scoring-formula)
14. [Using EgoLens as a Python Library](#14-using-egolens-as-a-python-library)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. What It Checks

| Detector | What it measures | Fails when… | Model |
|---|---|---|---|
| 🌑 **Brightness** | Mean YCrCb Y-channel luminance | Scene is too dark (< 30 % of full range) | YCrCb channel mean |
| 🔍 **Sharpness** | Laplacian variance of each frame | Image is blurry or defocused | OpenCV Laplacian |
| 📷 **Stability** | Farneback optical flow magnitude | Camera shakes continuously | Dense optical flow |
| 🤚 **Hand Detection** | 5-tier cascade: MP Hands → 100DOH → MP Pose → Skin → Phantom | No hands visible for > N seconds | MediaPipe + ResNet-101 |
| ⚡ **Hand Speed** | Wrist displacement between sampled frames | Hands stationary for > N seconds | Landmark velocity |
| 📍 **Hand Position** | Fraction of landmarks inside workspace box | Hands outside workspace for > N seconds | Landmark centroid zones |
| ↩️ **Hand Out-of-Frame** | Wrist proximity to frame edge + velocity direction | Wrist drifting off screen | Edge proximity + velocity check |
| 👤 **Face Detection** | Eye-region sharpness of any detected face (privacy) | Identifiable face visible for > N seconds | MediaPipe + RetinaFace |
| 🚶 **Person Detection** | Upper-body keypoints + multi-layer verification (privacy) | Another person visible for > N seconds | MediaPipe Pose + YOLOv8n |

The **5-tier hand detection cascade** ensures hands are found even when fingers are hidden (fists, tool use, dark skin):

```
Tier 1 — MediaPipe Hands     (primary; flip + CLAHE retries if weak)
Tier 2 — 100DOH ResNet-101   (Faster-RCNN; ~90% AP on egocentric data)
Tier 3 — MediaPipe Pose wrist (from arm geometry; works on any hand shape)
Tier 4 — YCbCr skin blob      (colour-based fallback; Kovac 2003 thresholds)
Tier 5 — Phantom coast         (velocity extrapolation through brief occlusion)
```

---

## 2. What's New — v2 (Recent Changes)

---

### 🗄️ PostgreSQL Integration (`db.py`, `auto_qc_runner.py`)

EgoLens now writes all results to a PostgreSQL table (`lightwheel.el_auto_qc`).

**What's stored per asset:**
- Scores and normalised values for all 9 detectors
- Per-detector `pass` boolean, overall score, fail reason text
- `el_failed_detectors` — PostgreSQL `text[]` array
- `el_violation_segments` — PostgreSQL **JSONB** with timestamps and bounding boxes
- Video metadata (fps, resolution, duration, stereo flag, frames sampled)

**New columns (migration `002_add_columns.sql`):**
```sql
el_brightness_score   NUMERIC(6,2)
el_brightness_norm    NUMERIC(8,4)
el_brightness_pass    BOOLEAN
el_violation_segments JSONB   -- GIN indexed
```

**Violation segments JSONB format:**
```json
[
  {
    "detector": "face_detection",
    "start_s": 3.2, "end_s": 4.5, "duration_s": 1.3,
    "bboxes": [{ "frame_s": 3.4, "bbox": [0.31, 0.12, 0.68, 0.47] }]
  }
]
```

---

### ☀️ Brightness Detector (`detectors/brightness_detector.py`)

New detector that flags frames too dark to be useful for training.

- **YCrCb Y-channel** mean luminance — more perceptually accurate than raw RGB
- Gate: `0.30` (mean Y < 76.5 / 255 = dark); Violation window: `2.0 s`
- Caught 2 real failures: `session_20260513_141945` (score 9.4) and `session_20260513_142211` (score 29.9)

---

### 🔒 RetinaFace Secondary Face Verifier (`detectors/face_verifier.py`)

Two-stage face detection: **MediaPipe flags → RetinaFace confirms**.

- **InsightFace `buffalo_sc`** model (`det_500m.onnx`, ~14 MB, auto-downloaded)
- Returns `(confirmed: bool, bbox_norm: [x1,y1,x2,y2])` in normalised `[0,1]` coords
- Graceful fallback to `(True, None)` if InsightFace not installed

```bash
pip install insightface onnxruntime
```

---

### 🚶 YOLOv8n Person Verifier — Multi-Layer Anti-FP (`detectors/person_verifier.py`)

Completely rewritten to prevent the camera wearer's own hands/arms being flagged as "another person".

**The problem:** At confidence 0.40, YOLOv8n detects the worker's forearms on a table as a "person" (wide flat box at the bottom of frame).

**Layered solution:**

```
HARD GATES (instant reject if any fails):
  ① conf < 0.60         → worker body scores 0.40–0.55 → rejected
  ② aspect h/w < 0.40   → flat box (hands on table) → rejected
  ③ area > 40%          → own torso filling lens → rejected
  ④ entire box below     → both y1 & y2 > 0.65h → rejected
     65% frame height     camera-tilt safe: spanning boxes pass

EVIDENCE SCORING (soft signals):
  A. Position score  = 1 − centroid_y/h        [0–1]
  B. Hand overlap veto  = −1.5  if bbox overlaps own hands ≥ 40%
                                (handshake guard: skip if person >> hand)
  C. Face-in-crop bonus = +1.5  if RetinaFace finds a face in the crop
                                (neutral if no face — back-turned person safe)

ACCEPT if: score ≥ 1.0  OR  face confirmed in crop
```

**Result:** `session_20260512_134656` previously failing (person_detection 49.2/100) → **✅ PASS 81.8/100** after fix.

---

### 📦 Violation Segments with Bounding Boxes (`duration_tracker.py`)

- Segments changed from `List[Tuple]` to `List[Dict]` with `start_s`, `end_s`, `bboxes`
- Up to 10 normalised bbox samples `[x1/w, y1/h, x2/w, y2/h]` per segment
- Stored in PostgreSQL JSONB and rendered as canvas overlay in the HTML dashboard

---

### 📊 Interactive HTML QC Dashboard (`generate_qc_report.py`)

Self-contained HTML (~185 KB), no external dependencies.

| Feature | Detail |
|---|---|
| Tabs | All / ✅ Pass / ❌ Fail with live counts |
| Asset cards | Score bars per detector, fail reason, metadata |
| ▶ Play | HTML5 video modal with pre-signed S3 URL (4h expiry) |
| Timeline | Coloured violation segments, click to jump |
| Bbox overlay | `<canvas>` drawn live on video — coloured boxes at violation frames |
| 📍 Badge | Number of stored bbox samples per segment |

```bash
.venv/bin/python3 -m egolens.generate_qc_report -o report.html
.venv/bin/python3 -m egolens.generate_qc_report --no-video   # no S3 needed
```

---

### 🔄 `--failed-only` Rerun Mode (`run_auto_qc.py`)

```bash
.venv/bin/python3 -m egolens.run_auto_qc --failed-only
```

Resets `el_overall_pass = false` / `el_status = 'error'` rows to `pending`, then reprocesses only those — never touches clean assets.

---

### 📡 XR Tracking Loader — Tier-0 Hand Boost (`tracking_loader.py`)

- Auto-discovers XR wrist-tracking `.txt` files alongside S3 video
- If wrist world-position confirmed → boosts `HandDetector` score to `min(score, 0.60)`
- Silent fallback if no tracking file found

---

### 🎯 Strict AND Scorer (`scorer.py`)

```
passed = (overall_score ≥ threshold)  AND  (all detector gates pass)
```

A video cannot pass overall if any detector fails its individual gate.

---

### ↩️ Velocity-Direction OOF Check (`detectors/hand_out_of_frame_detector.py`)

- Near edge + moving **toward** edge → full penalty
- Near edge + moving **inward** or stationary → lenient (0.5 credit)
- Eliminates false positives from natural wrist rotation near frame edges

---

## 3. Architecture

```
S3 Video Asset
    │
    ▼
S3Downloader          ← video + XR tracking file
    │
    ▼
VideoPreprocessor     ← stereo crop, stride calculation
    │
    ▼
EgoLensPipeline       ← 9 detectors per sampled frame
    │   ├── BrightnessDetector
    │   ├── SharpnessDetector
    │   ├── StabilityDetector
    │   ├── HandDetector (5-tier + Tier-0 XR boost)
    │   ├── HandSpeedDetector
    │   ├── HandPositionDetector
    │   ├── HandOutOfFrameDetector (velocity-direction aware)
    │   ├── FaceDetector ──► FaceVerifier (RetinaFace)
    │   └── PersonDetector ─► PersonVerifier (YOLOv8n + 4-layer anti-FP)
    │
    ▼
Scorer                ← weighted score + strict AND pass
    │
    ▼
AutoQCRunner          ← PostgreSQL (scores + violation JSONB)
    │
    ▼
HTML Dashboard        ← video playback + bbox canvas overlay
```

---

## 4. Project Layout

```
egolens/
├── README.md
├── config.yaml                        ← master config
├── requirements_autoqc.txt
├── auto_qc_runner.py                  ← batch S3 → pipeline → PostgreSQL
├── run_auto_qc.py                     ← CLI entry + --failed-only
├── generate_qc_report.py              ← HTML dashboard generator
├── db.py                              ← PostgreSQL helpers
├── s3_downloader.py                   ← S3 + tracking file downloader
├── tracking_loader.py                 ← XR wrist-tracking parser
├── pipeline.py                        ← EgoLensPipeline orchestrator
├── preprocessor.py                    ← stereo crop + downscale
├── duration_tracker.py                ← DurationViolationTracker (with bboxes)
├── scorer.py                          ← weighted score + strict AND
├── reporter.py                        ← console output
├── config.py                          ← ConfigLoader
└── detectors/
    ├── brightness_detector.py         ← NEW v2
    ├── sharpness.py
    ├── stability.py
    ├── hand_tracker.py                ← 5-tier cascade
    ├── hand_detector.py               ← + Tier-0 XR boost
    ├── hand_speed_detector.py
    ├── hand_position_detector.py
    ├── hand_out_of_frame_detector.py  ← velocity-direction aware
    ├── face_detector.py
    ├── face_verifier.py               ← NEW v2: RetinaFace
    ├── person_detector.py
    ├── person_verifier.py             ← NEW v2: YOLOv8n + 4-layer anti-FP
    ├── skin_detector.py
    └── doh_detector.py
```

---

## 5. Requirements & Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_autoqc.txt

# Privacy detector models (auto-download weights on first use)
pip install insightface onnxruntime   # RetinaFace
pip install ultralytics               # YOLOv8n
```

> ⚠️ **mediapipe must be 0.10.14** — always use `.venv/bin/python3`, not system Python.

---

## 6. Quick Start — Batch Auto-QC

Create `egolens/.env`:
```env
DB_HOST=...  DB_PORT=5432  DB_NAME=...  DB_USER=...  DB_PASSWORD=...
AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_REGION=us-east-1
```

```bash
# All pending assets
.venv/bin/python3 -m egolens.run_auto_qc

# Only failed assets
.venv/bin/python3 -m egolens.run_auto_qc --failed-only

# HTML report
.venv/bin/python3 -m egolens.generate_qc_report -o report.html && open report.html
```

---

## 7. Anti-False-Positive Design

### Face — Two-Stage
```
MediaPipe (primary + eye sharpness)
    └─ face detected → RetinaFace (InsightFace buffalo_sc)
          ├─ confirmed → ❌ violation + bbox stored
          └─ rejected  → ✅ false positive suppressed
```

### Person — Four-Layer
```
Hard gates: conf ≥ 0.60, h/w ≥ 0.40, area ≤ 40%, not entirely below 65% height
Soft score: position + hand-overlap veto + face-in-crop bonus
Accept if: score ≥ 1.0 OR face found in crop
```

---

## 8. Scoring Formula

```
overall_score = Σ(mean_norm_i × 100 × weight_i) / Σ(weight_i)
passed        = overall_score ≥ threshold  AND  all gates pass
```

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `mediapipe has no attribute 'solutions'` | Use `.venv/bin/python3` (mediapipe 0.10.14) |
| `insightface not installed` | `pip install insightface onnxruntime` |
| `Could not load YOLOv8n` | `pip install ultralytics` |
| All videos fail — low hand detection | Lower `hand_detection.gate` or install 100DOH |
| `hand_position` fails — hands visible | Loosen `workspace` bounds in config.yaml |
| `GL version: 2.1 (Metal)` warnings | Harmless — suppress with `2>/dev/null` |

---

## Current Results — 58 Assets

| | |
|---|---|
| ✅ Pass | **55** |
| ❌ Fail | **3** |

**3 genuine failures (hardware issues, not fixable by tuning):**

| Asset | Score | Reason |
|---|---|---|
| `session_20260513_141945` | 63.4 | 🌑 Video shot in near-darkness (brightness 9.4/100) |
| `session_20260513_142211` | 72.7 | 🌑 Partially dark video (brightness 29.9/100) |
| `Cleaning_phone_case` | 86.4 | ✋ Hand repeatedly drifts out of frame |
