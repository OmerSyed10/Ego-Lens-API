# EgoLens — Egocentric Video Quality Control Pipeline

EgoLens is a config-driven, offline QC pipeline for egocentric (head-worn camera)
training videos. It analyses a video against 8 independent quality checks, flags
violations that persist beyond a configurable time window, and returns a weighted
overall score and pass/fail verdict.

---

## Table of Contents

1. [What It Checks](#1-what-it-checks)
2. [Project Layout](#2-project-layout)
3. [Requirements](#3-requirements)P
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [CLI Reference](#6-cli-reference)
7. [Sample Output — Passed Video](#7-sample-output--passed-video)
8. [Sample Output — Failed Video (Stereo)](#8-sample-output--failed-video-stereo)
9. [config.yaml — Field-by-Field Guide](#9-configyaml--field-by-field-guide)
10. [Stereo Video Handling](#10-stereo-video-handling)
11. [Duration-Based Violations Explained](#11-duration-based-violations-explained)
12. [Scoring Formula](#12-scoring-formula)
13. [Using EgoLens as a Python Library](#13-using-egolens-as-a-python-library)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What It Checks

| Detector | What it measures | Fails when… |
|---|---|---|
| **Sharpness** | Laplacian variance of each frame | Image is blurry (camera shake, defocus) |
| **Stability** | Mean optical-flow magnitude between frames | Camera shakes or pans continuously |
| **Hand Detection** | 5-tier cascade: MP Hands → 100DOH → MP Pose → Skin → Phantom coast | No hands visible for > N seconds |
| **Hand Speed** | Mean wrist displacement between sampled frames (px/frame) | Hands stationary for > N seconds |
| **Hand Position** | Fraction of landmarks inside the workspace bounding box | Hands outside the workspace for > N seconds |
| **Hand Out-of-Frame** | Wrist proximity to any frame edge | Wrist near edge for > N seconds |
| **Face Detection** | Eye-region sharpness of any detected face | An identifiable face is visible for > N seconds |
| **Person Detection** | Upper-body keypoint visibility via MP Pose | Another person is visible for > N seconds |

The **5-tier hand detection cascade** ensures hands are found even when fingers are
hidden (fists, tool use, dark skin):
```
Tier 1 — MediaPipe Hands     (primary; flip + CLAHE retries if weak)
Tier 2 — 100DOH ResNet-101   (Faster-RCNN; ~90% AP on egocentric data)
Tier 3 — MediaPipe Pose wrist (from arm geometry; works on any hand shape)
Tier 4 — YCbCr skin blob      (colour-based fallback; Kovac 2003 thresholds)
Tier 5 — Phantom coast         (velocity extrapolation through brief occlusion)
```

---

## 2. Project Layout

```
egolens/
├── README.md                        ← you are here
├── config.yaml                      ← master config (edit this)
├── __init__.py
├── __main__.py                      ← python -m egolens entry point
├── cli.py                           ← argument parsing
├── pipeline.py                      ← EgoLensPipeline orchestrator
├── preprocessor.py                  ← stereo crop + downscale
├── duration_tracker.py              ← DurationViolationTracker
├── scorer.py                        ← weighted score computation
├── reporter.py                      ← formatted console output
├── config.py                        ← ConfigLoader (YAML → dataclasses)
└── detectors/
    ├── sharpness.py
    ├── stability.py
    ├── hand_tracker.py              ← 5-tier cascade implementation
    ├── hand_detector.py             ← hand detection + duration tracking
    ├── hand_speed_detector.py
    ├── hand_position_detector.py
    ├── hand_out_of_frame_detector.py
    ├── face_detector.py
    ├── person_detector.py
    ├── skin_detector.py             ← YCbCr Tier 4
    └── doh_detector.py              ← 100DOH Tier 2 (optional)
```

---

## 3. Requirements

| Package | Minimum version | Purpose |
|---|---|---|
| Python | 3.9+ | — |
| `opencv-python` | 4.8.0 | Frame I/O, image processing, optical flow |
| `mediapipe` | 0.10.0 | Hand landmarks, Pose, FaceDetection |
| `numpy` | 1.24.0 | Numerical operations |
| `pyyaml` | 6.0 | Config file loading |
| `tqdm` | 4.60 | Per-frame progress bar |

**Optional (Tier 2 — 100DOH):**

| Package | Purpose |
|---|---|
| `torch` ≥ 2.0 | ResNet-101 Faster-RCNN inference |
| `torchvision` ≥ 0.15 | Model utilities |

> **Note:** Without `torch`, EgoLens skips the 100DOH tier and falls back to
> MediaPipe Pose (Tier 3). All other checks remain fully functional.

---

## 4. Installation

### Step 1 — Clone or enter the project

```bash
cd /path/to/
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install core dependencies

```bash
pip install --upgrade pip
pip install opencv-python>=4.8.0 mediapipe>=0.10.0 numpy>=1.24.0 pyyaml tqdm
```

### Step 4 — (Optional) Install PyTorch for 100DOH Tier 2

For CPU-only (recommended — MPS/GPU is too slow for this model):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 5 — (Optional) Download 100DOH weights

100DOH requires the pretrained ResNet-101 checkpoint:

```bash
pip install gdown
gdown --id 1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE \
      -O video_qc/vendors/weights/handobj_100K_ego.pth
```

If the weights file is missing, EgoLens automatically skips Tier 2 and logs:
```
[egolens] 100DOH not available — skipping Tier 2
```

### Step 6 — Verify the install

```bash
python -m egolens --help
```

Expected output:
```
usage: python -m egolens [-h] [--config CONFIG_YAML] [--no-doh] [--quiet] VIDEO [VIDEO ...]

EgoLens video QC pipeline

positional arguments:
  VIDEO                 Path(s) to video file(s) to analyse

options:
  --config, -c          Path to config YAML (default: egolens/config.yaml)
  --no-doh              Skip 100DOH detector (faster)
  --quiet, -q           Suppress per-frame progress bar
```

---

## 5. Quick Start

```bash
# Run on a single video
python -m egolens video.mp4

# Run on multiple videos at once
python -m egolens testdata/passed/*.mp4

# Run on multiple videos with a custom config
python -m egolens testdata/failed/*.mp4 --config egolens/config.yaml

# Skip 100DOH for faster results (3–4× speedup)
python -m egolens video.mp4 --no-doh

# Suppress progress bar (useful in scripts / CI)
python -m egolens video.mp4 --quiet

# Combine flags
python -m egolens batch/*.mp4 --no-doh --quiet --config strict.yaml
```

---

## 6. CLI Reference

```
python -m egolens VIDEO [VIDEO ...] [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `VIDEO` (positional) | — | One or more video file paths. Glob patterns supported via shell: `testdata/**/*.mp4` |
| `--config` / `-c` | `egolens/config.yaml` | Path to YAML config file |
| `--no-doh` | off | Skip 100DOH (Tier 2). Faster but weaker on fists / tool-use / gloves |
| `--quiet` / `-q` | off | Hide the tqdm progress bar |

**Exit code:**
- `0` — all videos passed
- `1` — one or more videos failed, or a file was not found

---

## 7. Sample Output — Passed Video

Running on a standard egocentric training video:

```
  → Analysing ego_view.mp4 …
    Non-stereo 1080×810 (unchanged)
    60.0 fps → stride 20 → 3.0 fps effective  |  ~62 frames
  Frames: 100%|█████████████████████████| 62/62 [00:09<00:00,  6.5fr/s]

══════════════════════════════════════════════════════════════════
  [PASSED] ego_view.mp4
  1080×810  non-stereo
  60.0 fps → stride 20 → 3.0 fps effective  |  62 frames
══════════════════════════════════════════════════════════════════
  ✅ Sharpness         score=100.0  norm=1.000  (62/62 frames pass)
  ✅ Stability         score= 72.3  norm=0.723  (36/62 frames pass)
  ✅ Hand Detection    score= 72.4  norm=0.724  (52/62 frames pass)
       violations: 8.0s–9.0s, 15.0s–17.3s  [total 3.3s]
       sources: hands=27%  none=18%  phantom=48%  pose=6%
  ✅ Face (Privacy)    score= 93.5  norm=0.935  (58/62 frames pass)
       violations: 3.0s–3.3s, 8.3s–8.7s  [total 1.3s]
  ✅ Person Detect.    score= 96.8  norm=0.968  (60/62 frames pass)
  ✅ Hand Speed        score= 44.3  norm=0.443  (48/62 frames pass)
  ✅ Hand Position     score= 59.4  norm=0.594  (41/62 frames pass)
       violations: 7.3s–9.0s, 14.0s–18.3s  [total 7.0s]
  ❌ Hand OOF          score= 75.0  norm=0.750  (44/62 frames pass)
       violations: 7.7s–9.0s, 13.7s–16.0s  [total 6.0s]
  ──────────────────────────────────────────────────────────────────
  ✅ Overall score : 75.3/100  (threshold=70  PASS)
```

**How to read this:**

| Column | Meaning |
|---|---|
| `✅ / ❌` | Per-detector pass/fail (score ≥ gate = ✅) |
| `score=100.0` | Weighted contribution score (0–100) |
| `norm=1.000` | Mean normalised value across all sampled frames (0.0–1.0) |
| `(62/62 frames pass)` | How many sampled frames individually passed the gate |
| `violations: 8.0s–9.0s` | Time segments where the bad condition lasted ≥ max_violation_s |
| `[total 3.3s]` | Total duration of all violation segments |
| `sources: hands=27% …` | For hand detection: % of detections credited to each cascade tier |

---

## 8. Sample Output — Failed Video (Stereo)

Running on a side-by-side stereo VR capture:

```
  → Analysing 3DVideo_2026-04-26-19-17-33-178.mp4 …
    Stereo 3248×1232 → right eye 1624×1232 → scaled 1424×1080
    50.0 fps → stride 17 → 2.9 fps effective  |  ~292 frames
  Frames: 100%|█████████████████████████| 291/291 [01:42<00:00,  2.8fr/s]

══════════════════════════════════════════════════════════════════
  [FAILED] 3DVideo_2026-04-26-19-17-33-178.mp4
  Stereo 3248×1232 → right eye 1624×1232 → scaled 1424×1080
  50.0 fps → stride 17 → 2.9 fps effective  |  291 frames
══════════════════════════════════════════════════════════════════
  ✅ Sharpness         score= 89.3  norm=0.893  (291/291 frames pass)
  ✅ Stability         score= 77.3  norm=0.773  (240/291 frames pass)
  ❌ Hand Detection    score= 12.4  norm=0.124  (39/291 frames pass)
       violations: 0.0s–12.6s, 13.9s–32.6s, 34.3s–39.8s  [total 85.3s]
       sources: hands=0%  none=83%  phantom=13%  pose=4%
  ✅ Face (Privacy)    score=100.0  norm=1.000  (291/291 frames pass)
  ✅ Person Detect.    score= 99.7  norm=0.997  (290/291 frames pass)
  ❌ Hand Speed        score=  5.0  norm=0.050  (29/291 frames pass)
       violations: 0.0s–33.0s, 35.0s–71.7s  [total 88.7s]
  ❌ Hand Position     score= 16.8  norm=0.168  (53/291 frames pass)
       violations: 0.0s–12.6s, 14.6s–32.6s  [total 80.6s]
  ✅ Hand OOF          score= 98.6  norm=0.986  (286/291 frames pass)
  ──────────────────────────────────────────────────────────────────
  ❌ Overall score : 56.9/100  (threshold=70  FAIL)
```

The pipeline automatically detected the 3248×1232 frame as side-by-side stereo
(width ≥ 2 × height), cropped the right eye, downscaled to 1080p, and then
correctly identified that hands are not visible in the right-eye view.

---

## 9. config.yaml — Field-by-Field Guide

The full config lives at `egolens/config.yaml`. Every value has a sensible default;
you only need to edit the sections that matter for your use case.

---

### `video` — Preprocessing

```yaml
video:
  stereo_detection: true
  stereo_eye: right
  stereo_target_height: 1080
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `stereo_detection` | bool | `true` | When `true`, EgoLens automatically detects side-by-side stereo video. Detection rule: `width ≥ 2 × height`. Set to `false` to treat every video as a flat frame. |
| `stereo_eye` | string | `"right"` | Which half of a stereo frame to use. `"right"` = right half of the frame. `"left"` = left half. Egocentric rigs typically place the primary/dominant eye on the **right**. |
| `stereo_target_height` | int | `1080` | After cropping the eye, downscale so the frame height equals this value (pixels). Only applied when the crop is **taller** than this value — if already ≤ target height the frame is used as-is. Non-stereo videos are **never** resized. `1080` = 1080p, `720` = 720p. |

**Example — force left eye and downscale to 720p:**
```yaml
video:
  stereo_detection: true
  stereo_eye: left
  stereo_target_height: 720
```

---

### `sampling` — Analysis Rate

```yaml
sampling:
  fps: 3.0
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `fps` | float | `3.0` | How many frames per second are analysed. EgoLens calculates a stride (`video_fps / sampling_fps`) and skips intermediate frames. Lower values are faster; higher values are more precise for short violations. |

**Tradeoffs:**

| `fps` | Suitable for | Speed |
|---|---|---|
| `1.0` | Long clips, rough pass/fail | Fastest |
| `3.0` | Standard QC (recommended) | Balanced |
| `6.0` | Short clips, precise timestamps | Slower |
| `10.0+` | Near frame-accurate violations | Slowest |

---

### `duration` — Global Violation Window

```yaml
duration:
  default_max_violation_s: 1.0
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `default_max_violation_s` | float | `1.0` | Minimum **continuous** seconds a bad condition must last before it is counted as a violation. Used by any detector whose own `max_violation_s` is set to `null`. Set to `0.0` to flag every single bad frame immediately (no grace period). |

**Why this matters:** A demonstrator's hand briefly leaving frame for half a second
during a natural wrist rotation is not a quality problem. Setting `1.0` second means
only sustained absences are flagged.

---

### `detectors` — Per-Detector Settings

Each detector block has the same four fields:

```yaml
detectors:
  <detector_name>:
    enabled: true          # true/false — set false to completely skip
    gate: 0.50             # 0.0–1.0 — minimum passing normalised score
    max_violation_s: null  # seconds (null = use duration.default_max_violation_s)
    weight: 1.0            # relative contribution to the overall score
    description: "..."     # human label (informational only)
```

---

#### `sharpness`

```yaml
sharpness:
  enabled: true
  gate: 0.05
  max_violation_s: null
  weight: 1.0
  description: "Image must not be blurry (Laplacian variance)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Run sharpness check. Set `false` to skip entirely (e.g. when filming through glass intentionally). |
| `gate` | `0.05` | Very lenient — only the blurriest frames (Laplacian variance near zero) will fail. Raise to `0.30`–`0.50` for stricter focus requirements. |
| `max_violation_s` | `null` | Uses `duration.default_max_violation_s` (1.0 s). |
| `weight` | `1.0` | Equal weight in overall score. |

**Calibration:**
- `norm ≈ 0.0–0.1` = heavily blurred / defocused
- `norm ≈ 0.3–0.5` = mildly soft
- `norm ≈ 0.8–1.0` = sharp

---

#### `stability`

```yaml
stability:
  enabled: true
  gate: 0.70
  max_violation_s: null
  weight: 1.0
  description: "Camera must not shake (Farneback optical flow, inverted)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Run stability check. |
| `gate` | `0.70` | A frame fails if mean optical flow > ~12 px/frame. Lower gate (e.g. `0.50`) tolerates more motion. |
| `max_violation_s` | `null` | Uses global default (1.0 s). |
| `weight` | `1.0` | Equal weight. |

**Calibration (norm = exp(−flow / 35)):**
- `norm ≈ 0.95+` = very stable (tripod / head mount at rest)
- `norm ≈ 0.80–0.95` = walking or slight head movement — normal
- `norm ≈ 0.70` = borderline (gate)
- `norm < 0.70` = shaky — fast head turns, accidental bumps

---

#### `hand_detection`

```yaml
hand_detection:
  enabled: true
  gate: 0.50
  max_violation_s: 1.0
  weight: 2.0
  description: "Hands must be visible (5-tier cascade: MP→DOH→Pose→Skin→Phantom)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Run hand detection. Disable for non-manual tasks (e.g. passive observation videos). |
| `gate` | `0.50` | Tracker confidence must be ≥ 0.5 to count as "hand detected". |
| `max_violation_s` | `1.0` | Hands absent for > 1 second = violation flagged. Use `2.0` to tolerate longer natural pauses. |
| `weight` | `2.0` | **Double weight** — this is the most critical dimension for a hand-demonstration video. |

The **sources** breakdown in the output shows which cascade tier detected the hand:
- `hands` = Tier 1 (MediaPipe Hands) — highest quality landmarks
- `pose` = Tier 3 (MediaPipe Pose wrist) — detected from arm geometry
- `phantom` = Tier 5 — coasted from last known position
- `none` = no detection at all (violation)

---

#### `hand_speed`

```yaml
hand_speed:
  enabled: true
  gate: 0.10
  max_violation_s: 1.0
  weight: 1.0
  description: "Hands must be moving (wrist velocity px/frame)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Run speed check. Disable if the task genuinely requires stationary hands (e.g. precision assembly with pauses). |
| `gate` | `0.10` | Very lenient — only hands with near-zero movement fail. A score of `0.10` corresponds to roughly 8 px/frame. Raise to `0.30` for tasks requiring active motion throughout. |
| `max_violation_s` | `1.0` | Stationary hands for > 1 second = violation. |
| `weight` | `1.0` | Standard weight. |

**Calibration (norm = tanh(speed / 80)):**
- `norm ≈ 0.0–0.10` = stationary or nearly still
- `norm ≈ 0.40–0.60` = moderate, purposeful movement
- `norm ≈ 0.80+` = fast / active demonstration

---

#### `hand_position`

```yaml
hand_position:
  enabled: true
  gate: 0.50
  max_violation_s: 1.0
  weight: 1.5
  description: "Hands must stay inside the workspace bounding box"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Check that hand landmarks stay inside the workspace rectangle. |
| `gate` | `0.50` | At least 50% of hand landmarks must be inside the workspace. Lower this for tasks where hands frequently reach to the edges. |
| `max_violation_s` | `1.0` | Hands outside workspace for > 1 second = violation. |
| `weight` | `1.5` | Slightly elevated — consistent frame composition is important. |

The workspace rectangle is set separately under the **`workspace`** key (see below).

---

#### `hand_out_of_frame`

```yaml
hand_out_of_frame:
  enabled: true
  gate: 0.90
  max_violation_s: 1.0
  weight: 1.5
  description: "Hands must not leave the frame (wrist near edge = bad)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Check whether wrists are dangerously close to any frame edge. |
| `gate` | `0.90` | At least 90% of wrists must be well inside the frame. High because even one wrist near the edge is a problem. |
| `max_violation_s` | `1.0` | Wrist near edge for > 1 second = violation. |
| `weight` | `1.5` | Elevated — an out-of-frame hand provides no training signal. |

The "danger zone" margin is set under the **`frame_edge`** key (see below).

---

#### `face_detection`

```yaml
face_detection:
  enabled: true
  gate: 0.90
  max_violation_s: 1.0
  weight: 1.0
  description: "No face must be visible for privacy (MP FaceMesh)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Run privacy check. Disable only in a controlled, consent-given environment. |
| `gate` | `0.90` | Score must be ≥ 0.90 (i.e. the frame is clean of identifiable faces). The metric returns `1.0` when no face is found and `0.0` when a sharp, identifiable face is present. The high gate means even a brief clear face = fail. |
| `max_violation_s` | `1.0` | A face visible for > 1 second triggers the violation. Set to `0.5` for stricter privacy. |
| `weight` | `1.0` | Standard weight. |

**Important:** The detector uses **eye-region sharpness** to distinguish a truly
identifiable face from a blurry background face or a reflection. Small, blurry,
or very distant faces are tolerated.

---

#### `person_detection`

```yaml
person_detection:
  enabled: true
  gate: 0.50
  max_violation_s: 1.0
  weight: 1.0
  description: "No other person must be visible (MP Pose body detection)"
```

| Field | Value | Meaning |
|---|---|---|
| `enabled` | `true` | Check for a third party entering the frame. |
| `gate` | `0.50` | Score ≥ 0.50 = no person detected. Score < 0.50 = upper body keypoints (shoulders/hips) are visible = another person present. |
| `max_violation_s` | `1.0` | Person visible for > 1 second = violation. |
| `weight` | `1.0` | Standard weight. |

**Note:** In egocentric video the camera wearer's own hands/arms appear, but their
full body does not. This detector only flags when both shoulders (or shoulders +
hips) are visible — indicative of a third party walking into frame.

---

### `workspace` — Hand Position Bounds

```yaml
workspace:
  x_lo: 0.10
  x_hi: 0.90
  y_lo: 0.10
  y_hi: 0.90
```

Defines the valid area of the frame where hands should be, in normalised
coordinates (`0.0` = left/top edge, `1.0` = right/bottom edge).

| Field | Default | Meaning |
|---|---|---|
| `x_lo` | `0.10` | Left boundary — hands must be at least 10% from the left edge |
| `x_hi` | `0.90` | Right boundary — hands must be at most 90% from the left edge |
| `y_lo` | `0.10` | Top boundary — hands must be at least 10% from the top |
| `y_hi` | `0.90` | Bottom boundary — hands must be at most 90% from the top |

**Example — tighter workspace for close-up tasks:**
```yaml
workspace:
  x_lo: 0.20
  x_hi: 0.80
  y_lo: 0.20
  y_hi: 0.85
```

**Example — relaxed workspace for wide-angle rigs:**
```yaml
workspace:
  x_lo: 0.05
  x_hi: 0.95
  y_lo: 0.05
  y_hi: 0.95
```

---

### `frame_edge` — Out-of-Frame Margin

```yaml
frame_edge:
  margin: 0.05
```

| Field | Default | Meaning |
|---|---|---|
| `margin` | `0.05` | Any wrist within this fraction of any frame edge is considered "out of frame". `0.05` = 5% margin on all four sides. For a 1920×1080 frame that is a 96px / 54px strip around the perimeter. |

**Example — stricter margin:**
```yaml
frame_edge:
  margin: 0.10   # 10% — flag hands within 192px of any edge
```

---

### `scoring` — Pass Threshold

```yaml
scoring:
  pass_threshold: 70
```

| Field | Default | Meaning |
|---|---|---|
| `pass_threshold` | `70` | The weighted overall score (0–100) must meet or exceed this value to receive a **PASS** verdict. Scores below this result in **FAIL**. |

**Score formula:**
```
per_detector_score = mean_norm × 100
overall_score = Σ(per_detector_score × weight) / Σ(weight)
```

**Threshold guidance:**

| Threshold | Strictness |
|---|---|
| `50` | Lenient — only grossly poor videos fail |
| `70` | Standard (recommended) |
| `80` | Strict — suitable for final dataset curation |
| `90` | Very strict — near-broadcast quality only |

---

### Complete Example — Strict Production Config

```yaml
# egolens/strict.yaml — for final dataset release
version: "1.0"

video:
  stereo_detection: true
  stereo_eye: right
  stereo_target_height: 1080

sampling:
  fps: 5.0              # more precise violation timestamps

duration:
  default_max_violation_s: 0.5   # half-second grace period only

detectors:
  sharpness:
    enabled: true
    gate: 0.30          # must be reasonably sharp
    weight: 1.0

  stability:
    enabled: true
    gate: 0.80          # must be quite stable
    weight: 1.0

  hand_detection:
    enabled: true
    gate: 0.60
    max_violation_s: 0.5
    weight: 3.0         # critical

  hand_speed:
    enabled: true
    gate: 0.20
    max_violation_s: 0.5
    weight: 1.0

  hand_position:
    enabled: true
    gate: 0.70
    max_violation_s: 0.5
    weight: 2.0

  hand_out_of_frame:
    enabled: true
    gate: 0.95
    max_violation_s: 0.5
    weight: 2.0

  face_detection:
    enabled: true
    gate: 0.95
    max_violation_s: 0.5
    weight: 1.5

  person_detection:
    enabled: true
    gate: 0.70
    max_violation_s: 0.5
    weight: 1.0

workspace:
  x_lo: 0.15
  x_hi: 0.85
  y_lo: 0.15
  y_hi: 0.85

frame_edge:
  margin: 0.08

scoring:
  pass_threshold: 80
```

```bash
python -m egolens video.mp4 --config egolens/strict.yaml
```

---

### Complete Example — Fast Screening Config

Use this to do a quick first-pass triage before committing to full analysis:

```yaml
# egolens/fast.yaml — quick pass/fail only
version: "1.0"

video:
  stereo_detection: true
  stereo_eye: right
  stereo_target_height: 720    # 720p is enough for screening

sampling:
  fps: 1.0                     # 1 frame per second

duration:
  default_max_violation_s: 3.0 # only flag sustained problems

detectors:
  sharpness:
    enabled: true
    gate: 0.05
    weight: 1.0

  stability:
    enabled: false              # skip for speed

  hand_detection:
    enabled: true
    gate: 0.40
    max_violation_s: 3.0
    weight: 2.0

  hand_speed:
    enabled: false              # skip for speed

  hand_position:
    enabled: false              # skip for speed

  hand_out_of_frame:
    enabled: false              # skip for speed

  face_detection:
    enabled: true
    gate: 0.90
    max_violation_s: 2.0
    weight: 1.0

  person_detection:
    enabled: false              # skip for speed

workspace:
  x_lo: 0.05
  x_hi: 0.95
  y_lo: 0.05
  y_hi: 0.95

frame_edge:
  margin: 0.05

scoring:
  pass_threshold: 60
```

```bash
python -m egolens batch/*.mp4 --config egolens/fast.yaml --no-doh --quiet
```

---

## 10. Stereo Video Handling

EgoLens automatically handles side-by-side (SBS) stereo video:

```
Detection rule:  native_width  ≥  2 × native_height
```

**Processing pipeline for a stereo frame:**

```
Raw frame  3248 × 1232
     │
     ▼  Split at midpoint
Left half  1624 × 1232   │   Right half  1624 × 1232
                         │
     stereo_eye: right   ▼
                    1624 × 1232
                         │
     stereo_target_height: 1080  (1232 > 1080, so downscale)
                         │
                         ▼
                    1424 × 1080   ← used for all detectors
```

- **Non-stereo videos** are never resized — they are passed through exactly as decoded.
- If the right-eye crop is already ≤ `stereo_target_height`, no downscale is applied.
- To force left-eye: set `stereo_eye: left` in config.
- To disable stereo detection entirely: set `stereo_detection: false`.

---

## 11. Duration-Based Violations Explained

Every detector uses a `DurationViolationTracker`. A frame is only **flagged** when
the bad condition has been **continuously present** for at least `max_violation_s`
seconds:

```
Frame timeline (3 fps, max_violation_s = 1.0 s):

  t=0.0  GOOD  ──────────────
  t=0.3  GOOD
  t=0.7  BAD   ← run starts (t_start = 0.7)
  t=1.0  BAD     continuous = 0.3 s  < 1.0 s → not flagged yet
  t=1.3  BAD     continuous = 0.6 s  < 1.0 s → not flagged yet
  t=1.7  BAD     continuous = 1.0 s  ≥ 1.0 s → FLAGGED ✓
  t=2.0  GOOD  ← violation segment (0.7 s → 2.0 s) recorded
  t=2.3  GOOD
```

**Violation segments** in the report (`violations: 0.7s–2.0s`) are the
raw start/end timestamps — independent of the flagging threshold.

Setting `max_violation_s: 0.0` flags every bad frame immediately (no tolerance).

---

## 12. Scoring Formula

```
step 1:  norm_i        =  mean normalised score across all sampled frames  [0, 1]
step 2:  det_score_i   =  norm_i × 100                                     [0, 100]
step 3:  overall_score =  Σ(det_score_i × weight_i) / Σ(weight_i)          [0, 100]
step 4:  passed        =  overall_score ≥ pass_threshold
```

Disabled detectors (`enabled: false`) contribute `0` to both the numerator and
denominator — they do not drag the score down.

**Normalisation curves (fixed calibration):**

| Detector | Formula | Notes |
|---|---|---|
| Sharpness | `sigmoid((raw − 27) / 10.5)` | raw = Laplacian variance |
| Stability | `exp(−raw / 35)` | raw = mean optical flow (px/frame) |
| Hand speed | `tanh(raw / 80)` | raw = mean landmark displacement (px/frame) |
| All others | raw value already in [0, 1] | face/person: 1=OK, 0=fail |

---

## 13. Using EgoLens as a Python Library

```python
from egolens.config   import load_config
from egolens.pipeline import EgoLensPipeline
from egolens.scorer   import score
from egolens.reporter import print_report

# 1. Load config (uses egolens/config.yaml by default)
cfg = load_config()                              # or load_config("my_config.yaml")

# 2. Run pipeline
pipeline = EgoLensPipeline(cfg)
result   = pipeline.run("video.mp4")            # verbose=False to suppress progress

# 3. Score
scored = score(result, cfg)
print(f"Overall: {scored.overall_score:.1f}/100  passed={scored.passed}")

# 4. Print full report
print_report(result, scored, pass_threshold=cfg.scoring.pass_threshold)

# 5. Access per-detector results
for name, det in result.detectors.items():
    print(f"{name}: norm={det.mean_norm:.3f}  pass_rate={det.pass_rate:.1%}")

# 6. Access violation segments
hand = result.detectors["hand_detection"]
if hand.violation:
    for start, end in hand.violation.violation_segments:
        print(f"  Hand absent: {start:.1f}s → {end:.1f}s")
```

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'mediapipe'`
```bash
pip install mediapipe
```

### `[egolens] 100DOH not available — skipping Tier 2`
Normal message — it means `torch` is not installed or the weights file is missing.
All other tiers still run. Install PyTorch + download weights (Step 4–5 above) to
enable Tier 2.

### `IOError: Cannot open video: video.mp4`
Check the file path. On macOS/Linux use `file video.mp4` to verify it is a valid
video container.

### Progress bar stuck or very slow
- Use `--no-doh` to skip the heavy Tier 2 detector.
- Lower `sampling.fps` in config (e.g. `1.0`) to analyse fewer frames.
- Disable expensive detectors (`stability`, `person_detection`) in config.

### All videos show `[FAILED]` with low hand detection
Common causes:
1. Gloves — YCbCr skin detector cannot detect gloved hands; MediaPipe Hands
   still works if the gloves are not black. Consider lowering `hand_detection.gate`.
2. Very dark skin + poor lighting — lower `gate` or increase `sampling.fps` to
   give the phantom coasting tier more frames to work with.
3. Stereo video not being split — verify that `stereo_detection: true` and that
   `native_width ≥ 2 × native_height`.

### `hand_position` fails even though hands are visible
The workspace bounds may be too tight. Inspect the bounds:
```yaml
workspace:
  x_lo: 0.05   # loosen from 0.10
  x_hi: 0.95   # loosen from 0.90
  y_lo: 0.05
  y_hi: 0.95
```

### macOS warning: `GL version: 2.1 (Metal)`
Harmless MediaPipe initialisation log. Suppress with:
```bash
python -m egolens video.mp4 2>/dev/null
```
(stderr only — the report still prints to stdout)
