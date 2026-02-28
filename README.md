# FaceSlim

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-0078D4)
![Python](https://img.shields.io/badge/Python-3.9--3.12-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-active-success)

> AI-powered real-time face slimming with MediaPipe 478-landmark detection, smooth RBF warping, and temporal stabilization. Webcam preview, video file processing, A/B comparison, and one-click export.

![Screenshot](screenshot.png)
<!-- Add your own screenshot.png to the repo root -->

## Installation

```bash
git clone https://github.com/SysAdminDoc/FaceSlim.git
cd FaceSlim
python FaceSlim.py  # Auto-installs dependencies and downloads model on first run
```

Fully turnkey — the script auto-bootstraps PyQt5, OpenCV, MediaPipe, NumPy, and SciPy on first launch, then downloads the 3.7 MB face landmarker model from Google's servers. No manual setup required.

## Features

| Feature | Description | Default |
|---------|-------------|---------|
| Jaw Slimming | Narrows the jawline contour using 9 bilateral landmark pairs | 0% (0–100) |
| Cheek Slimming | Reduces cheek fullness via 12 bilateral landmark pairs | 0% (0–100) |
| Chin Reshape | Lifts and narrows the chin region across 15 landmarks | 0% (0–100) |
| Overall Width | Reduces total face width along the full jaw contour | 0% (0–100) |
| Warp Smoothing | Controls RBF displacement field smoothness | 50% (10–100) |
| A/B Compare | Split-screen original vs slimmed with labeled divider | Off |
| Face Landmarks | Color-coded overlay of jaw, cheek, and chin landmark groups | Off |
| Quick Presets | Subtle / Moderate / Strong / Reset one-click presets | — |
| Video Export | Frame-by-frame processing with progress bar and audio mux | — |
| Settings Persistence | Remembers all slider positions between sessions via QSettings | Enabled |
| Auto-Bootstrap | Installs all Python dependencies automatically on first run | Enabled |
| Auto-Model Download | Downloads MediaPipe face landmarker model if not present | Enabled |
| Crash Logging | Writes full traceback to `crash.log` on unhandled exceptions | Enabled |
| Dark Theme | Catppuccin Mocha dark UI with accent-colored controls | Enabled |
| FPS Counter | Live color-coded FPS display (green/yellow/red thresholds) | Enabled |

## Usage

### Webcam Mode

Click **Webcam** to start real-time face slimming with your default camera. Adjust sliders and see changes applied live.

### Video File Mode

Click **Load Video** to open MP4, AVI, MOV, MKV, WebM, or WMV files. The video loops continuously with face slimming applied in real-time. This also enables the **Export Video** button.

### A/B Comparison

Toggle **A/B Compare** to see a split-screen view — original on the left half, slimmed on the right, separated by a labeled blue divider line.

### Presets

| Preset | Jaw | Cheeks | Chin | Width |
|--------|-----|--------|------|-------|
| Subtle | 15% | 10% | 5% | 10% |
| Moderate | 35% | 25% | 15% | 20% |
| Strong | 60% | 45% | 30% | 35% |
| Reset | 0% | 0% | 0% | 0% |

### Exporting

1. Load a video file
2. Adjust sliders to desired effect
3. Click **Export Video** and choose output path
4. Processing runs frame-by-frame with a progress bar
5. Audio is automatically preserved if FFmpeg is available on PATH

## How It Works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Video Input      │────>│  MediaPipe        │────>│  RBF Warp         │────>│  Display          │
│                  │     │  FaceLandmarker   │     │  Engine           │     │  (PyQt5)          │
│  cv2.VideoCapture│     │  478 3D landmarks │     │                  │     │                  │
│  webcam or file  │     │  + One-Euro       │     │  TPS interpolation│     │  A/B split view  │
│                  │     │  temporal filter  │     │  cv2.remap cubic  │     │  FPS counter     │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

**Pipeline per frame:**

1. **Landmark Detection** — MediaPipe Face Landmarker (Tasks API, `face_landmarker.task` model) detects 478 3D face landmarks
2. **Temporal Smoothing** — One-Euro adaptive filter stabilizes landmarks across frames, eliminating inter-frame jitter via adaptive cutoff frequency
3. **Displacement Computation** — Jaw (9 pts), cheek (12 pts), chin (15 pts), and contour landmarks are shifted toward face center proportional to slider values. Nose bridge and forehead landmarks are anchored. 8 edge anchors prevent border distortion
4. **RBF Interpolation** — `scipy.interpolate.RBFInterpolator` with thin-plate spline kernel creates a smooth continuous displacement field from sparse control points, computed at 1/4 resolution then bicubic-upscaled
5. **Gaussian Smoothing** — 7x7 Gaussian blur on the displacement field for artifact-free transitions
6. **Remap** — `cv2.remap` with `INTER_CUBIC` interpolation and `BORDER_REFLECT_101` applies the final warp

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.9 – 3.12 (MediaPipe does not ship 3.13 wheels) |
| OS | Windows 10/11, macOS, Linux |
| GPU | Not required — CPU inference via XNNPACK. NVIDIA GPU improves OpenCV performance |
| FFmpeg | Optional — required only for audio preservation during video export |
| Internet | Required on first run only to download the 3.7 MB face landmarker model |
| Admin | Not required |

## Configuration

All settings are persisted automatically via `QSettings` under the `FaceSlim/FaceSlim` key:

| Setting | Registry/Config Key | Type |
|---------|-------------------|------|
| Jaw Slimming | `s/jaw` | int 0–100 |
| Cheek Slimming | `s/cheeks` | int 0–100 |
| Chin Reshape | `s/chin` | int 0–100 |
| Overall Width | `s/face_width` | int 0–100 |
| Warp Smoothing | `s/smooth` | int 10–100 |

On Windows these are stored in the registry under `HKCU\Software\FaceSlim`. On macOS/Linux they use the platform-native QSettings backend.

## File Structure

```
FaceSlim/
├── FaceSlim.py              # Main application (single-file, turnkey)
├── face_landmarker.task     # MediaPipe model (auto-downloaded on first run)
├── crash.log                # Created on unhandled exceptions
├── README.md
└── LICENSE
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Failed to open video source" | Webcam in use by another app, or video file uses unsupported codec. Close other camera apps or try a different file |
| Low FPS / sluggish preview | RBF computation is CPU-bound. Reduce source resolution or close GPU-intensive apps |
| No audio in exported video | Install [FFmpeg](https://ffmpeg.org/download.html) and ensure it's on your PATH. Without FFmpeg, video exports silently without audio |
| MediaPipe import errors on Python 3.13 | Downgrade to Python 3.12. MediaPipe doesn't ship 3.13 wheels yet |
| Model download fails | Download manually from [Google Storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) and place `face_landmarker.task` next to `FaceSlim.py` |
| `crash.log` created on launch | Open `crash.log` for the full traceback. Most common cause is a missing or corrupt model file — delete `face_landmarker.task` and relaunch to re-download |

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Face Detection | MediaPipe Face Landmarker 0.10.32 (Apache 2.0) | 478 3D landmark detection via Tasks API |
| Warping | SciPy RBFInterpolator | Thin-plate spline smooth displacement field |
| Video I/O | OpenCV | Capture, remap with bicubic interpolation, VideoWriter |
| Temporal Filter | One-Euro Filter | Adaptive low-pass filter on landmark positions |
| GUI | PyQt5 + Fusion style | Desktop interface with Catppuccin Mocha dark theme |
| Audio Mux | FFmpeg (optional) | Copies audio track from source to exported video |

## What It Does and Doesn't Do

**Does:**
- Real-time face landmark detection and smooth warping
- Adjustable jaw, cheek, chin, and overall width slimming
- Temporal stabilization to prevent frame-to-frame jitter
- Export processed video with optional audio preservation
- Persist settings between sessions

**Doesn't:**
- Modify skin texture, color, or lighting
- Apply AI face restoration or enhancement (no GFPGAN/CodeFormer)
- Process multiple faces simultaneously (single face only)
- Provide GPU-accelerated warping (CPU RBF interpolation only in v0.1.0)
- Work offline on first run (needs internet to download model once)

## License

MIT License. See `LICENSE` for details.

Issues and PRs welcome.
