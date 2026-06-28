# FaceSlim

![Version](https://img.shields.io/badge/version-1.15.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)

> AI-powered face slimming, reshaping, and beautification suite with real-time preview, GPU acceleration, and CLI batch processing. 

## Quick Start

```bash
git clone https://github.com/SysAdminDoc/FaceSlim.git
cd FaceSlim
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
python FaceSlim_v1.py
```

On first launch, FaceSlim downloads the face landmarker (~3.7 MB), the selected BiSeNet parsing model (~50-94 MB), and MODNet (~25 MB) only when matting refinement is enabled. Python dependencies are installed through `requirements.txt`.

## Features

### Face Reshaping (Warp-Based)

| Feature | Description | Method |
|---------|-------------|--------|
| Jaw Slimming | Narrows the jawline | TPS warp toward face center |
| Cheek Slimming | Reduces cheek fullness | TPS warp toward face center |
| Chin Reshape | Lifts and narrows the chin | TPS warp with vertical bias |
| Overall Width | Reduces overall face width | Full jaw contour inward warp |
| Forehead Slim | Narrows the forehead | TPS warp on forehead landmarks |
| Nose Slim | Narrows the nose bridge and tip | Horizontal-only warp toward nose centerline |
| Eye Enlarge | Enlarges eyes from iris center | Radial outward push on eye contour |
| Lip Plump | Plumps lips outward | Directional push (upper=up, lower=down) + radial |
| Expression Neutralize | Softens frowns and raised/lowered brows | MediaPipe blendshape-guided local warp |

### AI Beauty (BiSeNet Parsing-Based)

| Feature | Description | Method |
|---------|-------------|--------|
| Skin Smoothing | Smooths skin while preserving texture | Frequency-separation bilateral filter on skin-only mask |
| Skin Tone Even | Reduces redness and blotchiness | LAB color correction + mean-color blending |
| Teeth Whitening | Brightens teeth naturally | HSV saturation/brightness on mouth interior only |
| Teeth Target Mask | Shows the exact whitening target in preview | Visual-only BiSeNet mouth-interior overlay |
| Eye Sharpen | Sharpens iris and brow detail | Unsharp mask on eye/brow parsing region |
| Lip Color | Boosts lip saturation and warmth | HSV saturation boost on lip parsing region |
| Under-Eye Smooth | Smooths the infraorbital region separately | Landmark-guided under-eye mask + skin smoother |
| Hair Controls | Hue, saturation, and density hints | BiSeNet hair-class masking |
| Makeup Overlays | Blush, lip gloss, and eye shadow | Procedural opacity-controlled masks |

### Pipeline & Performance

| Feature | Description |
|---------|-------------|
| BiSeNet Face Parsing | 19-class pixel-level segmentation via ONNX Runtime |
| Parser Model Selector | Toggle BiSeNet ResNet18 or ResNet34 ONNX masks in Settings/CLI |
| MODNet Matting Refinement | Optional portrait matte edge refinement for ROI warp masks |
| Background Protection | ROI-isolated warp + seamless clone composite |
| Temporal Mask Smoothing | EMA filter prevents parsing mask flicker on video |
| Optical Flow Propagation | Warp displacement propagation between TPS keyframes |
| GPU Acceleration | PyTorch TPS warping on CUDA (auto-detected) |
| DirectML Face Parsing | ONNX Runtime DirectML provider on Windows when available |
| Multi-Face Support | Up to 5 simultaneous faces with per-face caching |
| Per-Face Overrides | CLI/manifest can apply different presets or values per face index |
| Real-Time Preview | Webcam/video with live slider adjustment |
| Video Timeline Scrubber | Per-second seeking with current slider-state preview |
| OBS Virtual Camera | Stream the processed preview to an OBS-compatible virtual webcam |
| A/B Compare | Draggable split-screen before/after overlay |
| Batch Processing | Folder/multi-file image and video processing |
| Batch Manifest | JSON jobs with per-file preset, face overrides, output, watermark, and compare settings |
| Batch Queue Dialog | Per-file status, progress, ETA, and single-job cancellation |
| CLI Mode | Headless with presets and per-param control |
| Docker CLI Image | Headless container build for farm rendering |
| Metadata Preservation | Image exports preserve EXIF/ICC metadata by default |
| Disclosure Watermark | Optional exported media badge declaring AI modification |
| Responsible-Use Gate | First-launch acknowledgement for consent, disclosure, and platform-rule expectations |
| Preset System | 9 built-in + unlimited custom presets (JSON) |
| Before/After GIF | One-click animated comparison export |
| Drag & Drop | Drop images/videos directly onto the window |
| Undo/Redo | 50-level parameter history |
| Settings Persistence | Slider values saved between sessions |

## How It Works

```
┌──────────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────────┐
│ Input Frame  │───>│  MediaPipe    │───>│  TPS Warp     │───>│  Post-Warp   │
│  (RGB)       │    │ 478 Landmarks │    │  (GPU/CPU)    │    │  Effects     │
│              │    │  + One-Euro   │    │  ROI-isolated │    │              │
│              │    │  filtering    │    │  + Seamless   │    │  BiSeNet     │
│              │    │               │    │  Clone        │    │  Parsing     │
└──────────────┘    └───────────────┘    └───────────────┘    └──────┬───────┘
                    ┌───────────────┐    ┌───────────────┐           │
                    │  Output Frame │<───│  Skin Smooth  │<──────────┘
                    │  (RGB)        │    │  Tone Even    │
                    │               │    │  Teeth Whiten │
                    │               │    │  Eye Sharpen  │
                    │               │    │  Lip Color    │
                    └───────────────┘    └───────────────┘

Video Mode Only:
┌──────────────────────────────────────────────┐
│  Optical Flow Propagator                     │
│  Keyframe (every 4 frames) → full TPS solve  │
│  Interim frames → Farneback flow warp of     │
│  cached displacement field                   │
└──────────────────────────────────────────────┘
```

## Usage

### GUI Mode

```bash
python FaceSlim_v1.py
```

The interface has three tabs: **Reshape** (sliders for all effects plus preview overlays), **Presets** (built-in/custom preset management), and **Export** (video export, screenshots, batch, GIF). File videos expose a per-second timeline scrubber under the preview.

### CLI Mode

```bash
# Single image with preset
python FaceSlim_v1.py --input photo.jpg --preset Moderate

# AI beauty only
python FaceSlim_v1.py --input photo.jpg --preset Beauty

# Full glamour (reshaping + beauty)
python FaceSlim_v1.py --input photo.jpg --preset Glamour

# Custom parameters
python FaceSlim_v1.py --input video.mp4 --jaw 50 --eye-enlarge 30 --lip-plump 20 --skin-smooth 40

# Batch folder processing
python FaceSlim_v1.py --input ./photos/ --output ./results/ --preset Strong

# GUI batch processing
# Export tab -> Select Files for Batch / Process Entire Folder / Run Manifest opens the queue dialog

# Multi-face processing
python FaceSlim_v1.py --input group.jpg --faces 3 --preset "Full Sculpt"

# Per-face overrides
python FaceSlim_v1.py --input group.jpg --faces 3 --face-preset 1=Subtle --face-param 2:jaw=45

# Batch manifest with per-file settings
python FaceSlim_v1.py --manifest batch.json --output ./results/

# Split-screen video export with disclosure watermark
python FaceSlim_v1.py --input video.mp4 --preset Glamour --video-compare split --watermark

# Higher-quality parser model
python FaceSlim_v1.py --input photo.jpg --parser-model bisenet_resnet34 --skin-smooth 35

# Cleaner face/background boundary on strong warps
python FaceSlim_v1.py --input photo.jpg --jaw 45 --matting-refine 70

# Blendshape-guided expression softening
python FaceSlim_v1.py --input photo.jpg --expression-neutralize 65

# List available presets
python FaceSlim_v1.py --list-presets
```

### CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--input`, `-i` | path(s) | Input image(s), video(s), or folder(s) |
| `--output`, `-o` | path | Output directory (default: `faceslim_output/`) |
| `--preset`, `-p` | string | Named preset (built-in or custom) |
| `--jaw` | 0-100 | Jawline slimming |
| `--cheeks` | 0-100 | Cheek slimming |
| `--chin` | 0-100 | Chin reshaping |
| `--face-width` | 0-100 | Face width reduction |
| `--forehead` | 0-100 | Forehead narrowing |
| `--nose` | 0-100 | Nose slimming |
| `--eye-enlarge` | 0-100 | Eye enlargement |
| `--lip-plump` | 0-100 | Lip plumping |
| `--expression-neutralize` | 0-100 | Blendshape-guided frown and brow dampening |
| `--skin-smooth` | 0-100 | AI skin smoothing |
| `--skin-tone-even` | 0-100 | AI skin tone evening |
| `--teeth-whiten` | 0-100 | AI teeth whitening |
| `--eye-sharpen` | 0-100 | AI eye sharpening |
| `--lip-color` | 0-100 | AI lip color enhancement |
| `--under-eye` | 0-100 | Dedicated under-eye smoothing |
| `--hair-hue` | 0-100 | Hair hue shift |
| `--hair-saturation` | 0-100 | Hair saturation boost |
| `--hair-density` | 0-100 | Hair density hint |
| `--blush` | 0-100 | Procedural blush overlay |
| `--lip-gloss` | 0-100 | Lip gloss overlay |
| `--eye-shadow` | 0-100 | Eye shadow overlay |
| `--smoothing` | 10-100 | Warp field smoothness |
| `--temporal` | 0-100 | Temporal landmark smoothing |
| `--bg-protect` | 0-100 | Background protection |
| `--matting-refine` | 0-100 | MODNet mask edge refinement |
| `--faces` | 1-5 | Max faces to process |
| `--face-preset` | `FACE=PRESET` | Apply a preset to one face index |
| `--face-param` | `FACE:key=value` | Override one parameter for one face index |
| `--manifest` | path | JSON batch manifest |
| `--watermark` | flag | Add AI modification disclosure watermark |
| `--strip-metadata` | flag | Do not preserve image EXIF/ICC metadata |
| `--video-compare` | `none`, `split`, `side_by_side` | Export processed, split-screen, or side-by-side video |
| `--parser-model` | `bisenet_resnet18`, `bisenet_resnet34` | Face parsing model for beauty masks |
| `--list-presets` | flag | List all available presets |

### Batch Manifest

```json
{
  "preset": "Moderate",
  "parser_model": "bisenet_resnet34",
  "faces": 3,
  "watermark": true,
  "video_compare": "split",
  "files": [
    {
      "input": "photos/group.jpg",
      "params": {"skin_smooth": 35},
      "face_params": {"2": {"jaw": 45, "cheeks": 20}}
    },
    {
      "input": "clips/demo.mp4",
      "preset": "Glamour",
      "output": "results/demo_split.mp4"
    }
  ]
}
```

### Built-In Presets

| Preset | Focus | Key Parameters |
|--------|-------|----------------|
| Subtle | Light face slimming | jaw=15, cheeks=10 |
| Moderate | Balanced slimming | jaw=35, cheeks=25, chin=15 |
| Strong | Aggressive slimming | jaw=60, cheeks=45, chin=30 |
| V-Shape | Jaw-focused sculpting | jaw=70, chin=40 |
| Oval | Rounded face shape | jaw=40, cheeks=35, face_width=30 |
| Slim Nose | Nose only | nose=50 |
| Full Sculpt | All reshaping combined | jaw=50, cheeks=40, nose=30 |
| Beauty | AI beautification only | skin_smooth=40, teeth=20, eyes=30, lips=20 |
| Glamour | Full reshaping + beautification | All effects combined |

## Configuration

Custom presets are stored as JSON in:

| OS | Location |
|----|----------|
| Windows | `%APPDATA%\.faceslim\presets\` |
| macOS/Linux | `~/.faceslim/presets/` |

Slider values persist between sessions via Qt settings. Crash logs are written to `crash.log` in the application directory.
The first GUI launch shows a responsible-use acknowledgement and stores it in Qt settings after acceptance.

## Models (Auto-Downloaded)

| Model | Expected size | SHA-256 prefix | Source | Purpose |
|-------|---------------|---------------|--------|---------|
| `face_landmarker.task` | 3,758,596 bytes | `64184e229b26` | Google MediaPipe | 478-point face landmarks |
| `bisenet_face_parsing.onnx` | 53,205,364 bytes | `0d9bd318e469` | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) | Fast 19-class face segmentation (CelebAMask-HQ) |
| `bisenet_resnet34.onnx` | 93,632,554 bytes | `5b805bba7b56` | [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) | Higher-quality 19-class face segmentation (CelebAMask-HQ) |
| `modnet_photographic.onnx` | 25,969,398 bytes | `5069a5e306b9` | [yakhyo/modnet](https://github.com/yakhyo/modnet) | Optional portrait matte for ROI edge refinement |

The face landmarker and selected parser model are downloaded automatically on first use. Each cached or downloaded model must match the versioned in-app manifest by exact byte size and SHA-256 before it is used; corrupt caches are deleted and downloaded again through an atomic temporary file. MODNet downloads only when matting refinement is enabled. If the selected BiSeNet model fails to download or verify, the app falls back to landmark-based polygon masking (reshaping still works, AI beauty effects are disabled).

## Requirements

- **Python 3.9+**
- **Install:** `python -m pip install -r requirements.txt`
- **Optional:** PyTorch + CUDA for GPU acceleration (auto-detected)
- **Optional:** FFmpeg for audio muxing on video exports
- **Optional:** OBS Virtual Camera driver for the Virtual Cam preview output

## Build

```bash
python -m pip install -r requirements.txt
pyinstaller FaceSlim.spec --noconfirm --clean
```

Run the local regression suite with:

```bash
python -m unittest discover -s tests
```

The Windows executable is emitted at `dist/FaceSlim.exe`. The spec includes a multiprocessing freeze guard to prevent frozen MediaPipe/OpenCV worker relaunch loops.

## Docker CLI

```bash
docker build -t faceslim-cli .
docker run --rm -v "%cd%:/data" faceslim-cli --input /data/photo.jpg --output /data/faceslim_output --preset Moderate
```

The container uses `requirements-docker.txt`, `opencv-python-headless`, and `QT_QPA_PLATFORM=offscreen` for CLI-only rendering.

## Supported Formats

| Type | Extensions |
|------|------------|
| Images | `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp` |
| Videos | `.mp4` `.avi` `.mov` `.mkv` `.webm` `.wmv` `.flv` `.m4v` |

## FAQ / Troubleshooting

**BiSeNet model fails to download**
The app falls back to landmark-based masking automatically. AI beauty effects (skin smooth, teeth whiten, etc.) require a BiSeNet model. You can manually download `resnet18.onnx` or `resnet34.onnx` from the [face-parsing releases](https://github.com/yakhyo/face-parsing/releases) and place it in the app directory as `bisenet_face_parsing.onnx` or `bisenet_resnet34.onnx`.

**Low FPS in real-time preview**
Use the Preview Scale dropdown (75% or 50%) in the Quality group. Install PyTorch with CUDA for GPU acceleration. Optical flow propagation kicks in automatically for video, computing full TPS only every 4th frame.

**Warp affects the background**
Increase the Background Protection slider. At 70+ (default), warping is confined to the face ROI and blended back with seamless clone.

**Face/background boundary looks soft on a strong warp**
Increase the Matting Refine slider. FaceSlim downloads MODNet on first use and multiplies the ROI blend mask by a portrait matte to reduce background bleed.

**Video export has no audio**
Install FFmpeg and make sure it's on your PATH. FaceSlim automatically muxes audio from the original file when FFmpeg is available.

**Faces not detected**
Ensure the face is reasonably front-facing and well-lit. MediaPipe requires a minimum face size — try loading a higher resolution source. For multi-face scenes, increase Max Faces (up to 5).

**Crash on startup**
Check `crash.log` in the application directory. Common causes: incompatible Python version (<3.9), missing system libraries for PyQt5 on Linux (install `libxcb-xinerama0`), or corrupted model files (delete `.task`/`.onnx` files and relaunch to re-download).

## License

MIT License. See [LICENSE](LICENSE) for details. Issues and PRs welcome.
