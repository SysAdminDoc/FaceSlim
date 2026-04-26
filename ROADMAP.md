# ROADMAP

Backlog for FaceSlim. Keep the MediaPipe + BiSeNet backbone; expand model options, video export
quality, and batch pipelines.

## Planned Features

### Models
- **MediaPipe Face Mesh V2 upgrade path** when the next landmarker revision ships (smaller + more
  stable landmarks). Keep v1 as fallback.
- **Alternative face-parsing models** — swap BiSeNet for `face-parsing.PyTorch` ResNet-50 or the
  newer FaRL segmenter, toggleable in Settings.
- **3DMM-based reshaping option** — add DECA/FLAME inference as an experimental backend for
  geometry-aware slimming that respects pose.
- **Matting refinement** — run MODNet or RVM on the face ROI before warp to get a cleaner
  background boundary than seamless-clone can deliver alone.
- **ONNX Runtime DirectML provider** on Windows for GPU accel without CUDA.

### Editing features
- **Expression neutralization slider** — blendshape-driven dampening of frowns / raised brows.
- **Teeth alignment hint mask** (visual only) to show what teeth-whiten is actually targeting.
- **Under-eye smoothing** — dedicated slider separate from skin-smooth, scoped to the
  infraorbital parsing region.
- **Hair-region controls** — hue shift, saturation, density hint using BiSeNet hair class.
- **Makeup overlay layer** — procedural blush, lip gloss, eye shadow with opacity sliders.

### Pipeline
- **Multi-face presets** — apply a different preset per face index when multiple subjects are
  detected, not a global setting.
- **Batch manifest file** — JSON describing per-file preset + per-face overrides for reproducible
  bulk runs.
- **Queue dialog** with progress per file, ETA, cancel-one-without-aborting-batch.
- **Timeline in video mode** — scrubber with per-second preview of the current slider state.
- **Before/after split-screen video export** (side-by-side or slider-reveal).

### Safety
- **Deepfake disclosure watermark option** — opt-in small corner badge on exported video declaring
  AI modification.
- **Source-metadata preservation** — carry EXIF/XMP forward on image output so provenance isn't
  stripped.
- **Consent/ToS gate** on first launch about responsible use.

### Distribution
- **PyInstaller exe with `multiprocessing.freeze_support()` guard** (critical: MediaPipe uses
  multiprocessing on some backends; frozen launch will fork-bomb without it).
- **macOS `.app` build** via py2app with signed notarized bundle.
- **Docker/CLI-only image** for headless farm rendering.

## Competitive Research

- **BeautyCam / Meitu / YouCam Perfect** — mobile apps with aggressive retouch presets and AR
  makeup. Feature cues: one-tap presets, skin-tone sliders, AR makeup.
- **Remini / Topaz Photo AI** — upscale + face-enhance pipelines. Borrow their superres stage as
  an optional post step.
- **Open-source `face-parsing.PyTorch` and `DECA`** — primary technical references for upgrades
  noted above.
- **Avatarify / Deep-Live-Cam** — real-time face manipulation frameworks; cues for low-latency
  preview, but out of scope (FaceSlim is reshape, not swap).

## Nice-to-Haves

- **OBS virtual-camera output** — stream the processed feed as a virtual webcam.
- **Preset marketplace** — signed JSON presets with thumbnails, community upload.
- **Photoshop plugin** via UXP that calls a local FaceSlim CLI for automation inside Photoshop
  actions.
- **Automatic preset suggestion** based on face shape analysis (oval / round / square / heart).
- **Undo stack export** as a recipe file — replay an edit on a different photo.
- **Timeline keyframes** for video — ramp sliders over time.

## Open-Source Research (Round 2)

### Related OSS Projects
- **RetouchML** — https://github.com/ju-leon/RetouchML — StyleGAN2 latent-space beautification; face detected, normalized, then gradient-ascended toward a "prettier" attractiveness classifier.
- **photo-enhancer (nuwandda)** — https://github.com/nuwandda/photo-enhancer — Wraps GFPGAN (faces) + RealESRGAN (background) in one pipeline; good reference for split face/bg processing.
- **FaceEnhancementAndMakeup** — https://github.com/ZainabZaman/FaceEnhancementAndMakeup — DLIB landmarks for makeup + CodeFormer for face restoration.
- **facefusion** — https://github.com/facefusion/facefusion — Industry-scale face manipulation platform; model zoo and pipeline graph are the takeaway.
- **Awesome-Face-Restoration** — https://github.com/sczhou/Awesome-Face-Restoration — Curated index of face restoration papers/weights; treat as the SOTA tracker.
- **GFPGAN** — https://github.com/TencentARC/GFPGAN — Canonical face-perfector GAN; still the baseline for skin-smooth + detail-preserve.
- **CodeFormer** — https://github.com/sczhou/CodeFormer — Robust face restoration under heavy degradation; better than GFPGAN on blurry/low-res input.

### Features to Borrow
- Latent-space slimming slider from `RetouchML` — move along the StyleGAN2 "face width" direction instead of mesh-warping; preserves identity better than liquify.
- Two-stage face/bg pipeline from `nuwandda/photo-enhancer` — segment face, enhance separately, alpha-composite back. Avoids background smoothing artifacts.
- DLIB 68-point landmark gating from `FaceEnhancementAndMakeup` — run reshape only inside the face polygon; skip ears/hair.
- Model-zoo selector like `facefusion` — let users pick between GFPGAN / CodeFormer / RestoreFormer at runtime.
- CodeFormer's "fidelity weight" slider — user dial between strict identity preservation and aggressive restoration.
- Batch CLI parity (`facefusion` style): every GUI operation also exposed as a CLI flag for headless runs.

### Patterns & Architectures Worth Studying
- **StyleGAN2 inversion + direction editing** (`RetouchML`): invert face → edit latent → regenerate. Cleaner for geometry edits than landmark warp.
- **ONNX Runtime model-swap** (`facefusion`): ship models as ONNX, let users download alternates. Avoids PyTorch install footprint.
- **Face-parsing mask composite**: BiSeNet face-parsing → per-region weight maps → blend restored face back into original pixels. Used by both `CodeFormer` and `GFPGAN` demos; prevents halo artifacts.
- **Tile-based inference for >4K images** (`RealESRGAN`): split into overlapping tiles, process, seam-blend. Needed if users drop in DSLR RAWs.
