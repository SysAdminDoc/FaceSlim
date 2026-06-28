# Research - FaceSlim

## Executive Summary
FaceSlim is a local Python/PyQt desktop and CLI suite for landmark-based face reshaping, parsing-mask beauty effects, batch rendering, Docker CLI use, and PyInstaller distribution. Its strongest current shape is a privacy-preserving local tool with unusually broad GUI/CLI parity for a small project; the highest-value direction is to harden trust, reproducibility, provenance, testing, and large-media reliability before adding heavier restoration models. Priority opportunities: verified model downloads, automated regression fixtures, observable export failures, reproducible environment setup, C2PA/IPTC disclosure metadata, accessibility metadata, large-media preflight/tiling, provider benchmarking, dependency upgrade strategy, and model provenance UI.

## Product Map
- Core workflows: live webcam/file preview, slider-driven face reshaping, parsing-based beauty edits, batch image/video processing, manifest-driven CLI jobs.
- User personas: privacy-sensitive creators, batch editors processing portraits/video clips, developers wanting a local CLI/Docker renderer, streamers testing real-time virtual camera output.
- Platforms and distribution: Python 3.9+ source, Windows/macOS/Linux claimed, PyInstaller Windows spec, Docker CLI image, local ONNX/MediaPipe model downloads.
- Key integrations and data flows: MediaPipe Face Landmarker -> TPS/ROI warp -> BiSeNet/MODNet ONNX masks -> OpenCV/Pillow export -> optional FFmpeg audio mux -> optional pyvirtualcam preview output.

## Competitive Landscape
- FaceFusion: strong model-zoo, provider, and job-pipeline surface for face manipulation. Learn its explicit model/runtime selection and batch ergonomics; avoid face-swap-first positioning that would dilute FaceSlim's reshape/retouch identity.
- GFPGAN and CodeFormer: strong face restoration baselines with identity/fidelity tradeoffs. Learn optional restoration as a post-stage with a fidelity slider; avoid making restoration mandatory or silently changing identity.
- Real-ESRGAN and Warlock-Studio: strong tiling/upscale/batch patterns for large media and GPU workflows. Learn tile preflight, estimated cost, and multi-model batch orchestration; avoid bloating the base install with every model.
- RetouchML: shows latent-space beautification can preserve smooth geometry better than mesh warps in some cases. Keep as an experimental path only because inversion cost and identity drift conflict with FaceSlim's fast local workflow.
- Facetune, YouCam Perfect, Meitu, Remini, Topaz Photo AI: commercial tools package one-tap presets, face/body controls, enhancer/upscale, and mobile-first UX. Learn guided presets, visual previews, and paid-tier signals; avoid subscription/account/cloud dependencies.
- Adobe Photoshop UXP: best integration path for professional editors. Learn local CLI/service automation from Photoshop actions; avoid building a plugin before the local API/manifest contract is stable.

## Security, Privacy, and Reliability
- Verified: `FaceSlim_v1.py` downloads `face_landmarker.task`, BiSeNet, and MODNet with HTTPS plus minimum byte checks only; no pinned SHA-256, signed manifest, cache versioning, or provenance UI.
- Verified: `ExportThread._mux_audio()` swallows all non-`FileNotFoundError` FFmpeg failures, so audio mux problems can disappear without a user-visible diagnostic.
- Verified: `save_rgb_image()` preserves EXIF/ICC by default, but exports do not embed C2PA/IPTC provenance metadata; the optional visual watermark can be cropped or omitted.
- Verified: large videos/images are accepted without preflight checks for frame count, resolution, expected disk space, codec support, memory pressure, or output writability.
- Verified: current `.venv` Python points at `C:\Users\--\AppData\Local\Programs\Python\Python311\python.exe`, so local verification through repo `.venv` is broken on this machine.
- Missing guardrails: checksum verification, retry/resume for model downloads, dependency audit job, structured log file per render, crash-safe partial output cleanup, export rollback on mux failure.

## Architecture Assessment
- `FaceSlim_v1.py` is a 3,700+ line single module containing dependency checks, downloads, model inference, image effects, GUI, threads, batch workers, and CLI parsing; extracting `models.py`, `pipeline.py`, `exporters.py`, `cli.py`, and `ui/` would make testing and future plugin/API work practical.
- `FaceParsingEngine` and `MattingRefinementEngine` auto-select CUDA/DML/CPU providers, but there is no benchmarked provider selector, diagnostics panel, or CoreML/OpenVINO/TensorRT strategy.
- GUI controls are mostly plain `QLabel`/`QPushButton`/`QSlider` instances with tooltips but no explicit accessible names/descriptions, screen-reader labels, or automated contrast/focus checks.
- No tests are present in the tracked tree; only `--list-presets` gives a cheap smoke path. Add deterministic CLI tests, manifest parser tests, image fixture golden checks, and worker cancellation tests before major model work.
- Documentation is broad and current for features, but model licenses, hashes, cache location, provider diagnostics, C2PA/IPTC disclosure behavior, and known platform limitations need a user-facing section after implementation.

## Rejected Ideas
- Face swap pipeline parity with Deep-Live-Cam/InsightFace: rejected because it conflicts with FaceSlim's retouch/reshape purpose and increases abuse risk.
- Cloud account sync/subscription features copied from mobile editors: rejected because local privacy is a differentiator.
- Mandatory GFPGAN/CodeFormer restore on every export: rejected because it can change identity and adds heavyweight dependencies; keep optional.
- Immediate 3DMM/DECA implementation: rejected for now because `Roadmap_Blocked.md` already records unresolved model/legal distribution decisions.
- Mobile app rewrite: rejected because current architecture is PyQt/OpenCV desktop/CLI; focus first on distribution quality and local plugin/API integration.

## Sources
Repo and OSS:
- https://github.com/SysAdminDoc/FaceSlim
- https://github.com/facefusion/facefusion
- https://github.com/TencentARC/GFPGAN
- https://github.com/sczhou/CodeFormer
- https://github.com/xinntao/Real-ESRGAN
- https://github.com/ju-leon/RetouchML
- https://github.com/Ivan-Ayub97/Warlock-Studio
- https://github.com/yakhyo/face-parsing
- https://github.com/yakhyo/modnet
- https://github.com/sczhou/Awesome-Face-Restoration

Commercial and integration:
- https://www.facetuneapp.com/
- https://www.perfectcorp.com/consumer/apps/ypc
- https://www.meitu.com/en/
- https://remini.ai/
- https://www.topazlabs.com/topaz-photo-ai
- https://developer.adobe.com/photoshop/uxp/

Standards and platform policy:
- https://c2pa.org/specifications/specifications/2.2/index.html
- https://iptc.org/std/photometadata/specification/IPTC-PhotoMetadata
- https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- https://support.google.com/youtube/answer/14328491

Dependencies and advisories:
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- https://github.com/microsoft/onnxruntime/releases
- https://pypi.org/project/onnxruntime/
- https://pypi.org/project/mediapipe/
- https://pypi.org/project/opencv-python/
- https://pypi.org/project/numpy/
- https://devguide.python.org/versions/
- https://osv.dev/
- https://github.com/advisories?query=ecosystem%3Apip

## Open Questions
- Which model licenses and redistribution terms are acceptable for optional restoration/upscale models shipped or downloaded by FaceSlim?
- Should C2PA/IPTC disclosure be opt-out for all exported edits, or tied to the existing optional visual watermark?
