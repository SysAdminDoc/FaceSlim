# Changelog

## FaceSlim v1.20.0 - 2026-06-28

- Added shared media preflight checks for video export, batch jobs, and CLI processing.
- Reported resolution, frame count/duration, output size, memory/time estimates, free disk space, output writability, and MP4 writer availability before long renders start.
- Refused blocked or unsafe jobs before inference and logged preflight failures to `render.log`.

## FaceSlim v1.19.0 - 2026-06-28

- Added accessible names and descriptions across primary GUI controls, sliders, toggles, combo boxes, and export/batch actions.
- Added explicit tab-order metadata for the main workflow.
- Added local accessibility QA tests for control metadata and dark-theme contrast ratios.

## FaceSlim v1.18.0 - 2026-06-28

- Added always-on FaceSlim provenance metadata to image exports.
- Embedded IPTC/XMP DigitalSourceType, edit timestamp, tool version, disclosure-watermark state, and source-preservation state in PNG/JPEG outputs.
- Updated `--strip-metadata` semantics so source metadata can be removed while FaceSlim provenance remains.

## FaceSlim v1.17.0 - 2026-06-28

- Added `tools/bootstrap_dev.py` to create or repair `.venv`, reinstall requirements, and run local verification from one command.
- Added stale `pyvenv.cfg` base-interpreter detection so copied or broken virtual environments are rebuilt automatically.
- Documented the bootstrap command in Quick Start and requirements setup.

## FaceSlim v1.16.0 - 2026-06-28

- Added JSON-lines `render.log` diagnostics for export, batch, CLI, frame-processing, and FFmpeg mux failures.
- Kept rendered video output intact when FFmpeg audio muxing fails and surfaced the warning in GUI status.
- Made CLI processing return a non-zero exit code when any requested media job fails.

## FaceSlim v1.15.0 - 2026-06-28

- Added exact-size and SHA-256 verification for all auto-downloaded model artifacts.
- Added atomic model download replacement with corrupt-cache retry handling.
- Added a local `unittest` regression suite for CLI, manifest, batch, cancellation, and image export paths.

## FaceSlim v1.14.0 - 2026-06-28

- Added an OBS-compatible Virtual Cam preview toggle powered by `pyvirtualcam`.
- Added graceful UI feedback when the local virtual-camera backend is unavailable.

## FaceSlim v1.13.0 - 2026-06-28

- Added a Docker CLI image path with a slim Dockerfile, `.dockerignore`, and reduced runtime requirements.
- Documented headless container usage for farm rendering.

## FaceSlim v1.12.0 - 2026-06-28

- Added a first-launch responsible-use gate for consent, disclosure, and platform-rule expectations.
- Persisted the acknowledgement in Qt settings while leaving CLI mode unchanged.

## FaceSlim v1.11.0 - 2026-06-28

- Added a per-second video timeline scrubber for file playback.
- Added seek handling that reprocesses the selected frame with the current slider state.

## FaceSlim v1.10.0 - 2026-06-28

- Added a batch queue dialog with per-file status, progress, ETA, and single-job cancellation.
- Added worker-side skip handling so queued or current video jobs can be cancelled without aborting the batch.

## FaceSlim v1.9.0 - 2026-06-28

- Added a visual-only Teeth Mask preview overlay for the BiSeNet mouth-interior whitening target.
- Kept the hint overlay out of exported image/video data by storing masks separately from processed frames.

## FaceSlim v1.8.0 - 2026-06-28

- Added expression neutralization with MediaPipe blendshape scores to soften frowns and raised/lowered brows.
- Added GUI and CLI controls for `expression_neutralize`.

## FaceSlim v1.7.0 - 2026-06-27

- Added optional MODNet ONNX matting refinement for ROI warp masks.
- Added GUI and CLI controls for `matting_refine` and documented the MODNet model.

## FaceSlim v1.6.0 - 2026-06-27

- Added selectable BiSeNet ResNet18/ResNet34 ONNX parser models for GUI, CLI, exports, and batch manifests.
- Persisted the parser-model setting and documented the new `--parser-model` option.

## FaceSlim v1.5.0 - 2026-06-27

- Added DirectML-backed ONNX Runtime provider selection on Windows.
- Added under-eye smoothing, hair controls, and procedural makeup overlays across GUI and CLI.
- Added per-face overrides, batch manifests, disclosure watermarks, split/side-by-side video export, and image metadata preservation.
- Added PyInstaller packaging with multiprocessing freeze guards and a compatibility launcher.
