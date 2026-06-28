# Changelog

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
