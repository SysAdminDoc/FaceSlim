# Changelog

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
