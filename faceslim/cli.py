#!/usr/bin/env python3
"""Command-line entry point for FaceSlim."""

import argparse
import json
import math
import os
import sys
import time
import traceback

import cv2
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from .i18n import set_locale, tr
from . import runtime
from .runtime import VERSION, log_render_event
from .models import *
from .pipeline import *
from .ui import DARK_STYLE, FaceSlimApp

def cli_process(args):
    """Headless command-line processing."""
    if not ensure_model():
        print(f"ERROR: Model not available. Download from:\n  {MODEL_URL}")
        sys.exit(1)
    parser_model = parser_model_key(args.parser_model)
    onnx_provider = onnx_provider_key(getattr(args, "onnx_provider", DEFAULT_ONNX_PROVIDER))
    ensure_parsing_model(parser_model)  # Non-fatal - falls back to landmark mask
    overrides = {k: getattr(args, k) for k in CLI_PARAM_KEYS
                 if hasattr(args, k) and getattr(args, k) is not None}
    if getattr(args, "post_stage_model", None):
        overrides["post_stage_model"] = args.post_stage_model
    try:
        params = params_from_preset_and_overrides(args.preset, overrides)
        face_overrides = parse_face_overrides(args.face_preset, args.face_param)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if face_overrides:
        params["face_params"] = face_overrides

    max_faces = args.faces or 1

    if args.manifest:
        jobs = load_batch_manifest(args.manifest, args.output, parser_model, onnx_provider)
        inputs = [job["input"] for job in jobs]
    else:
        inputs = media_files_from_paths(args.input)
        jobs = None

    if not inputs:
        print("No input files found"); sys.exit(1)

    output_dir = args.output or os.path.join(os.path.dirname(inputs[0]), 'faceslim_output')
    os.makedirs(output_dir, exist_ok=True)
    if jobs is None:
        jobs = jobs_from_files(inputs, output_dir, params, max_faces, args.watermark,
                               not args.strip_metadata, args.video_compare, parser_model,
                               onnx_provider)

    print(f"\nFaceSlim v{VERSION} - CLI Mode")
    print(f"  GPU: {'ON (' + GPU_NAME + ')' if USE_GPU else 'OFF'}")
    print(f"  ONNX Provider: {onnx_provider_label(onnx_provider)}")
    print(f"  Post-stage: {post_stage_model_label(params.get('post_stage_model'))}")
    print(f"  Files: {len(inputs)}")
    print(f"  Params: {json.dumps({k: v for k, v in params.items() if v != 0}, indent=2)}")
    print(f"  Output: {output_dir}\n")

    processed = failed = 0
    for i, job in enumerate(jobs):
        filepath = job["input"]
        job_params = job.get("params", params)
        job_max_faces = job.get("max_faces", max_faces)
        watermark = job.get("watermark", args.watermark)
        preserve_metadata = job.get("preserve_metadata", not args.strip_metadata)
        compare_mode = job.get("compare_mode", args.video_compare)
        job_parser_model = job.get("parser_model", parser_model)
        job_onnx_provider = job.get("onnx_provider", onnx_provider)
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        print(f"  [{i+1}/{len(jobs)}] {fname}")
        eng = None
        cap = None
        writer = None

        try:
            if ext in IMAGE_EXTS:
                out_path = media_job_output_path(job, output_dir, filepath, ".png")
                preflight = preflight_media_job(filepath, out_path, compare_mode, job_params)
                print(f"    {format_preflight_summary(preflight)}")
                if not preflight["ok"]:
                    msg = preflight_failure_message(preflight)
                    print(f"    FAIL preflight: {msg}")
                    log_render_event("cli_preflight_failed", msg, {
                        "input": filepath,
                        "output": out_path,
                        "job_index": i,
                        "preflight": preflight,
                    })
                    failed += 1
                    continue
                eng = FaceWarpEngine('image', job_max_faces, job_parser_model, job_onnx_provider)
                img = cv2.imread(filepath)
                if img is None: raise ValueError("Cannot read image")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result, faces = eng.warp_single_image(rgb, job_params)
                result = apply_disclosure_watermark(result, watermark)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                save_rgb_image(out_path, result, filepath, preserve_metadata, watermark)
                eng.close(); eng = None
                print(f"    OK ({len(faces)} face{'s' if len(faces) != 1 else ''})")
                processed += 1

            elif ext in VIDEO_EXTS:
                out_path = media_job_output_path(job, output_dir, filepath, ".mp4")
                preflight = preflight_media_job(filepath, out_path, compare_mode, job_params)
                print(f"    {format_preflight_summary(preflight)}")
                if not preflight["ok"]:
                    msg = preflight_failure_message(preflight)
                    print(f"    FAIL preflight: {msg}")
                    log_render_event("cli_preflight_failed", msg, {
                        "input": filepath,
                        "output": out_path,
                        "job_index": i,
                        "preflight": preflight,
                    })
                    failed += 1
                    continue
                eng = FaceWarpEngine('video', job_max_faces, job_parser_model, job_onnx_provider)
                eng.grid_scale = 6
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened(): raise ValueError("Cannot open video")
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or not math.isfinite(fps) or fps <= 0:
                    fps = 30
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                out_w, out_h = video_output_size(w, h, compare_mode, job_params)
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
                if not writer.isOpened():
                    raise ValueError(f"Cannot open output writer: {out_path}")
                has_fx = has_effective_processing(job_params)
                t0 = time.time()
                fi = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = eng.detect(rgb)
                    proc = eng.warp(rgb, faces, job_params) if has_fx else rgb
                    proc = compose_compare_frame(rgb, proc, compare_mode)
                    proc = apply_disclosure_watermark(proc, watermark)
                    writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                    fi += 1
                    if total > 0 and fi % 30 == 0:
                        eta = ((time.time() - t0) / fi) * (total - fi)
                        print(f"\r  [{i+1}/{len(jobs)}] {fname}... {fi}/{total} frames (ETA: {int(eta)}s)", end='', flush=True)
                cap.release(); cap = None
                writer.release(); writer = None
                eng.close(); eng = None
                elapsed = time.time() - t0
                print(f"\r  [{i+1}/{len(jobs)}] {fname}... OK ({fi} frames, {elapsed:.1f}s)")
                processed += 1
            else:
                print(f"    SKIP (unsupported)")
                log_render_event("cli_unsupported_media", "Unsupported file type", {
                    "input": filepath,
                    "job_index": i,
                })
                failed += 1
        except Exception as e:
            print(f"FAIL ({e})")
            log_render_event("cli_job_failed", str(e), {
                "input": filepath,
                "job_index": i,
                "output_dir": output_dir,
            }, traceback.format_exc())
            failed += 1
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if eng is not None:
                eng.close()

    print(f"\nDone: {processed} processed, {failed} failed")
    print(f"Output: {output_dir}")
    if failed:
        print(f"Render diagnostics: {runtime.RENDER_LOG_PATH}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        prog='FaceSlim',
        description=f"{tr('FaceSlim')} v{VERSION} - {tr('AI Face Slimming & Reshaping Suite')}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python FaceSlim.py                                    # Launch GUI
  python FaceSlim.py --input video.mp4 --preset Moderate
  python FaceSlim.py --input photo.jpg --jaw 50 --cheeks 30
  python FaceSlim.py --input ./photos/ --output ./results/ --preset Strong
  python FaceSlim.py --input a.mp4 b.jpg --faces 3 --preset "Full Sculpt"
""")
    parser.add_argument('--input', '-i', nargs='+', help=tr('Input file(s) or folder(s)'))
    parser.add_argument('--output', '-o', help=tr('Output directory'))
    parser.add_argument('--preset', '-p', help=tr('Apply named preset'))
    parser.add_argument('--jaw', type=int, help='Jaw slimming (0-100)')
    parser.add_argument('--cheeks', type=int, help='Cheek slimming (0-100)')
    parser.add_argument('--chin', type=int, help='Chin reshape (0-100)')
    parser.add_argument('--face-width', type=int, dest='face_width', help='Face width (0-100)')
    parser.add_argument('--forehead', type=int, help='Forehead slimming (0-100)')
    parser.add_argument('--nose', type=int, help='Nose slimming (0-100)')
    parser.add_argument('--eye-enlarge', type=int, dest='eye_enlarge', help='Eye enlargement (0-100)')
    parser.add_argument('--lip-plump', type=int, dest='lip_plump', help='Lip plumping (0-100)')
    parser.add_argument('--expression-neutralize', type=int, dest='expression_neutralize',
                        help='Blendshape-guided frown and brow dampening (0-100)')
    parser.add_argument('--skin-smooth', type=int, dest='skin_smooth', help='AI skin smoothing (0-100)')
    parser.add_argument('--teeth-whiten', type=int, dest='teeth_whiten', help='AI teeth whitening (0-100)')
    parser.add_argument('--eye-sharpen', type=int, dest='eye_sharpen', help='AI eye sharpening (0-100)')
    parser.add_argument('--skin-tone-even', type=int, dest='skin_tone_even', help='AI skin tone evening (0-100)')
    parser.add_argument('--lip-color', type=int, dest='lip_color', help='AI lip color enhancement (0-100)')
    parser.add_argument('--under-eye', type=int, dest='under_eye', help='Under-eye smoothing (0-100)')
    parser.add_argument('--hair-hue', type=int, dest='hair_hue', help='Hair hue shift (0-100)')
    parser.add_argument('--hair-saturation', type=int, dest='hair_saturation', help='Hair saturation boost (0-100)')
    parser.add_argument('--hair-density', type=int, dest='hair_density', help='Hair density hint (0-100)')
    parser.add_argument('--blush', type=int, help='Procedural blush overlay (0-100)')
    parser.add_argument('--lip-gloss', type=int, dest='lip_gloss', help='Lip gloss overlay (0-100)')
    parser.add_argument('--eye-shadow', type=int, dest='eye_shadow', help='Eye shadow overlay (0-100)')
    parser.add_argument('--smoothing', type=int, help='Warp smoothing (10-100)')
    parser.add_argument('--temporal', type=int, help='Temporal landmark smoothing (0-100)')
    parser.add_argument('--bg-protect', type=int, dest='bg_protect', help='Background protection (0-100)')
    parser.add_argument('--matting-refine', type=int, dest='matting_refine', help='MODNet mask edge refinement (0-100)')
    parser.add_argument('--post-stage', dest='post_stage_model', choices=list(POST_STAGE_OPTIONS.keys()),
                        help='Optional post-stage model: off, GFPGAN, Real-ESRGAN 2x, or both')
    parser.add_argument('--post-strength', type=int, dest='post_stage_strength',
                        help='Post-stage blend strength (0-100)')
    parser.add_argument('--post-fidelity', type=int, dest='post_stage_fidelity',
                        help='Identity fidelity for face restoration (0-100; higher keeps more original detail)')
    parser.add_argument('--post-tile', type=int, dest='post_stage_tile',
                        help='Real-ESRGAN tile size for large frames (128-1024 recommended)')
    parser.add_argument('--faces', type=int, help='Max faces to process (1-5)')
    parser.add_argument('--face-preset', action='append', default=[], metavar='FACE=PRESET',
                        help='Apply a preset to one face index, e.g. 2=Beauty')
    parser.add_argument('--face-param', action='append', default=[], metavar='FACE:key=value',
                        help='Override one parameter for a face, e.g. 1:jaw=35')
    parser.add_argument('--manifest', help='JSON batch manifest with per-file/per-face settings')
    parser.add_argument('--watermark', action='store_true', help='Add AI modification disclosure watermark')
    parser.add_argument('--strip-metadata', action='store_true', help='Do not preserve source image EXIF/XMP/ICC metadata')
    parser.add_argument('--video-compare', choices=['none', 'split', 'side_by_side'], default='none',
                        help='Export processed video normally, split-screen, or side-by-side')
    parser.add_argument('--parser-model', choices=list(PARSER_MODELS.keys()), default=DEFAULT_PARSER_MODEL,
                        help='Face parsing model for beauty masks')
    parser.add_argument('--onnx-provider', choices=list(ONNX_PROVIDER_OPTIONS.keys()), default=None,
                        help='ONNX Runtime provider override (default: saved setting or auto)')
    parser.add_argument('--provider-diagnostics', action='store_true',
                        help='Show ONNX provider availability, fallback, and one-frame benchmark')
    parser.add_argument('--list-models', action='store_true',
                        help='List model source, license, verification, provider, and cache status')
    parser.add_argument('--redownload-model', choices=["all"] + [cfg["key"] for cfg in all_model_configs()],
                        help='Delete and re-download one model artifact, or all model artifacts')
    parser.add_argument('--locale', choices=['en', 'pseudo', 'qps'], default=None,
                        help=tr('Locale override for CLI output and GUI labels'))
    parser.add_argument('--list-presets', action='store_true', help=tr('List available presets'))

    args = parser.parse_args()
    if args.locale:
        set_locale(args.locale)
    saved_provider = QSettings("FaceSlim", "FaceSlim").value(
        "onnx_provider", DEFAULT_ONNX_PROVIDER, type=str)
    args.onnx_provider = onnx_provider_key(args.onnx_provider or saved_provider)

    if args.list_presets:
        print(f"\n{tr('FaceSlim')} v{VERSION} - {tr('Available Presets')}:\n")
        print(f"{tr('Built-in')}:")
        for name, vals in BUILT_IN_PRESETS.items():
            desc = ', '.join(f"{k}={v}" for k, v in vals.items() if v > 0)
            print(f"  {name:15s} {desc}")
        custom = PresetManager.list_custom()
        if custom:
            print(f"\n{tr('Custom')}:")
            for name, vals in custom.items():
                desc = ', '.join(f"{k}={v}" for k, v in vals.items() if v > 0)
                print(f"  {name:15s} {desc}")
        sys.exit(0)

    if args.list_models:
        print(f"\n{tr('FaceSlim')} v{VERSION} - {tr('Model Inventory')}\n")
        print(model_inventory_text(args.onnx_provider))
        sys.exit(0)

    if args.redownload_model:
        print(f"\nFaceSlim v{VERSION} - Model Redownload\n")
        results = redownload_models(args.redownload_model)
        for key, ok in results.items():
            print(f"  {key}: {'OK' if ok else 'FAILED'}")
        print("\n" + model_inventory_text(args.onnx_provider))
        if not all(results.values()):
            sys.exit(1)
        sys.exit(0)

    if args.provider_diagnostics:
        print(f"\nFaceSlim v{VERSION} - ONNX Provider Diagnostics\n")
        print(provider_diagnostics_text(
            args.onnx_provider, args.parser_model,
            run_benchmark=True, ensure_parser=True))
        sys.exit(0)

    if not ensure_model():
        print(f"ERROR: Could not obtain model.\nDownload from:\n  {MODEL_URL}\nPlace at:\n  {MODEL_PATH}")
        sys.exit(1)
    if args.input or args.manifest:
        cli_process(args)
    else:
        gui_settings = QSettings("FaceSlim", "FaceSlim")
        gui_parser = parser_model_key(gui_settings.value("parser_model", DEFAULT_PARSER_MODEL, type=str))
        ensure_parsing_model(gui_parser)  # Non-fatal - falls back to landmark mask
        # GUI mode
        app = QApplication(sys.argv)
        app.setStyle("Fusion"); app.setStyleSheet(DARK_STYLE)
        pal = QPalette()
        for role, col in [(QPalette.ColorRole.Window,"#1e1e2e"),(QPalette.ColorRole.WindowText,"#cdd6f4"),
            (QPalette.ColorRole.Base,"#313244"),(QPalette.ColorRole.Text,"#cdd6f4"),
            (QPalette.ColorRole.Button,"#313244"),(QPalette.ColorRole.ButtonText,"#cdd6f4"),
            (QPalette.ColorRole.Highlight,"#89b4fa"),(QPalette.ColorRole.HighlightedText,"#1e1e2e")]:
            pal.setColor(role, QColor(col))
        app.setPalette(pal)
        w = FaceSlimApp(); w.show()
        sys.exit(app.exec())
