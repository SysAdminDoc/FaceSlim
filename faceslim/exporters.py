#!/usr/bin/env python3
"""Qt worker threads for preview, export, batch, and model diagnostics."""

import math
import os
import subprocess
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from PIL import Image as PILImage
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import pyvirtualcam
    HAS_VIRTUALCAM = True
except Exception:
    pyvirtualcam = None
    HAS_VIRTUALCAM = False

from .runtime import _decode_process_output, log_render_event
from .models import (
    DEFAULT_ONNX_PROVIDER, DEFAULT_PARSER_MODEL, onnx_provider_key,
    parser_model_key, provider_diagnostics_text, redownload_models,
)
from .pipeline import *

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list, object)  # orig, proc, faces, teeth_hint_rois
    fps_update = pyqtSignal(float)
    timeline_update = pyqtSignal(float, float)  # current seconds, duration seconds
    virtualcam_update = pyqtSignal(bool, str)
    status_update = pyqtSignal(str)
    error_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = self.show_landmarks = self.show_confidence = self.show_teeth_hint = False
        self.paused = False
        self.source = None
        self.params = DEFAULT_PARAMS.copy()
        self.max_faces = 1
        self.parser_model = DEFAULT_PARSER_MODEL
        self.onnx_provider = DEFAULT_ONNX_PROVIDER
        self.preview_scale = 1.0  # 1.0 = full res, 0.5 = half
        self.virtualcam_enabled = False
        self._seek_seconds = None
        self._seek_lock = threading.Lock()
        self._last_frame_error_log = 0.0

    def set_source(self, s): self.source = s
    def update_params(self, p): self.params = p.copy()
    def request_seek(self, seconds):
        with self._seek_lock:
            self._seek_seconds = max(0.0, float(seconds))

    def _take_seek(self):
        with self._seek_lock:
            seconds = self._seek_seconds
            self._seek_seconds = None
            return seconds

    def run(self):
        eng = None
        cap = None
        virtual_cam = None
        virtual_cam_shape = None
        try:
            self.running = True
            eng = FaceWarpEngine('video', self.max_faces, self.parser_model, self.onnx_provider)
            # Apply temporal beta
            beta = self.params.get('temporal', 50) / 5000.0
            eng.set_temporal_beta(beta)

            cap = cv2.VideoCapture(0 if self.source is None else self.source)
            if not cap.isOpened():
                self.status_update.emit("Failed to open video source")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (frame_count / fps) if (self.source is not None and frame_count > 0 and fps > 0) else 0.0
            self.timeline_update.emit(0.0, duration)
            fps_t = deque(maxlen=30)
            while self.running:
                seek_to = self._take_seek()
                if seek_to is not None and self.source is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, seek_to * 1000.0)
                    if eng is not None:
                        eng.close()
                    eng = FaceWarpEngine('video', self.max_faces, self.parser_model, self.onnx_provider)
                    eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)
                if self.paused and seek_to is None:
                    self.msleep(50); continue
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    if self.source is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        eng.close(); eng = FaceWarpEngine(
                            'video', self.max_faces, self.parser_model, self.onnx_provider)
                        eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)
                        self.timeline_update.emit(0.0, duration)
                        continue
                    break

                try:
                    if self.source is not None:
                        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        current_sec = (pos_msec / 1000.0) if pos_msec > 0 else (pos_frame / fps if fps > 0 else 0.0)
                        self.timeline_update.emit(current_sec, duration)

                    # Preview scaling
                    if self.preview_scale < 1.0:
                        h, w = frame.shape[:2]
                        nh, nw = int(h * self.preview_scale), int(w * self.preview_scale)
                        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = eng.detect(rgb)
                    has_fx = has_effective_processing(self.params)

                    # Update temporal beta if changed
                    eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)

                    proc = eng.warp(rgb, faces, self.params) if has_fx else rgb.copy()
                    teeth_hint_rois = eng.teeth_hint_rois(proc, faces) if (self.show_teeth_hint and faces) else []
                    if self.virtualcam_enabled:
                        try:
                            vh, vw = proc.shape[:2]
                            if virtual_cam is None or virtual_cam_shape != (vw, vh):
                                if virtual_cam is not None:
                                    virtual_cam.close()
                                virtual_cam = pyvirtualcam.Camera(
                                    width=vw, height=vh, fps=max(1, int(fps or 30)),
                                    fmt=pyvirtualcam.PixelFormat.RGB)
                                virtual_cam_shape = (vw, vh)
                                self.virtualcam_update.emit(True, f"Virtual camera: {vw}x{vh}")
                            virtual_cam.send(np.ascontiguousarray(proc))
                        except Exception as e:
                            self.virtualcam_enabled = False
                            if virtual_cam is not None:
                                virtual_cam.close()
                                virtual_cam = None
                            self.virtualcam_update.emit(False, f"Virtual camera unavailable: {e}")

                    el = time.time() - t0
                    fps_t.append(el)
                    if len(fps_t) > 2:
                        self.fps_update.emit(1.0 / max(sum(fps_t)/len(fps_t), 0.001))

                    self.frame_ready.emit(rgb, proc, faces, teeth_hint_rois)
                except Exception as e:
                    msg = f"Frame processing error: {e}"
                    now = time.time()
                    if now - self._last_frame_error_log > 2.0:
                        log_render_event("video_frame_error", msg, {
                            "source": self.source if self.source is not None else "webcam",
                        }, traceback.format_exc())
                        self.status_update.emit(f"{msg} (see render.log)")
                        self._last_frame_error_log = now
                    print(msg)

                self.msleep(1)
        except Exception as e:
            msg = f"Video error: {e}"
            log_render_event("video_thread_error", msg, {
                "source": self.source if self.source is not None else "webcam",
            }, traceback.format_exc())
            self.error_update.emit(f"{msg} (see render.log)")
            print(f"VideoThread fatal: {e}")
        finally:
            if cap is not None:
                cap.release()
            if eng is not None:
                eng.close()
            if virtual_cam is not None:
                virtual_cam.close()

    def stop(self):
        self.running = False; self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT THREAD (video)
# ═══════════════════════════════════════════════════════════════════════════
class ExportThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, inp, out, params, max_faces=1, watermark=False,
                 compare_mode='none', parser_model=None, onnx_provider=None):
        super().__init__()
        self.inp, self.out, self.params = inp, out, params
        self.max_faces = max_faces
        self.watermark = watermark
        self.compare_mode = compare_mode
        self.parser_model = parser_model_key(parser_model)
        self.onnx_provider = onnx_provider_key(onnx_provider)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        eng = None
        cap = None
        writer = None
        try:
            preflight = preflight_media_job(self.inp, self.out, self.compare_mode, self.params)
            self.status.emit(format_preflight_summary(preflight))
            if not preflight["ok"]:
                msg = preflight_failure_message(preflight)
                log_render_event("export_preflight_failed", msg, {
                    "input": self.inp,
                    "output": self.out,
                    "preflight": preflight,
                })
                self.error.emit(f"Preflight failed: {msg} (see render.log)")
                return
            eng = FaceWarpEngine('video', self.max_faces, self.parser_model, self.onnx_provider)
            eng.grid_scale = 6
            cap = cv2.VideoCapture(self.inp)
            if not cap.isOpened():
                msg = "Cannot open input"
                log_render_event("export_open_failed", msg, {"input": self.inp, "output": self.out})
                self.error.emit(f"{msg} (see render.log)"); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or not math.isfinite(fps) or fps <= 0:
                fps = 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w, out_h = video_output_size(w, h, self.compare_mode, self.params)
            writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            if not writer.isOpened():
                raise ValueError(f"Cannot open output writer: {self.out}")
            self.status.emit(f"Exporting {total} frames...")
            has_fx = has_effective_processing(self.params)
            t_start = time.time()
            for i in range(total):
                if self.cancelled:
                    self.status.emit("Export cancelled"); break
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = eng.detect(rgb)
                proc = eng.warp(rgb, faces, self.params) if has_fx else rgb
                proc = compose_compare_frame(rgb, proc, self.compare_mode)
                proc = apply_disclosure_watermark(proc, self.watermark)
                writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                if total > 0:
                    self.progress.emit(int((i+1)/total*100))
                    elapsed = time.time() - t_start
                    if i > 0:
                        eta = (elapsed / (i+1)) * (total - i - 1)
                        mins, secs = divmod(int(eta), 60)
                        self.status.emit(f"Frame {i+1}/{total} | ETA: {mins}m {secs}s")
            cap.release(); cap = None
            writer.release(); writer = None
            eng.close(); eng = None
            if not self.cancelled:
                self._mux_audio()
                elapsed_total = time.time() - t_start
                mins, secs = divmod(int(elapsed_total), 60)
                self.status.emit(f"Done in {mins}m {secs}s")
                self.finished.emit(self.out)
        except Exception as e:
            log_render_event("export_failed", str(e), {
                "input": self.inp,
                "output": self.out,
                "compare_mode": self.compare_mode,
            }, traceback.format_exc())
            self.error.emit(f"{e} (see render.log)")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if eng is not None:
                eng.close()

    def _mux_audio(self):
        base, ext = os.path.splitext(self.out)
        tmp = base + '_mux' + ext
        try:
            r = subprocess.run(['ffmpeg','-y','-i',self.out,'-i',self.inp,
                '-c:v','copy','-c:a','aac','-map','0:v:0','-map','1:a:0?',
                '-shortest',tmp], capture_output=True, timeout=600)
            if r.returncode == 0 and os.path.exists(tmp):
                os.replace(tmp, self.out)
                self.status.emit("Audio mux complete")
                return True
            stderr = _decode_process_output(r.stderr).strip()
            log_render_event("audio_mux_failed", "FFmpeg audio mux failed; keeping rendered video.", {
                "input": self.inp,
                "output": self.out,
                "returncode": r.returncode,
                "stderr": stderr[-4000:],
            })
            self.status.emit("Audio mux failed; video kept (see render.log)")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            return False
        except FileNotFoundError:
            log_render_event("audio_mux_unavailable", "FFmpeg not found; keeping rendered video without audio.", {
                "input": self.inp,
                "output": self.out,
            })
            self.status.emit("FFmpeg not found; video kept without audio (see render.log)")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            return False
        except Exception as e:
            log_render_event("audio_mux_error", str(e), {
                "input": self.inp,
                "output": self.out,
            }, traceback.format_exc())
            self.status.emit("Audio mux error; video kept (see render.log)")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            return False


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING THREAD
# ═══════════════════════════════════════════════════════════════════════════
class BatchThread(QThread):
    progress = pyqtSignal(int, int, str)             # current, total, filename
    job_update = pyqtSignal(int, str, int, str, str) # row, status, progress, detail, eta
    overall_update = pyqtSignal(int, int, str)       # completed, total, eta
    finished = pyqtSignal(int, int, int)             # processed, failed, skipped
    error = pyqtSignal(str)

    def __init__(self, files, output_dir, params, max_faces=1, watermark=False,
                 preserve_metadata=True, compare_mode='none', jobs=None,
                 parser_model=None, onnx_provider=None):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self.max_faces = max_faces
        self.watermark = watermark
        self.preserve_metadata = preserve_metadata
        self.compare_mode = compare_mode
        self.jobs = jobs
        self.parser_model = parser_model_key(parser_model)
        self.onnx_provider = onnx_provider_key(onnx_provider)
        self.cancelled = False
        self._cancelled_jobs = set()
        self._cancel_lock = threading.Lock()

    def cancel(self):
        self.cancelled = True

    def cancel_job(self, index):
        with self._cancel_lock:
            self._cancelled_jobs.add(int(index))

    def _job_cancelled(self, index):
        with self._cancel_lock:
            return int(index) in self._cancelled_jobs

    def _emit_overall(self, completed, total, started_at):
        eta = "--"
        if completed > 0 and completed < total:
            eta = format_duration(((time.time() - started_at) / completed) * (total - completed))
        self.overall_update.emit(completed, total, eta)

    def run(self):
        processed = failed = skipped = 0
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            jobs = self.jobs or jobs_from_files(self.files, self.output_dir, self.params,
                                                self.max_faces, self.watermark,
                                                self.preserve_metadata, self.compare_mode,
                                                self.parser_model, self.onnx_provider)
            total = len(jobs)
            started_at = time.time()
            self._emit_overall(0, total, started_at)

            for i, job in enumerate(jobs):
                if self.cancelled:
                    break
                filepath = job["input"]
                fname = os.path.basename(filepath)
                if self._job_cancelled(i):
                    skipped += 1
                    self.job_update.emit(i, "Skipped", 100, "Skipped before start", "--")
                    self._emit_overall(processed + failed + skipped, total, started_at)
                    continue

                self.progress.emit(i + 1, total, fname)
                self.job_update.emit(i, "Processing", 0, "Starting", "--")
                ext = os.path.splitext(filepath)[1].lower()

                try:
                    if ext in IMAGE_EXTS:
                        ok = self._process_image(job, i)
                    elif ext in VIDEO_EXTS:
                        ok = self._process_video(job, i)
                    else:
                        ok = False
                        failed += 1
                        log_render_event("batch_unsupported_media", "Unsupported file type", {
                            "input": filepath,
                            "job_index": i,
                        })
                        self.job_update.emit(i, "Failed", 100, "Unsupported file type", "--")
                    if ext in IMAGE_EXTS | VIDEO_EXTS:
                        if ok:
                            processed += 1
                            self.job_update.emit(i, "Done", 100, "Complete", "0s")
                        else:
                            skipped += 1
                            self.job_update.emit(i, "Skipped", 100, "Cancelled", "--")
                except Exception as e:
                    print(f"Batch error {fname}: {e}")
                    log_render_event("batch_job_failed", str(e), {
                        "input": filepath,
                        "job_index": i,
                    }, traceback.format_exc())
                    failed += 1
                    self.job_update.emit(i, "Failed", 100, str(e), "--")

                self._emit_overall(processed + failed + skipped, total, started_at)

            self.finished.emit(processed, failed, skipped)
        except Exception as e:
            log_render_event("batch_failed", str(e), {
                "output_dir": self.output_dir,
            }, traceback.format_exc())
            self.error.emit(str(e))

    def _process_image(self, job, index):
        if self.cancelled or self._job_cancelled(index):
            return False
        filepath = job["input"]
        params = job.get("params", self.params)
        eng = None
        out_path = media_job_output_path(job, self.output_dir, filepath, ".png")
        try:
            preflight = preflight_media_job(filepath, out_path, job.get("compare_mode", self.compare_mode), params)
            self.job_update.emit(index, "Processing", 5, format_preflight_summary(preflight), "--")
            if not preflight["ok"]:
                msg = preflight_failure_message(preflight)
                log_render_event("batch_preflight_failed", msg, {
                    "input": filepath,
                    "output": out_path,
                    "job_index": index,
                    "preflight": preflight,
                })
                raise ValueError(f"Preflight failed: {msg}")
            self.job_update.emit(index, "Processing", 20, "Reading image", "--")
            eng = FaceWarpEngine('image', job.get("max_faces", self.max_faces),
                                 job.get("parser_model", self.parser_model),
                                 job.get("onnx_provider", self.onnx_provider))
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(f"Cannot read {filepath}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.job_update.emit(index, "Processing", 55, "Processing image", "--")
            result, _ = eng.warp_single_image(rgb, params)
            if self.cancelled or self._job_cancelled(index):
                return False
            result = apply_disclosure_watermark(result, job.get("watermark", self.watermark))
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.job_update.emit(index, "Processing", 85, "Saving image", "--")
            save_rgb_image(out_path, result, filepath, job.get("preserve_metadata", self.preserve_metadata),
                           job.get("watermark", self.watermark))
            return True
        finally:
            if eng is not None:
                eng.close()

    def _process_video(self, job, index):
        if self.cancelled or self._job_cancelled(index):
            return False
        filepath = job["input"]
        params = job.get("params", self.params)
        max_faces = job.get("max_faces", self.max_faces)
        watermark = job.get("watermark", self.watermark)
        compare_mode = job.get("compare_mode", self.compare_mode)
        eng = None
        cap = None
        writer = None
        out_path = media_job_output_path(job, self.output_dir, filepath, ".mp4")
        try:
            preflight = preflight_media_job(filepath, out_path, compare_mode, params)
            self.job_update.emit(index, "Processing", 5, format_preflight_summary(preflight), "--")
            if not preflight["ok"]:
                msg = preflight_failure_message(preflight)
                log_render_event("batch_preflight_failed", msg, {
                    "input": filepath,
                    "output": out_path,
                    "job_index": index,
                    "preflight": preflight,
                })
                raise ValueError(f"Preflight failed: {msg}")
            eng = FaceWarpEngine('video', max_faces, job.get("parser_model", self.parser_model),
                                 job.get("onnx_provider", self.onnx_provider))
            eng.grid_scale = 6
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValueError(f"Cannot open {filepath}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or not math.isfinite(fps) or fps <= 0:
                fps = 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_w, out_h = video_output_size(w, h, compare_mode, params)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            if not writer.isOpened():
                raise ValueError(f"Cannot open output writer: {out_path}")
            has_fx = has_effective_processing(params)
            total = max(total, 1)
            started_at = time.time()
            for frame_idx in range(total):
                if self.cancelled or self._job_cancelled(index):
                    return False
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = eng.detect(rgb)
                proc = eng.warp(rgb, faces, params) if has_fx else rgb
                proc = compose_compare_frame(rgb, proc, compare_mode)
                proc = apply_disclosure_watermark(proc, watermark)
                writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                pct = int(((frame_idx + 1) / total) * 100)
                if frame_idx == 0 or frame_idx == total - 1 or frame_idx % max(1, total // 50) == 0:
                    elapsed = time.time() - started_at
                    eta = "--"
                    if frame_idx > 0:
                        eta = format_duration((elapsed / (frame_idx + 1)) * (total - frame_idx - 1))
                    self.job_update.emit(index, "Processing", pct,
                                         f"Frame {frame_idx + 1}/{total}", eta)
            return True
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if eng is not None:
                eng.close()
            if (self.cancelled or self._job_cancelled(index)) and os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass


# ═══════════════════════════════════════════════════════════════════════════
# GIF EXPORT (before/after comparison)
# ═══════════════════════════════════════════════════════════════════════════
class GifExportThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, original, processed, output_path, duration=800):
        super().__init__()
        self.original = original.copy()
        self.processed = processed.copy()
        self.output_path = output_path
        self.duration = duration

    def run(self):
        try:
            h, w = self.original.shape[:2]
            # Add labels
            orig_labeled = self.original.copy()
            proc_labeled = self.processed.copy()
            for img, text in [(orig_labeled, "Before"), (proc_labeled, "After")]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(img, (8, 8), (tw + 20, th + 20), (30, 30, 46), -1)
                cv2.putText(img, text, (14, th + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)

            frames = [PILImage.fromarray(orig_labeled), PILImage.fromarray(proc_labeled)]
            frames[0].save(self.output_path, save_all=True, append_images=frames[1:],
                          duration=self.duration, loop=0, optimize=True)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK DRAWING
# ═══════════════════════════════════════════════════════════════════════════

class ProviderDiagnosticsThread(QThread):
    diagnostics_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, provider_preference, parser_model):
        super().__init__()
        self.provider_preference = onnx_provider_key(provider_preference)
        self.parser_model = parser_model_key(parser_model)

    def run(self):
        try:
            text = provider_diagnostics_text(
                self.provider_preference, self.parser_model,
                run_benchmark=True, ensure_parser=True)
            self.diagnostics_ready.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class ModelRedownloadThread(QThread):
    download_finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_key):
        super().__init__()
        self.model_key = model_key

    def run(self):
        try:
            results = redownload_models(self.model_key)
            failed = [key for key, ok in results.items() if not ok]
            if failed:
                self.error.emit("Failed to download: " + ", ".join(failed))
            else:
                self.download_finished.emit("Downloaded: " + ", ".join(results.keys()))
        except Exception as e:
            self.error.emit(str(e))
