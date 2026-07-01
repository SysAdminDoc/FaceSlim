#!/usr/bin/env python3
"""FaceSlim image pipeline, effects, preflight, presets, and warp engine."""

import glob
import json
import math
import os
import shutil
import threading
import time
import traceback
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from PIL import Image as PILImage, PngImagePlugin
from scipy.interpolate import RBFInterpolator

try:
    import torch
    import torch.nn.functional as TF
    HAS_TORCH = True
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
    GPU_NAME = torch.cuda.get_device_name(0) if USE_GPU else "CPU"
    if USE_GPU:
        print(f"  GPU Acceleration: ON ({GPU_NAME})")
    else:
        print("  PyTorch loaded (CPU only)")
except Exception:
    HAS_TORCH = False
    USE_GPU = False
    DEVICE = None
    GPU_NAME = "CPU"
    print("  PyTorch not available - using CPU mode (install torch for GPU acceleration)")

from .runtime import (
    APP_DIR, IMAGE_EXTS, IPTC_DIGITAL_SOURCE_TYPE, PRESETS_DIR, RENDER_LOG_PATH,
    VERSION, VIDEO_EXTS, log_render_event,
)
from .models import *
from .models import _model_path, _remove_quietly

JAW_CONTOUR = [10,338,297,332,284,251,389,356,454,323,361,288,
               397,365,379,378,400,377,152,148,176,149,150,136,
               172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_CHEEK  = [330,329,328,326,427,411,376,352,345,372,383,353]
RIGHT_CHEEK = [101,100,99,97,207,187,147,123,116,143,156,124]
CHIN        = [152,148,176,149,150,136,172,58,132,377,378,400,379,365,397]
LEFT_JAW    = [454,323,361,288,397,365,379,378,400]
RIGHT_JAW   = [234,93,132,58,172,136,150,149,176]
NOSE_BRIDGE = [6,197,195,5,4,1,19,94,2]
FOREHEAD    = [10,338,297,109,67,103,54,21,162,127]
FACE_CENTER = [1,4,5,6,168,197,195]
# Additional landmark regions
LEFT_EYE_BROW  = [336,296,334,293,300,276,283,282,295,285]
RIGHT_EYE_BROW = [107,66,105,63,70,46,53,52,65,55]
NOSE_TIP    = [1,2,98,327,168]

# Eye contours for enlargement warp (iris centers + surrounding mesh)
LEFT_EYE_CONTOUR  = [362,385,387,263,373,380,374,386,388,466,249,390]
RIGHT_EYE_CONTOUR = [33,160,158,133,153,144,145,159,157,246,7,161]
LEFT_IRIS_CENTER  = [468]   # MediaPipe iris landmark
RIGHT_IRIS_CENTER = [473]   # MediaPipe iris landmark

# Lip contours for plumping warp (outer lip boundary)
# Corner landmarks 291 and 61 shared between upper/lower; only include in upper to avoid TPS duplicates
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [146, 91, 181, 84, 17, 314, 405, 321, 375]  # corners excluded
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [95, 88, 178, 87, 14, 317, 402, 318, 324]  # corners excluded

# Labels for eye sharpening
EYE_SHARPEN_LABELS = {PARSE_L_EYE, PARSE_R_EYE, PARSE_L_BROW, PARSE_R_BROW}
# Labels for lip color
LIP_LABELS = {PARSE_U_LIP, PARSE_L_LIP}

FACE_OVAL = JAW_CONTOUR

# ═══════════════════════════════════════════════════════════════════════════
# BUILT-IN PRESETS
# ═══════════════════════════════════════════════════════════════════════════
BUILT_IN_PRESETS = {
    'Subtle':      {'jaw': 15, 'cheeks': 10, 'chin': 5,  'face_width': 10, 'forehead': 0, 'nose': 0},
    'Moderate':    {'jaw': 35, 'cheeks': 25, 'chin': 15, 'face_width': 20, 'forehead': 0, 'nose': 0},
    'Strong':      {'jaw': 60, 'cheeks': 45, 'chin': 30, 'face_width': 35, 'forehead': 0, 'nose': 0},
    'V-Shape':     {'jaw': 70, 'cheeks': 20, 'chin': 40, 'face_width': 15, 'forehead': 0, 'nose': 0},
    'Oval':        {'jaw': 40, 'cheeks': 35, 'chin': 20, 'face_width': 30, 'forehead': 10, 'nose': 0},
    'Slim Nose':   {'jaw': 0,  'cheeks': 0,  'chin': 0,  'face_width': 0,  'forehead': 0, 'nose': 50},
    'Full Sculpt': {'jaw': 50, 'cheeks': 40, 'chin': 25, 'face_width': 25, 'forehead': 15, 'nose': 30},
    'Beauty':      {'skin_smooth': 40, 'skin_tone_even': 25, 'teeth_whiten': 20, 'eye_sharpen': 30, 'lip_color': 20},
    'Glamour':     {'jaw': 30, 'cheeks': 20, 'nose': 20, 'eye_enlarge': 25, 'lip_plump': 20,
                    'skin_smooth': 50, 'teeth_whiten': 30, 'eye_sharpen': 35, 'lip_color': 30},
}

DEFAULT_PARAMS = {'jaw': 0, 'cheeks': 0, 'chin': 0, 'face_width': 0,
                  'forehead': 0, 'nose': 0, 'eye_enlarge': 0, 'lip_plump': 0,
                  'smoothing': 50, 'temporal': 50, 'bg_protect': 70,
                  'skin_smooth': 0, 'teeth_whiten': 0, 'eye_sharpen': 0,
                  'skin_tone_even': 0, 'lip_color': 0, 'under_eye': 0,
                  'hair_hue': 0, 'hair_saturation': 0, 'hair_density': 0,
                  'blush': 0, 'lip_gloss': 0, 'eye_shadow': 0,
                  'expression_neutralize': 0,
                  'matting_refine': 0,
                  'post_stage_model': POST_STAGE_OFF,
                  'post_stage_strength': 50,
                  'post_stage_fidelity': 70,
                  'post_stage_tile': 512}

EFFECT_PARAM_KEYS = (
    'jaw', 'cheeks', 'chin', 'face_width', 'forehead', 'nose', 'eye_enlarge',
    'lip_plump', 'skin_smooth', 'teeth_whiten', 'eye_sharpen',
    'skin_tone_even', 'lip_color', 'under_eye', 'hair_hue',
    'hair_saturation', 'hair_density', 'blush', 'lip_gloss', 'eye_shadow',
    'expression_neutralize'
)

POST_STAGE_PARAM_KEYS = ('post_stage_strength', 'post_stage_fidelity', 'post_stage_tile')
CLI_PARAM_KEYS = EFFECT_PARAM_KEYS + ('smoothing', 'temporal', 'bg_protect', 'matting_refine') + POST_STAGE_PARAM_KEYS


def post_stage_model_key(value=None):
    key = str(value or POST_STAGE_OFF).strip()
    return key if key in POST_STAGE_OPTIONS else POST_STAGE_OFF


def post_stage_model_label(value=None):
    return POST_STAGE_OPTIONS[post_stage_model_key(value)]["label"]


def post_stage_model_keys(value=None):
    return POST_STAGE_OPTIONS[post_stage_model_key(value)]["models"]


def post_stage_enabled(params):
    return post_stage_model_key((params or {}).get("post_stage_model")) != POST_STAGE_OFF


def post_stage_upscale_factor(params):
    return 2 if "real_esrgan_x2" in post_stage_model_keys((params or {}).get("post_stage_model")) else 1


def has_effective_processing(params):
    return any(abs((params or {}).get(k, 0)) > 0 for k in EFFECT_PARAM_KEYS) or post_stage_enabled(params)

# ═══════════════════════════════════════════════════════════════════════════
# ONE-EURO FILTER
# ═══════════════════════════════════════════════════════════════════════════
class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq, self.mincutoff, self.beta, self.dcutoff = freq, mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def set_beta(self, beta):
        self.beta = beta

    def _alpha(self, cutoff):
        r = 2 * math.pi * cutoff / self.freq
        return r / (r + 1)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t if t is not None else time.time()
            return x.copy()
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if dt > 0: self.freq = 1.0 / dt
        self.t_prev = t if t is not None else time.time()
        a_d = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = np.vectorize(self._alpha)(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat.copy(), dx_hat.copy()
        return x_hat

# ═══════════════════════════════════════════════════════════════════════════
# FACE PARSING ENGINE (BiSeNet ONNX)
# ═══════════════════════════════════════════════════════════════════════════
class FaceParsingEngine:
    """BiSeNet-based semantic face segmentation via ONNX Runtime.
    Returns per-pixel class labels (19 classes from CelebAMask-HQ)."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _INPUT_SIZE = 512

    def __init__(self, model_key=None, onnx_provider=None):
        self.model_key = parser_model_key(model_key)
        self.onnx_provider = onnx_provider_key(onnx_provider)
        if not ensure_parsing_model(self.model_key):
            raise RuntimeError(f"{parser_model_label(self.model_key)} unavailable")
        self.session, self.provider_resolution = create_onnx_session(
            parser_model_path(self.model_key), self.onnx_provider, parser_model_label(self.model_key))
        self.input_name = self.session.get_inputs()[0].name
        self._cache = {}  # {(h, w, key): parsing_map}
        print(
            f"  Face parsing: {parser_model_label(self.model_key)} loaded "
            f"({self.provider_resolution['selected_provider']}; "
            f"{self.provider_resolution['fallback_reason']})"
        )

    def parse(self, face_rgb, cache_key=None):
        """Run face parsing on RGB image. Returns (h, w) int array of class labels 0-18."""
        h, w = face_rgb.shape[:2]

        # Check cache
        if cache_key is not None:
            cached = self._cache.get((h, w, cache_key))
            if cached is not None:
                return cached

        # Preprocess: resize, normalize, NCHW
        img = cv2.resize(face_rgb, (self._INPUT_SIZE, self._INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)

        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        logits = outputs[0]  # (1, 19, 512, 512)

        # Argmax + resize back to original size
        parsing = np.argmax(logits[0], axis=0).astype(np.uint8)  # (512, 512)
        if (h, w) != (self._INPUT_SIZE, self._INPUT_SIZE):
            parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

        # Cache result
        if cache_key is not None:
            self._cache[(h, w, cache_key)] = parsing
            # Keep cache bounded
            if len(self._cache) > 8:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return parsing

    def get_mask(self, parsing, label_set, feather=0):
        """Create a binary float32 mask from parsing map for given label set.
        Optional Gaussian feathering."""
        mask = np.isin(parsing, list(label_set)).astype(np.float32)
        if feather > 0:
            k = max(3, feather) | 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask


class MattingRefinementEngine:
    """MODNet portrait matting via ONNX Runtime for cleaner ROI edge masks."""

    _INPUT_SIZE = 512

    def __init__(self, onnx_provider=None):
        self.onnx_provider = onnx_provider_key(onnx_provider)
        if not ensure_matting_model():
            raise RuntimeError("MODNet matting model unavailable")
        self.session, self.provider_resolution = create_onnx_session(
            MODNET_PATH, self.onnx_provider, MATTE_MODEL["label"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._cache = {}
        print(
            f"  Matting refinement: MODNet loaded "
            f"({self.provider_resolution['selected_provider']}; "
            f"{self.provider_resolution['fallback_reason']})"
        )

    def _preprocess(self, rgb):
        orig_h, orig_w = rgb.shape[:2]
        if max(orig_h, orig_w) < self._INPUT_SIZE or min(orig_h, orig_w) > self._INPUT_SIZE:
            if orig_w >= orig_h:
                new_h = self._INPUT_SIZE
                new_w = int(orig_w / max(orig_h, 1) * self._INPUT_SIZE)
            else:
                new_w = self._INPUT_SIZE
                new_h = int(orig_h / max(orig_w, 1) * self._INPUT_SIZE)
        else:
            new_h, new_w = orig_h, orig_w
        new_h = max(32, new_h - (new_h % 32))
        new_w = max(32, new_w - (new_w % 32))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))
        return np.expand_dims(x, axis=0), orig_h, orig_w

    def matte(self, rgb, cache_key=None):
        h, w = rgb.shape[:2]
        if cache_key is not None:
            cached = self._cache.get((h, w, cache_key))
            if cached is not None:
                return cached
        tensor, orig_h, orig_w = self._preprocess(rgb)
        out = self.session.run(self.output_names, {self.input_name: tensor})[0]
        matte = np.squeeze(out).astype(np.float32)
        matte = cv2.resize(matte, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        matte = np.clip(matte, 0.0, 1.0)
        if cache_key is not None:
            self._cache[(h, w, cache_key)] = matte
            if len(self._cache) > 8:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return matte


def ensure_restoration_model(model_key):
    cfg = RESTORATION_MODELS[model_key]
    return _download_verified_model(cfg, f"{cfg['label']} post-stage disabled")


def _onnx_output_to_rgb(output):
    arr = np.asarray(output)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return None
    if arr.min() < -0.05:
        arr = (arr + 1.0) * 127.5
    elif arr.max() <= 2.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class FaceRestorationPostStage:
    """GFPGAN ONNX face restoration for detected face ROIs."""

    _INPUT_SIZE = 512

    def __init__(self, onnx_provider=None):
        self.onnx_provider = onnx_provider_key(onnx_provider)
        if not ensure_restoration_model("gfpgan_1.4"):
            raise RuntimeError("GFPGAN post-stage model unavailable")
        cfg = RESTORATION_MODELS["gfpgan_1.4"]
        self.session, self.provider_resolution = create_onnx_session(
            _model_path(cfg), self.onnx_provider, cfg["label"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def restore(self, face_rgb):
        if face_rgb.size == 0:
            return face_rgb
        h, w = face_rgb.shape[:2]
        resized = cv2.resize(face_rgb, (self._INPUT_SIZE, self._INPUT_SIZE), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 127.5 - 1.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
        output = self.session.run(self.output_names, {self.input_name: tensor})[0]
        restored = _onnx_output_to_rgb(output)
        if restored is None:
            return face_rgb
        if restored.shape[:2] != (h, w):
            restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_CUBIC)
        return restored


class UpscalePostStage:
    """Real-ESRGAN ONNX tiled 2x upscaler."""

    def __init__(self, onnx_provider=None):
        self.onnx_provider = onnx_provider_key(onnx_provider)
        if not ensure_restoration_model("real_esrgan_x2"):
            raise RuntimeError("Real-ESRGAN post-stage model unavailable")
        cfg = RESTORATION_MODELS["real_esrgan_x2"]
        self.session, self.provider_resolution = create_onnx_session(
            _model_path(cfg), self.onnx_provider, cfg["label"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _run_tile(self, tile_rgb):
        tensor = tile_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]
        output = self.session.run(self.output_names, {self.input_name: tensor})[0]
        upscaled = _onnx_output_to_rgb(output)
        if upscaled is None:
            return cv2.resize(tile_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return upscaled

    def upscale(self, rgb, tile_size=512):
        h, w = rgb.shape[:2]
        tile_size = int(max(128, min(2048, tile_size or 512)))
        if h <= tile_size and w <= tile_size:
            return self._run_tile(rgb)

        overlap = min(32, max(8, tile_size // 8))
        first = rgb[0:min(tile_size, h), 0:min(tile_size, w)]
        first_up = self._run_tile(first)
        scale_y = max(1, int(round(first_up.shape[0] / max(first.shape[0], 1))))
        scale_x = max(1, int(round(first_up.shape[1] / max(first.shape[1], 1))))
        out_h, out_w = h * scale_y, w * scale_x
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
        step = max(1, tile_size - overlap)

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                tile = rgb[y0:y1, x0:x1]
                up = first_up if (y0 == 0 and x0 == 0) else self._run_tile(tile)
                oy0, ox0 = y0 * scale_y, x0 * scale_x
                oy1, ox1 = oy0 + up.shape[0], ox0 + up.shape[1]
                accum[oy0:oy1, ox0:ox1] += up.astype(np.float32)
                weight[oy0:oy1, ox0:ox1] += 1.0

        weight = np.maximum(weight, 1.0)
        return np.clip(accum / weight, 0, 255).astype(np.uint8)


def face_components(face):
    if len(face) >= 3:
        return face[0], face[1], face[2] or {}
    return face[0], face[1], {}


def scale_faces_for_frame(faces, source_shape, target_shape):
    if not faces or not source_shape or not target_shape:
        return faces
    src_h, src_w = source_shape[:2]
    dst_h, dst_w = target_shape[:2]
    if src_h <= 0 or src_w <= 0 or (src_h, src_w) == (dst_h, dst_w):
        return faces
    scale = np.array([dst_w / src_w, dst_h / src_h], dtype=np.float64)
    scaled = []
    for face in faces:
        lms, conf, blendshapes = face_components(face)
        scaled.append((lms * scale, conf, blendshapes))
    return scaled


def blend_score(blendshapes, *names):
    if not blendshapes:
        return 0.0
    return max((float(blendshapes.get(name, 0.0)) for name in names), default=0.0)


def apply_skin_smoothing(frame, skin_mask, strength):
    """Frequency-separation skin smoothing using bilateral filter.
    frame: RGB uint8 (h, w, 3)
    skin_mask: float32 (h, w) with values 0-1
    strength: 0-100
    Returns smoothed RGB uint8."""
    if strength <= 0 or skin_mask is None:
        return frame

    s = strength / 100.0

    # Bilateral filter params scale with strength
    d = int(5 + s * 10)         # diameter: 5-15
    sigma_color = 20 + s * 55   # 20-75
    sigma_space = 20 + s * 55   # 20-75

    # Apply bilateral filter (smooths skin, preserves edges like eyes/lips)
    smooth = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

    # Frequency separation: extract high-frequency detail
    # Low freq = bilateral output, High freq = original - bilateral
    # Result = bilateral + (high_freq * texture_retention)
    texture_retain = max(0.0, 1.0 - s * 0.8)  # Keep 20-100% of texture
    high_freq = frame.astype(np.float32) - smooth.astype(np.float32)
    result = smooth.astype(np.float32) + high_freq * texture_retain
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Blend with original using skin mask (only smooth skin pixels)
    mask_3ch = skin_mask[:, :, np.newaxis]
    blended = result.astype(np.float32) * mask_3ch + frame.astype(np.float32) * (1.0 - mask_3ch)
    return np.clip(blended, 0, 255).astype(np.uint8)


def teeth_target_mask(parsing, feather=5):
    """Return the mouth-interior mask used by teeth whitening."""
    if parsing is None:
        return None
    mask = (parsing == PARSE_MOUTH).astype(np.float32)
    if feather > 0:
        k = max(3, feather) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def apply_teeth_hint_overlay(frame, mask):
    """Preview-only overlay for the teeth whitening target mask."""
    if mask is None or mask.max() < 0.01:
        return frame
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    overlay_color = np.array([137, 180, 250], dtype=np.float32)
    result = frame.astype(np.float32)
    mask_3 = mask[:, :, np.newaxis]
    result = result * (1.0 - mask_3 * 0.38) + overlay_color * (mask_3 * 0.38)
    edge = cv2.Canny((mask > 0.08).astype(np.uint8) * 255, 40, 120)
    result[edge > 0] = np.array([243, 139, 168], dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_teeth_hint_rois(frame, hint_rois):
    if not hint_rois:
        return frame
    result = frame.copy()
    for roi_bounds, mask in hint_rois:
        rx1, ry1, rx2, ry2 = roi_bounds
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        roi = result[ry1:ry2, rx1:rx2]
        if roi.size == 0 or mask.shape != roi.shape[:2]:
            continue
        result[ry1:ry2, rx1:rx2] = apply_teeth_hint_overlay(roi, mask)
    return result


def apply_teeth_whitening(frame, parsing, strength):
    """Whiten teeth within mouth interior mask.
    Uses HSV: increase V, decrease S in mouth interior only (not lips)."""
    if strength <= 0 or parsing is None:
        return frame

    # Only mouth interior (teeth/tongue visible area) - NOT lips
    mask = teeth_target_mask(parsing, feather=5)

    if mask.max() < 0.01:
        return frame

    s = strength / 100.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - s * 0.5 * mask)
    hsv[:, :, 2] = hsv[:, :, 2] + s * 35.0 * mask
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Soft blend back using mask
    blended = result.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_eye_sharpen(frame, parsing, strength):
    """Sharpen eye and brow region using unsharp mask.
    Enhances iris detail and brow definition via parsing mask."""
    if strength <= 0 or parsing is None:
        return frame

    mask = np.isin(parsing, list(EYE_SHARPEN_LABELS)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    if mask.max() < 0.01:
        return frame

    s = strength / 100.0
    # Unsharp mask: sharpen = original + (original - blurred) * amount
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(frame, 1.0 + s * 1.5, blurred, -s * 1.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    mask_3 = mask[:, :, np.newaxis]
    blended = sharpened.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_skin_tone_even(frame, skin_mask, strength):
    """Even out skin tone by blending toward the mean skin color.
    Reduces redness, blotchiness, and uneven pigmentation within skin mask."""
    if strength <= 0 or skin_mask is None:
        return frame
    if skin_mask.max() < 0.01:
        return frame

    s = strength / 100.0

    # Compute mean skin color (weighted by mask)
    mask_sum = skin_mask.sum()
    if mask_sum < 10:
        return frame
    mask_3 = skin_mask[:, :, np.newaxis]
    mean_color = (frame.astype(np.float32) * mask_3).sum(axis=(0, 1)) / mask_sum

    # Create a uniform color layer at mean skin tone
    uniform = np.full_like(frame, mean_color, dtype=np.float32)

    # Blend: push skin pixels toward mean color (reduces variation)
    # Use a mild blend - full replacement would look plastic
    blend_strength = s * 0.35  # max 35% blend toward uniform
    evened = frame.astype(np.float32) * (1.0 - blend_strength * mask_3) + uniform * (blend_strength * mask_3)

    # Additionally reduce redness: if R channel is dominant, pull it down
    lab = cv2.cvtColor(np.clip(evened, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    # a channel: positive = red, negative = green. Reduce positive bias.
    lab[:, :, 1] = lab[:, :, 1] - s * 8.0 * skin_mask  # reduce redness
    lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    blended = result.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_lip_color(frame, parsing, strength):
    """Enhance lip color - boost saturation and warm hue in lip mask.
    Creates a natural lip tint effect."""
    if strength <= 0 or parsing is None:
        return frame

    mask = np.isin(parsing, list(LIP_LABELS)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    if mask.max() < 0.01:
        return frame

    s = strength / 100.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Boost saturation in lip region
    hsv[:, :, 1] = hsv[:, :, 1] + s * 40.0 * mask
    # Slight warmth: push hue toward red/pink (reduce if already warm)
    hsv[:, :, 2] = hsv[:, :, 2] + s * 10.0 * mask  # slight brightness
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    mask_3 = mask[:, :, np.newaxis]
    blended = result.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _mask_to_blend(mask):
    if mask is None:
        return None
    return np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, np.newaxis]


def apply_color_overlay(frame, mask, color_rgb, opacity):
    if opacity <= 0 or mask is None or mask.max() < 0.01:
        return frame
    strength = min(max(opacity / 100.0, 0.0), 1.0)
    mask_3 = _mask_to_blend(mask) * strength
    color = np.zeros_like(frame, dtype=np.float32)
    color[:, :] = np.array(color_rgb, dtype=np.float32)
    blended = frame.astype(np.float32) * (1.0 - mask_3) + color * mask_3
    return np.clip(blended, 0, 255).astype(np.uint8)


def make_under_eye_mask(lms, roi, shape, parsing=None):
    rx1, ry1, _, _ = roi
    h, w = shape
    offset = np.array([rx1, ry1], dtype=np.float64)
    face_h = max(24.0, float(np.ptp(lms[:min(468, len(lms)), 1])))
    mask = np.zeros((h, w), dtype=np.float32)
    for contour in (LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR):
        pts = [lms[i] - offset for i in contour if i < len(lms)]
        if len(pts) < 4:
            continue
        pts = np.array(pts, dtype=np.float64)
        center_y = float(np.mean(pts[:, 1]))
        lower = pts[pts[:, 1] >= center_y]
        if len(lower) < 2:
            lower = pts
        lower = lower[np.argsort(lower[:, 0])]
        drop = np.array([0.0, face_h * 0.08], dtype=np.float64)
        poly = np.vstack([lower, lower[::-1] + drop]).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1.0, cv2.LINE_AA)
    if parsing is not None:
        skin = np.isin(parsing, list(SKIN_SMOOTH_LABELS)).astype(np.float32)
        mask *= skin
    return cv2.GaussianBlur(mask, (13, 13), 0)


def make_blush_mask(lms, roi, shape):
    rx1, ry1, _, _ = roi
    h, w = shape
    offset = np.array([rx1, ry1], dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.float32)
    for cheek in (LEFT_CHEEK, RIGHT_CHEEK):
        pts = [lms[i] - offset for i in cheek if i < len(lms)]
        if len(pts) < 3:
            continue
        pts = np.array(pts, dtype=np.float64)
        center = np.mean(pts, axis=0).astype(int)
        radius_x = max(10, int(np.ptp(pts[:, 0]) * 0.45))
        radius_y = max(8, int(np.ptp(pts[:, 1]) * 0.55))
        cv2.ellipse(mask, tuple(center), (radius_x, radius_y), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (25, 25), 0)


def make_eye_shadow_mask(parsing):
    if parsing is None:
        return None
    mask = np.isin(parsing, [PARSE_L_BROW, PARSE_R_BROW, PARSE_L_EYE, PARSE_R_EYE]).astype(np.float32)
    if mask.max() < 0.01:
        return None
    dilated = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return cv2.GaussianBlur(dilated, (15, 15), 0)


def apply_hair_controls(frame, parsing, hue_shift, saturation, density):
    if parsing is None or (hue_shift <= 0 and saturation <= 0 and density <= 0):
        return frame
    mask = (parsing == PARSE_HAIR).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    if mask.max() < 0.01:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    if hue_shift > 0:
        hsv[:, :, 0] = (hsv[:, :, 0] + (hue_shift / 100.0) * 36.0 * mask) % 180.0
    if saturation > 0:
        hsv[:, :, 1] = hsv[:, :, 1] + (saturation / 100.0) * 55.0 * mask
    if density > 0:
        hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - (density / 100.0) * 0.28 * mask)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    mask_3 = _mask_to_blend(mask)
    blended = adjusted.astype(np.float32) * mask_3 + frame.astype(np.float32) * (1.0 - mask_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_lip_gloss(frame, parsing, strength):
    if strength <= 0 or parsing is None:
        return frame
    mask = np.isin(parsing, list(LIP_LABELS)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    if mask.max() < 0.01:
        return frame
    s = strength / 100.0
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + 20.0 * s * mask, 0, 255)
    glossy = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return apply_color_overlay(glossy, mask, (255, 215, 220), strength * 0.12)


def apply_disclosure_watermark(frame, enabled=True):
    if not enabled:
        return frame
    out = frame.copy()
    text = "AI modified"
    h, w = out.shape[:2]
    scale = max(0.45, min(w, h) / 900.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(6, int(8 * scale))
    x1 = max(0, w - tw - pad * 2 - 10)
    y1 = max(0, h - th - base - pad * 2 - 10)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (w - 10, h - 10), (17, 17, 27), -1)
    out = cv2.addWeighted(overlay, 0.68, out, 0.32, 0)
    cv2.putText(out, text, (x1 + pad, h - 10 - pad - base),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (205, 214, 244), thick, cv2.LINE_AA)
    return out


def compose_compare_frame(original, processed, mode):
    if original.shape[:2] != processed.shape[:2]:
        ph, pw = processed.shape[:2]
        original = cv2.resize(original, (pw, ph), interpolation=cv2.INTER_CUBIC)
    if mode == 'side_by_side':
        return np.concatenate([original, processed], axis=1)
    if mode == 'split':
        h, w = original.shape[:2]
        sx = max(1, min(w - 1, w // 2))
        out = np.empty_like(processed)
        out[:, :sx] = original[:, :sx]
        out[:, sx:] = processed[:, sx:]
        cv2.line(out, (sx, 0), (sx, h), (203, 166, 247), 3)
        return out
    return processed


def post_stage_output_dimensions(width, height, params=None):
    factor = post_stage_upscale_factor(params or {})
    return int(width) * factor, int(height) * factor


def video_output_size(width, height, mode, params=None):
    width, height = post_stage_output_dimensions(width, height, params)
    if mode == 'side_by_side':
        return width * 2, height
    return width, height


def _provenance_context(source_path=None, preserve_metadata=True, watermark=False):
    return {
        "tool": f"FaceSlim v{VERSION}",
        "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_file": os.path.basename(source_path) if source_path else "",
        "source_metadata_preserved": bool(preserve_metadata),
        "disclosure_watermark": bool(watermark),
        "digital_source_type": IPTC_DIGITAL_SOURCE_TYPE,
        "description": "AI modified by FaceSlim face reshaping and retouch pipeline.",
    }


def _provenance_description(ctx):
    watermark = "yes" if ctx["disclosure_watermark"] else "no"
    preserved = "yes" if ctx["source_metadata_preserved"] else "no"
    return (
        f"{ctx['description']} Tool: {ctx['tool']}; edited: {ctx['edited_at']}; "
        f"visual watermark: {watermark}; source metadata preserved: {preserved}."
    )


def build_provenance_xmp(source_path=None, preserve_metadata=True, watermark=False):
    ctx = _provenance_context(source_path, preserve_metadata, watermark)
    desc = _provenance_description(ctx)
    attrs = {
        "xmp:CreatorTool": ctx["tool"],
        "photoshop:Instructions": desc,
        "Iptc4xmpExt:DigitalSourceType": ctx["digital_source_type"],
        "faceslim:Version": VERSION,
        "faceslim:EditedAt": ctx["edited_at"],
        "faceslim:SourceFile": ctx["source_file"],
        "faceslim:SourceMetadataPreserved": str(ctx["source_metadata_preserved"]).lower(),
        "faceslim:DisclosureWatermark": str(ctx["disclosure_watermark"]).lower(),
    }
    attr_text = "\n   ".join(f'{key}="{xml_escape(str(value))}"' for key, value in attrs.items())
    return f'''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:xmp="http://ns.adobe.com/xap/1.0/"
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
   xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
   xmlns:faceslim="https://github.com/SysAdminDoc/FaceSlim/ns/1.0/"
   {attr_text}>
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{xml_escape(desc)}</rdf:li>
    </rdf:Alt>
   </dc:description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''.encode("utf-8")


def _apply_exif_provenance(save_kwargs, source_exif, source_path=None,
                           preserve_metadata=True, watermark=False):
    exif = source_exif if source_exif is not None else PILImage.Exif()
    ctx = _provenance_context(source_path, preserve_metadata, watermark)
    desc = _provenance_description(ctx)
    exif[270] = desc  # ImageDescription
    exif[305] = ctx["tool"]  # Software
    exif[37510] = b"ASCII\0\0\0" + desc.encode("ascii", errors="replace")  # UserComment
    save_kwargs["exif"] = exif.tobytes()


def _jpeg_without_existing_xmp(data):
    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    if not data.startswith(b"\xff\xd8"):
        return data
    out = bytearray(data[:2])
    i = 2
    while i < len(data):
        if data[i:i + 1] != b"\xff" or i + 4 > len(data):
            out.extend(data[i:])
            break
        marker = data[i:i + 2]
        if marker == b"\xff\xda":
            out.extend(data[i:])
            break
        length = int.from_bytes(data[i + 2:i + 4], "big")
        end = i + 2 + length
        if end > len(data):
            out.extend(data[i:])
            break
        payload = data[i + 4:end]
        if marker == b"\xff\xe1" and payload.startswith(xmp_header):
            i = end
            continue
        out.extend(data[i:end])
        i = end
    return bytes(out)


def _write_jpeg_xmp(path, xmp_bytes):
    xmp_header = b"http://ns.adobe.com/xap/1.0/\x00"
    payload = xmp_header + xmp_bytes
    if len(payload) + 2 > 65535:
        return
    with open(path, "rb") as f:
        data = f.read()
    data = _jpeg_without_existing_xmp(data)
    if not data.startswith(b"\xff\xd8"):
        return
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    with open(path, "wb") as f:
        f.write(data[:2] + segment + data[2:])


def save_rgb_image(output_path, rgb, source_path=None, preserve_metadata=True, watermark=False):
    ext = os.path.splitext(output_path)[1].lower()
    if not ext:
        output_path += ".png"
        ext = ".png"
    image = PILImage.fromarray(rgb)
    save_kwargs = {}
    png_info = None
    source_exif = None
    if preserve_metadata and source_path and os.path.exists(source_path):
        try:
            with PILImage.open(source_path) as src:
                source_exif = src.getexif()
                if src.info.get("icc_profile"):
                    save_kwargs["icc_profile"] = src.info["icc_profile"]
                if ext == ".png":
                    png_info = PngImagePlugin.PngInfo()
                    for key, value in src.info.items():
                        if isinstance(key, str) and isinstance(value, str):
                            png_info.add_text(key, value)
        except Exception as e:
            print(f"Metadata preserve warning: {e}")
    xmp = build_provenance_xmp(source_path, preserve_metadata, watermark)
    if ext == ".png":
        if png_info is None:
            png_info = PngImagePlugin.PngInfo()
        ctx = _provenance_context(source_path, preserve_metadata, watermark)
        png_info.add_text("XML:com.adobe.xmp", xmp.decode("utf-8"))
        png_info.add_text("Software", ctx["tool"])
        png_info.add_text("Description", _provenance_description(ctx))
        png_info.add_text("IPTC:DigitalSourceType", ctx["digital_source_type"])
        png_info.add_text("FaceSlim:SourceMetadataPreserved", str(ctx["source_metadata_preserved"]).lower())
        png_info.add_text("FaceSlim:DisclosureWatermark", str(ctx["disclosure_watermark"]).lower())
    if ext in (".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        _apply_exif_provenance(save_kwargs, source_exif, source_path, preserve_metadata, watermark)
    if png_info is not None:
        save_kwargs["pnginfo"] = png_info
    if ext in (".jpg", ".jpeg"):
        save_kwargs.setdefault("quality", 95)
        save_kwargs.setdefault("subsampling", 1)
    image.save(output_path, **save_kwargs)
    if ext in (".jpg", ".jpeg"):
        _write_jpeg_xmp(output_path, xmp)
    return output_path


def resolve_preset(name):
    if not name:
        return {}
    if name in BUILT_IN_PRESETS:
        return BUILT_IN_PRESETS[name].copy()
    custom = PresetManager.list_custom()
    if name in custom:
        return custom[name].copy()
    raise ValueError(f"Unknown preset: {name}")


def params_from_preset_and_overrides(preset_name=None, overrides=None):
    params = DEFAULT_PARAMS.copy()
    if preset_name:
        params.update(resolve_preset(preset_name))
    if overrides:
        params.update(coerce_param_overrides(overrides))
    return params


def coerce_param_overrides(values):
    overrides = {}
    for key, value in (values or {}).items():
        normalized = str(key).strip().replace("-", "_")
        if normalized in CLI_PARAM_KEYS:
            overrides[normalized] = max(0, min(int(value), 1024 if normalized == "post_stage_tile" else 100))
        elif normalized == "post_stage_model":
            overrides[normalized] = post_stage_model_key(value)
    return overrides


def parse_param_overrides(raw):
    overrides = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in CLI_PARAM_KEYS:
            raise ValueError(f"Unknown parameter: {key}")
        overrides[key] = max(0, min(int(value), 1024 if key == "post_stage_tile" else 100))
    return overrides


def parse_face_overrides(face_presets=None, face_params=None):
    face_overrides = {}
    for item in face_presets or []:
        if "=" not in item:
            raise ValueError(f"Expected FACE=PRESET, got {item}")
        face, preset = item.split("=", 1)
        idx = max(0, int(face) - 1)
        face_overrides.setdefault(idx, {}).update(resolve_preset(preset.strip()))
    for item in face_params or []:
        if ":" not in item or "=" not in item:
            raise ValueError(f"Expected FACE:key=value, got {item}")
        face, rest = item.split(":", 1)
        key, value = rest.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in CLI_PARAM_KEYS:
            raise ValueError(f"Unknown parameter: {key}")
        idx = max(0, int(face) - 1)
        face_overrides.setdefault(idx, {})[key] = int(value)
    return face_overrides


def media_files_from_paths(paths):
    inputs = []
    for inp in paths or []:
        if os.path.isdir(inp):
            for f in os.listdir(inp):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS | VIDEO_EXTS:
                    inputs.append(os.path.join(inp, f))
        elif os.path.isfile(inp):
            inputs.append(inp)
        else:
            print(f"Warning: {inp} not found, skipping")
    return sorted(inputs)


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_timecode(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


PREFLIGHT_DISK_RESERVE_BYTES = 64 * 1024 * 1024
PREFLIGHT_DISK_SAFETY_MULTIPLIER = 1.25
PREFLIGHT_VIDEO_BYTES_PER_PIXEL_FRAME = 0.08
PREFLIGHT_IMAGE_BUFFER_MULTIPLIER = 5


def format_bytes(size_bytes):
    if size_bytes is None:
        return "--"
    try:
        value = float(size_bytes)
    except (TypeError, ValueError):
        return "--"
    if value < 0:
        return "--"
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def media_job_output_path(job, output_dir, source_path, output_ext):
    return job.get("output") or os.path.join(
        job.get("output_dir", output_dir),
        os.path.splitext(os.path.basename(source_path))[0] + f"_slimmed{output_ext}",
    )


def _safe_file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _existing_parent(path):
    target = os.path.abspath(path or ".")
    if os.path.isdir(target):
        return target
    target = os.path.dirname(target) or os.getcwd()
    while target and not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    return target or os.getcwd()


def _disk_free_bytes(path):
    try:
        return shutil.disk_usage(_existing_parent(path)).free
    except OSError:
        return None


def _probe_writable_output(output_path):
    if not output_path:
        return False, "Missing output path"
    if os.path.isdir(output_path):
        return False, f"Output path is a directory: {output_path}"
    parent = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        return False, f"Cannot create output directory: {e}"
    test_path = os.path.join(parent, f".faceslim-write-test-{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        with open(test_path, "wb") as f:
            f.write(b"")
        return True, "writable"
    except OSError as e:
        return False, f"Output is not writable: {e}"
    finally:
        _remove_quietly(test_path)


def _probe_video_writer(output_path, fps, width, height):
    writable, detail = _probe_writable_output(output_path)
    if not writable:
        return False, detail
    if width <= 0 or height <= 0:
        return False, "Cannot verify codec without output dimensions"
    parent = os.path.dirname(os.path.abspath(output_path)) or os.getcwd()
    probe_path = os.path.join(parent, f".faceslim-writer-probe-{os.getpid()}-{threading.get_ident()}.mp4")
    writer = None
    try:
        writer = cv2.VideoWriter(probe_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps or 30), (int(width), int(height)))
        ok = bool(writer.isOpened())
        return ok, "mp4v writer available" if ok else "OpenCV mp4v writer is unavailable"
    except Exception as e:
        return False, f"OpenCV writer probe failed: {e}"
    finally:
        if writer is not None:
            writer.release()
        _remove_quietly(probe_path)


def _add_disk_guardrail(report, output_path):
    report["available_disk_bytes"] = _disk_free_bytes(output_path)
    estimate = report.get("estimated_output_bytes")
    free_bytes = report.get("available_disk_bytes")
    if estimate is None or free_bytes is None:
        return
    required = int(estimate * PREFLIGHT_DISK_SAFETY_MULTIPLIER) + PREFLIGHT_DISK_RESERVE_BYTES
    report["required_disk_bytes"] = required
    if free_bytes < required:
        report["errors"].append(
            f"Insufficient disk space: need about {format_bytes(required)}, "
            f"available {format_bytes(free_bytes)}"
        )


def preflight_media_job(input_path, output_path, compare_mode="none", params=None):
    report = {
        "ok": False,
        "input": input_path,
        "output": output_path,
        "kind": "unknown",
        "errors": [],
        "warnings": [],
        "input_size_bytes": _safe_file_size(input_path),
        "estimated_output_bytes": None,
        "estimated_memory_bytes": None,
        "estimated_time_seconds": None,
        "available_disk_bytes": None,
        "required_disk_bytes": None,
        "codec_ok": None,
        "output_writable": False,
    }
    ext = os.path.splitext(input_path or "")[1].lower()
    if not input_path or not os.path.exists(input_path):
        report["errors"].append(f"Input not found: {input_path}")
        return report
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        report["errors"].append(f"Unsupported media type: {ext or 'none'}")
        return report

    if ext in IMAGE_EXTS:
        report["kind"] = "image"
        try:
            with PILImage.open(input_path) as image:
                width, height = image.size
                bands = len(image.getbands()) or 3
        except Exception as e:
            report["errors"].append(f"Cannot read image header: {e}")
            return report
        width, height = post_stage_output_dimensions(width, height, params)
        pixel_bytes = max(1, int(width) * int(height) * max(3, bands))
        out_ext = os.path.splitext(output_path or "")[1].lower()
        if out_ext in {".jpg", ".jpeg", ".webp"}:
            estimated_output = max(64 * 1024, int(pixel_bytes * 0.7))
        else:
            estimated_output = max(64 * 1024, int(width) * int(height) * 4)
        megapixels = (int(width) * int(height)) / 1_000_000
        writable, detail = _probe_writable_output(output_path)
        report.update({
            "width": int(width),
            "height": int(height),
            "frame_count": 1,
            "duration_seconds": 0,
            "fps": None,
            "estimated_output_bytes": estimated_output,
            "estimated_memory_bytes": int(pixel_bytes * PREFLIGHT_IMAGE_BUFFER_MULTIPLIER),
            "estimated_time_seconds": max(1, int(math.ceil(max(0.2, megapixels * 0.8)))),
            "output_writable": writable,
            "output_detail": detail,
        })
        if not writable:
            report["errors"].append(detail)
        _add_disk_guardrail(report, output_path)
        report["ok"] = not report["errors"]
        return report

    report["kind"] = "video"
    cap = cv2.VideoCapture(input_path)
    try:
        if not cap.isOpened():
            report["errors"].append(f"Cannot open video: {input_path}")
            return report
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            report["errors"].append("Cannot determine video resolution")
        if fps <= 0 or not math.isfinite(fps):
            fps = 30.0
            report["warnings"].append("Video FPS is unknown; estimating at 30 FPS")
        if frame_count <= 0:
            report["warnings"].append("Frame count is unknown; progress and size estimates are conservative")
        out_w, out_h = video_output_size(width, height, compare_mode, params)
        estimated_frames = max(frame_count, 1)
        estimated_output = max(
            report["input_size_bytes"] or 0,
            int(out_w * out_h * estimated_frames * PREFLIGHT_VIDEO_BYTES_PER_PIXEL_FRAME),
            256 * 1024,
        )
        megapixels = (max(out_w, 1) * max(out_h, 1)) / 1_000_000
        codec_ok, codec_detail = _probe_video_writer(output_path, fps, out_w, out_h)
        report.update({
            "width": width,
            "height": height,
            "output_width": out_w,
            "output_height": out_h,
            "frame_count": frame_count,
            "duration_seconds": (frame_count / fps) if frame_count > 0 else None,
            "fps": fps,
            "estimated_output_bytes": estimated_output,
            "estimated_memory_bytes": int(max(out_w, 1) * max(out_h, 1) * 3 * 8),
            "estimated_time_seconds": int(math.ceil(estimated_frames * max(0.03, megapixels * 0.08))),
            "codec_ok": codec_ok,
            "codec_detail": codec_detail,
            "output_writable": codec_ok,
        })
        if not codec_ok:
            report["errors"].append(codec_detail)
        _add_disk_guardrail(report, output_path)
        report["ok"] = not report["errors"]
        return report
    finally:
        cap.release()


def format_preflight_summary(report):
    if not report:
        return "Preflight unavailable"
    kind = report.get("kind", "media")
    size = f"{report.get('width', 0)}x{report.get('height', 0)}"
    output_size = format_bytes(report.get("estimated_output_bytes"))
    memory = format_bytes(report.get("estimated_memory_bytes"))
    disk = format_bytes(report.get("available_disk_bytes"))
    eta = format_duration(report.get("estimated_time_seconds"))
    if kind == "video":
        frames = report.get("frame_count") or "unknown"
        fps = report.get("fps")
        fps_text = f"{fps:.2f} fps" if fps else "unknown fps"
        duration = format_timecode(report.get("duration_seconds"))
        out_size = f"{report.get('output_width', 0)}x{report.get('output_height', 0)}"
        codec = "OK" if report.get("codec_ok") else "failed"
        return (
            f"Preflight: video {size}->{out_size}, {frames} frames, {duration}, {fps_text}; "
            f"est output {output_size}, est memory {memory}, est time {eta}, free disk {disk}, codec {codec}"
        )
    return (
        f"Preflight: image {size}; est output {output_size}, est memory {memory}, "
        f"est time {eta}, free disk {disk}, output {'writable' if report.get('output_writable') else 'blocked'}"
    )


def preflight_failure_message(report):
    if not report:
        return "preflight unavailable"
    return "; ".join(report.get("errors") or ["unknown preflight failure"])


def _manifest_path(base_dir, path_value):
    if not path_value:
        return path_value
    return path_value if os.path.isabs(path_value) else os.path.abspath(os.path.join(base_dir, path_value))


def load_batch_manifest(manifest_path, fallback_output=None, fallback_parser_model=None,
                        fallback_onnx_provider=None):
    with open(manifest_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    default_params = params_from_preset_and_overrides(data.get("preset"), data.get("params"))
    if data.get("post_stage_model"):
        default_params["post_stage_model"] = post_stage_model_key(data.get("post_stage_model"))
    default_parser_model = parser_model_key(data.get("parser_model") or fallback_parser_model)
    default_onnx_provider = onnx_provider_key(data.get("onnx_provider") or fallback_onnx_provider)
    default_face_params = data.get("face_params") or {}
    if default_face_params:
        default_params["face_params"] = {int(k) - 1 if str(k).isdigit() else k: v
                                         for k, v in default_face_params.items()}
    default_output = _manifest_path(base_dir, data.get("output") or fallback_output or "faceslim_output")
    jobs = []
    entries = data.get("files") or data.get("jobs") or []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"input": entry}
        inp = _manifest_path(base_dir, entry.get("input") or entry.get("path"))
        params = default_params.copy()
        if entry.get("preset"):
            params.update(resolve_preset(entry["preset"]))
        if entry.get("params"):
            params.update(coerce_param_overrides(entry["params"]))
        if entry.get("post_stage_model"):
            params["post_stage_model"] = post_stage_model_key(entry.get("post_stage_model"))
        if entry.get("face_params"):
            params["face_params"] = {int(k) - 1 if str(k).isdigit() else k: v
                                     for k, v in entry["face_params"].items()}
        out_dir = _manifest_path(base_dir, entry.get("output_dir") or default_output)
        jobs.append({
            "input": inp,
            "output_dir": out_dir,
            "output": _manifest_path(base_dir, entry.get("output")) if entry.get("output") else None,
            "params": params,
            "max_faces": int(entry.get("faces", data.get("faces", 1))),
            "watermark": bool(entry.get("watermark", data.get("watermark", False))),
            "preserve_metadata": bool(entry.get("preserve_metadata", data.get("preserve_metadata", True))),
            "compare_mode": entry.get("video_compare", data.get("video_compare", "none")),
            "parser_model": parser_model_key(entry.get("parser_model") or default_parser_model),
            "onnx_provider": onnx_provider_key(entry.get("onnx_provider") or default_onnx_provider),
        })
    return jobs


def jobs_from_files(files, output_dir, params, max_faces=1, watermark=False,
                    preserve_metadata=True, compare_mode='none', parser_model=None,
                    onnx_provider=None):
    return [{
        "input": f,
        "output_dir": output_dir,
        "output": None,
        "params": params.copy(),
        "max_faces": max_faces,
        "watermark": watermark,
        "preserve_metadata": preserve_metadata,
        "compare_mode": compare_mode,
        "parser_model": parser_model_key(parser_model),
        "onnx_provider": onnx_provider_key(onnx_provider),
    } for f in files]


class TemporalMaskSmoother:
    """EMA-based mask smoothing to prevent parsing mask flicker on video.
    Smooths the float32 mask over time so boundaries don't jitter frame-to-frame."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Higher = more responsive, lower = smoother
        self._prev_masks = {}  # {face_idx: prev_mask}

    def smooth(self, mask, face_idx=0):
        """Apply temporal EMA smoothing to mask. Returns smoothed float32 mask."""
        prev = self._prev_masks.get(face_idx)
        if prev is None or prev.shape != mask.shape:
            self._prev_masks[face_idx] = mask.copy()
            return mask

        smoothed = self.alpha * mask + (1.0 - self.alpha) * prev
        self._prev_masks[face_idx] = smoothed.copy()
        return smoothed

    def reset(self):
        self._prev_masks.clear()


class OpticalFlowPropagator:
    """Propagates warp displacement fields between TPS keyframes using optical flow.
    Instead of computing full TPS every frame, compute every N frames and
    use Farneback optical flow to warp the displacement field for interim frames."""

    def __init__(self, keyframe_interval=5):
        self.interval = keyframe_interval
        self._frame_count = 0
        self._prev_gray = None
        self._last_displacement = None  # (dx, dy) maps from last keyframe
        self._last_keyframe_gray = None

    def should_compute_full(self):
        """Returns True if this frame should be a full TPS keyframe."""
        is_key = (self._frame_count % self.interval == 0)
        self._frame_count += 1
        return is_key

    def store_keyframe(self, frame_gray, warped, original):
        """Store displacement field from keyframe warp result."""
        h, w = frame_gray.shape[:2]
        self._last_keyframe_gray = frame_gray.copy()
        self._prev_gray = frame_gray.copy()
        # Compute displacement as difference between warped and original pixel positions
        # We store the actual warped frame for compositing
        self._last_displacement = warped.astype(np.float32) - original.astype(np.float32)

    def propagate(self, frame_gray, original_frame):
        """Propagate last keyframe's warp to current frame via optical flow.
        Returns warped frame or None if no keyframe stored."""
        if self._last_displacement is None or self._prev_gray is None:
            return None
        if self._prev_gray.shape != frame_gray.shape:
            self._prev_gray = frame_gray.copy()
            return None

        try:
            # Compute flow from previous frame to current
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, frame_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Warp the displacement field using the flow
            h, w = frame_gray.shape[:2]
            map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                        np.arange(h, dtype=np.float32))
            map_x += flow[:, :, 0]
            map_y += flow[:, :, 1]

            warped_disp = cv2.remap(self._last_displacement, map_x, map_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # Apply propagated displacement to current frame
            result = original_frame.astype(np.float32) + warped_disp
            result = np.clip(result, 0, 255).astype(np.uint8)

            self._prev_gray = frame_gray.copy()
            return result
        except Exception:
            self._prev_gray = frame_gray.copy()
            return None

    def reset(self):
        self._frame_count = 0
        self._prev_gray = None
        self._last_displacement = None
        self._last_keyframe_gray = None


# ═══════════════════════════════════════════════════════════════════════════
# FACE WARP ENGINE (GPU + CPU dual path, multi-face)
# ═══════════════════════════════════════════════════════════════════════════
class FaceWarpEngine:
    def __init__(self, mode='video', max_faces=1, parser_model=None, onnx_provider=None):
        self.parser_model = parser_model_key(parser_model)
        self.onnx_provider = onnx_provider_key(onnx_provider)
        rm = vision.RunningMode.VIDEO if mode == 'video' else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=rm, num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.mode = mode
        self.max_faces = max_faces
        self.lm_filters = [OneEuroFilter(freq=30, mincutoff=1.5, beta=0.01) for _ in range(max_faces)]
        self.ts = 0
        self.grid_scale = 4
        self.use_gpu = HAS_TORCH and USE_GPU
        # Per-face cache
        self._caches = [{} for _ in range(max_faces)]
        # Face parsing engine (optional, for better masks + skin smooth)
        self.parser = None
        self.matter = None
        self.face_restorer = None
        self.upscaler = None
        if ensure_parsing_model(self.parser_model):
            try:
                self.parser = FaceParsingEngine(self.parser_model, self.onnx_provider)
            except Exception as e:
                print(f"  Face parsing init failed: {e} (using landmark fallback)")
                self.parser = None
        # Temporal mask smoother (prevents mask boundary flicker on video)
        self.mask_smoother = TemporalMaskSmoother(alpha=0.35) if mode == 'video' else None
        # Optical flow propagator (video performance: skip TPS on interim frames)
        self.flow_prop = OpticalFlowPropagator(keyframe_interval=4) if mode == 'video' else None

    def set_temporal_beta(self, beta):
        for f in self.lm_filters:
            f.set_beta(beta)

    def detect(self, frame_rgb):
        """Returns list of (landmarks, confidence, blendshape_scores) tuples."""
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.mode == 'video':
            self.ts += 33
            res = self.landmarker.detect_for_video(mpi, self.ts)
        else:
            res = self.landmarker.detect(mpi)
        if not res.face_landmarks:
            return []
        h, w = frame_rgb.shape[:2]
        faces = []
        blendshape_results = getattr(res, 'face_blendshapes', None) or []
        for i, face_lms in enumerate(res.face_landmarks):
            pts = np.array([(lm.x * w, lm.y * h) for lm in face_lms], dtype=np.float64)
            # Compute confidence from visibility (presence can be None in some MediaPipe versions)
            try:
                pvals = [lm.presence for lm in face_lms if getattr(lm, 'presence', None) is not None]
                conf = float(np.mean(pvals)) if pvals else 0.9
            except Exception:
                conf = 0.9
            if i < len(self.lm_filters):
                pts = self.lm_filters[i](pts, time.time())
            blendshapes = {}
            if i < len(blendshape_results):
                for category in blendshape_results[i]:
                    name = getattr(category, 'category_name', None)
                    if name:
                        blendshapes[name] = float(getattr(category, 'score', 0.0))
            faces.append((pts, float(conf), blendshapes))
        return faces

    def _compute_control_points(self, lms, params, h, w, blendshapes=None):
        center = np.mean(lms[[i for i in FACE_CENTER if i < len(lms)]], axis=0)
        src, tgt = [], []
        jaw_s     = params.get('jaw', 0) / 100.0
        cheek_s   = params.get('cheeks', 0) / 100.0
        chin_s    = params.get('chin', 0) / 100.0
        width_s   = params.get('face_width', 0) / 100.0
        forehead_s = params.get('forehead', 0) / 100.0
        nose_s    = params.get('nose', 0) / 100.0
        neutral_s = params.get('expression_neutralize', 0) / 100.0

        def shift(indices, strength, vbias=0.0):
            for idx in indices:
                if idx >= len(lms): continue
                pt = lms[idx]
                vec = center - pt
                d = np.linalg.norm(vec)
                if d < 1: continue
                disp = (vec / d) * strength * min(d / 100.0, 1.0) * 15.0
                if vbias: disp[1] += vbias * strength * 8.0
                src.append(pt.copy()); tgt.append(pt + disp)

        def shift_horizontal(indices, strength):
            """Shift landmarks horizontally toward nose center (for nose slimming)."""
            nose_center_x = np.mean([lms[i][0] for i in NOSE_TIP if i < len(lms)])
            for idx in indices:
                if idx >= len(lms): continue
                pt = lms[idx]
                dx = nose_center_x - pt[0]
                disp = np.array([dx * strength * 0.3, 0.0])
                src.append(pt.copy()); tgt.append(pt + disp)

        if abs(jaw_s) > 0.01:
            shift(LEFT_JAW, jaw_s); shift(RIGHT_JAW, jaw_s)
        if abs(cheek_s) > 0.01:
            shift(LEFT_CHEEK, cheek_s); shift(RIGHT_CHEEK, cheek_s)
        if abs(chin_s) > 0.01:
            shift(CHIN, chin_s * 0.5, vbias=-1.0)
        if abs(width_s) > 0.01:
            shift(list(set(JAW_CONTOUR) - set(FOREHEAD)), width_s)
        if abs(forehead_s) > 0.01:
            shift(FOREHEAD, forehead_s * 0.5)
        if abs(nose_s) > 0.01:
            # Nose side landmarks only - exclude bridge centerline (those are anchors)
            nose_sides = [48, 64, 98, 327, 278, 294,
                          129, 358, 236, 456, 198, 420, 131, 360]
            shift_horizontal(nose_sides, nose_s)

        if neutral_s > 0.01 and blendshapes:
            face_h = max(24.0, float(np.ptp(lms[:min(468, len(lms)), 1])))

            def expression_shift(indices, dy):
                if abs(dy) < 0.1:
                    return
                for idx in indices:
                    if idx < len(lms):
                        pt = lms[idx]
                        src.append(pt.copy())
                        tgt.append(pt + np.array([0.0, dy], dtype=np.float64))

            left_brow_down = blend_score(blendshapes, 'browDownLeft')
            right_brow_down = blend_score(blendshapes, 'browDownRight')
            brow_inner_up = blend_score(blendshapes, 'browInnerUp')
            left_brow_up = max(brow_inner_up, blend_score(blendshapes, 'browOuterUpLeft'))
            right_brow_up = max(brow_inner_up, blend_score(blendshapes, 'browOuterUpRight'))
            expression_shift(LEFT_EYE_BROW, neutral_s * face_h * (left_brow_up * 0.025 - left_brow_down * 0.030))
            expression_shift(RIGHT_EYE_BROW, neutral_s * face_h * (right_brow_up * 0.025 - right_brow_down * 0.030))

            left_frown = blend_score(blendshapes, 'mouthFrownLeft')
            right_frown = blend_score(blendshapes, 'mouthFrownRight')
            expression_shift([61, 146, 91, 181], -neutral_s * face_h * left_frown * 0.030)
            expression_shift([291, 375, 321, 405], -neutral_s * face_h * right_frown * 0.030)

        # Eye enlargement: push eye contour outward from eye center
        eye_s = params.get('eye_enlarge', 0) / 100.0
        if abs(eye_s) > 0.01:
            for eye_contour, iris_center_idx in [(LEFT_EYE_CONTOUR, LEFT_IRIS_CENTER),
                                                  (RIGHT_EYE_CONTOUR, RIGHT_IRIS_CENTER)]:
                valid_contour = [i for i in eye_contour if i < len(lms)]
                valid_iris = [i for i in iris_center_idx if i < len(lms)]
                if len(valid_contour) < 3:
                    continue
                # Eye center: use iris if available, otherwise contour centroid
                if valid_iris:
                    eye_center = lms[valid_iris[0]]
                else:
                    eye_center = np.mean(lms[valid_contour], axis=0)
                for idx in valid_contour:
                    pt = lms[idx]
                    vec = pt - eye_center  # outward from eye center
                    d = np.linalg.norm(vec)
                    if d < 0.5: continue
                    disp = (vec / d) * eye_s * min(d * 0.5, 8.0)
                    src.append(pt.copy()); tgt.append(pt + disp)

        # Lip plumping: push outer lip contour outward from lip center
        lip_s = params.get('lip_plump', 0) / 100.0
        if abs(lip_s) > 0.01:
            all_lip = UPPER_LIP_OUTER + LOWER_LIP_OUTER
            valid_lip = [i for i in all_lip if i < len(lms)]
            if len(valid_lip) >= 6:
                lip_center = np.mean(lms[valid_lip], axis=0)
                # Upper lip pushes up, lower lip pushes down
                for idx in UPPER_LIP_OUTER:
                    if idx >= len(lms): continue
                    pt = lms[idx]
                    vec = pt - lip_center
                    d = np.linalg.norm(vec)
                    if d < 0.5: continue
                    # Bias upward for upper lip
                    disp = np.array([0.0, -lip_s * 4.0]) + (vec / d) * lip_s * min(d * 0.3, 4.0)
                    src.append(pt.copy()); tgt.append(pt + disp)
                for idx in LOWER_LIP_OUTER:
                    if idx >= len(lms): continue
                    pt = lms[idx]
                    vec = pt - lip_center
                    d = np.linalg.norm(vec)
                    if d < 0.5: continue
                    # Bias downward for lower lip
                    disp = np.array([0.0, lip_s * 4.0]) + (vec / d) * lip_s * min(d * 0.3, 4.0)
                    src.append(pt.copy()); tgt.append(pt + disp)

        # Anchors
        anchor_indices = set(NOSE_BRIDGE + FOREHEAD)
        if abs(forehead_s) > 0.01:
            anchor_indices -= set(FOREHEAD)
        if abs(nose_s) > 0.01:
            anchor_indices -= set(NOSE_BRIDGE)
        for idx in anchor_indices:
            if idx < len(lms):
                src.append(lms[idx].copy()); tgt.append(lms[idx].copy())

        # Edge anchors
        fmin, fmax = np.min(lms, axis=0), np.max(lms, axis=0)
        m = 80
        for c in [[fmin[0]-m, fmin[1]-m], [fmax[0]+m, fmin[1]-m],
                   [fmin[0]-m, fmax[1]+m], [fmax[0]+m, fmax[1]+m],
                   [center[0], fmin[1]-m*2], [center[0], fmax[1]+m*2],
                   [fmin[0]-m*2, center[1]], [fmax[0]+m*2, center[1]]]:
            src.append(np.array(c, dtype=np.float64))
            tgt.append(np.array(c, dtype=np.float64))

        return np.array(src), np.array(tgt)

    def _cache_key(self, src, tgt):
        return ((tgt - src) / 5.0).astype(np.int32).tobytes()

    def _params_for_face(self, params, face_idx):
        face_params = params.get('face_params') or {}
        override = face_params.get(face_idx)
        if override is None:
            override = face_params.get(str(face_idx + 1))
        if not override:
            return params
        merged = params.copy()
        merged.update(override)
        merged.pop('face_params', None)
        return merged

    def teeth_hint_rois(self, frame, faces):
        if self.parser is None or not faces:
            return []
        h, w = frame.shape[:2]
        hint_rois = []
        for i, face in enumerate(faces):
            lms, _conf, _blendshapes = face_components(face)
            try:
                roi_bounds = self._compute_roi(lms, h, w, pad_ratio=0.30)
                rx1, ry1, rx2, ry2 = roi_bounds
                roi_region = frame[ry1:ry2, rx1:rx2]
                if roi_region.size == 0:
                    continue
                parsing = self.parser.parse(roi_region)
                mask = teeth_target_mask(parsing, feather=5)
                if mask is not None and mask.max() >= 0.01:
                    hint_rois.append((roi_bounds, mask))
            except Exception as e:
                print(f"Teeth hint mask error face {i}: {e}")
        return hint_rois

    # ── GPU TPS ─────────────────────────────────────────────────
    def _tps_kernel(self, src_t, eval_t):
        diff = eval_t.unsqueeze(1) - src_t.unsqueeze(0)
        r2 = (diff ** 2).sum(dim=2)
        return 0.5 * r2 * torch.log(r2.clamp(min=1e-8))

    def _solve_tps_gpu(self, src, tgt, smoothing):
        n = len(src)
        src_t = torch.tensor(src, dtype=torch.float32, device=DEVICE)
        disp_t = torch.tensor(tgt - src, dtype=torch.float32, device=DEVICE)
        K = self._tps_kernel(src_t, src_t)
        K += max(0.1, smoothing / 100.0 * 50.0) * torch.eye(n, device=DEVICE)
        ones = torch.ones(n, 1, device=DEVICE)
        P = torch.cat([ones, src_t], dim=1)
        Z = torch.zeros(3, 3, device=DEVICE)
        L = torch.cat([torch.cat([K, P], 1), torch.cat([P.T, Z], 1)], 0)
        rhs = torch.cat([disp_t, torch.zeros(3, 2, device=DEVICE)], 0)
        params = torch.linalg.solve(L, rhs)
        return src_t, params[:n], params[n:]

    def _eval_tps_grid_gpu(self, src_t, weights, affine, h, w):
        sc = self.grid_scale
        gh, gw = h // sc, w // sc
        gy = torch.linspace(0, h-1, gh, device=DEVICE)
        gx = torch.linspace(0, w-1, gw, device=DEVICE)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        eval_pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        K_eval = self._tps_kernel(src_t, eval_pts)
        P_eval = torch.cat([torch.ones(len(eval_pts), 1, device=DEVICE), eval_pts], 1)
        disp = K_eval @ weights + P_eval @ affine
        dx = TF.interpolate(disp[:, 0].reshape(1, 1, gh, gw), (h, w), mode='bicubic', align_corners=False)
        dy = TF.interpolate(disp[:, 1].reshape(1, 1, gh, gw), (h, w), mode='bicubic', align_corners=False)
        k, pad = 5, 2
        dx = TF.avg_pool2d(TF.pad(dx, [pad]*4, mode='reflect'), k, stride=1)
        dy = TF.avg_pool2d(TF.pad(dy, [pad]*4, mode='reflect'), k, stride=1)
        base_x = torch.linspace(-1, 1, w, device=DEVICE).view(1, 1, w).expand(1, h, w)
        base_y = torch.linspace(-1, 1, h, device=DEVICE).view(1, h, 1).expand(1, h, w)
        grid_x = base_x.squeeze() - dx.squeeze() * 2.0 / max(w-1, 1)
        grid_y = base_y.squeeze() - dy.squeeze() * 2.0 / max(h-1, 1)
        return torch.stack([grid_x, grid_y], dim=2).unsqueeze(0)

    def _warp_gpu(self, frame, src, tgt, params):
        h, w = frame.shape[:2]
        with torch.no_grad():
            src_t, weights, affine = self._solve_tps_gpu(src, tgt, params.get('smoothing', 50))
            grid = self._eval_tps_grid_gpu(src_t, weights, affine, h, w)
            frame_t = torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1).unsqueeze(0).float()
            warped_t = TF.grid_sample(frame_t, grid, mode='bicubic',
                                       padding_mode='reflection', align_corners=False)
            return warped_t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy(), grid

    # ── CPU Fallback ────────────────────────────────────────────
    def _warp_cpu(self, frame, src, tgt, params):
        h, w = frame.shape[:2]
        disp = tgt - src
        smooth = max(0.1, params.get('smoothing', 50) / 100.0 * 50.0)
        sc = self.grid_scale
        gh, gw = h // sc, w // sc
        gy, gx = np.mgrid[0:h:sc, 0:w:sc].astype(np.float64)
        gpts = np.column_stack([gx.ravel(), gy.ravel()])
        dx_s = RBFInterpolator(src, disp[:, 0], kernel='thin_plate_spline',
                                smoothing=smooth)(gpts).reshape(gh, gw).astype(np.float32)
        dy_s = RBFInterpolator(src, disp[:, 1], kernel='thin_plate_spline',
                                smoothing=smooth)(gpts).reshape(gh, gw).astype(np.float32)
        dx = cv2.GaussianBlur(cv2.resize(dx_s, (w, h), interpolation=cv2.INTER_CUBIC), (7, 7), 0)
        dy = cv2.GaussianBlur(cv2.resize(dy_s, (w, h), interpolation=cv2.INTER_CUBIC), (7, 7), 0)
        my, mx = np.mgrid[0:h, 0:w].astype(np.float32)
        return cv2.remap(frame, mx - dx, my - dy, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101), (mx - dx, my - dy)

    # ── Face Region Mask (mesh-based) ─────────────────────────────
    def _compute_face_mask(self, lms, h, w, bg_protect, roi_offset=None):
        """Create a feathered mask from the face mesh oval contour.
        Uses ordered polygon fill (not convex hull) for a precise skin boundary.
        roi_offset: (ox, oy) to translate landmarks into ROI coordinates."""
        oval_pts = []
        for idx in FACE_OVAL:
            if idx < len(lms):
                pt = lms[idx].astype(np.float64).copy()
                if roi_offset is not None:
                    pt -= roi_offset
                oval_pts.append(pt)
        if len(oval_pts) < 10:
            return None

        oval_pts = np.array(oval_pts, dtype=np.float64)
        centroid = np.mean(oval_pts, axis=0)
        face_size = max(np.ptp(oval_pts[:, 0]), np.ptp(oval_pts[:, 1]))

        # Expand contour outward from centroid
        # Lower bg_protect = more expansion (larger blended region)
        expand_ratio = 0.12 + (1.0 - bg_protect / 100.0) * 0.25
        expanded = []
        for pt in oval_pts:
            vec = pt - centroid
            expanded.append(pt + vec * expand_ratio)
        expanded = np.array(expanded, dtype=np.int32)

        # Fill ordered polygon (follows actual face contour, not convex hull)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(mask, [expanded], 1.0, cv2.LINE_AA)

        # Feather edges - kernel scales with face size
        blur_radius = int(face_size * (0.06 + bg_protect / 100.0 * 0.16))
        blur_radius = max(5, blur_radius) | 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        return mask

    # ── ROI Computation ────────────────────────────────────────────
    def _compute_roi(self, lms, h, w, pad_ratio=0.35):
        """Compute face bounding box with padding, clamped to frame bounds."""
        face_pts = lms[:min(468, len(lms))]
        x_min, y_min = np.min(face_pts, axis=0)
        x_max, y_max = np.max(face_pts, axis=0)
        face_w, face_h = x_max - x_min, y_max - y_min
        pad = max(face_w, face_h) * pad_ratio

        rx1 = max(0, int(x_min - pad))
        ry1 = max(0, int(y_min - pad))
        rx2 = min(w, int(x_max + pad))
        ry2 = min(h, int(y_max + pad))

        if rx2 - rx1 < 64 or ry2 - ry1 < 64:
            return 0, 0, w, h
        return rx1, ry1, rx2, ry2

    # ── ROI-local Control Points ───────────────────────────────────
    def _translate_control_points_to_roi(self, src, tgt, roi):
        """Translate control points into ROI coordinates, filter out-of-bounds,
        and add ROI edge anchors."""
        rx1, ry1, rx2, ry2 = roi
        rw, rh = rx2 - rx1, ry2 - ry1
        offset = np.array([rx1, ry1], dtype=np.float64)

        src_roi = src - offset
        tgt_roi = tgt - offset

        # Keep points within padded ROI bounds
        margin = 30
        valid = ((src_roi[:, 0] > -margin) & (src_roi[:, 0] < rw + margin) &
                 (src_roi[:, 1] > -margin) & (src_roi[:, 1] < rh + margin))
        src_roi = src_roi[valid]
        tgt_roi = tgt_roi[valid]

        # Add ROI edge anchors (identity - keep edges pinned)
        edge_pts = np.array([
            [0, 0], [rw-1, 0], [0, rh-1], [rw-1, rh-1],
            [rw//2, 0], [rw//2, rh-1], [0, rh//2], [rw-1, rh//2],
            [rw//4, 0], [3*rw//4, 0], [rw//4, rh-1], [3*rw//4, rh-1],
            [0, rh//4], [0, 3*rh//4], [rw-1, rh//4], [rw-1, 3*rh//4],
        ], dtype=np.float64)
        src_roi = np.vstack([src_roi, edge_pts])
        tgt_roi = np.vstack([tgt_roi, edge_pts])

        return src_roi, tgt_roi

    # ── Seamless Composite ─────────────────────────────────────────
    def _composite_roi(self, frame, warped_roi, mask_roi, roi):
        """Blend warped ROI back into frame using seamless clone or alpha fallback."""
        rx1, ry1, rx2, ry2 = roi

        # Convert mask to uint8 for seamlessClone
        mask_u8 = (mask_roi * 255).astype(np.uint8)

        # Clear mask borders (seamlessClone requirement)
        border = 3
        mask_u8[:border, :] = 0; mask_u8[-border:, :] = 0
        mask_u8[:, :border] = 0; mask_u8[:, -border:] = 0

        # Face center in frame coordinates
        cx = int((rx1 + rx2) / 2)
        cy = int((ry1 + ry2) / 2)

        # Place warped ROI into full-frame canvas
        warped_full = frame.copy()
        warped_full[ry1:ry2, rx1:rx2] = warped_roi

        mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_full[ry1:ry2, rx1:rx2] = mask_u8

        if np.sum(mask_full) < 100:
            return warped_full

        try:
            return cv2.seamlessClone(warped_full, frame, mask_full,
                                      (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            # Fallback to alpha blend
            mask_f = mask_full.astype(np.float32) / 255.0
            mask_3ch = mask_f[:, :, np.newaxis]
            blended = (warped_full.astype(np.float32) * mask_3ch +
                        frame.astype(np.float32) * (1.0 - mask_3ch))
            return np.clip(blended, 0, 255).astype(np.uint8)

    # ── Main Warp (multi-face) ──────────────────────────────────
    def warp(self, frame, faces, params):
        """Warp all detected faces in frame. faces = list of (landmarks, conf)."""
        result = frame.copy()

        # ── Optical flow: decide keyframe at frame level (not per-face) ──
        has_any_warp = False
        for i, face in enumerate(faces):
            lms, conf, blendshapes = face_components(face)
            fp = self._params_for_face(params, i)
            h, w = result.shape[:2]
            src, tgt = self._compute_control_points(lms, fp, h, w, blendshapes)
            if len(src) >= 4 and np.max(np.linalg.norm(tgt - src, axis=1)) >= 0.5:
                has_any_warp = True
                break

        use_flow_this_frame = False
        if has_any_warp and self.flow_prop is not None:
            if not self.flow_prop.should_compute_full():
                gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                propagated = self.flow_prop.propagate(gray, result)
                if propagated is not None:
                    result = propagated
                    use_flow_this_frame = True

        pre_warp = result.copy() if (has_any_warp and not use_flow_this_frame and self.flow_prop is not None) else None

        for i, face in enumerate(faces):
            lms, conf, blendshapes = face_components(face)
            fp = self._params_for_face(params, i)
            h, w = result.shape[:2]
            cache = self._caches[min(i, len(self._caches)-1)]

            # Clear frame-dependent cache entries (prevent stale parsing/masks from prior frames)
            # TPS grid caches (roi_key, roi_grid, roi_maps, key, grid, maps) are kept since
            # they depend on control-point geometry which is invalidated by _cache_key()
            for _ck in ('parsing_roi', 'skin_mask_roi', 'roi_bounds'):
                cache.pop(_ck, None)

            # Compute warp control points
            src, tgt = self._compute_control_points(lms, fp, h, w, blendshapes)
            has_warp = len(src) >= 4 and np.max(np.linalg.norm(tgt - src, axis=1)) >= 0.5

            # Apply geometric warp if needed (full TPS on keyframes)
            if has_warp and not use_flow_this_frame:
                try:
                    bg_protect = fp.get('bg_protect', 0)
                    if bg_protect > 0:
                        result = self._warp_with_roi(result, lms, src, tgt, fp, cache, bg_protect, face_idx=i)
                    else:
                        result = self._warp_full_frame(result, src, tgt, fp, cache)
                except Exception as e:
                    print(f"Warp error face {i}: {e}")

            # ── Post-warp effects (parsing-based beautification) ──
            skin_smooth = fp.get('skin_smooth', 0)
            teeth_whiten = fp.get('teeth_whiten', 0)
            eye_sharpen = fp.get('eye_sharpen', 0)
            skin_tone_even = fp.get('skin_tone_even', 0)
            lip_color = fp.get('lip_color', 0)
            under_eye = fp.get('under_eye', 0)
            hair_hue = fp.get('hair_hue', 0)
            hair_saturation = fp.get('hair_saturation', 0)
            hair_density = fp.get('hair_density', 0)
            blush = fp.get('blush', 0)
            lip_gloss = fp.get('lip_gloss', 0)
            eye_shadow = fp.get('eye_shadow', 0)
            needs_parsing = (skin_smooth > 0 or teeth_whiten > 0 or eye_sharpen > 0
                             or skin_tone_even > 0 or lip_color > 0 or under_eye > 0
                             or hair_hue > 0 or hair_saturation > 0 or hair_density > 0
                             or blush > 0 or lip_gloss > 0 or eye_shadow > 0) and self.parser is not None
            if needs_parsing:
                try:
                    # Get or compute face ROI and parsing
                    roi_bounds = cache.get('roi_bounds')
                    if roi_bounds is None:
                        roi_bounds = self._compute_roi(lms, h, w, pad_ratio=0.30)
                        cache['roi_bounds'] = roi_bounds
                    rx1, ry1, rx2, ry2 = roi_bounds
                    roi_region = result[ry1:ry2, rx1:rx2]

                    # Get or compute parsing for this ROI
                    cached_parsing = cache.get('parsing_roi')
                    if cached_parsing is not None and cached_parsing.shape == roi_region.shape[:2]:
                        parsing = cached_parsing
                    else:
                        parsing = self.parser.parse(roi_region)
                        cache['parsing_roi'] = parsing

                    # Skin smoothing
                    if skin_smooth > 0:
                        skin_mask = cache.get('skin_mask_roi')
                        if skin_mask is None or skin_mask.shape != roi_region.shape[:2]:
                            skin_mask = self.parser.get_mask(parsing, SKIN_SMOOTH_LABELS, feather=5)
                            if self.mask_smoother is not None:
                                skin_mask = self.mask_smoother.smooth(skin_mask, face_idx=i)
                            cache['skin_mask_roi'] = skin_mask
                        roi_region = apply_skin_smoothing(roi_region, skin_mask, skin_smooth)

                    # Skin tone evening
                    if skin_tone_even > 0:
                        skin_mask = cache.get('skin_mask_roi')
                        if skin_mask is None or skin_mask.shape != roi_region.shape[:2]:
                            skin_mask = self.parser.get_mask(parsing, SKIN_SMOOTH_LABELS, feather=5)
                            cache['skin_mask_roi'] = skin_mask
                        roi_region = apply_skin_tone_even(roi_region, skin_mask, skin_tone_even)

                    # Teeth whitening
                    if teeth_whiten > 0:
                        roi_region = apply_teeth_whitening(roi_region, parsing, teeth_whiten)

                    # Eye sharpening
                    if eye_sharpen > 0:
                        roi_region = apply_eye_sharpen(roi_region, parsing, eye_sharpen)

                    # Lip color enhancement
                    if lip_color > 0:
                        roi_region = apply_lip_color(roi_region, parsing, lip_color)

                    if under_eye > 0:
                        under_eye_mask = make_under_eye_mask(lms, roi_bounds, roi_region.shape[:2], parsing)
                        roi_region = apply_skin_smoothing(roi_region, under_eye_mask, under_eye)

                    if hair_hue > 0 or hair_saturation > 0 or hair_density > 0:
                        roi_region = apply_hair_controls(roi_region, parsing, hair_hue, hair_saturation, hair_density)

                    if blush > 0:
                        blush_mask = make_blush_mask(lms, roi_bounds, roi_region.shape[:2])
                        roi_region = apply_color_overlay(roi_region, blush_mask, (255, 130, 150), blush * 0.28)

                    if lip_gloss > 0:
                        roi_region = apply_lip_gloss(roi_region, parsing, lip_gloss)

                    if eye_shadow > 0:
                        eye_shadow_mask = make_eye_shadow_mask(parsing)
                        roi_region = apply_color_overlay(roi_region, eye_shadow_mask, (112, 92, 160), eye_shadow * 0.22)

                    result[ry1:ry2, rx1:rx2] = roi_region
                except Exception as e:
                    print(f"Post-warp effects error face {i}: {e}")

        # Store keyframe for optical flow after ALL faces are warped
        if pre_warp is not None and has_any_warp and not use_flow_this_frame:
            gray = cv2.cvtColor(pre_warp, cv2.COLOR_RGB2GRAY)
            self.flow_prop.store_keyframe(gray, result, pre_warp)

        return self.apply_post_stage(frame, result, faces, params)

    def _warp_full_frame(self, frame, src, tgt, params, cache):
        """Legacy full-frame warp (bg_protect=0)."""
        h, w = frame.shape[:2]
        ck = self._cache_key(src, tgt)

        if cache.get('key') == ck and cache.get('dims') == (h, w):
            if self.use_gpu and 'grid' in cache:
                with torch.no_grad():
                    ft = torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1).unsqueeze(0).float()
                    wt = TF.grid_sample(ft, cache['grid'], mode='bicubic',
                                         padding_mode='reflection', align_corners=False)
                    return wt.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            elif 'maps' in cache:
                return cv2.remap(frame, cache['maps'][0], cache['maps'][1],
                                 cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
        else:
            if self.use_gpu:
                result, grid = self._warp_gpu(frame, src, tgt, params)
                cache.update({'key': ck, 'dims': (h, w), 'grid': grid})
                return result
            else:
                result, maps = self._warp_cpu(frame, src, tgt, params)
                cache.update({'key': ck, 'dims': (h, w), 'maps': maps})
                return result
        return frame

    def _warp_with_roi(self, frame, lms, src, tgt, params, cache, bg_protect, face_idx=0):
        """ROI-based warp: crop face region, warp only that, seamless-clone back."""
        h, w = frame.shape[:2]

        # 1. Compute face ROI with padding
        roi = self._compute_roi(lms, h, w, pad_ratio=0.40)
        rx1, ry1, rx2, ry2 = roi
        rh, rw = ry2 - ry1, rx2 - rx1

        # 2. Crop ROI from current result
        roi_crop = frame[ry1:ry2, rx1:rx2].copy()

        # 3. Translate control points to ROI-local coordinates
        src_roi, tgt_roi = self._translate_control_points_to_roi(src, tgt, roi)
        if len(src_roi) < 4:
            return frame

        # 4. Warp the ROI (separate cache namespace)
        roi_ck = self._cache_key(src_roi, tgt_roi)
        roi_cache_key = roi_ck + bytes(f'{rw}x{rh}'.encode())

        if cache.get('roi_key') == roi_cache_key and cache.get('roi_dims') == (rh, rw):
            if self.use_gpu and 'roi_grid' in cache:
                with torch.no_grad():
                    ft = torch.from_numpy(roi_crop).to(DEVICE).permute(2, 0, 1).unsqueeze(0).float()
                    wt = TF.grid_sample(ft, cache['roi_grid'], mode='bicubic',
                                         padding_mode='reflection', align_corners=False)
                    warped_roi = wt.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
            elif 'roi_maps' in cache:
                warped_roi = cv2.remap(roi_crop, cache['roi_maps'][0], cache['roi_maps'][1],
                                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            else:
                return frame
        else:
            if self.use_gpu:
                warped_roi, grid = self._warp_gpu(roi_crop, src_roi, tgt_roi, params)
                cache.update({'roi_key': roi_cache_key, 'roi_dims': (rh, rw), 'roi_grid': grid})
            else:
                warped_roi, maps = self._warp_cpu(roi_crop, src_roi, tgt_roi, params)
                cache.update({'roi_key': roi_cache_key, 'roi_dims': (rh, rw), 'roi_maps': maps})

        # 5. Compute face mask - prefer BiSeNet parsing, fallback to landmark polygon
        mask_roi = None
        if self.parser is not None:
            try:
                # Parse the ORIGINAL roi_crop (pre-warp) for accurate segmentation
                parse_key = (lms[:10] / 8.0).astype(np.int16).tobytes()  # cheap hash
                parsing = self.parser.parse(roi_crop, cache_key=parse_key)
                mask_roi = self.parser.get_mask(parsing, FACE_MASK_LABELS,
                                                feather=int(15 + bg_protect * 0.2))
                # Apply temporal mask smoothing for video
                if self.mask_smoother is not None:
                    mask_roi = self.mask_smoother.smooth(mask_roi, face_idx=face_idx)
                # Cache parsing and skin mask for post-warp effects
                cache['parsing_roi'] = parsing
                cache['skin_mask_roi'] = self.parser.get_mask(parsing, SKIN_SMOOTH_LABELS, feather=5)
                cache['roi_bounds'] = roi
            except Exception as e:
                print(f"Parsing fallback: {e}")
                mask_roi = None

        if mask_roi is None:
            # Landmark polygon fallback
            oval_indices = [idx for idx in FACE_OVAL if idx < len(lms)]
            oval_hash = (lms[oval_indices] / 4.0).astype(np.int16).tobytes()
            mask_cache_key = (int(bg_protect), rw, rh, oval_hash)
            if cache.get('mask_key') == mask_cache_key and 'mask' in cache:
                mask_roi = cache['mask']
            else:
                offset = np.array([rx1, ry1], dtype=np.float64)
                mask_roi = self._compute_face_mask(lms, rh, rw, bg_protect, roi_offset=offset)
                if mask_roi is not None:
                    cache.update({'mask_key': mask_cache_key, 'mask': mask_roi})

        if mask_roi is None:
            result = frame.copy()
            result[ry1:ry2, rx1:rx2] = warped_roi
            return result

        matting_refine = params.get('matting_refine', 0)
        if matting_refine > 0:
            try:
                if self.matter is None:
                    self.matter = MattingRefinementEngine(self.onnx_provider)
                matte_key = (lms[:10] / 8.0).astype(np.int16).tobytes()
                matte = self.matter.matte(roi_crop, cache_key=matte_key)
                strength = min(max(matting_refine / 100.0, 0.0), 1.0)
                mask_roi = mask_roi * (1.0 - strength) + (mask_roi * matte) * strength
                mask_roi = np.clip(mask_roi, 0.0, 1.0)
            except Exception as e:
                print(f"Matting refinement fallback: {e}")

        # 6. Composite via seamless clone
        return self._composite_roi(frame, warped_roi, mask_roi, roi)

    def _get_face_restorer(self):
        if self.face_restorer is None:
            self.face_restorer = FaceRestorationPostStage(self.onnx_provider)
        return self.face_restorer

    def _get_upscaler(self):
        if self.upscaler is None:
            self.upscaler = UpscalePostStage(self.onnx_provider)
        return self.upscaler

    def _face_restore_mask(self, roi_rgb, lms, roi_bounds, face_idx=0):
        rh, rw = roi_rgb.shape[:2]
        if self.parser is not None:
            try:
                parsing = self.parser.parse(roi_rgb)
                mask = self.parser.get_mask(parsing, FACE_MASK_LABELS, feather=17)
                if self.mask_smoother is not None:
                    mask = self.mask_smoother.smooth(mask, face_idx=face_idx)
                return np.clip(mask, 0.0, 1.0)
            except Exception as e:
                print(f"Post-stage parsing fallback: {e}")
        offset = np.array([roi_bounds[0], roi_bounds[1]], dtype=np.float64)
        mask = self._compute_face_mask(lms, rh, rw, 70, roi_offset=offset)
        if mask is None:
            mask = np.ones((rh, rw), dtype=np.float32)
        return np.clip(mask, 0.0, 1.0)

    def _apply_face_restoration_post_stage(self, frame, faces, params):
        if not faces:
            return frame
        strength = max(0.0, min(1.0, params.get("post_stage_strength", 50) / 100.0))
        fidelity = max(0.0, min(1.0, params.get("post_stage_fidelity", 70) / 100.0))
        # Higher fidelity keeps more original pixels; lower fidelity lets GFPGAN dominate.
        blend_strength = strength * (1.0 - fidelity * 0.75)
        if blend_strength <= 0.01:
            return frame
        restorer = self._get_face_restorer()
        result = frame.copy()
        h, w = result.shape[:2]
        for i, face in enumerate(faces):
            lms, _conf, _blendshapes = face_components(face)
            try:
                roi_bounds = self._compute_roi(lms, h, w, pad_ratio=0.36)
                rx1, ry1, rx2, ry2 = roi_bounds
                roi_rgb = result[ry1:ry2, rx1:rx2].copy()
                if roi_rgb.size == 0:
                    continue
                restored = restorer.restore(roi_rgb)
                if restored.shape[:2] != roi_rgb.shape[:2]:
                    restored = cv2.resize(restored, (roi_rgb.shape[1], roi_rgb.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
                mask = self._face_restore_mask(roi_rgb, lms, roi_bounds, i)
                alpha = (mask * blend_strength)[:, :, np.newaxis]
                blended = restored.astype(np.float32) * alpha + roi_rgb.astype(np.float32) * (1.0 - alpha)
                result[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"Face restoration post-stage error face {i}: {e}")
        return result

    def apply_post_stage(self, original_frame, processed_frame, faces, params):
        model_key = post_stage_model_key((params or {}).get("post_stage_model"))
        if model_key == POST_STAGE_OFF:
            return processed_frame
        result = processed_frame
        try:
            if "gfpgan_1.4" in post_stage_model_keys(model_key):
                result = self._apply_face_restoration_post_stage(result, faces, params)
            if "real_esrgan_x2" in post_stage_model_keys(model_key):
                tile = int((params or {}).get("post_stage_tile", 512) or 512)
                result = self._get_upscaler().upscale(result, tile)
        except Exception as e:
            log_render_event("post_stage_failed", str(e), {
                "post_stage_model": model_key,
                "original_shape": list(original_frame.shape[:2]) if original_frame is not None else None,
                "processed_shape": list(processed_frame.shape[:2]) if processed_frame is not None else None,
            }, traceback.format_exc())
            print(f"Post-stage failed: {e}")
        return result

    def warp_single_image(self, frame_rgb, params):
        """Convenience for single-image processing."""
        faces = self.detect(frame_rgb)
        if not faces:
            result = self.apply_post_stage(frame_rgb, frame_rgb.copy(), [], params)
            return result, []
        return self.warp(frame_rgb, faces, params), faces

    def close(self):
        self.landmarker.close()

# ═══════════════════════════════════════════════════════════════════════════
# PRESET MANAGER
# ═══════════════════════════════════════════════════════════════════════════
class PresetManager:
    @staticmethod
    def list_custom():
        presets = {}
        for f in glob.glob(os.path.join(PRESETS_DIR, '*.json')):
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                presets[Path(f).stem] = data
            except Exception:
                pass
        return presets

    @staticmethod
    def save(name, params):
        safe_name = os.path.basename(name).replace('..', '_').strip('. ')
        if not safe_name:
            return None
        path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        return path

    @staticmethod
    def delete(name):
        safe_name = os.path.basename(name).replace('..', '_').strip('. ')
        if not safe_name:
            return False
        path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    @staticmethod
    def export_all(filepath):
        data = {'built_in': BUILT_IN_PRESETS, 'custom': PresetManager.list_custom()}
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def import_presets(filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        imported = 0
        for name, params in data.get('custom', {}).items():
            PresetManager.save(name, params)
            imported += 1
        return imported


# ═══════════════════════════════════════════════════════════════════════════
# UNDO/REDO HISTORY
# ═══════════════════════════════════════════════════════════════════════════
class ParamHistory:
    def __init__(self, max_size=50):
        self._stack = []
        self._pos = -1
        self._max = max_size
        self._frozen = False

    def push(self, params):
        if self._frozen:
            return
        # Trim future states
        self._stack = self._stack[:self._pos + 1]
        self._stack.append(params.copy())
        if len(self._stack) > self._max:
            self._stack.pop(0)
        self._pos = len(self._stack) - 1

    def undo(self):
        if self._pos > 0:
            self._pos -= 1
            return self._stack[self._pos].copy()
        return None

    def redo(self):
        if self._pos < len(self._stack) - 1:
            self._pos += 1
            return self._stack[self._pos].copy()
        return None

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @property
    def can_undo(self):
        return self._pos > 0

    @property
    def can_redo(self):
        return self._pos < len(self._stack) - 1


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO THREAD (real-time preview)
# ═══════════════════════════════════════════════════════════════════════════

def draw_landmarks(frame, faces, show_confidence=False, show_landmarks=True):
    ov = frame.copy()
    for face_idx, face in enumerate(faces):
        lms, conf, _blendshapes = face_components(face)
        if show_landmarks:
            for group, col in [(LEFT_JAW,(166,227,161)),(RIGHT_JAW,(166,227,161)),
                                (LEFT_CHEEK,(249,226,175)),(RIGHT_CHEEK,(249,226,175)),
                                (CHIN,(243,139,168)),(NOSE_TIP,(137,180,250))]:
                pts = [lms[i].astype(int) for i in group if i < len(lms)]
                for p in pts: cv2.circle(ov, tuple(p), 2, col, -1, cv2.LINE_AA)
                for i in range(len(pts)-1):
                    cv2.line(ov, tuple(pts[i]), tuple(pts[i+1]), col, 1, cv2.LINE_AA)
        if show_confidence:
            cx, cy = int(np.mean(lms[:, 0])), int(np.min(lms[:, 1])) - 15
            color = (166,227,161) if conf > 0.7 else (249,226,175) if conf > 0.4 else (243,139,168)
            text = f"Face {face_idx+1}: {conf:.0%}"
            cv2.putText(ov, text, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cv2.addWeighted(ov, 0.7, frame, 0.3, 0)

# ═══════════════════════════════════════════════════════════════════════════
# TOAST NOTIFICATION
# ═══════════════════════════════════════════════════════════════════════════
