#!/usr/bin/env python3
"""
FaceSlim v1.4.0 - AI Face Slimming & Reshaping Suite
GPU-accelerated face reshaping with MediaPipe 478-landmark detection,
PyTorch TPS warping, real-time preview, batch processing, CLI mode,
image+video support, preset management, and before/after GIF export.

Phase 1: BiSeNet face parsing, skin smoothing
Phase 2: Temporal mask smoothing, optical flow propagation,
         eye enlargement, teeth whitening
Phase 3: Lip plumping, eye sharpening, skin tone evening, lip color

Turnkey: auto-installs all dependencies and downloads models on first run.
"""

import sys, os, subprocess, time, json, math, traceback, urllib.request, argparse, glob
from collections import deque
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════
def _bootstrap():
    if sys.version_info < (3, 9):
        print("Python 3.9+ required"); sys.exit(1)
    try:
        import pip
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'ensurepip', '--default-pip'])
    required = {
        'PyQt5': 'PyQt5', 'cv2': 'opencv-python', 'mediapipe': 'mediapipe',
        'numpy': 'numpy', 'scipy': 'scipy', 'PIL': 'Pillow',
        'onnxruntime': 'onnxruntime',
    }
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"  Installing {pkg}...")
            for flags in [[], ['--user'], ['--break-system-packages']]:
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install', pkg, '-q'] + flags,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except subprocess.CalledProcessError:
                    continue

_bootstrap()

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.interpolate import RBFInterpolator
from PIL import Image as PILImage
import onnxruntime as ort

# -- Qt Imports (must precede QThread subclasses) --
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QGroupBox,
    QProgressBar, QCheckBox, QGridLayout, QSizePolicy,
    QTabWidget, QScrollArea, QInputDialog, QSpinBox, QComboBox,
    QFrame
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSettings, QTimer
)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QCursor

# -- Optional GPU Acceleration (PyTorch) --
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
        print(f"  PyTorch loaded (CPU only)")
except Exception:
    HAS_TORCH = False
    USE_GPU = False
    DEVICE = None
    GPU_NAME = "CPU"
    print(f"  PyTorch not available - using CPU mode (install torch for GPU acceleration)")

VERSION = "1.4.1"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
CONFIG_DIR = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), '.faceslim')
PRESETS_DIR = os.path.join(CONFIG_DIR, 'presets')
os.makedirs(PRESETS_DIR, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v'}

# ── Crash Handler ───────────────────────────────────────────────────────
def exception_handler(exc_type, exc_value, exc_tb):
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(os.path.join(APP_DIR, 'crash.log'), 'a') as f:
            f.write(f"\n{'='*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n{msg}\n")
    except Exception:
        pass
    print(f"CRASH: {msg}")
    sys.exit(1)

sys.excepthook = exception_handler

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return True
    print("Downloading face landmarker model (~3.7 MB)...")
    tmp_path = MODEL_PATH + '.tmp'
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        if os.path.getsize(tmp_path) > 1_000_000:
            os.replace(tmp_path, MODEL_PATH)
            return True
        else:
            os.remove(tmp_path)
            print("Downloaded file too small - may be corrupt")
            return False
    except Exception as e:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        print(f"Failed to download model: {e}")
        return False

# ── BiSeNet Face Parsing Model ─────────────────────────────────────────
BISENET_PATH = os.path.join(APP_DIR, "bisenet_face_parsing.onnx")
BISENET_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
# CelebAMask-HQ label indices
PARSE_BACKGROUND = 0
PARSE_SKIN = 1
PARSE_L_BROW = 2
PARSE_R_BROW = 3
PARSE_L_EYE = 4
PARSE_R_EYE = 5
PARSE_GLASSES = 6
PARSE_L_EAR = 7
PARSE_R_EAR = 8
PARSE_EARRING = 9
PARSE_NOSE = 10
PARSE_MOUTH = 11
PARSE_U_LIP = 12
PARSE_L_LIP = 13
PARSE_NECK = 14
PARSE_NECKLACE = 15
PARSE_CLOTH = 16
PARSE_HAIR = 17
PARSE_HAT = 18

# Labels that constitute the "face surface" for warp masking
FACE_MASK_LABELS = {PARSE_SKIN, PARSE_L_BROW, PARSE_R_BROW, PARSE_L_EYE, PARSE_R_EYE,
                    PARSE_GLASSES, PARSE_NOSE, PARSE_MOUTH, PARSE_U_LIP, PARSE_L_LIP}
# Labels for skin smoothing (skin only, not eyes/lips/brows)
SKIN_SMOOTH_LABELS = {PARSE_SKIN}

HAS_BISENET = False

def ensure_parsing_model():
    global HAS_BISENET
    if os.path.exists(BISENET_PATH) and os.path.getsize(BISENET_PATH) > 1_000_000:
        HAS_BISENET = True
        return True
    print("Downloading BiSeNet face parsing model (~50 MB)...")
    tmp = BISENET_PATH + '.tmp'
    try:
        urllib.request.urlretrieve(BISENET_URL, tmp)
        if os.path.getsize(tmp) > 1_000_000:
            os.replace(tmp, BISENET_PATH)
            HAS_BISENET = True
            print("  Face parsing model ready")
            return True
        else:
            os.remove(tmp)
            print("  Downloaded file too small")
            return False
    except Exception as e:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        print(f"  Face parsing download failed: {e} (will use landmark fallback)")
        return False

# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════════════
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

# MediaPipe face mesh oval - ordered contour tracing the face boundary
# Used for precise face-region masking (much tighter than convex hull)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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
                  'skin_tone_even': 0, 'lip_color': 0}

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
            self.t_prev = t or time.time()
            return x.copy()
        if t and self.t_prev:
            dt = t - self.t_prev
            if dt > 0: self.freq = 1.0 / dt
        self.t_prev = t or time.time()
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

    def __init__(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        self.session = ort.InferenceSession(BISENET_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._cache = {}  # {(h, w, key): parsing_map}
        print(f"  Face parsing: BiSeNet loaded ({providers[0]})")

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


def apply_teeth_whitening(frame, parsing, strength):
    """Whiten teeth within mouth interior mask.
    Uses HSV: increase V, decrease S in mouth interior only (not lips)."""
    if strength <= 0 or parsing is None:
        return frame

    # Only mouth interior (teeth/tongue visible area) - NOT lips
    mask = (parsing == PARSE_MOUTH).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    if mask.max() < 0.01:
        return frame

    s = strength / 100.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase value (brightness), decrease saturation in teeth region
    mask_3 = mask[:, :, np.newaxis]
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - s * 0.5 * mask)   # reduce saturation (yellowness)
    hsv[:, :, 2] = hsv[:, :, 2] + s * 35.0 * mask            # increase brightness
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
    def __init__(self, mode='video', max_faces=1):
        rm = vision.RunningMode.VIDEO if mode == 'video' else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=rm, num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
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
        if HAS_BISENET:
            try:
                self.parser = FaceParsingEngine()
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
        """Returns list of (landmarks, confidence) tuples for each detected face."""
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
            faces.append((pts, float(conf)))
        return faces

    def _compute_control_points(self, lms, params, h, w):
        center = np.mean(lms[[i for i in FACE_CENTER if i < len(lms)]], axis=0)
        src, tgt = [], []
        jaw_s     = params.get('jaw', 0) / 100.0
        cheek_s   = params.get('cheeks', 0) / 100.0
        chin_s    = params.get('chin', 0) / 100.0
        width_s   = params.get('face_width', 0) / 100.0
        forehead_s = params.get('forehead', 0) / 100.0
        nose_s    = params.get('nose', 0) / 100.0

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
        bg_protect = params.get('bg_protect', 0)
        skin_smooth = params.get('skin_smooth', 0)
        teeth_whiten = params.get('teeth_whiten', 0)

        # ── Optical flow: decide keyframe at frame level (not per-face) ──
        has_any_warp = False
        for lms, conf in faces:
            h, w = result.shape[:2]
            src, tgt = self._compute_control_points(lms, params, h, w)
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

        for i, (lms, conf) in enumerate(faces):
            h, w = result.shape[:2]
            cache = self._caches[min(i, len(self._caches)-1)]

            # Clear frame-dependent cache entries (prevent stale parsing/masks from prior frames)
            # TPS grid caches (roi_key, roi_grid, roi_maps, key, grid, maps) are kept since
            # they depend on control-point geometry which is invalidated by _cache_key()
            for _ck in ('parsing_roi', 'skin_mask_roi', 'roi_bounds'):
                cache.pop(_ck, None)

            # Compute warp control points
            src, tgt = self._compute_control_points(lms, params, h, w)
            has_warp = len(src) >= 4 and np.max(np.linalg.norm(tgt - src, axis=1)) >= 0.5

            # Apply geometric warp if needed (full TPS on keyframes)
            if has_warp and not use_flow_this_frame:
                try:
                    if bg_protect > 0:
                        result = self._warp_with_roi(result, lms, src, tgt, params, cache, bg_protect, face_idx=i)
                    else:
                        result = self._warp_full_frame(result, src, tgt, params, cache)
                except Exception as e:
                    print(f"Warp error face {i}: {e}")

            # ── Post-warp effects (parsing-based beautification) ──
            eye_sharpen = params.get('eye_sharpen', 0)
            skin_tone_even = params.get('skin_tone_even', 0)
            lip_color = params.get('lip_color', 0)
            needs_parsing = (skin_smooth > 0 or teeth_whiten > 0 or eye_sharpen > 0
                             or skin_tone_even > 0 or lip_color > 0) and self.parser is not None
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

                    result[ry1:ry2, rx1:rx2] = roi_region
                except Exception as e:
                    print(f"Post-warp effects error face {i}: {e}")

        # Store keyframe for optical flow after ALL faces are warped
        if pre_warp is not None and has_any_warp and not use_flow_this_frame:
            gray = cv2.cvtColor(pre_warp, cv2.COLOR_RGB2GRAY)
            self.flow_prop.store_keyframe(gray, result, pre_warp)

        return result

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

        # 6. Composite via seamless clone
        return self._composite_roi(frame, warped_roi, mask_roi, roi)

    def warp_single_image(self, frame_rgb, params):
        """Convenience for single-image processing."""
        faces = self.detect(frame_rgb)
        if not faces:
            return frame_rgb, []
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
                with open(f) as fp:
                    data = json.load(fp)
                presets[Path(f).stem] = data
            except Exception:
                pass
        return presets

    @staticmethod
    def save(name, params):
        # Sanitize name: strip path separators and dangerous chars
        safe_name = os.path.basename(name).replace('..', '_').strip('. ')
        if not safe_name:
            return None
        path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        return path

    @staticmethod
    def delete(name):
        path = os.path.join(PRESETS_DIR, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    @staticmethod
    def export_all(filepath):
        data = {'built_in': BUILT_IN_PRESETS, 'custom': PresetManager.list_custom()}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def import_presets(filepath):
        with open(filepath) as f:
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
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, list)  # orig, proc, faces
    fps_update = pyqtSignal(float)
    status_update = pyqtSignal(str)
    error_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = self.show_landmarks = self.show_confidence = False
        self.paused = False
        self.source = None
        self.params = DEFAULT_PARAMS.copy()
        self.max_faces = 1
        self.preview_scale = 1.0  # 1.0 = full res, 0.5 = half

    def set_source(self, s): self.source = s
    def update_params(self, p): self.params = p.copy()

    def run(self):
        eng = None
        cap = None
        try:
            self.running = True
            eng = FaceWarpEngine('video', self.max_faces)
            # Apply temporal beta
            beta = self.params.get('temporal', 50) / 5000.0
            eng.set_temporal_beta(beta)

            cap = cv2.VideoCapture(0 if self.source is None else self.source)
            if not cap.isOpened():
                self.status_update.emit("Failed to open video source")
                return

            fps_t = deque(maxlen=30)
            while self.running:
                if self.paused: self.msleep(50); continue
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    if self.source is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        eng.close(); eng = FaceWarpEngine('video', self.max_faces)
                        eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)
                        continue
                    break

                try:
                    # Preview scaling
                    if self.preview_scale < 1.0:
                        h, w = frame.shape[:2]
                        nh, nw = int(h * self.preview_scale), int(w * self.preview_scale)
                        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = eng.detect(rgb)
                    has_fx = any(abs(self.params.get(k, 0)) > 0
                                 for k in ['jaw','cheeks','chin','face_width','forehead','nose','eye_enlarge','lip_plump','skin_smooth','teeth_whiten','eye_sharpen','skin_tone_even','lip_color'])

                    # Update temporal beta if changed
                    eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)

                    proc = eng.warp(rgb, faces, self.params) if (faces and has_fx) else rgb.copy()

                    el = time.time() - t0
                    fps_t.append(el)
                    if len(fps_t) > 2:
                        self.fps_update.emit(1.0 / max(sum(fps_t)/len(fps_t), 0.001))

                    self.frame_ready.emit(rgb, proc, faces)
                except Exception as e:
                    print(f"Frame processing error: {e}")

                self.msleep(1)
        except Exception as e:
            self.error_update.emit(f"Video error: {e}")
            print(f"VideoThread fatal: {e}")
        finally:
            if cap is not None:
                cap.release()
            if eng is not None:
                eng.close()

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

    def __init__(self, inp, out, params, max_faces=1):
        super().__init__()
        self.inp, self.out, self.params = inp, out, params
        self.max_faces = max_faces
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            eng = FaceWarpEngine('video', self.max_faces)
            eng.grid_scale = 6
            cap = cv2.VideoCapture(self.inp)
            if not cap.isOpened():
                self.error.emit("Cannot open input"); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.status.emit(f"Exporting {total} frames...")
            has_fx = any(abs(self.params.get(k,0)) > 0
                         for k in ['jaw','cheeks','chin','face_width','forehead','nose','eye_enlarge','lip_plump','skin_smooth','teeth_whiten','eye_sharpen','skin_tone_even','lip_color'])
            t_start = time.time()
            for i in range(total):
                if self.cancelled:
                    self.status.emit("Export cancelled"); break
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = eng.detect(rgb)
                proc = eng.warp(rgb, faces, self.params) if (faces and has_fx) else rgb
                writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                if total > 0:
                    self.progress.emit(int((i+1)/total*100))
                    elapsed = time.time() - t_start
                    if i > 0:
                        eta = (elapsed / (i+1)) * (total - i - 1)
                        mins, secs = divmod(int(eta), 60)
                        self.status.emit(f"Frame {i+1}/{total} | ETA: {mins}m {secs}s")
            cap.release(); writer.release(); eng.close()
            if not self.cancelled:
                self._mux_audio()
                elapsed_total = time.time() - t_start
                mins, secs = divmod(int(elapsed_total), 60)
                self.status.emit(f"Done in {mins}m {secs}s")
                self.finished.emit(self.out)
        except Exception as e:
            self.error.emit(str(e))

    def _mux_audio(self):
        try:
            base, ext = os.path.splitext(self.out)
            tmp = base + '_mux' + ext
            r = subprocess.run(['ffmpeg','-y','-i',self.out,'-i',self.inp,
                '-c:v','copy','-c:a','aac','-map','0:v:0','-map','1:a:0?',
                '-shortest',tmp], capture_output=True, timeout=600)
            if r.returncode == 0 and os.path.exists(tmp):
                os.replace(tmp, self.out)
            elif os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
        except FileNotFoundError:
            pass  # ffmpeg not installed
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING THREAD
# ═══════════════════════════════════════════════════════════════════════════
class BatchThread(QThread):
    progress = pyqtSignal(int, int, str)  # current, total, filename
    finished = pyqtSignal(int, int)       # processed, failed
    error = pyqtSignal(str)

    def __init__(self, files, output_dir, params, max_faces=1):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self.max_faces = max_faces
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        processed = failed = 0
        os.makedirs(self.output_dir, exist_ok=True)

        for i, filepath in enumerate(self.files):
            if self.cancelled: break
            fname = os.path.basename(filepath)
            self.progress.emit(i + 1, len(self.files), fname)
            ext = os.path.splitext(filepath)[1].lower()

            try:
                if ext in IMAGE_EXTS:
                    self._process_image(filepath)
                    processed += 1
                elif ext in VIDEO_EXTS:
                    self._process_video(filepath)
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Batch error {fname}: {e}")
                failed += 1

        self.finished.emit(processed, failed)

    def _process_image(self, filepath):
        eng = FaceWarpEngine('image', self.max_faces)
        img = cv2.imread(filepath)
        if img is None: raise ValueError(f"Cannot read {filepath}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result, _ = eng.warp_single_image(rgb, self.params)
        out_path = os.path.join(self.output_dir,
            os.path.splitext(os.path.basename(filepath))[0] + '_slimmed.png')
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        eng.close()

    def _process_video(self, filepath):
        eng = FaceWarpEngine('video', self.max_faces)
        eng.grid_scale = 6
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): raise ValueError(f"Cannot open {filepath}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.join(self.output_dir,
            os.path.splitext(os.path.basename(filepath))[0] + '_slimmed.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        has_fx = any(abs(self.params.get(k,0)) > 0
                     for k in ['jaw','cheeks','chin','face_width','forehead','nose','eye_enlarge','lip_plump','skin_smooth','teeth_whiten','eye_sharpen','skin_tone_even','lip_color'])
        for _ in range(total):
            if self.cancelled: break
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = eng.detect(rgb)
            proc = eng.warp(rgb, faces, self.params) if (faces and has_fx) else rgb
            writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
        cap.release(); writer.release(); eng.close()


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
def draw_landmarks(frame, faces, show_confidence=False, show_landmarks=True):
    ov = frame.copy()
    for face_idx, (lms, conf) in enumerate(faces):
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
class Toast(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background-color: rgba(49, 50, 68, 230); color: #cdd6f4; "
            "border-radius: 8px; padding: 10px 20px; font-size: 13px; font-weight: bold;")
        self.setFixedHeight(40)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)

    def show_message(self, msg, duration=2000):
        self.setText(msg)
        self.adjustSize()
        self.setMinimumWidth(min(len(msg) * 9 + 40, 500))
        if self.parent():
            pw = self.parent().width()
            self.move((pw - self.width()) // 2, 10)
        self.show()
        self.raise_()
        self._timer.start(duration)

    def _fade_out(self):
        self.hide()


# ═══════════════════════════════════════════════════════════════════════════
# DRAGGABLE A/B COMPARE LABEL
# ═══════════════════════════════════════════════════════════════════════════
class CompareVideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.divider_ratio = 0.5
        self._dragging = False
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True; self._update_ratio(e.pos())
    def mouseMoveEvent(self, e):
        if self._dragging: self._update_ratio(e.pos())
        p = self.parent()
        while p and not isinstance(p, FaceSlimApp): p = p.parent()
        self.setCursor(QCursor(Qt.CursorShape.SplitHCursor if (p and p.comparison_mode) else Qt.CursorShape.ArrowCursor))
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self._dragging = False
    def _update_ratio(self, pos):
        if self.width() > 0:
            self.divider_ratio = max(0.05, min(0.95, pos.x() / self.width()))

    # Drag-and-drop support
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            p = self.parent()
            while p and not isinstance(p, FaceSlimApp): p = p.parent()
            if p: p._handle_drop(path)


# ═══════════════════════════════════════════════════════════════════════════
# STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
QTabWidget::pane { border: 1px solid #45475a; background: #1e1e2e; border-radius: 4px; }
QTabBar::tab { background: #181825; color: #6c7086; padding: 8px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #cdd6f4; border-bottom-color: #89b4fa; }
QTabBar::tab:hover { color: #bac2de; }
QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 1.2em;
    padding: 12px 8px 8px 8px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #89b4fa; }
QPushButton { background-color: #89b4fa; color: #1e1e2e; border: none;
    padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; }
QPushButton:hover { background-color: #74c7ec; }
QPushButton:pressed { background-color: #89dceb; }
QPushButton:disabled { background-color: #45475a; color: #6c7086; }
QPushButton[danger="true"] { background-color: #f38ba8; }
QPushButton[danger="true"]:hover { background-color: #eba0ac; }
QPushButton[success="true"] { background-color: #a6e3a1; color: #1e1e2e; }
QPushButton[success="true"]:hover { background-color: #94e2d5; }
QPushButton[secondary="true"] { background-color: #45475a; color: #cdd6f4; }
QPushButton[secondary="true"]:hover { background-color: #585b70; }
QPushButton:checked { background-color: #cba6f7; color: #1e1e2e; }
QSlider::groove:horizontal { height: 6px; background: #313244; border-radius: 3px; }
QSlider::handle:horizontal { background: #89b4fa; width: 18px; height: 18px; margin: -6px 0; border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #74c7ec; }
QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
QProgressBar { background-color: #313244; border: none; border-radius: 4px;
    text-align: center; color: #cdd6f4; height: 20px; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
QCheckBox { spacing: 8px; color: #cdd6f4; }
QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid #45475a; border-radius: 4px; background: #313244; }
QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
QStatusBar { background-color: #181825; color: #6c7086; font-size: 11px; }
QSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 4px; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 6px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background-color: #1e1e2e; color: #cdd6f4;
    border: 1px solid #45475a; selection-background-color: #89b4fa; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #181825; width: 8px; border: none; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #585b70; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QFrame#separator { background-color: #45475a; max-height: 1px; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════
class FaceSlimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FaceSlim v{VERSION}")
        self.setMinimumSize(1150, 720); self.resize(1360, 800)
        self.setAcceptDrops(True)
        self.settings = QSettings("FaceSlim", "FaceSlim")
        self.video_thread = None
        self.source_path = None
        self.comparison_mode = False
        self.current_original = None
        self.current_processed = None
        self.current_faces = []
        self.history = ParamHistory()
        self.image_mode = False  # True when viewing a single image
        self._image_engine = None  # Cached engine for image mode
        self._image_engine_faces = 0
        self._build_ui()
        self._load_settings()
        self.history.push(self._p())

    # ── Slider Factory ──────────────────────────────────────────
    def _make_slider(self, layout, key, label, max_v=100, default=0, tip=""):
        row = QVBoxLayout(); row.setSpacing(2)
        lr = QHBoxLayout(); lr.addWidget(QLabel(label)); lr.addStretch()
        vl = QLabel(f"{default}%")
        vl.setStyleSheet("color:#f9e2af; font-size:12px; font-weight:bold; min-width:36px;")
        lr.addWidget(vl); row.addLayout(lr)
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, max_v); s.setValue(default)
        if tip: s.setToolTip(tip)
        s.valueChanged.connect(lambda v, k=key, lab=vl: self._on_slider(k, v, lab))
        s.sliderReleased.connect(lambda: self.history.push(self._p()))
        row.addWidget(s); layout.addLayout(row)
        self.sliders[key] = (s, vl)

    def _on_slider(self, key, value, label):
        label.setText(f"{value}%")
        self._push()
        if self.image_mode:
            self._process_current_image()

    # ── Build UI ────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setSpacing(12); root.setContentsMargins(12,12,12,12)

        # ── Left: Video ──
        left = QVBoxLayout(); left.setSpacing(8)
        hdr = QHBoxLayout()
        t = QLabel("FaceSlim")
        t.setStyleSheet("font-size: 20px; font-weight: bold; color: #89b4fa;")
        hdr.addWidget(t)
        self.face_count_label = QLabel("")
        self.face_count_label.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        hdr.addWidget(self.face_count_label)
        hdr.addStretch()
        self.fps_label = QLabel("-- FPS")
        self.fps_label.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold;")
        hdr.addWidget(self.fps_label)
        left.addLayout(hdr)

        self.video_label = CompareVideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            "background-color:#11111b; border:2px solid #313244; border-radius:8px; "
            "color:#6c7086; font-size:14px;")
        self.video_label.setText("Drop a file here, or use the buttons below to start")
        left.addWidget(self.video_label, 1)

        # Toast overlay
        self.toast = Toast(self.video_label)

        # Transport
        tr = QHBoxLayout(); tr.setSpacing(6)
        self.btn_webcam = QPushButton("Webcam"); self.btn_webcam.clicked.connect(self.start_webcam)
        tr.addWidget(self.btn_webcam)
        self.btn_load = QPushButton("Load File"); self.btn_load.setProperty("secondary", True)
        self.btn_load.clicked.connect(self.load_file); tr.addWidget(self.btn_load)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setProperty("danger", True)
        self.btn_stop.clicked.connect(self.stop_video); self.btn_stop.setEnabled(False)
        tr.addWidget(self.btn_stop); tr.addStretch()

        self.btn_undo = QPushButton("Undo"); self.btn_undo.setProperty("secondary", True)
        self.btn_undo.clicked.connect(self._undo); self.btn_undo.setEnabled(False)
        tr.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("Redo"); self.btn_redo.setProperty("secondary", True)
        self.btn_redo.clicked.connect(self._redo); self.btn_redo.setEnabled(False)
        tr.addWidget(self.btn_redo)

        self.btn_cmp = QPushButton("A/B Compare"); self.btn_cmp.setCheckable(True)
        self.btn_cmp.toggled.connect(self._toggle_compare); tr.addWidget(self.btn_cmp)
        left.addLayout(tr)
        root.addLayout(left, 3)

        # ── Right: Tabbed Controls ──
        rw = QWidget(); rw.setMaximumWidth(370); rw.setMinimumWidth(300)
        rl = QVBoxLayout(rw); rl.setSpacing(0); rl.setContentsMargins(0,0,0,0)

        tabs = QTabWidget()
        self.sliders = {}

        # ─── Tab 1: Reshape ───
        tab_reshape = QWidget()
        tab_reshape_scroll = QScrollArea()
        tab_reshape_scroll.setWidget(tab_reshape)
        tab_reshape_scroll.setWidgetResizable(True)
        t1 = QVBoxLayout(tab_reshape); t1.setSpacing(8); t1.setContentsMargins(8,8,8,8)

        g1 = QGroupBox("Face Reshaping"); g1l = QVBoxLayout(g1); g1l.setSpacing(6)
        self._make_slider(g1l, 'jaw', 'Jaw Slimming', tip='Narrows the jawline')
        self._make_slider(g1l, 'cheeks', 'Cheek Slimming', tip='Reduces cheek fullness')
        self._make_slider(g1l, 'chin', 'Chin Reshape', tip='Lifts and narrows the chin')
        self._make_slider(g1l, 'face_width', 'Overall Width', tip='Reduces overall face width')
        self._make_slider(g1l, 'forehead', 'Forehead Slim', tip='Narrows the forehead')
        self._make_slider(g1l, 'nose', 'Nose Slim', tip='Narrows the nose bridge and tip')
        self._make_slider(g1l, 'eye_enlarge', 'Eye Enlarge', tip='Enlarges eyes outward from iris center')
        self._make_slider(g1l, 'lip_plump', 'Lip Plump', tip='Plumps lips outward from lip center')
        t1.addWidget(g1)

        g_beauty = QGroupBox("AI Beauty"); g_bl = QVBoxLayout(g_beauty); g_bl.setSpacing(6)
        self._make_slider(g_bl, 'skin_smooth', 'Skin Smoothing', default=0, tip='Frequency-separation bilateral filter on skin mask')
        self._make_slider(g_bl, 'skin_tone_even', 'Skin Tone Even', default=0, tip='Reduces redness and blotchiness via LAB color correction')
        self._make_slider(g_bl, 'teeth_whiten', 'Teeth Whitening', default=0, tip='HSV brightness boost in mouth region')
        self._make_slider(g_bl, 'eye_sharpen', 'Eye Sharpen', default=0, tip='Unsharp mask on eyes and brows for crisp detail')
        self._make_slider(g_bl, 'lip_color', 'Lip Color', default=0, tip='Boosts lip saturation and warmth')
        # Face parsing status indicator
        parsing_lbl = QLabel("  Face Parsing: " + ("BiSeNet ONNX" if HAS_BISENET else "Landmark Fallback"))
        parsing_lbl.setStyleSheet(f"color: {'#a6e3a1' if HAS_BISENET else '#f9e2af'}; font-size: 10px;")
        g_bl.addWidget(parsing_lbl)
        t1.addWidget(g_beauty)

        g2 = QGroupBox("Quality"); g2l = QVBoxLayout(g2); g2l.setSpacing(6)
        self._make_slider(g2l, 'smoothing', 'Warp Smoothing', default=50, tip='Displacement field smoothness')
        self._make_slider(g2l, 'temporal', 'Temporal Stability', default=50, tip='Landmark jitter reduction (higher=smoother)')
        self._make_slider(g2l, 'bg_protect', 'Background Protection', default=70, tip='Prevents warping background - blends face region only')

        opts_row = QHBoxLayout()
        self.chk_lm = QCheckBox("Landmarks"); self.chk_lm.toggled.connect(self._tog_lm)
        opts_row.addWidget(self.chk_lm)
        self.chk_conf = QCheckBox("Confidence"); self.chk_conf.toggled.connect(self._tog_conf)
        opts_row.addWidget(self.chk_conf)
        g2l.addLayout(opts_row)

        faces_row = QHBoxLayout()
        faces_row.addWidget(QLabel("Max Faces:"))
        self.spin_faces = QSpinBox(); self.spin_faces.setRange(1, 5); self.spin_faces.setValue(1)
        self.spin_faces.valueChanged.connect(self._on_faces_changed)
        faces_row.addWidget(self.spin_faces); faces_row.addStretch()
        faces_row.addWidget(QLabel("Preview Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100%", "75%", "50%"])
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        faces_row.addWidget(self.combo_scale)
        g2l.addLayout(faces_row)
        t1.addWidget(g2)

        t1.addStretch()
        tabs.addTab(tab_reshape_scroll, "Reshape")

        # ─── Tab 2: Presets ───
        tab_presets = QWidget()
        t2 = QVBoxLayout(tab_presets); t2.setSpacing(8); t2.setContentsMargins(8,8,8,8)

        g3 = QGroupBox("Built-in Presets"); g3l = QGridLayout(g3); g3l.setSpacing(6)
        for i, (name, vals) in enumerate(BUILT_IN_PRESETS.items()):
            b = QPushButton(name); b.setProperty("secondary", True)
            b.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            g3l.addWidget(b, i // 2, i % 2)
        btn_reset = QPushButton("Reset All"); btn_reset.setProperty("danger", True)
        btn_reset.clicked.connect(lambda: self._apply_preset(DEFAULT_PARAMS.copy()))
        g3l.addWidget(btn_reset, (len(BUILT_IN_PRESETS)) // 2, (len(BUILT_IN_PRESETS)) % 2)
        t2.addWidget(g3)

        g4 = QGroupBox("Custom Presets"); g4l = QVBoxLayout(g4); g4l.setSpacing(6)
        self.preset_combo = QComboBox(); self._refresh_presets()
        g4l.addWidget(self.preset_combo)
        pc_row = QHBoxLayout(); pc_row.setSpacing(6)
        btn_load_preset = QPushButton("Load"); btn_load_preset.setProperty("secondary", True)
        btn_load_preset.clicked.connect(self._load_custom_preset); pc_row.addWidget(btn_load_preset)
        btn_save_preset = QPushButton("Save Current"); btn_save_preset.setProperty("success", True)
        btn_save_preset.clicked.connect(self._save_custom_preset); pc_row.addWidget(btn_save_preset)
        btn_del_preset = QPushButton("Delete"); btn_del_preset.setProperty("danger", True)
        btn_del_preset.clicked.connect(self._delete_custom_preset); pc_row.addWidget(btn_del_preset)
        g4l.addLayout(pc_row)

        io_row = QHBoxLayout(); io_row.setSpacing(6)
        btn_export_presets = QPushButton("Export All"); btn_export_presets.setProperty("secondary", True)
        btn_export_presets.clicked.connect(self._export_presets); io_row.addWidget(btn_export_presets)
        btn_import_presets = QPushButton("Import"); btn_import_presets.setProperty("secondary", True)
        btn_import_presets.clicked.connect(self._import_presets); io_row.addWidget(btn_import_presets)
        g4l.addLayout(io_row)
        t2.addWidget(g4)
        t2.addStretch()
        tabs.addTab(tab_presets, "Presets")

        # ─── Tab 3: Export ───
        tab_export = QWidget()
        t3 = QVBoxLayout(tab_export); t3.setSpacing(8); t3.setContentsMargins(8,8,8,8)

        g5 = QGroupBox("Single Export"); g5l = QVBoxLayout(g5); g5l.setSpacing(6)
        self.btn_exp_video = QPushButton("Export Video"); self.btn_exp_video.setProperty("success", True)
        self.btn_exp_video.clicked.connect(self.export_video); self.btn_exp_video.setEnabled(False)
        g5l.addWidget(self.btn_exp_video)
        self.btn_exp_cancel = QPushButton("Cancel Export")
        self.btn_exp_cancel.setProperty("danger", True)
        self.btn_exp_cancel.clicked.connect(self._cancel_export)
        self.btn_exp_cancel.setEnabled(False)
        g5l.addWidget(self.btn_exp_cancel)
        self.btn_exp_img = QPushButton("Save Screenshot (PNG)"); self.btn_exp_img.setProperty("success", True)
        self.btn_exp_img.clicked.connect(self.save_screenshot); self.btn_exp_img.setEnabled(False)
        g5l.addWidget(self.btn_exp_img)
        self.btn_exp_gif = QPushButton("Export Before/After GIF")
        self.btn_exp_gif.clicked.connect(self.export_gif); self.btn_exp_gif.setEnabled(False)
        g5l.addWidget(self.btn_exp_gif)
        self.exp_prog = QProgressBar(); self.exp_prog.setVisible(False); g5l.addWidget(self.exp_prog)
        self.exp_stat = QLabel(""); self.exp_stat.setStyleSheet("color:#6c7086; font-size:11px;")
        g5l.addWidget(self.exp_stat)
        t3.addWidget(g5)

        g6 = QGroupBox("Batch Processing"); g6l = QVBoxLayout(g6); g6l.setSpacing(6)
        self.btn_batch = QPushButton("Select Files for Batch")
        self.btn_batch.clicked.connect(self.start_batch); g6l.addWidget(self.btn_batch)
        self.btn_batch_folder = QPushButton("Process Entire Folder")
        self.btn_batch_folder.setProperty("secondary", True)
        self.btn_batch_folder.clicked.connect(self.start_batch_folder); g6l.addWidget(self.btn_batch_folder)
        self.btn_batch_cancel = QPushButton("Cancel Batch"); self.btn_batch_cancel.setProperty("danger", True)
        self.btn_batch_cancel.clicked.connect(self._cancel_batch); self.btn_batch_cancel.setEnabled(False)
        g6l.addWidget(self.btn_batch_cancel)
        self.batch_prog = QProgressBar(); self.batch_prog.setVisible(False); g6l.addWidget(self.batch_prog)
        self.batch_stat = QLabel(""); self.batch_stat.setStyleSheet("color:#6c7086; font-size:11px;")
        g6l.addWidget(self.batch_stat)
        t3.addWidget(g6)
        t3.addStretch()
        tabs.addTab(tab_export, "Export")

        rl.addWidget(tabs)
        root.addWidget(rw, 1)

        # Status bar
        self.statusBar().showMessage("Ready - Drop a file or use buttons to start")
        gpu_text = f"GPU: {GPU_NAME}" if USE_GPU else "CPU Mode"
        gpu_label = QLabel(f"  {gpu_text}  ")
        gpu_label.setStyleSheet(
            f"color: {'#a6e3a1' if USE_GPU else '#f9e2af'}; font-size: 11px; font-weight: bold; "
            f"background-color: {'#1e3a2e' if USE_GPU else '#3a2e1e'}; border-radius: 4px; padding: 2px 8px;")
        self.statusBar().addPermanentWidget(gpu_label)

    # ── Parameter Management ────────────────────────────────────
    def _push(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.update_params(self._p())
        self._update_undo_buttons()

    def _p(self):
        p = {k: s.value() for k, (s, _) in self.sliders.items()}
        return p

    def _set_params(self, params):
        self.history.freeze()
        for k, v in params.items():
            if k in self.sliders:
                self.sliders[k][0].setValue(v)
        self.history.unfreeze()
        self._push()
        if self.image_mode:
            self._process_current_image()

    def _apply_preset(self, vals):
        self._set_params(vals)
        self.history.push(self._p())
        self.toast.show_message("Preset applied")

    def _undo(self):
        params = self.history.undo()
        if params:
            self._set_params(params)
            self.toast.show_message("Undo")
        self._update_undo_buttons()

    def _redo(self):
        params = self.history.redo()
        if params:
            self._set_params(params)
            self.toast.show_message("Redo")
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        self.btn_undo.setEnabled(self.history.can_undo)
        self.btn_redo.setEnabled(self.history.can_redo)

    # ── Toggles ─────────────────────────────────────────────────
    def _tog_lm(self, on):
        if self.video_thread: self.video_thread.show_landmarks = on
        if self.image_mode and self.current_original is not None and self.current_processed is not None:
            self._display_frame(self.current_original, self.current_processed, self.current_faces)
    def _tog_conf(self, on):
        if self.video_thread: self.video_thread.show_confidence = on
        if self.image_mode and self.current_original is not None and self.current_processed is not None:
            self._display_frame(self.current_original, self.current_processed, self.current_faces)
    def _toggle_compare(self, checked):
        self.comparison_mode = checked
        if checked: self.video_label.divider_ratio = 0.5
        if self.image_mode and self.current_original is not None and self.current_processed is not None:
            self._display_frame(self.current_original, self.current_processed, self.current_faces)

    def _on_faces_changed(self, val):
        if self.video_thread and self.video_thread.isRunning():
            src = self.video_thread.source
            self.video_thread.max_faces = val
            # Restart to apply new face count
            self._go(src)
        elif self.image_mode:
            self._process_current_image()
        self.toast.show_message(f"Max faces: {val}")

    def _on_scale_changed(self, idx):
        scales = [1.0, 0.75, 0.5]
        if self.video_thread:
            self.video_thread.preview_scale = scales[idx]

    # ── Preset Management ───────────────────────────────────────
    def _refresh_presets(self):
        self.preset_combo.clear()
        custom = PresetManager.list_custom()
        if custom:
            for name in sorted(custom.keys()):
                self.preset_combo.addItem(name)
        else:
            self.preset_combo.addItem("(no custom presets)")

    def _save_custom_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            PresetManager.save(name.strip(), self._p())
            self._refresh_presets()
            self.toast.show_message(f"Preset '{name.strip()}' saved")

    def _load_custom_preset(self):
        name = self.preset_combo.currentText()
        if name and name != "(no custom presets)":
            custom = PresetManager.list_custom()
            if name in custom:
                self._apply_preset(custom[name])

    def _delete_custom_preset(self):
        name = self.preset_combo.currentText()
        if name and name != "(no custom presets)":
            PresetManager.delete(name)
            self._refresh_presets()
            self.toast.show_message(f"Preset '{name}' deleted")

    def _export_presets(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Presets", "faceslim_presets.json", "JSON (*.json)")
        if path:
            PresetManager.export_all(path)
            self.toast.show_message("Presets exported")

    def _import_presets(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Presets", "", "JSON (*.json)")
        if path:
            n = PresetManager.import_presets(path)
            self._refresh_presets()
            self.toast.show_message(f"Imported {n} presets")

    # ── File Loading ────────────────────────────────────────────
    def start_webcam(self):
        self.image_mode = False
        self._go(None)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)")
        if path: self._handle_drop(path)

    def _handle_drop(self, path):
        ext = os.path.splitext(path)[1].lower()
        self.source_path = path
        if ext in IMAGE_EXTS:
            self.image_mode = True
            self.stop_video()
            self._load_image(path)
            self.btn_exp_img.setEnabled(True)
            self.btn_exp_gif.setEnabled(True)
            self.btn_exp_video.setEnabled(False)
        elif ext in VIDEO_EXTS:
            self.image_mode = False
            self._go(path)
            self.btn_exp_video.setEnabled(True)
            self.btn_exp_img.setEnabled(True)
            self.btn_exp_gif.setEnabled(True)
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None: return
        self.current_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Reset cached engine so filter state doesn't bleed between images
        if self._image_engine is not None:
            self._image_engine.close()
            self._image_engine = None
        self._process_current_image()

    def _process_current_image(self):
        if self.current_original is None: return
        nf = self.spin_faces.value()
        if self._image_engine is None or self._image_engine_faces != nf:
            if self._image_engine is not None:
                self._image_engine.close()
            self._image_engine = FaceWarpEngine('image', nf)
            self._image_engine_faces = nf
        result, faces = self._image_engine.warp_single_image(self.current_original, self._p())
        self.current_processed = result
        self.current_faces = faces
        self.face_count_label.setText(f"  {len(faces)} face{'s' if len(faces) != 1 else ''} detected")
        self._display_frame(self.current_original, result, faces)

    def _go(self, src):
        self.stop_video()
        self.video_thread = VideoThread()
        self.video_thread.set_source(src)
        self.video_thread.update_params(self._p())
        self.video_thread.show_landmarks = self.chk_lm.isChecked()
        self.video_thread.show_confidence = self.chk_conf.isChecked()
        self.video_thread.max_faces = self.spin_faces.value()
        scales = [1.0, 0.75, 0.5]
        self.video_thread.preview_scale = scales[self.combo_scale.currentIndex()]
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.fps_update.connect(self._on_fps)
        self.video_thread.status_update.connect(self.statusBar().showMessage)
        self.video_thread.error_update.connect(self._on_video_error)
        self.video_thread.start()
        self.btn_stop.setEnabled(True)
        nm = "Webcam" if src is None else os.path.basename(src)
        self.statusBar().showMessage(f"Playing: {nm}")

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop(); self.video_thread = None
        self.btn_stop.setEnabled(False); self.fps_label.setText("-- FPS")

    # ── Frame Display ───────────────────────────────────────────
    def _on_frame(self, orig, proc, faces):
        self.current_original = orig
        self.current_processed = proc
        self.current_faces = faces
        self.face_count_label.setText(f"  {len(faces)} face{'s' if len(faces) != 1 else ''}")
        self.btn_exp_img.setEnabled(True)
        self.btn_exp_gif.setEnabled(len(faces) > 0)
        self._display_frame(orig, proc, faces)

    def _display_frame(self, orig, proc, faces):
        if orig is None or proc is None:
            return
        if self.comparison_mode:
            h, w = orig.shape[:2]
            sx = max(1, min(w - 1, int(w * self.video_label.divider_ratio)))
            show = np.empty_like(orig)
            show[:, :sx] = orig[:, :sx]; show[:, sx:] = proc[:, sx:]
            cv2.line(show, (sx, 0), (sx, h), (203, 166, 247), 3)
            cy = h // 2
            cv2.fillPoly(show, [np.array([[sx-8,cy-12],[sx+8,cy-12],[sx+8,cy+12],[sx-8,cy+12]])], (203,166,247))
            cv2.line(show, (sx, cy-8), (sx, cy+8), (30,30,46), 2)
            for text, tx, color in [("Original", 10, (255,255,255)), ("Slimmed", sx+10, (137,180,250))]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(show, (tx-4, 10), (tx+tw+4, th+22), (30,30,46), -1)
                cv2.putText(show, text, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            show = proc

        show_lm = (self.video_thread and self.video_thread.show_landmarks) if self.video_thread else self.chk_lm.isChecked()
        show_conf = (self.video_thread and self.video_thread.show_confidence) if self.video_thread else self.chk_conf.isChecked()
        if faces and (show_lm or show_conf):
            show = draw_landmarks(show.copy() if show is proc else show, faces, show_conf, show_lm)

        show = np.ascontiguousarray(show)
        h, w, ch = show.shape
        qi = QImage(show.data, w, h, ch * w, QImage.Format.Format_RGB888)
        # CRITICAL: QImage references show.data - must keep ref until pixmap is created
        px = QPixmap.fromImage(qi.copy())  # .copy() detaches from numpy buffer
        px = px.scaled(self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(px)

    def _on_fps(self, fps):
        c = "#a6e3a1" if fps >= 20 else "#f9e2af" if fps >= 10 else "#f38ba8"
        self.fps_label.setText(f"{fps:.1f} FPS")
        self.fps_label.setStyleSheet(f"color:{c}; font-size:11px; font-weight:bold;")

    def _on_video_error(self, msg):
        self.statusBar().showMessage(msg)
        self.toast.show_message(msg, 4000)
        self.btn_stop.setEnabled(False)
        self.fps_label.setText("-- FPS")

    # ── Export: Video ───────────────────────────────────────────
    def export_video(self):
        if not self.source_path: return
        base = os.path.splitext(os.path.basename(self.source_path))[0]
        out, _ = QFileDialog.getSaveFileName(self, "Save", f"{base}_slimmed.mp4", "MP4 (*.mp4)")
        if not out: return
        self.exp_prog.setVisible(True); self.exp_prog.setValue(0)
        self.btn_exp_video.setEnabled(False); self.btn_exp_cancel.setEnabled(True)
        self._et = ExportThread(self.source_path, out, self._p(), self.spin_faces.value())
        self._et.progress.connect(self.exp_prog.setValue)
        self._et.finished.connect(lambda p: (self.exp_prog.setValue(100),
            self.btn_exp_video.setEnabled(True), self.btn_exp_cancel.setEnabled(False),
            self.exp_stat.setText(f"Saved: {os.path.basename(p)}"),
            self.toast.show_message(f"Export complete")))
        self._et.error.connect(lambda m: (self.btn_exp_video.setEnabled(True),
            self.btn_exp_cancel.setEnabled(False),
            self.exp_prog.setVisible(False), self.exp_stat.setText(f"Error: {m}")))
        self._et.status.connect(self.exp_stat.setText)
        self._et.start()

    def _cancel_export(self):
        if hasattr(self, '_et') and self._et.isRunning():
            self._et.cancel()
            self.btn_exp_cancel.setEnabled(False)
            self.toast.show_message("Cancelling export...")

    # ── Export: Screenshot ──────────────────────────────────────
    def save_screenshot(self):
        if self.current_processed is None:
            self.toast.show_message("No frame available"); return
        default = "screenshot.png"
        if self.source_path:
            default = os.path.splitext(os.path.basename(self.source_path))[0] + "_slimmed.png"
        out, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", default, "PNG (*.png);;JPEG (*.jpg)")
        if out:
            bgr = cv2.cvtColor(self.current_processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out, bgr)
            self.toast.show_message(f"Screenshot saved")

    # ── Export: GIF ─────────────────────────────────────────────
    def export_gif(self):
        if self.current_original is None or self.current_processed is None:
            self.toast.show_message("No frame available"); return
        default = "before_after.gif"
        if self.source_path:
            default = os.path.splitext(os.path.basename(self.source_path))[0] + "_comparison.gif"
        out, _ = QFileDialog.getSaveFileName(self, "Save GIF", default, "GIF (*.gif)")
        if not out: return
        self._gif_thread = GifExportThread(self.current_original, self.current_processed, out)
        self._gif_thread.finished.connect(lambda p: self.toast.show_message("GIF saved"))
        self._gif_thread.error.connect(lambda m: self.toast.show_message(f"GIF error: {m}"))
        self._gif_thread.start()

    # ── Batch Processing ────────────────────────────────────────
    def start_batch(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)")
        if files: self._run_batch(files)

    def start_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder: return
        files = []
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS | VIDEO_EXTS:
                files.append(os.path.join(folder, f))
        if files:
            self._run_batch(sorted(files))
        else:
            self.toast.show_message("No media files found in folder")

    def _run_batch(self, files):
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir: return
        self.batch_prog.setVisible(True); self.batch_prog.setValue(0)
        self.btn_batch.setEnabled(False); self.btn_batch_folder.setEnabled(False)
        self.btn_batch_cancel.setEnabled(True)
        self._batch_thread = BatchThread(files, output_dir, self._p(), self.spin_faces.value())
        self._batch_thread.progress.connect(lambda c, t, f: (
            self.batch_prog.setValue(int(c / t * 100)),
            self.batch_stat.setText(f"Processing {c}/{t}: {f}")))
        self._batch_thread.finished.connect(lambda p, f: (
            self.btn_batch.setEnabled(True), self.btn_batch_folder.setEnabled(True),
            self.btn_batch_cancel.setEnabled(False),
            self.batch_stat.setText(f"Done: {p} processed, {f} failed"),
            self.toast.show_message(f"Batch complete: {p} files")))
        self._batch_thread.error.connect(lambda m: (
            self.btn_batch.setEnabled(True), self.btn_batch_folder.setEnabled(True),
            self.batch_stat.setText(f"Error: {m}")))
        self._batch_thread.start()

    def _cancel_batch(self):
        if hasattr(self, '_batch_thread') and self._batch_thread.isRunning():
            self._batch_thread.cancel()
            self.toast.show_message("Cancelling batch...")

    # ── Drag and Drop ───────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._handle_drop(path)

    # ── Settings Persistence ────────────────────────────────────
    def _load_settings(self):
        for k, (s, _) in self.sliders.items():
            s.setValue(self.settings.value(f"s/{k}", s.value(), type=int))
        self.spin_faces.setValue(self.settings.value("max_faces", 1, type=int))

    def _save_settings(self):
        for k, (s, _) in self.sliders.items():
            self.settings.setValue(f"s/{k}", s.value())
        self.settings.setValue("max_faces", self.spin_faces.value())

    def closeEvent(self, e):
        self._save_settings(); self.stop_video()
        if self._image_engine is not None:
            self._image_engine.close()
        e.accept()

# ═══════════════════════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════════════════════
def cli_process(args):
    """Headless command-line processing."""
    if not ensure_model():
        print(f"ERROR: Model not available. Download from:\n  {MODEL_URL}")
        sys.exit(1)
    ensure_parsing_model()  # Non-fatal - falls back to landmark mask
    params = DEFAULT_PARAMS.copy()
    if args.preset:
        key = args.preset
        if key in BUILT_IN_PRESETS:
            params.update(BUILT_IN_PRESETS[key])
        else:
            custom = PresetManager.list_custom()
            if key in custom:
                params.update(custom[key])
            else:
                print(f"Unknown preset: {key}")
                print(f"Available: {', '.join(list(BUILT_IN_PRESETS.keys()) + list(custom.keys()))}")
                sys.exit(1)

    if args.jaw is not None: params['jaw'] = args.jaw
    if args.cheeks is not None: params['cheeks'] = args.cheeks
    if args.chin is not None: params['chin'] = args.chin
    if args.face_width is not None: params['face_width'] = args.face_width
    if args.forehead is not None: params['forehead'] = args.forehead
    if args.nose is not None: params['nose'] = args.nose
    if args.eye_enlarge is not None: params['eye_enlarge'] = args.eye_enlarge
    if args.lip_plump is not None: params['lip_plump'] = args.lip_plump
    if args.skin_smooth is not None: params['skin_smooth'] = args.skin_smooth
    if args.teeth_whiten is not None: params['teeth_whiten'] = args.teeth_whiten
    if args.eye_sharpen is not None: params['eye_sharpen'] = args.eye_sharpen
    if args.skin_tone_even is not None: params['skin_tone_even'] = args.skin_tone_even
    if args.lip_color is not None: params['lip_color'] = args.lip_color
    if args.smoothing is not None: params['smoothing'] = args.smoothing

    max_faces = args.faces or 1

    # Collect input files
    inputs = []
    for inp in args.input:
        if os.path.isdir(inp):
            for f in os.listdir(inp):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS | VIDEO_EXTS:
                    inputs.append(os.path.join(inp, f))
        elif os.path.isfile(inp):
            inputs.append(inp)
        else:
            print(f"Warning: {inp} not found, skipping")

    if not inputs:
        print("No input files found"); sys.exit(1)

    output_dir = args.output or os.path.join(os.path.dirname(inputs[0]), 'faceslim_output')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nFaceSlim v{VERSION} - CLI Mode")
    print(f"  GPU: {'ON (' + GPU_NAME + ')' if USE_GPU else 'OFF'}")
    print(f"  Files: {len(inputs)}")
    print(f"  Params: {json.dumps({k: v for k, v in params.items() if v != 0}, indent=2)}")
    print(f"  Output: {output_dir}\n")

    processed = failed = 0
    for i, filepath in enumerate(sorted(inputs)):
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        print(f"  [{i+1}/{len(inputs)}] {fname}...", end=' ', flush=True)

        try:
            if ext in IMAGE_EXTS:
                eng = FaceWarpEngine('image', max_faces)
                img = cv2.imread(filepath)
                if img is None: raise ValueError("Cannot read image")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result, faces = eng.warp_single_image(rgb, params)
                out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '_slimmed.png')
                cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                eng.close()
                print(f"OK ({len(faces)} face{'s' if len(faces) != 1 else ''})")
                processed += 1

            elif ext in VIDEO_EXTS:
                eng = FaceWarpEngine('video', max_faces)
                eng.grid_scale = 6
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened(): raise ValueError("Cannot open video")
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '_slimmed.mp4')
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                has_fx = any(abs(params.get(k,0)) > 0
                             for k in ['jaw','cheeks','chin','face_width','forehead','nose','eye_enlarge','lip_plump','skin_smooth','teeth_whiten','eye_sharpen','skin_tone_even','lip_color'])
                t0 = time.time()
                for fi in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = eng.detect(rgb)
                    proc = eng.warp(rgb, faces, params) if (faces and has_fx) else rgb
                    writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                    if fi % 30 == 0 and fi > 0:
                        eta = ((time.time() - t0) / fi) * (total - fi)
                        print(f"\r  [{i+1}/{len(inputs)}] {fname}... {fi}/{total} frames (ETA: {int(eta)}s)", end='', flush=True)
                cap.release(); writer.release(); eng.close()
                elapsed = time.time() - t0
                print(f"\r  [{i+1}/{len(inputs)}] {fname}... OK ({total} frames, {elapsed:.1f}s)")
                processed += 1
            else:
                print(f"SKIP (unsupported)")
                failed += 1
        except Exception as e:
            print(f"FAIL ({e})")
            failed += 1

    print(f"\nDone: {processed} processed, {failed} failed")
    print(f"Output: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if not ensure_model():
        print(f"ERROR: Could not obtain model.\nDownload from:\n  {MODEL_URL}\nPlace at:\n  {MODEL_PATH}")
        sys.exit(1)
    ensure_parsing_model()  # Non-fatal - falls back to landmark mask

    parser = argparse.ArgumentParser(
        prog='FaceSlim',
        description=f'FaceSlim v{VERSION} - AI Face Slimming & Reshaping Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python FaceSlim.py                                    # Launch GUI
  python FaceSlim.py --input video.mp4 --preset Moderate
  python FaceSlim.py --input photo.jpg --jaw 50 --cheeks 30
  python FaceSlim.py --input ./photos/ --output ./results/ --preset Strong
  python FaceSlim.py --input a.mp4 b.jpg --faces 3 --preset "Full Sculpt"
""")
    parser.add_argument('--input', '-i', nargs='+', help='Input file(s) or folder(s)')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--preset', '-p', help='Apply named preset')
    parser.add_argument('--jaw', type=int, help='Jaw slimming (0-100)')
    parser.add_argument('--cheeks', type=int, help='Cheek slimming (0-100)')
    parser.add_argument('--chin', type=int, help='Chin reshape (0-100)')
    parser.add_argument('--face-width', type=int, dest='face_width', help='Face width (0-100)')
    parser.add_argument('--forehead', type=int, help='Forehead slimming (0-100)')
    parser.add_argument('--nose', type=int, help='Nose slimming (0-100)')
    parser.add_argument('--eye-enlarge', type=int, dest='eye_enlarge', help='Eye enlargement (0-100)')
    parser.add_argument('--lip-plump', type=int, dest='lip_plump', help='Lip plumping (0-100)')
    parser.add_argument('--skin-smooth', type=int, dest='skin_smooth', help='AI skin smoothing (0-100)')
    parser.add_argument('--teeth-whiten', type=int, dest='teeth_whiten', help='AI teeth whitening (0-100)')
    parser.add_argument('--eye-sharpen', type=int, dest='eye_sharpen', help='AI eye sharpening (0-100)')
    parser.add_argument('--skin-tone-even', type=int, dest='skin_tone_even', help='AI skin tone evening (0-100)')
    parser.add_argument('--lip-color', type=int, dest='lip_color', help='AI lip color enhancement (0-100)')
    parser.add_argument('--smoothing', type=int, help='Warp smoothing (10-100)')
    parser.add_argument('--faces', type=int, help='Max faces to process (1-5)')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')

    args = parser.parse_args()

    if args.list_presets:
        print(f"\nFaceSlim v{VERSION} - Available Presets:\n")
        print("Built-in:")
        for name, vals in BUILT_IN_PRESETS.items():
            desc = ', '.join(f"{k}={v}" for k, v in vals.items() if v > 0)
            print(f"  {name:15s} {desc}")
        custom = PresetManager.list_custom()
        if custom:
            print("\nCustom:")
            for name, vals in custom.items():
                desc = ', '.join(f"{k}={v}" for k, v in vals.items() if v > 0)
                print(f"  {name:15s} {desc}")
        sys.exit(0)

    if args.input:
        cli_process(args)
    else:
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


if __name__ == "__main__":
    main()
