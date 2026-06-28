#!/usr/bin/env python3
"""
FaceSlim v1.17.0 - AI Face Slimming & Reshaping Suite
GPU-accelerated face reshaping with MediaPipe 478-landmark detection,
PyTorch TPS warping, real-time preview, batch processing, CLI mode,
image+video support, preset management, and before/after GIF export.

Phase 1: BiSeNet face parsing, skin smoothing
Phase 2: Temporal mask smoothing, optical flow propagation,
         eye enlargement, teeth whitening
Phase 3: Lip plumping, eye sharpening, skin tone evening, lip color

Dependencies install from requirements.txt; models download on first use.
"""

import multiprocessing
multiprocessing.freeze_support()

import sys, os, subprocess, time, json, math, traceback, urllib.request, argparse, glob, threading, hashlib
from collections import deque
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════
def _bootstrap():
    if sys.version_info < (3, 9):
        print("Python 3.9+ required"); sys.exit(1)
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        return
    required = {
        'PyQt5': 'PyQt5', 'cv2': 'opencv-python', 'mediapipe': 'mediapipe',
        'numpy': 'numpy', 'scipy': 'scipy', 'PIL': 'Pillow',
        'onnxruntime': 'onnxruntime',
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies: " + ", ".join(sorted(missing)))
        print("Install with: python -m pip install -r requirements.txt")
        sys.exit(1)

_bootstrap()

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.interpolate import RBFInterpolator
from PIL import Image as PILImage, PngImagePlugin
import onnxruntime as ort

try:
    import pyvirtualcam
    HAS_VIRTUALCAM = True
except Exception:
    pyvirtualcam = None
    HAS_VIRTUALCAM = False

# -- Qt Imports (must precede QThread subclasses) --
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QGroupBox,
    QProgressBar, QCheckBox, QGridLayout, QSizePolicy,
    QTabWidget, QScrollArea, QInputDialog, QSpinBox, QComboBox,
    QFrame, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
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

VERSION = "1.17.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_LOG_PATH = os.path.join(APP_DIR, 'render.log')
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

def _decode_process_output(data):
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)

def log_render_event(kind, message, context=None, exc_text=None):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": VERSION,
        "kind": kind,
        "message": str(message),
        "context": context or {},
    }
    if exc_text:
        record["traceback"] = exc_text
    try:
        with open(RENDER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    except Exception:
        pass
    return RENDER_LOG_PATH

MODEL_MANIFEST_VERSION = 1
LANDMARK_MODEL = {
    "key": "face_landmarker",
    "label": "MediaPipe Face Landmarker",
    "filename": "face_landmarker.task",
    "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "size_bytes": 3_758_596,
    "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
MODEL_PATH = os.path.join(APP_DIR, LANDMARK_MODEL["filename"])
MODEL_URL = LANDMARK_MODEL["url"]

def _model_path(cfg):
    return os.path.join(APP_DIR, cfg["filename"])

def _format_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.1f} MB"

def _remove_quietly(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_model_artifact(path, cfg):
    if not os.path.exists(path):
        return False, "missing"
    actual_size = os.path.getsize(path)
    expected_size = int(cfg["size_bytes"])
    if actual_size != expected_size:
        return False, f"size {actual_size} bytes, expected {expected_size}"
    actual_hash = _sha256_file(path)
    expected_hash = cfg["sha256"].lower()
    if actual_hash.lower() != expected_hash:
        return False, f"sha256 {actual_hash}, expected {expected_hash}"
    return True, "verified"

def _download_verified_model(cfg, optional_failure_note=None):
    path = _model_path(cfg)
    ok, reason = validate_model_artifact(path, cfg)
    if ok:
        return True
    if os.path.exists(path):
        print(f"  Cached {cfg['label']} is invalid ({reason}); re-downloading.")
        _remove_quietly(path)

    tmp_path = f"{path}.download-{os.getpid()}.tmp"
    print(f"Downloading {cfg['label']} ({_format_size(cfg['size_bytes'])})...")
    try:
        _remove_quietly(tmp_path)
        urllib.request.urlretrieve(cfg["url"], tmp_path)
        ok, reason = validate_model_artifact(tmp_path, cfg)
        if not ok:
            _remove_quietly(tmp_path)
            print(f"  Downloaded {cfg['label']} failed verification: {reason}")
            return False
        os.replace(tmp_path, path)
        print(f"  {cfg['label']} verified: sha256 {cfg['sha256'][:12]}...")
        return True
    except Exception as e:
        _remove_quietly(tmp_path)
        note = f" ({optional_failure_note})" if optional_failure_note else ""
        print(f"  {cfg['label']} download failed: {e}{note}")
        return False

def ensure_model():
    return _download_verified_model(LANDMARK_MODEL)

# ── Face Parsing Models ────────────────────────────────────────────────
PARSER_MODELS = {
    "bisenet_resnet18": {
        "key": "bisenet_resnet18",
        "label": "BiSeNet ResNet18 (fast)",
        "filename": "bisenet_face_parsing.onnx",
        "url": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx",
        "size_bytes": 53_205_364,
        "sha256": "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
    },
    "bisenet_resnet34": {
        "key": "bisenet_resnet34",
        "label": "BiSeNet ResNet34 (quality)",
        "filename": "bisenet_resnet34.onnx",
        "url": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx",
        "size_bytes": 93_632_554,
        "sha256": "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
    },
}
DEFAULT_PARSER_MODEL = "bisenet_resnet18"
BISENET_PATH = os.path.join(APP_DIR, PARSER_MODELS[DEFAULT_PARSER_MODEL]["filename"])
BISENET_URL = PARSER_MODELS[DEFAULT_PARSER_MODEL]["url"]
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
ACTIVE_PARSER_MODEL = DEFAULT_PARSER_MODEL
ACTIVE_PARSER_PATH = BISENET_PATH

def parser_model_key(model_key=None):
    return model_key if model_key in PARSER_MODELS else DEFAULT_PARSER_MODEL

def parser_model_label(model_key=None):
    return PARSER_MODELS[parser_model_key(model_key)]["label"]

def parser_model_path(model_key=None):
    return os.path.join(APP_DIR, PARSER_MODELS[parser_model_key(model_key)]["filename"])

def parser_model_ready(model_key=None):
    cfg = PARSER_MODELS[parser_model_key(model_key)]
    path = parser_model_path(model_key)
    return validate_model_artifact(path, cfg)[0]

def ensure_parsing_model(model_key=None):
    global HAS_BISENET, ACTIVE_PARSER_MODEL, ACTIVE_PARSER_PATH
    key = parser_model_key(model_key)
    cfg = PARSER_MODELS[key]
    path = parser_model_path(key)
    if _download_verified_model(cfg, "will use landmark fallback"):
        HAS_BISENET = True
        ACTIVE_PARSER_MODEL = key
        ACTIVE_PARSER_PATH = path
        print(f"  Face parsing model ready: {cfg['label']}")
        return True
    HAS_BISENET = False
    return False

# ── MODNet Matting Refinement ──────────────────────────────────────────
MATTE_MODEL = {
    "key": "modnet_photographic",
    "label": "MODNet Photographic Matting",
    "filename": "modnet_photographic.onnx",
    "url": "https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx",
    "size_bytes": 25_969_398,
    "sha256": "5069a5e306b9f5e9f4f2b0360264c9f8ea13b257c7c39943c7cf6a2ec3a102ae",
}
MODNET_PATH = _model_path(MATTE_MODEL)
MODNET_URL = MATTE_MODEL["url"]

def ensure_matting_model():
    if _download_verified_model(MATTE_MODEL, "matting refinement disabled"):
        print("  MODNet matting model ready")
        return True
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
                  'skin_tone_even': 0, 'lip_color': 0, 'under_eye': 0,
                  'hair_hue': 0, 'hair_saturation': 0, 'hair_density': 0,
                  'blush': 0, 'lip_gloss': 0, 'eye_shadow': 0,
                  'expression_neutralize': 0,
                  'matting_refine': 0}

EFFECT_PARAM_KEYS = (
    'jaw', 'cheeks', 'chin', 'face_width', 'forehead', 'nose', 'eye_enlarge',
    'lip_plump', 'skin_smooth', 'teeth_whiten', 'eye_sharpen',
    'skin_tone_even', 'lip_color', 'under_eye', 'hair_hue',
    'hair_saturation', 'hair_density', 'blush', 'lip_gloss', 'eye_shadow',
    'expression_neutralize'
)

CLI_PARAM_KEYS = EFFECT_PARAM_KEYS + ('smoothing', 'temporal', 'bg_protect', 'matting_refine')

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

    def __init__(self, model_key=None):
        self.model_key = parser_model_key(model_key)
        if not ensure_parsing_model(self.model_key):
            raise RuntimeError(f"{parser_model_label(self.model_key)} unavailable")
        available = ort.get_available_providers()
        providers = []
        if USE_GPU and 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if sys.platform.startswith('win') and 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        providers = [p for p in providers if p in available]
        self.session = ort.InferenceSession(parser_model_path(self.model_key), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._cache = {}  # {(h, w, key): parsing_map}
        print(f"  Face parsing: {parser_model_label(self.model_key)} loaded ({providers[0]})")

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

    def __init__(self):
        if not ensure_matting_model():
            raise RuntimeError("MODNet matting model unavailable")
        available = ort.get_available_providers()
        providers = []
        if USE_GPU and 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if sys.platform.startswith('win') and 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        providers = [p for p in providers if p in available]
        self.session = ort.InferenceSession(MODNET_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._cache = {}
        print(f"  Matting refinement: MODNet loaded ({providers[0]})")

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


def face_components(face):
    if len(face) >= 3:
        return face[0], face[1], face[2] or {}
    return face[0], face[1], {}


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


def video_output_size(width, height, mode):
    if mode == 'side_by_side':
        return width * 2, height
    return width, height


def save_rgb_image(output_path, rgb, source_path=None, preserve_metadata=True):
    ext = os.path.splitext(output_path)[1].lower()
    if not ext:
        output_path += ".png"
        ext = ".png"
    image = PILImage.fromarray(rgb)
    save_kwargs = {}
    png_info = None
    if preserve_metadata and source_path and os.path.exists(source_path):
        try:
            with PILImage.open(source_path) as src:
                if src.info.get("exif"):
                    save_kwargs["exif"] = src.info["exif"]
                if src.info.get("icc_profile"):
                    save_kwargs["icc_profile"] = src.info["icc_profile"]
                if ext == ".png":
                    png_info = PngImagePlugin.PngInfo()
                    for key, value in src.info.items():
                        if isinstance(key, str) and isinstance(value, str):
                            png_info.add_text(key, value)
        except Exception as e:
            print(f"Metadata preserve warning: {e}")
    if png_info is not None:
        save_kwargs["pnginfo"] = png_info
    if ext in (".jpg", ".jpeg"):
        save_kwargs.setdefault("quality", 95)
        save_kwargs.setdefault("subsampling", 1)
    image.save(output_path, **save_kwargs)
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
        for key, value in overrides.items():
            if key in CLI_PARAM_KEYS:
                params[key] = int(value)
    return params


def parse_param_overrides(raw):
    overrides = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in CLI_PARAM_KEYS:
            raise ValueError(f"Unknown parameter: {key}")
        overrides[key] = int(value)
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


def _manifest_path(base_dir, path_value):
    if not path_value:
        return path_value
    return path_value if os.path.isabs(path_value) else os.path.abspath(os.path.join(base_dir, path_value))


def load_batch_manifest(manifest_path, fallback_output=None, fallback_parser_model=None):
    with open(manifest_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    default_params = params_from_preset_and_overrides(data.get("preset"), data.get("params"))
    default_parser_model = parser_model_key(data.get("parser_model") or fallback_parser_model)
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
            params.update({k: int(v) for k, v in entry["params"].items() if k in CLI_PARAM_KEYS})
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
        })
    return jobs


def jobs_from_files(files, output_dir, params, max_faces=1, watermark=False,
                    preserve_metadata=True, compare_mode='none', parser_model=None):
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
    def __init__(self, mode='video', max_faces=1, parser_model=None):
        self.parser_model = parser_model_key(parser_model)
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
        if ensure_parsing_model(self.parser_model):
            try:
                self.parser = FaceParsingEngine(self.parser_model)
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

        matting_refine = params.get('matting_refine', 0)
        if matting_refine > 0:
            try:
                if self.matter is None:
                    self.matter = MattingRefinementEngine()
                matte_key = (lms[:10] / 8.0).astype(np.int16).tobytes()
                matte = self.matter.matte(roi_crop, cache_key=matte_key)
                strength = min(max(matting_refine / 100.0, 0.0), 1.0)
                mask_roi = mask_roi * (1.0 - strength) + (mask_roi * matte) * strength
                mask_roi = np.clip(mask_roi, 0.0, 1.0)
            except Exception as e:
                print(f"Matting refinement fallback: {e}")

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
            eng = FaceWarpEngine('video', self.max_faces, self.parser_model)
            # Apply temporal beta
            beta = self.params.get('temporal', 50) / 5000.0
            eng.set_temporal_beta(beta)

            cap = cv2.VideoCapture(0 if self.source is None else self.source)
            if not cap.isOpened():
                self.status_update.emit("Failed to open video source")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
                    eng = FaceWarpEngine('video', self.max_faces, self.parser_model)
                    eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)
                if self.paused and seek_to is None:
                    self.msleep(50); continue
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    if self.source is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        eng.close(); eng = FaceWarpEngine('video', self.max_faces, self.parser_model)
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
                    has_fx = any(abs(self.params.get(k, 0)) > 0 for k in EFFECT_PARAM_KEYS)

                    # Update temporal beta if changed
                    eng.set_temporal_beta(self.params.get('temporal', 50) / 5000.0)

                    proc = eng.warp(rgb, faces, self.params) if (faces and has_fx) else rgb.copy()
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
                 compare_mode='none', parser_model=None):
        super().__init__()
        self.inp, self.out, self.params = inp, out, params
        self.max_faces = max_faces
        self.watermark = watermark
        self.compare_mode = compare_mode
        self.parser_model = parser_model_key(parser_model)
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        eng = None
        cap = None
        writer = None
        try:
            eng = FaceWarpEngine('video', self.max_faces, self.parser_model)
            eng.grid_scale = 6
            cap = cv2.VideoCapture(self.inp)
            if not cap.isOpened():
                msg = "Cannot open input"
                log_render_event("export_open_failed", msg, {"input": self.inp, "output": self.out})
                self.error.emit(f"{msg} (see render.log)"); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_w, out_h = video_output_size(w, h, self.compare_mode)
            writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            if not writer.isOpened():
                raise ValueError(f"Cannot open output writer: {self.out}")
            self.status.emit(f"Exporting {total} frames...")
            has_fx = any(abs(self.params.get(k, 0)) > 0 for k in EFFECT_PARAM_KEYS)
            t_start = time.time()
            for i in range(total):
                if self.cancelled:
                    self.status.emit("Export cancelled"); break
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = eng.detect(rgb)
                proc = eng.warp(rgb, faces, self.params) if (faces and has_fx) else rgb
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
                 parser_model=None):
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
                                                self.parser_model)
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
        try:
            self.job_update.emit(index, "Processing", 20, "Reading image", "--")
            eng = FaceWarpEngine('image', job.get("max_faces", self.max_faces),
                                 job.get("parser_model", self.parser_model))
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(f"Cannot read {filepath}")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.job_update.emit(index, "Processing", 55, "Processing image", "--")
            result, _ = eng.warp_single_image(rgb, params)
            if self.cancelled or self._job_cancelled(index):
                return False
            result = apply_disclosure_watermark(result, job.get("watermark", self.watermark))
            out_path = job.get("output") or os.path.join(job.get("output_dir", self.output_dir),
                os.path.splitext(os.path.basename(filepath))[0] + '_slimmed.png')
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.job_update.emit(index, "Processing", 85, "Saving image", "--")
            save_rgb_image(out_path, result, filepath, job.get("preserve_metadata", self.preserve_metadata))
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
        out_path = job.get("output") or os.path.join(job.get("output_dir", self.output_dir),
            os.path.splitext(os.path.basename(filepath))[0] + '_slimmed.mp4')
        try:
            eng = FaceWarpEngine('video', max_faces, job.get("parser_model", self.parser_model))
            eng.grid_scale = 6
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValueError(f"Cannot open {filepath}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_w, out_h = video_output_size(w, h, compare_mode)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            if not writer.isOpened():
                raise ValueError(f"Cannot open output writer: {out_path}")
            has_fx = any(abs(params.get(k, 0)) > 0 for k in EFFECT_PARAM_KEYS)
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
                proc = eng.warp(rgb, faces, params) if (faces and has_fx) else rgb
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
class ResponsibleUseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Responsible Use")
        self.setModal(True)
        self.setFixedWidth(520)
        layout = QVBoxLayout(self)
        title = QLabel("Responsible Use")
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#89b4fa;")
        layout.addWidget(title)
        body = QLabel(
            "Use FaceSlim only on media you own or have permission to edit. "
            "Do not misrepresent identity, consent, or material appearance changes. "
            "Disclose altered media when context, platform rules, or local law requires it."
        )
        body.setWordWrap(True)
        body.setStyleSheet("color:#cdd6f4; font-size:13px; line-height:1.35;")
        layout.addWidget(body)
        row = QHBoxLayout()
        row.addStretch()
        btn_exit = QPushButton("Exit")
        btn_exit.setProperty("secondary", True)
        btn_exit.clicked.connect(self.reject)
        row.addWidget(btn_exit)
        btn_accept = QPushButton("I Understand")
        btn_accept.setProperty("success", True)
        btn_accept.clicked.connect(self.accept)
        row.addWidget(btn_accept)
        layout.addLayout(row)


class BatchQueueDialog(QDialog):
    cancel_job_requested = pyqtSignal(int)

    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Queue")
        self.resize(780, 360)
        self._progress = []
        self._status_items = []
        self._eta_items = []
        self._buttons = []

        layout = QVBoxLayout(self)
        self.summary = QLabel(f"Queued: {len(jobs)} files")
        self.summary.setStyleSheet("color:#a6e3a1; font-size:12px; font-weight:bold;")
        layout.addWidget(self.summary)

        self.table = QTableWidget(len(jobs), 5)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Progress", "ETA", "Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        for row, job in enumerate(jobs):
            name = os.path.basename(job.get("input", "")) or str(job.get("input", ""))
            self.table.setItem(row, 0, QTableWidgetItem(name))
            status = QTableWidgetItem("Queued")
            eta = QTableWidgetItem("--")
            self.table.setItem(row, 1, status)
            self.table.setItem(row, 3, eta)
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFixedWidth(150)
            self.table.setCellWidget(row, 2, progress)
            btn = QPushButton("Cancel")
            btn.setProperty("danger", True)
            btn.clicked.connect(lambda _checked=False, idx=row: self._cancel_job(idx))
            self.table.setCellWidget(row, 4, btn)
            self._status_items.append(status)
            self._eta_items.append(eta)
            self._progress.append(progress)
            self._buttons.append(btn)

        layout.addWidget(self.table)

    def _cancel_job(self, index):
        if 0 <= index < len(self._buttons):
            self._buttons[index].setEnabled(False)
            self.update_job(index, "Skipped", self._progress[index].value(), "Cancel requested", "--")
            self.cancel_job_requested.emit(index)

    def update_job(self, index, status, progress, detail, eta):
        if not (0 <= index < len(self._progress)):
            return
        self._status_items[index].setText(status)
        self._progress[index].setValue(max(0, min(100, int(progress))))
        self._eta_items[index].setText(eta or "--")
        if detail and self.table.item(index, 0):
            self.table.item(index, 0).setToolTip(detail)
        if status in {"Done", "Failed", "Skipped"}:
            self._buttons[index].setEnabled(False)

    def update_summary(self, completed, total, eta):
        self.summary.setText(f"Completed: {completed}/{total} | ETA: {eta}")


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
    def __init__(self, show_responsible_gate=True):
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
        self.current_teeth_hint_rois = []
        self._timeline_dragging = False
        self._timeline_duration = 0.0
        self.history = ParamHistory()
        self.image_mode = False  # True when viewing a single image
        self._image_engine = None  # Cached engine for image mode
        self._image_engine_faces = 0
        self._image_engine_parser = DEFAULT_PARSER_MODEL
        self._build_ui()
        self._load_settings()
        self.history.push(self._p())
        if show_responsible_gate:
            QTimer.singleShot(0, self._show_responsible_use_gate)

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

    def _show_responsible_use_gate(self):
        if self.settings.value("responsible_use_ack_v1", False, type=bool):
            return
        dlg = ResponsibleUseDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.settings.setValue("responsible_use_ack_v1", True)
            self.toast.show_message("Responsible use acknowledged")
        else:
            self.close()

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

        timeline_row = QHBoxLayout(); timeline_row.setSpacing(8)
        timeline_title = QLabel("Timeline")
        timeline_title.setStyleSheet("color:#89b4fa; font-size:11px; font-weight:bold;")
        timeline_row.addWidget(timeline_title)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.sliderPressed.connect(self._timeline_pressed)
        self.timeline_slider.sliderMoved.connect(self._timeline_seek)
        self.timeline_slider.sliderReleased.connect(self._timeline_released)
        timeline_row.addWidget(self.timeline_slider, 1)
        self.timeline_label = QLabel("--:-- / --:--")
        self.timeline_label.setStyleSheet("color:#cdd6f4; font-size:11px; min-width:86px;")
        timeline_row.addWidget(self.timeline_label)
        left.addLayout(timeline_row)

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
        self.btn_virtualcam = QPushButton("Virtual Cam"); self.btn_virtualcam.setCheckable(True)
        self.btn_virtualcam.setProperty("secondary", True)
        self.btn_virtualcam.clicked.connect(self._toggle_virtualcam)
        self.btn_virtualcam.setEnabled(False)
        tr.addWidget(self.btn_virtualcam)
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
        self._make_slider(g1l, 'expression_neutralize', 'Expression Neutralize', default=0,
                          tip='Blendshape-guided dampening of frowns and raised or lowered brows')
        t1.addWidget(g1)

        g_beauty = QGroupBox("AI Beauty"); g_bl = QVBoxLayout(g_beauty); g_bl.setSpacing(6)
        self._make_slider(g_bl, 'skin_smooth', 'Skin Smoothing', default=0, tip='Frequency-separation bilateral filter on skin mask')
        self._make_slider(g_bl, 'skin_tone_even', 'Skin Tone Even', default=0, tip='Reduces redness and blotchiness via LAB color correction')
        self._make_slider(g_bl, 'teeth_whiten', 'Teeth Whitening', default=0, tip='HSV brightness boost in mouth region')
        self._make_slider(g_bl, 'eye_sharpen', 'Eye Sharpen', default=0, tip='Unsharp mask on eyes and brows for crisp detail')
        self._make_slider(g_bl, 'lip_color', 'Lip Color', default=0, tip='Boosts lip saturation and warmth')
        self._make_slider(g_bl, 'under_eye', 'Under-Eye Smooth', default=0, tip='Dedicated smoothing below the eyes')
        self._make_slider(g_bl, 'hair_hue', 'Hair Hue Shift', default=0, tip='Subtle hue shift on parsed hair region')
        self._make_slider(g_bl, 'hair_saturation', 'Hair Saturation', default=0, tip='Boosts parsed hair color intensity')
        self._make_slider(g_bl, 'hair_density', 'Hair Density Hint', default=0, tip='Darkens parsed hair region for a denser look')
        self._make_slider(g_bl, 'blush', 'Blush Overlay', default=0, tip='Procedural cheek blush opacity')
        self._make_slider(g_bl, 'lip_gloss', 'Lip Gloss', default=0, tip='Adds soft lip highlight and warmth')
        self._make_slider(g_bl, 'eye_shadow', 'Eye Shadow', default=0, tip='Procedural eye shadow opacity')
        parser_row = QHBoxLayout()
        parser_row.addWidget(QLabel("Parser Model:"))
        self.combo_parser_model = QComboBox()
        for key, cfg in PARSER_MODELS.items():
            self.combo_parser_model.addItem(cfg["label"], key)
        self.combo_parser_model.currentIndexChanged.connect(self._on_parser_model_changed)
        parser_row.addWidget(self.combo_parser_model)
        g_bl.addLayout(parser_row)
        self.parsing_lbl = QLabel("")
        self.parsing_lbl.setStyleSheet("color:#a6e3a1; font-size:10px;")
        g_bl.addWidget(self.parsing_lbl)
        t1.addWidget(g_beauty)

        g2 = QGroupBox("Quality"); g2l = QVBoxLayout(g2); g2l.setSpacing(6)
        self._make_slider(g2l, 'smoothing', 'Warp Smoothing', default=50, tip='Displacement field smoothness')
        self._make_slider(g2l, 'temporal', 'Temporal Stability', default=50, tip='Landmark jitter reduction (higher=smoother)')
        self._make_slider(g2l, 'bg_protect', 'Background Protection', default=70, tip='Prevents warping background - blends face region only')
        self._make_slider(g2l, 'matting_refine', 'Matting Refine', default=0, tip='MODNet portrait matte edge refinement for ROI warp masks')

        opts_row = QHBoxLayout()
        self.chk_lm = QCheckBox("Landmarks"); self.chk_lm.toggled.connect(self._tog_lm)
        opts_row.addWidget(self.chk_lm)
        self.chk_conf = QCheckBox("Confidence"); self.chk_conf.toggled.connect(self._tog_conf)
        opts_row.addWidget(self.chk_conf)
        self.chk_teeth_hint = QCheckBox("Teeth Mask"); self.chk_teeth_hint.toggled.connect(self._tog_teeth_hint)
        opts_row.addWidget(self.chk_teeth_hint)
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
        exp_opts = QHBoxLayout(); exp_opts.setSpacing(6)
        self.chk_watermark = QCheckBox("Disclosure watermark")
        exp_opts.addWidget(self.chk_watermark)
        exp_opts.addWidget(QLabel("Video:"))
        self.combo_video_compare = QComboBox()
        self.combo_video_compare.addItems(["Normal", "Split", "Side-by-side"])
        exp_opts.addWidget(self.combo_video_compare)
        g5l.addLayout(exp_opts)
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
        self.btn_batch_manifest = QPushButton("Run Manifest")
        self.btn_batch_manifest.setProperty("secondary", True)
        self.btn_batch_manifest.clicked.connect(self.start_batch_manifest); g6l.addWidget(self.btn_batch_manifest)
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
    def _tog_teeth_hint(self, on):
        if self.video_thread:
            self.video_thread.show_teeth_hint = on
        if self.image_mode and self.current_original is not None and self.current_processed is not None:
            self._compute_image_teeth_hint_rois()
            self._display_frame(self.current_original, self.current_processed, self.current_faces)
    def _toggle_compare(self, checked):
        self.comparison_mode = checked
        if checked: self.video_label.divider_ratio = 0.5
        if self.image_mode and self.current_original is not None and self.current_processed is not None:
            self._display_frame(self.current_original, self.current_processed, self.current_faces)

    def _toggle_virtualcam(self, checked):
        if checked and not HAS_VIRTUALCAM:
            self.btn_virtualcam.setChecked(False)
            self.toast.show_message("Install pyvirtualcam to enable virtual camera", 4000)
            return
        if checked and not self.video_thread:
            self.btn_virtualcam.setChecked(False)
            self.toast.show_message("Start webcam or video first")
            return
        if self.video_thread:
            self.video_thread.virtualcam_enabled = checked
        self.toast.show_message("Virtual camera starting..." if checked else "Virtual camera stopped")

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

    def _reset_timeline(self):
        self._timeline_duration = 0.0
        self._timeline_dragging = False
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.blockSignals(False)
        self.timeline_label.setText("--:-- / --:--")

    def _timeline_pressed(self):
        self._timeline_dragging = True

    def _timeline_seek(self, value):
        self.timeline_label.setText(
            f"{format_timecode(value)} / {format_timecode(self._timeline_duration)}")
        if self.video_thread and self.video_thread.source is not None:
            self.video_thread.request_seek(value)

    def _timeline_released(self):
        self._timeline_dragging = False
        self._timeline_seek(self.timeline_slider.value())

    def _on_timeline(self, current_sec, duration_sec):
        self._timeline_duration = max(0.0, float(duration_sec or 0.0))
        if self._timeline_duration <= 0:
            if not self._timeline_dragging:
                self.timeline_slider.setEnabled(False)
                self.timeline_slider.setRange(0, 0)
                self.timeline_label.setText("--:-- / --:--")
            return
        max_seconds = max(1, int(math.ceil(self._timeline_duration)))
        if self.timeline_slider.maximum() != max_seconds:
            self.timeline_slider.setRange(0, max_seconds)
        self.timeline_slider.setEnabled(True)
        current_value = max(0, min(max_seconds, int(round(current_sec))))
        if not self._timeline_dragging:
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(current_value)
            self.timeline_slider.blockSignals(False)
        self.timeline_label.setText(
            f"{format_timecode(current_value)} / {format_timecode(self._timeline_duration)}")

    def _video_compare_mode(self):
        return ['none', 'split', 'side_by_side'][self.combo_video_compare.currentIndex()]

    def _parser_model_key(self):
        if hasattr(self, "combo_parser_model"):
            return parser_model_key(self.combo_parser_model.currentData())
        return DEFAULT_PARSER_MODEL

    def _set_parser_model_status(self):
        key = self._parser_model_key()
        ready = parser_model_ready(key)
        self.parsing_lbl.setText(
            f"  Face Parsing: {parser_model_label(key)}"
            + (" ready" if ready else " downloads on first use")
        )
        self.parsing_lbl.setStyleSheet(f"color: {'#a6e3a1' if ready else '#f9e2af'}; font-size: 10px;")

    def _on_parser_model_changed(self, _idx):
        self._set_parser_model_status()
        if self._image_engine is not None:
            self._image_engine.close()
            self._image_engine = None
        if self.video_thread and self.video_thread.isRunning():
            self._go(self.video_thread.source)
        elif self.image_mode:
            self._process_current_image()
        if hasattr(self, "toast"):
            self.toast.show_message(f"Parser: {parser_model_label(self._parser_model_key())}")

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
        self._reset_timeline()
        # Reset cached engine so filter state doesn't bleed between images
        if self._image_engine is not None:
            self._image_engine.close()
            self._image_engine = None
        self._process_current_image()

    def _process_current_image(self):
        if self.current_original is None: return
        nf = self.spin_faces.value()
        parser_model = self._parser_model_key()
        if (self._image_engine is None or self._image_engine_faces != nf
                or self._image_engine_parser != parser_model):
            if self._image_engine is not None:
                self._image_engine.close()
            self._image_engine = FaceWarpEngine('image', nf, parser_model)
            self._image_engine_faces = nf
            self._image_engine_parser = parser_model
        result, faces = self._image_engine.warp_single_image(self.current_original, self._p())
        self.current_processed = result
        self.current_faces = faces
        self._compute_image_teeth_hint_rois()
        self.face_count_label.setText(f"  {len(faces)} face{'s' if len(faces) != 1 else ''} detected")
        self._display_frame(self.current_original, result, faces)

    def _compute_image_teeth_hint_rois(self):
        self.current_teeth_hint_rois = []
        if (not hasattr(self, "chk_teeth_hint") or not self.chk_teeth_hint.isChecked()
                or self._image_engine is None or self.current_processed is None or not self.current_faces):
            return
        self.current_teeth_hint_rois = self._image_engine.teeth_hint_rois(
            self.current_processed, self.current_faces)

    def _go(self, src):
        self.stop_video()
        self._reset_timeline()
        self.video_thread = VideoThread()
        self.video_thread.set_source(src)
        self.video_thread.update_params(self._p())
        self.video_thread.show_landmarks = self.chk_lm.isChecked()
        self.video_thread.show_confidence = self.chk_conf.isChecked()
        self.video_thread.show_teeth_hint = self.chk_teeth_hint.isChecked()
        self.video_thread.max_faces = self.spin_faces.value()
        self.video_thread.parser_model = self._parser_model_key()
        scales = [1.0, 0.75, 0.5]
        self.video_thread.preview_scale = scales[self.combo_scale.currentIndex()]
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.fps_update.connect(self._on_fps)
        self.video_thread.timeline_update.connect(self._on_timeline)
        self.video_thread.virtualcam_update.connect(self._on_virtualcam_update)
        self.video_thread.status_update.connect(self.statusBar().showMessage)
        self.video_thread.error_update.connect(self._on_video_error)
        self.video_thread.start()
        self.btn_stop.setEnabled(True)
        self.btn_virtualcam.setEnabled(True)
        nm = "Webcam" if src is None else os.path.basename(src)
        self.statusBar().showMessage(f"Playing: {nm}")

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop(); self.video_thread = None
        self._reset_timeline()
        self.btn_virtualcam.blockSignals(True)
        self.btn_virtualcam.setChecked(False)
        self.btn_virtualcam.setEnabled(False)
        self.btn_virtualcam.blockSignals(False)
        self.btn_stop.setEnabled(False); self.fps_label.setText("-- FPS")

    # ── Frame Display ───────────────────────────────────────────
    def _on_frame(self, orig, proc, faces, teeth_hint_rois=None):
        self.current_original = orig
        self.current_processed = proc
        self.current_faces = faces
        self.current_teeth_hint_rois = teeth_hint_rois or []
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
            proc_show = (apply_teeth_hint_rois(proc, self.current_teeth_hint_rois)
                         if self.chk_teeth_hint.isChecked() else proc)
            show = np.empty_like(orig)
            show[:, :sx] = orig[:, :sx]; show[:, sx:] = proc_show[:, sx:]
            cv2.line(show, (sx, 0), (sx, h), (203, 166, 247), 3)
            cy = h // 2
            cv2.fillPoly(show, [np.array([[sx-8,cy-12],[sx+8,cy-12],[sx+8,cy+12],[sx-8,cy+12]])], (203,166,247))
            cv2.line(show, (sx, cy-8), (sx, cy+8), (30,30,46), 2)
            for text, tx, color in [("Original", 10, (255,255,255)), ("Slimmed", sx+10, (137,180,250))]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(show, (tx-4, 10), (tx+tw+4, th+22), (30,30,46), -1)
                cv2.putText(show, text, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            show = (apply_teeth_hint_rois(proc, self.current_teeth_hint_rois)
                    if self.chk_teeth_hint.isChecked() else proc)

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

    def _on_virtualcam_update(self, enabled, msg):
        self.btn_virtualcam.blockSignals(True)
        self.btn_virtualcam.setChecked(enabled)
        self.btn_virtualcam.blockSignals(False)
        self.statusBar().showMessage(msg)
        self.toast.show_message(msg, 4000)

    # ── Export: Video ───────────────────────────────────────────
    def export_video(self):
        if not self.source_path: return
        base = os.path.splitext(os.path.basename(self.source_path))[0]
        out, _ = QFileDialog.getSaveFileName(self, "Save", f"{base}_slimmed.mp4", "MP4 (*.mp4)")
        if not out: return
        self.exp_prog.setVisible(True); self.exp_prog.setValue(0)
        self.btn_exp_video.setEnabled(False); self.btn_exp_cancel.setEnabled(True)
        self._et = ExportThread(self.source_path, out, self._p(), self.spin_faces.value(),
                                self.chk_watermark.isChecked(), self._video_compare_mode(),
                                self._parser_model_key())
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
            result = apply_disclosure_watermark(self.current_processed, self.chk_watermark.isChecked())
            save_rgb_image(out, result, self.source_path, True)
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

    def start_batch_manifest(self):
        manifest, _ = QFileDialog.getOpenFileName(self, "Open Batch Manifest", "", "JSON (*.json)")
        if not manifest:
            return
        try:
            jobs = load_batch_manifest(manifest, fallback_parser_model=self._parser_model_key())
        except Exception as e:
            self.toast.show_message(f"Manifest error: {e}", 5000)
            return
        if not jobs:
            self.toast.show_message("Manifest has no jobs")
            return
        output_dir = jobs[0].get("output_dir") or os.path.join(os.path.dirname(manifest), "faceslim_output")
        self._run_batch([job["input"] for job in jobs], output_dir=output_dir, jobs=jobs)

    def _run_batch(self, files, output_dir=None, jobs=None):
        output_dir = output_dir or QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir: return
        batch_jobs = jobs or jobs_from_files(files, output_dir, self._p(), self.spin_faces.value(),
                                             self.chk_watermark.isChecked(), True,
                                             self._video_compare_mode(), self._parser_model_key())
        self.batch_prog.setVisible(True); self.batch_prog.setValue(0)
        self.btn_batch.setEnabled(False); self.btn_batch_folder.setEnabled(False); self.btn_batch_manifest.setEnabled(False)
        self.btn_batch_cancel.setEnabled(True)
        self._batch_dialog = BatchQueueDialog(batch_jobs, self)
        self._batch_dialog.show()
        self._batch_thread = BatchThread(files, output_dir, self._p(), self.spin_faces.value(),
                                         self.chk_watermark.isChecked(), True,
                                         self._video_compare_mode(), batch_jobs,
                                         self._parser_model_key())
        self._batch_dialog.cancel_job_requested.connect(self._batch_thread.cancel_job)
        self._batch_thread.progress.connect(lambda c, t, f: (
            self.batch_prog.setValue(int(c / t * 100)),
            self.batch_stat.setText(f"Processing {c}/{t}: {f}")))
        self._batch_thread.job_update.connect(self._batch_dialog.update_job)
        self._batch_thread.overall_update.connect(lambda c, t, eta: (
            self.batch_prog.setValue(int(c / max(t, 1) * 100)),
            self.batch_stat.setText(f"Batch {c}/{t} | ETA: {eta}"),
            self._batch_dialog.update_summary(c, t, eta)))
        self._batch_thread.finished.connect(lambda p, f, s: (
            self.btn_batch.setEnabled(True), self.btn_batch_folder.setEnabled(True),
            self.btn_batch_manifest.setEnabled(True),
            self.btn_batch_cancel.setEnabled(False),
            self.batch_stat.setText(f"Done: {p} processed, {f} failed, {s} skipped"),
            self.toast.show_message(f"Batch complete: {p} files")))
        self._batch_thread.error.connect(lambda m: (
            self.btn_batch.setEnabled(True), self.btn_batch_folder.setEnabled(True),
            self.btn_batch_manifest.setEnabled(True),
            self.btn_batch_cancel.setEnabled(False),
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
        self.chk_watermark.setChecked(self.settings.value("watermark", False, type=bool))
        self.chk_teeth_hint.setChecked(self.settings.value("teeth_hint", False, type=bool))
        self.combo_video_compare.setCurrentIndex(self.settings.value("video_compare", 0, type=int))
        saved_parser = parser_model_key(self.settings.value("parser_model", DEFAULT_PARSER_MODEL, type=str))
        parser_idx = self.combo_parser_model.findData(saved_parser)
        if parser_idx >= 0:
            self.combo_parser_model.setCurrentIndex(parser_idx)
        self._set_parser_model_status()

    def _save_settings(self):
        for k, (s, _) in self.sliders.items():
            self.settings.setValue(f"s/{k}", s.value())
        self.settings.setValue("max_faces", self.spin_faces.value())
        self.settings.setValue("watermark", self.chk_watermark.isChecked())
        self.settings.setValue("teeth_hint", self.chk_teeth_hint.isChecked())
        self.settings.setValue("video_compare", self.combo_video_compare.currentIndex())
        self.settings.setValue("parser_model", self._parser_model_key())

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
    parser_model = parser_model_key(args.parser_model)
    ensure_parsing_model(parser_model)  # Non-fatal - falls back to landmark mask
    overrides = {k: getattr(args, k) for k in CLI_PARAM_KEYS
                 if hasattr(args, k) and getattr(args, k) is not None}
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
        jobs = load_batch_manifest(args.manifest, args.output, parser_model)
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
                               not args.strip_metadata, args.video_compare, parser_model)

    print(f"\nFaceSlim v{VERSION} - CLI Mode")
    print(f"  GPU: {'ON (' + GPU_NAME + ')' if USE_GPU else 'OFF'}")
    print(f"  Files: {len(inputs)}")
    print(f"  Params: {json.dumps({k: v for k, v in params.items() if v != 0}, indent=2)}")
    print(f"  Output: {output_dir}\n")

    processed = failed = 0
    for i, job in enumerate(jobs):
        filepath = job["input"]
        params = job.get("params", params)
        max_faces = job.get("max_faces", max_faces)
        watermark = job.get("watermark", args.watermark)
        preserve_metadata = job.get("preserve_metadata", not args.strip_metadata)
        compare_mode = job.get("compare_mode", args.video_compare)
        parser_model = job.get("parser_model", parser_model)
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        print(f"  [{i+1}/{len(jobs)}] {fname}...", end=' ', flush=True)
        eng = None
        cap = None
        writer = None

        try:
            if ext in IMAGE_EXTS:
                eng = FaceWarpEngine('image', max_faces, parser_model)
                img = cv2.imread(filepath)
                if img is None: raise ValueError("Cannot read image")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result, faces = eng.warp_single_image(rgb, params)
                result = apply_disclosure_watermark(result, watermark)
                out_path = job.get("output") or os.path.join(job.get("output_dir", output_dir),
                    os.path.splitext(fname)[0] + '_slimmed.png')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                save_rgb_image(out_path, result, filepath, preserve_metadata)
                eng.close(); eng = None
                print(f"OK ({len(faces)} face{'s' if len(faces) != 1 else ''})")
                processed += 1

            elif ext in VIDEO_EXTS:
                eng = FaceWarpEngine('video', max_faces, parser_model)
                eng.grid_scale = 6
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened(): raise ValueError("Cannot open video")
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = job.get("output") or os.path.join(job.get("output_dir", output_dir),
                    os.path.splitext(fname)[0] + '_slimmed.mp4')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                out_w, out_h = video_output_size(w, h, compare_mode)
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
                if not writer.isOpened():
                    raise ValueError(f"Cannot open output writer: {out_path}")
                has_fx = any(abs(params.get(k, 0)) > 0 for k in EFFECT_PARAM_KEYS)
                t0 = time.time()
                for fi in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = eng.detect(rgb)
                    proc = eng.warp(rgb, faces, params) if (faces and has_fx) else rgb
                    proc = compose_compare_frame(rgb, proc, compare_mode)
                    proc = apply_disclosure_watermark(proc, watermark)
                    writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                    if fi % 30 == 0 and fi > 0:
                        eta = ((time.time() - t0) / fi) * (total - fi)
                        print(f"\r  [{i+1}/{len(jobs)}] {fname}... {fi}/{total} frames (ETA: {int(eta)}s)", end='', flush=True)
                cap.release(); cap = None
                writer.release(); writer = None
                eng.close(); eng = None
                elapsed = time.time() - t0
                print(f"\r  [{i+1}/{len(jobs)}] {fname}... OK ({total} frames, {elapsed:.1f}s)")
                processed += 1
            else:
                print(f"SKIP (unsupported)")
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
        print(f"Render diagnostics: {RENDER_LOG_PATH}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
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
    parser.add_argument('--faces', type=int, help='Max faces to process (1-5)')
    parser.add_argument('--face-preset', action='append', default=[], metavar='FACE=PRESET',
                        help='Apply a preset to one face index, e.g. 2=Beauty')
    parser.add_argument('--face-param', action='append', default=[], metavar='FACE:key=value',
                        help='Override one parameter for a face, e.g. 1:jaw=35')
    parser.add_argument('--manifest', help='JSON batch manifest with per-file/per-face settings')
    parser.add_argument('--watermark', action='store_true', help='Add AI modification disclosure watermark')
    parser.add_argument('--strip-metadata', action='store_true', help='Do not preserve image EXIF/XMP/ICC metadata')
    parser.add_argument('--video-compare', choices=['none', 'split', 'side_by_side'], default='none',
                        help='Export processed video normally, split-screen, or side-by-side')
    parser.add_argument('--parser-model', choices=list(PARSER_MODELS.keys()), default=DEFAULT_PARSER_MODEL,
                        help='Face parsing model for beauty masks')
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


if __name__ == "__main__":
    main()
