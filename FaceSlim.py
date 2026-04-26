#!/usr/bin/env python3
"""
FaceSlim v0.2.0 - AI Face Slimming Application
Real-time face reshaping using MediaPipe 478-landmark Face Landmarker
and smooth RBF/TPS warping with One-Euro temporal stabilization.

Turnkey: auto-installs all dependencies and downloads model on first run.
"""

import sys, os, subprocess, time, json, math, traceback, urllib.request
from collections import deque
from pathlib import Path


# codex-branding:start
def _branding_icon_path() -> Path:
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "icon.png")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "icon.png")
    current = Path(__file__).resolve()
    candidates.extend([current.parent / "icon.png", current.parent.parent / "icon.png", current.parent.parent.parent / "icon.png"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("icon.png")
# codex-branding:end


# -- Auto-Bootstrap --
def _bootstrap():
    if sys.version_info < (3, 9):
        print("Python 3.9+ required"); sys.exit(1)
    try:
        import pip
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'ensurepip', '--default-pip'])
    required = {
        'PyQt5': 'PyQt5',
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'scipy': 'scipy',
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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QGroupBox,
    QStatusBar, QProgressBar, QCheckBox,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QCursor, QIcon

VERSION = "0.2.0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

def exception_handler(exc_type, exc_value, exc_tb):
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with open(os.path.join(APP_DIR, 'crash.log'), 'w') as f:
        f.write(msg)
    print(f"CRASH: {msg}")
    sys.exit(1)

sys.excepthook = exception_handler

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return True
    print("Downloading face landmarker model (~3.7 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

# -- MediaPipe Face Landmark Indices --
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


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq, self.mincutoff, self.beta, self.dcutoff = freq, mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def _alpha(self, cutoff):
        r = 2 * math.pi * cutoff / self.freq
        return r / (r + 1)

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


class FaceWarpEngine:
    def __init__(self, mode='video'):
        rm = vision.RunningMode.VIDEO if mode == 'video' else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=rm, num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.mode = mode
        self.lm_filter = OneEuroFilter(freq=30, mincutoff=1.5, beta=0.01)
        self.ts = 0
        # Displacement field cache
        self._cached_dx = None
        self._cached_dy = None
        self._cached_map_x = None
        self._cached_map_y = None
        self._cache_src_hash = None
        self._cache_dims = None

    def detect(self, frame_rgb):
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.mode == 'video':
            self.ts += 33
            res = self.landmarker.detect_for_video(mpi, self.ts)
        else:
            res = self.landmarker.detect(mpi)
        if not res.face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]], dtype=np.float64)
        return self.lm_filter(pts, time.time())

    def _compute_control_points(self, lms, params, h, w):
        """Compute source and target control points for the warp."""
        center = np.mean(lms[[i for i in FACE_CENTER if i < len(lms)]], axis=0)
        src, tgt = [], []
        jaw_s   = params.get('jaw', 0) / 100.0
        cheek_s = params.get('cheeks', 0) / 100.0
        chin_s  = params.get('chin', 0) / 100.0
        width_s = params.get('face_width', 0) / 100.0

        def shift(indices, strength, vbias=0.0):
            for idx in indices:
                if idx >= len(lms): continue
                pt = lms[idx]
                vec = center - pt
                d = np.linalg.norm(vec)
                if d < 1: continue
                disp = (vec / d) * strength * min(d / 100.0, 1.0) * 15.0
                if vbias: disp[1] += vbias * strength * 8.0
                src.append(pt.copy())
                tgt.append(pt + disp)

        if abs(jaw_s) > 0.01:
            shift(LEFT_JAW, jaw_s); shift(RIGHT_JAW, jaw_s)
        if abs(cheek_s) > 0.01:
            shift(LEFT_CHEEK, cheek_s); shift(RIGHT_CHEEK, cheek_s)
        if abs(chin_s) > 0.01:
            shift(CHIN, chin_s * 0.5, vbias=-1.0)
        if abs(width_s) > 0.01:
            shift(list(set(JAW_CONTOUR) - set(FOREHEAD)), width_s)

        for idx in set(NOSE_BRIDGE + FOREHEAD):
            if idx < len(lms):
                src.append(lms[idx].copy()); tgt.append(lms[idx].copy())

        fmin, fmax = np.min(lms, axis=0), np.max(lms, axis=0)
        m = 80
        for c in [[fmin[0]-m, fmin[1]-m], [fmax[0]+m, fmin[1]-m],
                   [fmin[0]-m, fmax[1]+m], [fmax[0]+m, fmax[1]+m],
                   [center[0], fmin[1]-m*2], [center[0], fmax[1]+m*2],
                   [fmin[0]-m*2, center[1]], [fmax[0]+m*2, center[1]]]:
            src.append(np.array(c, dtype=np.float64))
            tgt.append(np.array(c, dtype=np.float64))

        return np.array(src), np.array(tgt)

    def _compute_remap_maps(self, src, tgt, h, w, smoothing):
        """Compute the full displacement field and remap maps via RBF."""
        disp = tgt - src
        smooth = max(0.1, smoothing / 100.0 * 50.0)
        sc = 4
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
        return dx, dy, mx - dx, my - dy

    def _make_cache_key(self, src, tgt):
        """Create a hash to detect when control points change significantly."""
        disp = tgt - src
        # Quantize to nearest 2 pixels -- if movement is < 2px, reuse cache
        quantized = (disp / 2.0).astype(np.int32)
        return quantized.tobytes()

    def warp(self, frame, lms, params):
        h, w = frame.shape[:2]
        src, tgt = self._compute_control_points(lms, params, h, w)

        if len(src) < 4:
            return frame
        if np.max(np.linalg.norm(tgt - src, axis=1)) < 0.5:
            return frame

        try:
            cache_key = self._make_cache_key(src, tgt)
            dims = (h, w)

            # Reuse cached displacement field if landmarks haven't moved much
            if (self._cached_map_x is not None and
                self._cache_src_hash == cache_key and
                self._cache_dims == dims):
                map_x, map_y = self._cached_map_x, self._cached_map_y
            else:
                dx, dy, map_x, map_y = self._compute_remap_maps(
                    src, tgt, h, w, params.get('smoothing', 50))
                self._cached_dx = dx
                self._cached_dy = dy
                self._cached_map_x = map_x
                self._cached_map_y = map_y
                self._cache_src_hash = cache_key
                self._cache_dims = dims

            return cv2.remap(frame, map_x, map_y,
                             interpolation=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
        except Exception as e:
            print(f"Warp error: {e}")
            return frame

    def close(self):
        self.landmarker.close()


class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object)
    fps_update = pyqtSignal(float)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = self.show_landmarks = False
        self.paused = False
        self.source = None
        self.params = {'jaw':0, 'cheeks':0, 'chin':0, 'face_width':0, 'smoothing':50}

    def set_source(self, s): self.source = s
    def update_params(self, p): self.params = p.copy()

    def run(self):
        self.running = True
        eng = FaceWarpEngine('video')
        cap = cv2.VideoCapture(0 if self.source is None else self.source)
        if not cap.isOpened():
            self.status_update.emit("Failed to open video source")
            eng.close(); return
        fps_t = deque(maxlen=30)
        while self.running:
            if self.paused: self.msleep(50); continue
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                if self.source is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    eng.close(); eng = FaceWarpEngine('video')
                    continue
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = eng.detect(rgb)
            has_fx = any(abs(self.params.get(k, 0)) > 0 for k in ['jaw','cheeks','chin','face_width'])
            proc = eng.warp(rgb, lms, self.params) if (lms is not None and has_fx) else rgb.copy()
            el = time.time() - t0
            fps_t.append(el)
            if len(fps_t) > 2:
                self.fps_update.emit(1.0 / max(sum(fps_t)/len(fps_t), 0.001))
            self.frame_ready.emit(rgb, proc, lms)
            self.msleep(1)
        cap.release(); eng.close()

    def stop(self):
        self.running = False; self.wait(3000)


class ExportThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, inp, out, params):
        super().__init__()
        self.inp, self.out, self.params = inp, out, params

    def run(self):
        try:
            eng = FaceWarpEngine('video')
            cap = cv2.VideoCapture(self.inp)
            if not cap.isOpened():
                self.error.emit("Cannot open input"); return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.status.emit(f"Exporting {total} frames...")
            t_start = time.time()
            for i in range(total):
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lms = eng.detect(rgb)
                has_fx = any(abs(self.params.get(k,0)) > 0 for k in ['jaw','cheeks','chin','face_width'])
                proc = eng.warp(rgb, lms, self.params) if (lms is not None and has_fx) else rgb
                writer.write(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
                if total > 0:
                    pct = int((i+1)/total*100)
                    self.progress.emit(pct)
                    # ETA calculation
                    elapsed = time.time() - t_start
                    if i > 0:
                        eta = (elapsed / (i+1)) * (total - i - 1)
                        mins, secs = divmod(int(eta), 60)
                        self.status.emit(f"Frame {i+1}/{total} | ETA: {mins}m {secs}s")
            cap.release(); writer.release(); eng.close()
            # Mux audio
            try:
                tmp = self.out.replace('.mp4','_mux.mp4')
                r = subprocess.run(['ffmpeg','-y','-i',self.out,'-i',self.inp,
                    '-c:v','copy','-c:a','aac','-map','0:v:0','-map','1:a:0?',
                    '-shortest',tmp], capture_output=True, timeout=300)
                if r.returncode == 0 and os.path.exists(tmp):
                    os.replace(tmp, self.out)
            except Exception: pass
            elapsed_total = time.time() - t_start
            mins, secs = divmod(int(elapsed_total), 60)
            self.status.emit(f"Done in {mins}m {secs}s")
            self.finished.emit(self.out)
        except Exception as e:
            self.error.emit(str(e))


def draw_lm(frame, lms):
    ov = frame.copy()
    for group, col in [(LEFT_JAW,(166,227,161)),(RIGHT_JAW,(166,227,161)),
                        (LEFT_CHEEK,(249,226,175)),(RIGHT_CHEEK,(249,226,175)),
                        (CHIN,(243,139,168))]:
        pts = [lms[i].astype(int) for i in group if i < len(lms)]
        for p in pts: cv2.circle(ov, tuple(p), 2, col, -1, cv2.LINE_AA)
        for i in range(len(pts)-1):
            cv2.line(ov, tuple(pts[i]), tuple(pts[i+1]), col, 1, cv2.LINE_AA)
    return cv2.addWeighted(ov, 0.7, frame, 0.3, 0)


# -- Draggable A/B Compare Video Label --
class CompareVideoLabel(QLabel):
    """QLabel with draggable vertical divider for A/B comparison."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.divider_ratio = 0.5  # 0.0 = full original, 1.0 = full slimmed
        self._dragging = False
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_ratio(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_ratio(event.pos())
        # Show resize cursor when compare mode is active
        parent = self.parent()
        while parent and not isinstance(parent, FaceSlimApp):
            parent = parent.parent()
        if parent and parent.comparison_mode:
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def _update_ratio(self, pos):
        if self.width() > 0:
            self.divider_ratio = max(0.05, min(0.95, pos.x() / self.width()))


DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 1.2em;
    padding: 12px 8px 8px 8px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #89b4fa; }
QPushButton { background-color: #89b4fa; color: #1e1e2e; border: none;
    padding: 8px 20px; border-radius: 6px; font-weight: bold; font-size: 12px; }
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
"""


class FaceSlimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FaceSlim v{VERSION}")
        self.setMinimumSize(1100, 700); self.resize(1280, 780)
        self.settings = QSettings("FaceSlim", "FaceSlim")
        self.video_thread = None
        self.source_path = None
        self.comparison_mode = False
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setSpacing(12); root.setContentsMargins(12,12,12,12)

        # -- Left: Video --
        left = QVBoxLayout(); left.setSpacing(8)
        hdr = QHBoxLayout()
        t = QLabel("FaceSlim")
        t.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        hdr.addWidget(t); hdr.addStretch()
        self.fps_label = QLabel("-- FPS")
        self.fps_label.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold;")
        hdr.addWidget(self.fps_label)
        left.addLayout(hdr)

        self.video_label = CompareVideoLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            "background-color:#11111b; border:2px solid #313244; border-radius:8px; color:#6c7086; font-size:14px;")
        self.video_label.setText("Load a video or start webcam to begin")
        left.addWidget(self.video_label, 1)

        tr = QHBoxLayout(); tr.setSpacing(8)
        self.btn_webcam = QPushButton("Webcam")
        self.btn_webcam.clicked.connect(self.start_webcam); tr.addWidget(self.btn_webcam)
        self.btn_load = QPushButton("Load Video")
        self.btn_load.setProperty("secondary", True)
        self.btn_load.clicked.connect(self.load_video); tr.addWidget(self.btn_load)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.clicked.connect(self.stop_video); self.btn_stop.setEnabled(False)
        tr.addWidget(self.btn_stop); tr.addStretch()
        self.btn_cmp = QPushButton("A/B Compare")
        self.btn_cmp.setProperty("secondary", True); self.btn_cmp.setCheckable(True)
        self.btn_cmp.toggled.connect(self._toggle_compare)
        tr.addWidget(self.btn_cmp)
        left.addLayout(tr)
        root.addLayout(left, 3)

        # -- Right: Controls --
        rw = QWidget(); rw.setMaximumWidth(340); rw.setMinimumWidth(280)
        rl = QVBoxLayout(rw); rl.setSpacing(10); rl.setContentsMargins(0,0,0,0)

        g1 = QGroupBox("Face Reshaping"); g1l = QVBoxLayout(g1); g1l.setSpacing(6)
        self.sliders = {}
        for key, label, mx, dv, tip in [
            ('jaw','Jaw Slimming',100,0,'Narrows the jawline'),
            ('cheeks','Cheek Slimming',100,0,'Reduces cheek fullness'),
            ('chin','Chin Reshape',100,0,'Lifts and narrows the chin'),
            ('face_width','Overall Width',100,0,'Reduces overall face width'),
        ]:
            rv = QVBoxLayout(); rv.setSpacing(2)
            lr = QHBoxLayout(); lr.addWidget(QLabel(label)); lr.addStretch()
            vl = QLabel(f"{dv}%")
            vl.setStyleSheet("color:#f9e2af; font-size:12px; font-weight:bold; min-width:36px;")
            lr.addWidget(vl); rv.addLayout(lr)
            s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, mx); s.setValue(dv); s.setToolTip(tip)
            s.valueChanged.connect(lambda v, k=key, lab=vl: (lab.setText(f"{v}%"), self._push()))
            rv.addWidget(s); g1l.addLayout(rv)
            self.sliders[key] = (s, vl)
        rl.addWidget(g1)

        g2 = QGroupBox("Quality"); g2l = QVBoxLayout(g2); g2l.setSpacing(6)
        sr = QVBoxLayout(); sr.setSpacing(2)
        slr = QHBoxLayout(); slr.addWidget(QLabel("Warp Smoothing")); slr.addStretch()
        self.sv = QLabel("50%")
        self.sv.setStyleSheet("color:#f9e2af; font-size:12px; font-weight:bold;")
        slr.addWidget(self.sv); sr.addLayout(slr)
        self.ss = QSlider(Qt.Orientation.Horizontal); self.ss.setRange(10,100); self.ss.setValue(50)
        self.ss.valueChanged.connect(lambda v: (self.sv.setText(f"{v}%"), self._push()))
        sr.addWidget(self.ss); g2l.addLayout(sr)
        self.chk_lm = QCheckBox("Show Face Landmarks")
        self.chk_lm.toggled.connect(self._tog_lm); g2l.addWidget(self.chk_lm)
        rl.addWidget(g2)

        g3 = QGroupBox("Quick Presets"); g3l = QGridLayout(g3); g3l.setSpacing(6)
        for i, (nm, vl, pr) in enumerate([
            ("Subtle",{'jaw':15,'cheeks':10,'chin':5,'face_width':10},"secondary"),
            ("Moderate",{'jaw':35,'cheeks':25,'chin':15,'face_width':20},"secondary"),
            ("Strong",{'jaw':60,'cheeks':45,'chin':30,'face_width':35},"secondary"),
            ("Reset",{'jaw':0,'cheeks':0,'chin':0,'face_width':0},"danger"),
        ]):
            b = QPushButton(nm); b.setProperty(pr, True)
            b.clicked.connect(lambda _, v=vl: self._preset(v))
            g3l.addWidget(b, i//2, i%2)
        rl.addWidget(g3)

        g4 = QGroupBox("Export"); g4l = QVBoxLayout(g4); g4l.setSpacing(8)
        self.btn_exp = QPushButton("Export Video")
        self.btn_exp.setProperty("success", True)
        self.btn_exp.clicked.connect(self.export_video); self.btn_exp.setEnabled(False)
        g4l.addWidget(self.btn_exp)
        self.exp_prog = QProgressBar(); self.exp_prog.setVisible(False); g4l.addWidget(self.exp_prog)
        self.exp_stat = QLabel(""); self.exp_stat.setStyleSheet("color:#6c7086; font-size:11px;")
        g4l.addWidget(self.exp_stat)
        rl.addWidget(g4)

        rl.addStretch()
        root.addWidget(rw, 1)
        self.statusBar().showMessage("Ready - Load a video or start webcam")

    def _toggle_compare(self, checked):
        self.comparison_mode = checked
        if checked:
            self.video_label.divider_ratio = 0.5

    def _push(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.update_params(self._p())

    def _p(self):
        p = {k: s.value() for k, (s, _) in self.sliders.items()}
        p['smoothing'] = self.ss.value()
        return p

    def _tog_lm(self, on):
        if self.video_thread: self.video_thread.show_landmarks = on

    def _preset(self, v):
        for k, val in v.items():
            if k in self.sliders: self.sliders[k][0].setValue(val)

    def start_webcam(self): self._go(None)
    def load_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.wmv);;All Files (*)")
        if p:
            self.source_path = p; self._go(p); self.btn_exp.setEnabled(True)

    def _go(self, src):
        self.stop_video()
        self.video_thread = VideoThread()
        self.video_thread.set_source(src)
        self.video_thread.update_params(self._p())
        self.video_thread.show_landmarks = self.chk_lm.isChecked()
        self.video_thread.frame_ready.connect(self._frame)
        self.video_thread.fps_update.connect(self._fps)
        self.video_thread.status_update.connect(self.statusBar().showMessage)
        self.video_thread.start()
        self.btn_stop.setEnabled(True)
        nm = "Webcam" if src is None else os.path.basename(src)
        self.statusBar().showMessage(f"Playing: {nm}")

    def stop_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop(); self.video_thread = None
        self.btn_stop.setEnabled(False); self.fps_label.setText("-- FPS")

    def _frame(self, orig, proc, lms):
        if self.comparison_mode:
            h, w = orig.shape[:2]
            # Use divider_ratio from the draggable label
            split_x = int(w * self.video_label.divider_ratio)
            split_x = max(1, min(w - 1, split_x))
            # Build composite -- must use .copy() for contiguous memory
            show = np.empty_like(orig)
            show[:, :split_x] = orig[:, :split_x]
            show[:, split_x:] = proc[:, split_x:]
            # Draw divider line (3px wide, bright accent)
            cv2.line(show, (split_x, 0), (split_x, h), (203, 166, 247), 3)
            # Draw small handle triangle at center of divider
            cy = h // 2
            handle_pts = np.array([
                [split_x - 8, cy - 12], [split_x + 8, cy - 12],
                [split_x + 8, cy + 12], [split_x - 8, cy + 12]
            ], dtype=np.int32)
            cv2.fillPoly(show, [handle_pts], (203, 166, 247))
            cv2.line(show, (split_x, cy - 8), (split_x, cy + 8), (30, 30, 46), 2)
            # Labels with background for readability
            for text, tx, color in [("Original", 10, (255,255,255)), ("Slimmed", split_x + 10, (137,180,250))]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(show, (tx - 4, 10), (tx + tw + 4, 10 + th + 10), (30, 30, 46), -1)
                cv2.putText(show, text, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        else:
            show = proc

        if lms is not None and self.video_thread and self.video_thread.show_landmarks:
            show = draw_lm(show if show is proc else show.copy(), lms)

        # CRITICAL: ensure contiguous memory for QImage
        show = np.ascontiguousarray(show)
        h, w, ch = show.shape
        qi = QImage(show.data, w, h, ch * w, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qi).scaled(self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(px)

    def _fps(self, fps):
        c = "#a6e3a1" if fps >= 20 else "#f9e2af" if fps >= 10 else "#f38ba8"
        self.fps_label.setText(f"{fps:.1f} FPS")
        self.fps_label.setStyleSheet(f"color:{c}; font-size:11px; font-weight:bold;")

    def export_video(self):
        if not self.source_path: return
        base = os.path.splitext(os.path.basename(self.source_path))[0]
        out, _ = QFileDialog.getSaveFileName(self, "Save", f"{base}_slimmed.mp4", "MP4 (*.mp4)")
        if not out: return
        self.exp_prog.setVisible(True); self.exp_prog.setValue(0); self.btn_exp.setEnabled(False)
        self._et = ExportThread(self.source_path, out, self._p())
        self._et.progress.connect(self.exp_prog.setValue)
        self._et.finished.connect(lambda p: (self.exp_prog.setValue(100), self.btn_exp.setEnabled(True),
            self.exp_stat.setText(f"Saved: {os.path.basename(p)}"),
            self.statusBar().showMessage(f"Export complete: {p}")))
        self._et.error.connect(lambda m: (self.btn_exp.setEnabled(True),
            self.exp_prog.setVisible(False), self.exp_stat.setText(f"Error: {m}")))
        self._et.status.connect(self.exp_stat.setText)
        self._et.start()

    def _load_settings(self):
        for k, (s, _) in self.sliders.items():
            s.setValue(self.settings.value(f"s/{k}", s.value(), type=int))
        self.ss.setValue(self.settings.value("s/smooth", 50, type=int))

    def _save_settings(self):
        for k, (s, _) in self.sliders.items():
            self.settings.setValue(f"s/{k}", s.value())
        self.settings.setValue("s/smooth", self.ss.value())

    def closeEvent(self, e):
        self._save_settings(); self.stop_video(); e.accept()


def main():
    if not ensure_model():
        print(f"ERROR: Could not obtain model.\nDownload from:\n  {MODEL_URL}\nPlace at:\n  {MODEL_PATH}")
        sys.exit(1)
    app = QApplication(sys.argv)
    branding_icon = QIcon(str(_branding_icon_path()))
    app.setWindowIcon(branding_icon)
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
