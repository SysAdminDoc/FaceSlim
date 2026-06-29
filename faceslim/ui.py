#!/usr/bin/env python3
"""PyQt user interface for FaceSlim."""

import math
import os

import cv2
import numpy as np
from PyQt5.QtCore import QSettings, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QMainWindow, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .i18n import tr
from .runtime import VERSION
from .models import *
from .pipeline import *
from .exporters import (
    BatchThread, ExportThread, GifExportThread, HAS_VIRTUALCAM,
    ModelRedownloadThread, ProviderDiagnosticsThread, VideoThread,
)

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
INTERACTIVE_WIDGET_TYPES = (QPushButton, QSlider, QCheckBox, QComboBox, QSpinBox)
ACCESSIBILITY_CONTRAST_PAIRS = [
    ("Primary text on app background", "#cdd6f4", "#1e1e2e", 4.5),
    ("Muted text on app background", "#bac2de", "#1e1e2e", 4.5),
    ("Accent text on app background", "#89b4fa", "#1e1e2e", 4.5),
    ("Primary button text", "#1e1e2e", "#89b4fa", 4.5),
    ("Success button text", "#1e1e2e", "#a6e3a1", 4.5),
    ("Danger button text", "#1e1e2e", "#f38ba8", 4.5),
    ("Secondary button text", "#cdd6f4", "#45475a", 4.5),
]


def _hex_rgb(color):
    color = color.strip().lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _relative_luminance(color):
    def channel(value):
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(value) for value in _hex_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground, background):
    fg = _relative_luminance(foreground)
    bg = _relative_luminance(background)
    lighter, darker = max(fg, bg), min(fg, bg)
    return (lighter + 0.05) / (darker + 0.05)


def accessibility_contrast_report():
    return [{
        "name": name,
        "foreground": foreground,
        "background": background,
        "minimum": minimum,
        "ratio": contrast_ratio(foreground, background),
    } for name, foreground, background, minimum in ACCESSIBILITY_CONTRAST_PAIRS]


def accessibility_audit_window(window):
    missing = []
    for widget in window.findChildren(INTERACTIVE_WIDGET_TYPES):
        label = widget.objectName() or (widget.text() if hasattr(widget, "text") else type(widget).__name__)
        if not widget.accessibleName().strip():
            missing.append(f"{type(widget).__name__}:{label}")
        if not widget.accessibleDescription().strip():
            missing.append(f"{type(widget).__name__}:{label}:description")
    return missing



class FaceSlimApp(QMainWindow):
    def __init__(self, show_responsible_gate=True):
        super().__init__()
        self.setWindowTitle(f"{tr('FaceSlim')} v{VERSION}")
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
        self._image_engine_provider = DEFAULT_ONNX_PROVIDER
        self._provider_diag_thread = None
        self._model_redownload_thread = None
        self._build_ui()
        self._load_settings()
        self.history.push(self._p())
        if show_responsible_gate:
            QTimer.singleShot(0, self._show_responsible_use_gate)

    # ── Slider Factory ──────────────────────────────────────────
    def _set_accessible(self, widget, name, description=None):
        widget.setAccessibleName(name)
        widget.setAccessibleDescription(description or name)
        if description and hasattr(widget, "toolTip") and not widget.toolTip():
            widget.setToolTip(description)
        return widget

    def _apply_accessibility_metadata(self):
        self.setAccessibleName("FaceSlim main window")
        self.setAccessibleDescription("FaceSlim image and video face reshaping workspace.")
        self._set_accessible(self.video_label, "Preview canvas",
                             "Drop media here or review the processed before and after preview.")
        self._set_accessible(self.timeline_slider, "Video timeline scrubber",
                             "Seek through the loaded file video in seconds.")
        self._set_accessible(self.timeline_label, "Timeline position",
                             "Current video position and duration.")
        named = [
            (self.btn_webcam, "Start webcam", "Start live webcam preview."),
            (self.btn_load, "Load media file", "Open an image or video file."),
            (self.btn_stop, "Stop preview", "Stop the current webcam or video preview."),
            (self.btn_undo, "Undo adjustment", "Undo the last parameter change."),
            (self.btn_redo, "Redo adjustment", "Redo the next parameter change."),
            (self.btn_cmp, "A/B compare", "Toggle draggable split before and after comparison."),
            (self.btn_virtualcam, "Virtual camera", "Stream the processed preview to a virtual camera."),
            (self.combo_parser_model, "Parser model", "Choose the BiSeNet face parsing model."),
            (self.combo_onnx_provider, "ONNX runtime provider", "Choose automatic, CPU, CUDA, or DirectML ONNX inference."),
            (self.btn_provider_bench, "Benchmark ONNX provider", "Run a one-frame ONNX provider benchmark."),
            (self.combo_model_redownload, "Model redownload selector", "Choose one model or all models for redownload."),
            (self.btn_model_refresh, "Refresh model inventory", "Refresh model verification, source, license, and cache status."),
            (self.btn_model_redownload, "Redownload model", "Redownload the selected model artifact and verify its hash."),
            (self.combo_post_stage, "Post-stage model", "Choose optional GFPGAN restoration, Real-ESRGAN upscale, both, or off."),
            (self.chk_lm, "Show landmarks", "Overlay detected face landmarks on the preview."),
            (self.chk_conf, "Show confidence", "Overlay face detection confidence on the preview."),
            (self.chk_teeth_hint, "Show teeth mask", "Overlay the whitening target mask in preview only."),
            (self.spin_faces, "Maximum faces", "Choose how many faces FaceSlim should process."),
            (self.combo_scale, "Preview scale", "Reduce preview resolution for faster playback."),
            (self.preset_combo, "Custom preset selector", "Choose a saved custom preset."),
            (self.btn_exp_video, "Export video", "Render the current video with active settings."),
            (self.btn_exp_cancel, "Cancel export", "Cancel the active video export."),
            (self.btn_exp_img, "Save screenshot", "Save the current processed preview as a PNG image."),
            (self.btn_exp_gif, "Export before after GIF", "Export a two-frame before and after GIF."),
            (self.chk_watermark, "Disclosure watermark", "Add a visible AI modified disclosure badge."),
            (self.combo_video_compare, "Video export layout", "Choose normal, split, or side-by-side video export."),
            (self.btn_batch, "Select files for batch", "Choose image or video files for batch processing."),
            (self.btn_batch_folder, "Process folder", "Process every supported media file in a folder."),
            (self.btn_batch_manifest, "Run manifest", "Run a JSON batch manifest."),
            (self.btn_batch_cancel, "Cancel batch", "Cancel the active batch run."),
        ]
        for widget, name, description in named:
            self._set_accessible(widget, name, description)
        for widget in self.findChildren((QPushButton, QCheckBox)):
            if not widget.accessibleName().strip():
                text = widget.text().replace("&", "").strip()
                if text:
                    self._set_accessible(widget, text, widget.toolTip() or text)
        self._apply_tab_order()

    def _apply_tab_order(self):
        ordered = [
            self.btn_webcam, self.btn_load, self.btn_stop, self.btn_undo, self.btn_redo,
            self.btn_cmp, self.btn_virtualcam, self.timeline_slider,
            *[slider for slider, _label in self.sliders.values()],
            self.combo_parser_model, self.combo_onnx_provider, self.btn_provider_bench,
            self.combo_model_redownload, self.btn_model_refresh, self.btn_model_redownload,
            self.combo_post_stage,
            self.chk_lm, self.chk_conf, self.chk_teeth_hint,
            self.spin_faces, self.combo_scale, self.preset_combo,
            self.btn_exp_video, self.btn_exp_cancel, self.btn_exp_img, self.btn_exp_gif,
            self.chk_watermark, self.combo_video_compare,
            self.btn_batch, self.btn_batch_folder, self.btn_batch_manifest, self.btn_batch_cancel,
        ]
        for first, second in zip(ordered, ordered[1:]):
            QWidget.setTabOrder(first, second)

    def _make_slider(self, layout, key, label, max_v=100, default=0, tip=""):
        row = QVBoxLayout(); row.setSpacing(2)
        lr = QHBoxLayout()
        name_label = QLabel(label)
        lr.addWidget(name_label); lr.addStretch()
        vl = QLabel(f"{default}%")
        vl.setStyleSheet("color:#f9e2af; font-size:12px; font-weight:bold; min-width:36px;")
        lr.addWidget(vl); row.addLayout(lr)
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(0, max_v); s.setValue(default)
        if tip: s.setToolTip(tip)
        name_label.setBuddy(s)
        self._set_accessible(s, f"{label} slider", tip or f"Adjust {label}.")
        self._set_accessible(vl, f"{label} value", f"Current {label} slider value.")
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
        t = QLabel(tr("FaceSlim"))
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
        self.video_label.setText(tr("Drop a file here, or use the buttons below to start"))
        left.addWidget(self.video_label, 1)

        timeline_row = QHBoxLayout(); timeline_row.setSpacing(8)
        timeline_title = QLabel(tr("Timeline"))
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
        transport_row = QHBoxLayout(); transport_row.setSpacing(6)
        self.btn_webcam = QPushButton(tr("Webcam")); self.btn_webcam.clicked.connect(self.start_webcam)
        transport_row.addWidget(self.btn_webcam)
        self.btn_load = QPushButton(tr("Load File")); self.btn_load.setProperty("secondary", True)
        self.btn_load.clicked.connect(self.load_file); transport_row.addWidget(self.btn_load)
        self.btn_stop = QPushButton(tr("Stop")); self.btn_stop.setProperty("danger", True)
        self.btn_stop.clicked.connect(self.stop_video); self.btn_stop.setEnabled(False)
        transport_row.addWidget(self.btn_stop); transport_row.addStretch()

        self.btn_undo = QPushButton(tr("Undo")); self.btn_undo.setProperty("secondary", True)
        self.btn_undo.clicked.connect(self._undo); self.btn_undo.setEnabled(False)
        transport_row.addWidget(self.btn_undo)
        self.btn_redo = QPushButton(tr("Redo")); self.btn_redo.setProperty("secondary", True)
        self.btn_redo.clicked.connect(self._redo); self.btn_redo.setEnabled(False)
        transport_row.addWidget(self.btn_redo)

        self.btn_cmp = QPushButton(tr("A/B Compare")); self.btn_cmp.setCheckable(True)
        self.btn_cmp.toggled.connect(self._toggle_compare); transport_row.addWidget(self.btn_cmp)
        self.btn_virtualcam = QPushButton(tr("Virtual Cam")); self.btn_virtualcam.setCheckable(True)
        self.btn_virtualcam.setProperty("secondary", True)
        self.btn_virtualcam.clicked.connect(self._toggle_virtualcam)
        self.btn_virtualcam.setEnabled(False)
        transport_row.addWidget(self.btn_virtualcam)
        left.addLayout(transport_row)
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

        g1 = QGroupBox(tr("Face Reshaping")); g1l = QVBoxLayout(g1); g1l.setSpacing(6)
        self._make_slider(g1l, 'jaw', tr('Jaw Slimming'), tip='Narrows the jawline')
        self._make_slider(g1l, 'cheeks', tr('Cheek Slimming'), tip='Reduces cheek fullness')
        self._make_slider(g1l, 'chin', tr('Chin Reshape'), tip='Lifts and narrows the chin')
        self._make_slider(g1l, 'face_width', tr('Overall Width'), tip='Reduces overall face width')
        self._make_slider(g1l, 'forehead', tr('Forehead Slim'), tip='Narrows the forehead')
        self._make_slider(g1l, 'nose', tr('Nose Slim'), tip='Narrows the nose bridge and tip')
        self._make_slider(g1l, 'eye_enlarge', tr('Eye Enlarge'), tip='Enlarges eyes outward from iris center')
        self._make_slider(g1l, 'lip_plump', tr('Lip Plump'), tip='Plumps lips outward from lip center')
        self._make_slider(g1l, 'expression_neutralize', tr('Expression Neutralize'), default=0,
                          tip='Blendshape-guided dampening of frowns and raised or lowered brows')
        t1.addWidget(g1)

        g_beauty = QGroupBox(tr("AI Beauty")); g_bl = QVBoxLayout(g_beauty); g_bl.setSpacing(6)
        self._make_slider(g_bl, 'skin_smooth', tr('Skin Smoothing'), default=0, tip='Frequency-separation bilateral filter on skin mask')
        self._make_slider(g_bl, 'skin_tone_even', tr('Skin Tone Even'), default=0, tip='Reduces redness and blotchiness via LAB color correction')
        self._make_slider(g_bl, 'teeth_whiten', tr('Teeth Whitening'), default=0, tip='HSV brightness boost in mouth region')
        self._make_slider(g_bl, 'eye_sharpen', tr('Eye Sharpen'), default=0, tip='Unsharp mask on eyes and brows for crisp detail')
        self._make_slider(g_bl, 'lip_color', tr('Lip Color'), default=0, tip='Boosts lip saturation and warmth')
        self._make_slider(g_bl, 'under_eye', tr('Under-Eye Smooth'), default=0, tip='Dedicated smoothing below the eyes')
        self._make_slider(g_bl, 'hair_hue', tr('Hair Hue Shift'), default=0, tip='Subtle hue shift on parsed hair region')
        self._make_slider(g_bl, 'hair_saturation', tr('Hair Saturation'), default=0, tip='Boosts parsed hair color intensity')
        self._make_slider(g_bl, 'hair_density', tr('Hair Density Hint'), default=0, tip='Darkens parsed hair region for a denser look')
        self._make_slider(g_bl, 'blush', tr('Blush Overlay'), default=0, tip='Procedural cheek blush opacity')
        self._make_slider(g_bl, 'lip_gloss', tr('Lip Gloss'), default=0, tip='Adds soft lip highlight and warmth')
        self._make_slider(g_bl, 'eye_shadow', tr('Eye Shadow'), default=0, tip='Procedural eye shadow opacity')
        parser_row = QHBoxLayout()
        parser_row.addWidget(QLabel(tr("Parser Model:")))
        self.combo_parser_model = QComboBox()
        for key, cfg in PARSER_MODELS.items():
            self.combo_parser_model.addItem(cfg["label"], key)
        self.combo_parser_model.currentIndexChanged.connect(self._on_parser_model_changed)
        parser_row.addWidget(self.combo_parser_model)
        g_bl.addLayout(parser_row)
        self.parsing_lbl = QLabel("")
        self.parsing_lbl.setStyleSheet("color:#a6e3a1; font-size:10px;")
        g_bl.addWidget(self.parsing_lbl)
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel(tr("ONNX Provider:")))
        self.combo_onnx_provider = QComboBox()
        for key, cfg in ONNX_PROVIDER_OPTIONS.items():
            self.combo_onnx_provider.addItem(cfg["label"], key)
        self.combo_onnx_provider.currentIndexChanged.connect(self._on_onnx_provider_changed)
        provider_row.addWidget(self.combo_onnx_provider)
        self.btn_provider_bench = QPushButton(tr("Benchmark"))
        self.btn_provider_bench.setProperty("secondary", True)
        self.btn_provider_bench.clicked.connect(self._benchmark_provider)
        provider_row.addWidget(self.btn_provider_bench)
        g_bl.addLayout(provider_row)
        self.provider_lbl = QLabel("")
        self.provider_lbl.setWordWrap(True)
        self.provider_lbl.setStyleSheet("color:#bac2de; font-size:10px;")
        g_bl.addWidget(self.provider_lbl)
        model_row = QHBoxLayout()
        self.combo_model_redownload = QComboBox()
        self.combo_model_redownload.addItem("All models", "all")
        for cfg in all_model_configs():
            self.combo_model_redownload.addItem(cfg["label"], cfg["key"])
        model_row.addWidget(self.combo_model_redownload)
        self.btn_model_refresh = QPushButton(tr("Refresh Models"))
        self.btn_model_refresh.setProperty("secondary", True)
        self.btn_model_refresh.clicked.connect(self._set_model_inventory_status)
        model_row.addWidget(self.btn_model_refresh)
        self.btn_model_redownload = QPushButton(tr("Redownload"))
        self.btn_model_redownload.setProperty("secondary", True)
        self.btn_model_redownload.clicked.connect(self._redownload_selected_model)
        model_row.addWidget(self.btn_model_redownload)
        g_bl.addLayout(model_row)
        self.model_inventory_lbl = QLabel("")
        self.model_inventory_lbl.setWordWrap(True)
        self.model_inventory_lbl.setStyleSheet("color:#bac2de; font-size:10px;")
        g_bl.addWidget(self.model_inventory_lbl)
        post_row = QHBoxLayout()
        post_row.addWidget(QLabel(tr("Post-Stage:")))
        self.combo_post_stage = QComboBox()
        for key, cfg in POST_STAGE_OPTIONS.items():
            self.combo_post_stage.addItem(cfg["label"], key)
        self.combo_post_stage.currentIndexChanged.connect(self._on_post_stage_changed)
        post_row.addWidget(self.combo_post_stage)
        g_bl.addLayout(post_row)
        self._make_slider(g_bl, 'post_stage_strength', tr('Post Strength'), default=50,
                          tip='Blend amount for optional restoration or upscale stage')
        self._make_slider(g_bl, 'post_stage_fidelity', tr('Identity Fidelity'), default=70,
                          tip='Higher values preserve more original face detail')
        self._make_slider(g_bl, 'post_stage_tile', tr('Upscale Tile'), max_v=1024, default=512,
                          tip='Real-ESRGAN tile size for large images or video frames')
        t1.addWidget(g_beauty)

        g2 = QGroupBox(tr("Quality")); g2l = QVBoxLayout(g2); g2l.setSpacing(6)
        self._make_slider(g2l, 'smoothing', tr('Warp Smoothing'), default=50, tip='Displacement field smoothness')
        self._make_slider(g2l, 'temporal', tr('Temporal Stability'), default=50, tip='Landmark jitter reduction (higher=smoother)')
        self._make_slider(g2l, 'bg_protect', tr('Background Protection'), default=70, tip='Prevents warping background - blends face region only')
        self._make_slider(g2l, 'matting_refine', tr('Matting Refine'), default=0, tip='MODNet portrait matte edge refinement for ROI warp masks')

        opts_row = QHBoxLayout()
        self.chk_lm = QCheckBox(tr("Landmarks")); self.chk_lm.toggled.connect(self._tog_lm)
        opts_row.addWidget(self.chk_lm)
        self.chk_conf = QCheckBox(tr("Confidence")); self.chk_conf.toggled.connect(self._tog_conf)
        opts_row.addWidget(self.chk_conf)
        self.chk_teeth_hint = QCheckBox(tr("Teeth Mask")); self.chk_teeth_hint.toggled.connect(self._tog_teeth_hint)
        opts_row.addWidget(self.chk_teeth_hint)
        g2l.addLayout(opts_row)

        faces_row = QHBoxLayout()
        faces_row.addWidget(QLabel(tr("Max Faces:")))
        self.spin_faces = QSpinBox(); self.spin_faces.setRange(1, 5); self.spin_faces.setValue(1)
        self.spin_faces.valueChanged.connect(self._on_faces_changed)
        faces_row.addWidget(self.spin_faces); faces_row.addStretch()
        faces_row.addWidget(QLabel(tr("Preview Scale:")))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100%", "75%", "50%"])
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        faces_row.addWidget(self.combo_scale)
        g2l.addLayout(faces_row)
        t1.addWidget(g2)

        t1.addStretch()
        tabs.addTab(tab_reshape_scroll, tr("Reshape"))

        # ─── Tab 2: Presets ───
        tab_presets = QWidget()
        t2 = QVBoxLayout(tab_presets); t2.setSpacing(8); t2.setContentsMargins(8,8,8,8)

        g3 = QGroupBox(tr("Built-in Presets")); g3l = QGridLayout(g3); g3l.setSpacing(6)
        for i, (name, vals) in enumerate(BUILT_IN_PRESETS.items()):
            b = QPushButton(name); b.setProperty("secondary", True)
            b.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            g3l.addWidget(b, i // 2, i % 2)
        btn_reset = QPushButton(tr("Reset All")); btn_reset.setProperty("danger", True)
        btn_reset.clicked.connect(lambda: self._apply_preset(DEFAULT_PARAMS.copy()))
        g3l.addWidget(btn_reset, (len(BUILT_IN_PRESETS)) // 2, (len(BUILT_IN_PRESETS)) % 2)
        t2.addWidget(g3)

        g4 = QGroupBox(tr("Custom Presets")); g4l = QVBoxLayout(g4); g4l.setSpacing(6)
        self.preset_combo = QComboBox(); self._refresh_presets()
        g4l.addWidget(self.preset_combo)
        pc_row = QHBoxLayout(); pc_row.setSpacing(6)
        btn_load_preset = QPushButton(tr("Load")); btn_load_preset.setProperty("secondary", True)
        btn_load_preset.clicked.connect(self._load_custom_preset); pc_row.addWidget(btn_load_preset)
        btn_save_preset = QPushButton(tr("Save Current")); btn_save_preset.setProperty("success", True)
        btn_save_preset.clicked.connect(self._save_custom_preset); pc_row.addWidget(btn_save_preset)
        btn_del_preset = QPushButton(tr("Delete")); btn_del_preset.setProperty("danger", True)
        btn_del_preset.clicked.connect(self._delete_custom_preset); pc_row.addWidget(btn_del_preset)
        g4l.addLayout(pc_row)

        io_row = QHBoxLayout(); io_row.setSpacing(6)
        btn_export_presets = QPushButton(tr("Export All")); btn_export_presets.setProperty("secondary", True)
        btn_export_presets.clicked.connect(self._export_presets); io_row.addWidget(btn_export_presets)
        btn_import_presets = QPushButton(tr("Import")); btn_import_presets.setProperty("secondary", True)
        btn_import_presets.clicked.connect(self._import_presets); io_row.addWidget(btn_import_presets)
        g4l.addLayout(io_row)
        t2.addWidget(g4)
        t2.addStretch()
        tabs.addTab(tab_presets, tr("Presets"))

        # ─── Tab 3: Export ───
        tab_export = QWidget()
        t3 = QVBoxLayout(tab_export); t3.setSpacing(8); t3.setContentsMargins(8,8,8,8)

        g5 = QGroupBox(tr("Single Export")); g5l = QVBoxLayout(g5); g5l.setSpacing(6)
        self.btn_exp_video = QPushButton(tr("Export Video")); self.btn_exp_video.setProperty("success", True)
        self.btn_exp_video.clicked.connect(self.export_video); self.btn_exp_video.setEnabled(False)
        g5l.addWidget(self.btn_exp_video)
        self.btn_exp_cancel = QPushButton(tr("Cancel Export"))
        self.btn_exp_cancel.setProperty("danger", True)
        self.btn_exp_cancel.clicked.connect(self._cancel_export)
        self.btn_exp_cancel.setEnabled(False)
        g5l.addWidget(self.btn_exp_cancel)
        self.btn_exp_img = QPushButton(tr("Save Screenshot (PNG)")); self.btn_exp_img.setProperty("success", True)
        self.btn_exp_img.clicked.connect(self.save_screenshot); self.btn_exp_img.setEnabled(False)
        g5l.addWidget(self.btn_exp_img)
        self.btn_exp_gif = QPushButton(tr("Export Before/After GIF"))
        self.btn_exp_gif.clicked.connect(self.export_gif); self.btn_exp_gif.setEnabled(False)
        g5l.addWidget(self.btn_exp_gif)
        exp_opts = QHBoxLayout(); exp_opts.setSpacing(6)
        self.chk_watermark = QCheckBox(tr("Disclosure watermark"))
        exp_opts.addWidget(self.chk_watermark)
        exp_opts.addWidget(QLabel(tr("Video:")))
        self.combo_video_compare = QComboBox()
        self.combo_video_compare.addItems([tr("Normal"), tr("Split"), tr("Side-by-side")])
        exp_opts.addWidget(self.combo_video_compare)
        g5l.addLayout(exp_opts)
        self.exp_prog = QProgressBar(); self.exp_prog.setVisible(False); g5l.addWidget(self.exp_prog)
        self.exp_stat = QLabel(""); self.exp_stat.setStyleSheet("color:#6c7086; font-size:11px;")
        g5l.addWidget(self.exp_stat)
        t3.addWidget(g5)

        g6 = QGroupBox(tr("Batch Processing")); g6l = QVBoxLayout(g6); g6l.setSpacing(6)
        self.btn_batch = QPushButton(tr("Select Files for Batch"))
        self.btn_batch.clicked.connect(self.start_batch); g6l.addWidget(self.btn_batch)
        self.btn_batch_folder = QPushButton(tr("Process Entire Folder"))
        self.btn_batch_folder.setProperty("secondary", True)
        self.btn_batch_folder.clicked.connect(self.start_batch_folder); g6l.addWidget(self.btn_batch_folder)
        self.btn_batch_manifest = QPushButton(tr("Run Manifest"))
        self.btn_batch_manifest.setProperty("secondary", True)
        self.btn_batch_manifest.clicked.connect(self.start_batch_manifest); g6l.addWidget(self.btn_batch_manifest)
        self.btn_batch_cancel = QPushButton(tr("Cancel Batch")); self.btn_batch_cancel.setProperty("danger", True)
        self.btn_batch_cancel.clicked.connect(self._cancel_batch); self.btn_batch_cancel.setEnabled(False)
        g6l.addWidget(self.btn_batch_cancel)
        self.batch_prog = QProgressBar(); self.batch_prog.setVisible(False); g6l.addWidget(self.batch_prog)
        self.batch_stat = QLabel(""); self.batch_stat.setStyleSheet("color:#6c7086; font-size:11px;")
        g6l.addWidget(self.batch_stat)
        t3.addWidget(g6)
        t3.addStretch()
        tabs.addTab(tab_export, tr("Export"))

        rl.addWidget(tabs)
        root.addWidget(rw, 1)

        # Status bar
        self.statusBar().showMessage(tr("Ready - Drop a file or use buttons to start"))
        gpu_text = f"{tr('GPU')}: {GPU_NAME}" if USE_GPU else tr("CPU Mode")
        gpu_label = QLabel(f"  {gpu_text}  ")
        gpu_label.setStyleSheet(
            f"color: {'#a6e3a1' if USE_GPU else '#f9e2af'}; font-size: 11px; font-weight: bold; "
            f"background-color: {'#1e3a2e' if USE_GPU else '#3a2e1e'}; border-radius: 4px; padding: 2px 8px;")
        self.statusBar().addPermanentWidget(gpu_label)
        self._apply_accessibility_metadata()

    # ── Parameter Management ────────────────────────────────────
    def _push(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.update_params(self._p())
        self._update_undo_buttons()

    def _p(self):
        p = {k: s.value() for k, (s, _) in self.sliders.items()}
        p["post_stage_model"] = self._post_stage_model_key()
        return p

    def _set_params(self, params):
        self.history.freeze()
        for k, v in params.items():
            if k in self.sliders:
                self.sliders[k][0].setValue(v)
        if "post_stage_model" in params and hasattr(self, "combo_post_stage"):
            idx = self.combo_post_stage.findData(post_stage_model_key(params.get("post_stage_model")))
            if idx >= 0:
                self.combo_post_stage.setCurrentIndex(idx)
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

    def _onnx_provider_key(self):
        if hasattr(self, "combo_onnx_provider"):
            return onnx_provider_key(self.combo_onnx_provider.currentData())
        return DEFAULT_ONNX_PROVIDER

    def _post_stage_model_key(self):
        if hasattr(self, "combo_post_stage"):
            return post_stage_model_key(self.combo_post_stage.currentData())
        return POST_STAGE_OFF

    def _set_parser_model_status(self):
        key = self._parser_model_key()
        ready = parser_model_ready(key)
        self.parsing_lbl.setText(
            f"  Face Parsing: {parser_model_label(key)}"
            + (" ready" if ready else " downloads on first use")
        )
        self.parsing_lbl.setStyleSheet(f"color: {'#a6e3a1' if ready else '#f9e2af'}; font-size: 10px;")
        self._set_provider_status()

    def _set_provider_status(self, text=None):
        if text is None:
            text = provider_diagnostics_text(
                self._onnx_provider_key(), self._parser_model_key(),
                run_benchmark=False, ensure_parser=False)
        self.provider_lbl.setText(text)
        self._set_model_inventory_status()

    def _set_model_inventory_status(self):
        if hasattr(self, "model_inventory_lbl"):
            self.model_inventory_lbl.setText(model_inventory_text(self._onnx_provider_key()))

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

    def _on_onnx_provider_changed(self, _idx):
        self._set_provider_status()
        if self._image_engine is not None:
            self._image_engine.close()
            self._image_engine = None
        if self.video_thread and self.video_thread.isRunning():
            self._go(self.video_thread.source)
        elif self.image_mode:
            self._process_current_image()
        if hasattr(self, "toast"):
            self.toast.show_message(f"ONNX provider: {onnx_provider_label(self._onnx_provider_key())}")

    def _on_post_stage_changed(self, _idx):
        self._push()
        if self.image_mode:
            self._process_current_image()
        if hasattr(self, "toast"):
            self.toast.show_message(f"Post-stage: {post_stage_model_label(self._post_stage_model_key())}")

    def _benchmark_provider(self):
        if self._provider_diag_thread and self._provider_diag_thread.isRunning():
            return
        self.btn_provider_bench.setEnabled(False)
        self._set_provider_status("Benchmarking ONNX provider...")
        self._provider_diag_thread = ProviderDiagnosticsThread(
            self._onnx_provider_key(), self._parser_model_key())
        self._provider_diag_thread.diagnostics_ready.connect(self._on_provider_benchmark_done)
        self._provider_diag_thread.error.connect(self._on_provider_benchmark_error)
        self._provider_diag_thread.start()

    def _on_provider_benchmark_done(self, text):
        self.btn_provider_bench.setEnabled(True)
        self._set_provider_status(text)
        self.statusBar().showMessage("ONNX provider benchmark complete")
        self.toast.show_message("Provider benchmark complete")

    def _on_provider_benchmark_error(self, message):
        self.btn_provider_bench.setEnabled(True)
        self._set_provider_status(f"Provider benchmark failed: {message}")
        self.toast.show_message("Provider benchmark failed", 4000)

    def _redownload_selected_model(self):
        if self._model_redownload_thread and self._model_redownload_thread.isRunning():
            return
        key = self.combo_model_redownload.currentData()
        self.btn_model_redownload.setEnabled(False)
        self.model_inventory_lbl.setText(f"Downloading {key}...")
        self._model_redownload_thread = ModelRedownloadThread(key)
        self._model_redownload_thread.download_finished.connect(self._on_model_redownload_done)
        self._model_redownload_thread.error.connect(self._on_model_redownload_error)
        self._model_redownload_thread.start()

    def _on_model_redownload_done(self, message):
        self.btn_model_redownload.setEnabled(True)
        self._set_model_inventory_status()
        self.statusBar().showMessage(message)
        self.toast.show_message("Model download complete")

    def _on_model_redownload_error(self, message):
        self.btn_model_redownload.setEnabled(True)
        self._set_model_inventory_status()
        self.statusBar().showMessage(f"Model download failed: {message}")
        self.toast.show_message("Model download failed", 5000)

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
        onnx_provider = self._onnx_provider_key()
        if (self._image_engine is None or self._image_engine_faces != nf
                or self._image_engine_parser != parser_model
                or self._image_engine_provider != onnx_provider):
            if self._image_engine is not None:
                self._image_engine.close()
            self._image_engine = FaceWarpEngine('image', nf, parser_model, onnx_provider)
            self._image_engine_faces = nf
            self._image_engine_parser = parser_model
            self._image_engine_provider = onnx_provider
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
        self.video_thread.onnx_provider = self._onnx_provider_key()
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
        source_shape = orig.shape
        if self.comparison_mode:
            if orig.shape[:2] != proc.shape[:2]:
                ph, pw = proc.shape[:2]
                orig = cv2.resize(orig, (pw, ph), interpolation=cv2.INTER_CUBIC)
            h, w = proc.shape[:2]
            sx = max(1, min(w - 1, int(w * self.video_label.divider_ratio)))
            proc_show = (apply_teeth_hint_rois(proc, self.current_teeth_hint_rois)
                         if self.chk_teeth_hint.isChecked() else proc)
            show = np.empty_like(proc_show)
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
            draw_faces = scale_faces_for_frame(faces, source_shape, show.shape)
            show = draw_landmarks(show.copy() if show is proc else show, draw_faces, show_conf, show_lm)

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
                                self._parser_model_key(), self._onnx_provider_key())
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
            save_rgb_image(out, result, self.source_path, True, self.chk_watermark.isChecked())
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
            jobs = load_batch_manifest(manifest, fallback_parser_model=self._parser_model_key(),
                                       fallback_onnx_provider=self._onnx_provider_key())
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
                                             self._video_compare_mode(), self._parser_model_key(),
                                             self._onnx_provider_key())
        self.batch_prog.setVisible(True); self.batch_prog.setValue(0)
        self.btn_batch.setEnabled(False); self.btn_batch_folder.setEnabled(False); self.btn_batch_manifest.setEnabled(False)
        self.btn_batch_cancel.setEnabled(True)
        self._batch_dialog = BatchQueueDialog(batch_jobs, self)
        self._batch_dialog.show()
        self._batch_thread = BatchThread(files, output_dir, self._p(), self.spin_faces.value(),
                                         self.chk_watermark.isChecked(), True,
                                         self._video_compare_mode(), batch_jobs,
                                         self._parser_model_key(), self._onnx_provider_key())
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
        saved_provider = onnx_provider_key(
            self.settings.value("onnx_provider", DEFAULT_ONNX_PROVIDER, type=str))
        provider_idx = self.combo_onnx_provider.findData(saved_provider)
        if provider_idx >= 0:
            self.combo_onnx_provider.setCurrentIndex(provider_idx)
        saved_post_stage = post_stage_model_key(
            self.settings.value("post_stage_model", POST_STAGE_OFF, type=str))
        post_stage_idx = self.combo_post_stage.findData(saved_post_stage)
        if post_stage_idx >= 0:
            self.combo_post_stage.setCurrentIndex(post_stage_idx)
        self._set_parser_model_status()

    def _save_settings(self):
        for k, (s, _) in self.sliders.items():
            self.settings.setValue(f"s/{k}", s.value())
        self.settings.setValue("max_faces", self.spin_faces.value())
        self.settings.setValue("watermark", self.chk_watermark.isChecked())
        self.settings.setValue("teeth_hint", self.chk_teeth_hint.isChecked())
        self.settings.setValue("video_compare", self.combo_video_compare.currentIndex())
        self.settings.setValue("parser_model", self._parser_model_key())
        self.settings.setValue("onnx_provider", self._onnx_provider_key())
        self.settings.setValue("post_stage_model", self._post_stage_model_key())

    def closeEvent(self, e):
        self._save_settings(); self.stop_video()
        if self._image_engine is not None:
            self._image_engine.close()
        e.accept()

# ═══════════════════════════════════════════════════════════════════════════
