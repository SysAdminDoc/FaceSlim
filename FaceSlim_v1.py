#!/usr/bin/env python3
"""
FaceSlim v1.26.0 - AI Face Slimming & Reshaping Suite
Compatibility launcher for the modular FaceSlim package.
"""

import multiprocessing
import sys

multiprocessing.freeze_support()


def _bootstrap():
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)
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

import faceslim.cli as cli
import faceslim.exporters as exporters
import faceslim.i18n as i18n
import faceslim.models as models
import faceslim.pipeline as pipeline
import faceslim.runtime as runtime
import faceslim.ui as ui
from faceslim.runtime import *
from faceslim.i18n import *
from faceslim.models import *
from faceslim.pipeline import *
from faceslim.exporters import *
from faceslim.ui import *
from faceslim.cli import cli_process, main


if __name__ == "__main__":
    main()
