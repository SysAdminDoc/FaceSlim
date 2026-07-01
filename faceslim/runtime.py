#!/usr/bin/env python3
"""Runtime paths, version, and diagnostics for FaceSlim."""

import json
import os
import sys
import time
import traceback

VERSION = "1.26.0"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PACKAGE_DIR) if os.path.basename(PACKAGE_DIR) == 'faceslim' else PACKAGE_DIR
CONFIG_DIR = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), '.faceslim')
PRESETS_DIR = os.path.join(CONFIG_DIR, 'presets')
MODEL_DIR = os.path.join(CONFIG_DIR, 'models') if getattr(sys, "frozen", False) else APP_DIR
RENDER_LOG_PATH = os.path.join(APP_DIR, 'render.log')
IPTC_DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"
os.makedirs(PRESETS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v'}

def exception_handler(exc_type, exc_value, exc_tb):
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open(os.path.join(APP_DIR, 'crash.log'), 'a', encoding="utf-8") as f:
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
