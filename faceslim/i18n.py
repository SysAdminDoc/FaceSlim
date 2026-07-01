#!/usr/bin/env python3
"""Localization helpers for FaceSlim UI and CLI strings."""

import os

TRANSLATIONS = {
    "en": {},
}
_CURRENT_LOCALE = os.environ.get("FACESLIM_LOCALE", "en").strip().lower() or "en"


def set_locale(locale):
    global _CURRENT_LOCALE
    _CURRENT_LOCALE = str(locale or "en").strip().lower() or "en"


def current_locale():
    return _CURRENT_LOCALE


def pseudo_localize(text):
    return f"[!! {text} !!]"


def tr(text):
    if _CURRENT_LOCALE in {"pseudo", "qps"}:
        return pseudo_localize(text)
    return TRANSLATIONS.get(_CURRENT_LOCALE, TRANSLATIONS["en"]).get(text, text)


MAIN_UI_TEXT_LIMITS = {
    "Webcam": 24,
    "Load File": 26,
    "Stop": 20,
    "Undo": 20,
    "Redo": 20,
    "A/B Compare": 30,
    "Virtual Cam": 30,
    "Refresh Models": 34,
    "Redownload": 30,
    "Post-Stage:": 30,
    "Parser Model:": 34,
    "ONNX Provider:": 34,
    "Benchmark": 30,
    "Export Video": 32,
    "Cancel Export": 34,
    "Save Screenshot": 36,
    "Before/After GIF": 38,
}


def pseudo_locale_overflow_report(text_limits=None):
    report = []
    for text, limit in (text_limits or MAIN_UI_TEXT_LIMITS).items():
        localized = pseudo_localize(text)
        if len(localized) > limit:
            report.append({"text": text, "localized": localized, "limit": limit})
    return report
