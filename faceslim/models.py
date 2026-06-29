#!/usr/bin/env python3
"""Model manifests, verified downloads, and ONNX provider diagnostics."""

import hashlib
import os
import time
import urllib.request

import numpy as np
import onnxruntime as ort

from .runtime import MODEL_DIR

MODEL_MANIFEST_VERSION = 1
LANDMARK_MODEL = {
    "key": "face_landmarker",
    "label": "MediaPipe Face Landmarker",
    "kind": "landmark",
    "filename": "face_landmarker.task",
    "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "source": "Google MediaPipe",
    "source_url": "https://github.com/google-ai-edge/mediapipe",
    "license": "Apache-2.0",
    "license_url": "https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    "size_bytes": 3_758_596,
    "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}
MODEL_URL = LANDMARK_MODEL["url"]

def _model_path(cfg):
    return os.path.join(MODEL_DIR, cfg["filename"])

MODEL_PATH = _model_path(LANDMARK_MODEL)

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        "kind": "parser",
        "filename": "bisenet_face_parsing.onnx",
        "url": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx",
        "source": "yakhyo/face-parsing",
        "source_url": "https://github.com/yakhyo/face-parsing",
        "license": "MIT",
        "license_url": "https://github.com/yakhyo/face-parsing/blob/main/LICENSE",
        "size_bytes": 53_205_364,
        "sha256": "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
    },
    "bisenet_resnet34": {
        "key": "bisenet_resnet34",
        "label": "BiSeNet ResNet34 (quality)",
        "kind": "parser",
        "filename": "bisenet_resnet34.onnx",
        "url": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx",
        "source": "yakhyo/face-parsing",
        "source_url": "https://github.com/yakhyo/face-parsing",
        "license": "MIT",
        "license_url": "https://github.com/yakhyo/face-parsing/blob/main/LICENSE",
        "size_bytes": 93_632_554,
        "sha256": "5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e",
    },
}
DEFAULT_PARSER_MODEL = "bisenet_resnet18"
BISENET_PATH = _model_path(PARSER_MODELS[DEFAULT_PARSER_MODEL])
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
    return _model_path(PARSER_MODELS[parser_model_key(model_key)])

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
    "kind": "matting",
    "filename": "modnet_photographic.onnx",
    "url": "https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx",
    "source": "yakhyo/modnet",
    "source_url": "https://github.com/yakhyo/modnet",
    "license": "Apache-2.0",
    "license_url": "https://github.com/yakhyo/modnet/blob/main/LICENSE",
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


RESTORATION_MODELS = {
    "gfpgan_1.4": {
        "key": "gfpgan_1.4",
        "label": "GFPGAN 1.4 Face Restoration",
        "kind": "face_restoration",
        "filename": "gfpgan_1.4.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
        "source": "TencentARC/GFPGAN via FaceFusion assets",
        "source_url": "https://github.com/TencentARC/GFPGAN",
        "license": "Apache-2.0",
        "license_url": "https://github.com/TencentARC/GFPGAN/blob/master/LICENSE",
        "size_bytes": 340_299_087,
        "sha256": "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
    },
    "real_esrgan_x2": {
        "key": "real_esrgan_x2",
        "label": "Real-ESRGAN 2x Upscale",
        "kind": "upscale",
        "filename": "real_esrgan_x2.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/real_esrgan_x2.onnx",
        "source": "xinntao/Real-ESRGAN via FaceFusion assets",
        "source_url": "https://github.com/xinntao/Real-ESRGAN",
        "license": "BSD-3-Clause",
        "license_url": "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
        "size_bytes": 69_552_244,
        "sha256": "5be2d62ab3b0357986efb20a19638334c0e7a2a055ccbc52c77063da0482b3bb",
    },
}

POST_STAGE_OFF = "off"
POST_STAGE_GFPGAN = "gfpgan_1.4"
POST_STAGE_REALESRGAN = "real_esrgan_x2"
POST_STAGE_GFPGAN_REALESRGAN = "gfpgan_1.4_real_esrgan_x2"
POST_STAGE_OPTIONS = {
    POST_STAGE_OFF: {"label": "Off", "models": ()},
    POST_STAGE_GFPGAN: {"label": "GFPGAN face restore", "models": ("gfpgan_1.4",)},
    POST_STAGE_REALESRGAN: {"label": "Real-ESRGAN 2x", "models": ("real_esrgan_x2",)},
    POST_STAGE_GFPGAN_REALESRGAN: {
        "label": "GFPGAN + Real-ESRGAN 2x",
        "models": ("gfpgan_1.4", "real_esrgan_x2"),
    },
}


ONNX_PROVIDER_OPTIONS = {
    "auto": {"label": "Auto", "provider": None},
    "cpu": {"label": "CPU", "provider": "CPUExecutionProvider"},
    "cuda": {"label": "CUDA", "provider": "CUDAExecutionProvider"},
    "directml": {"label": "DirectML", "provider": "DmlExecutionProvider"},
}
DEFAULT_ONNX_PROVIDER = "auto"
ONNX_AUTO_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)


def onnx_provider_key(value=None):
    if not value:
        return DEFAULT_ONNX_PROVIDER
    raw = str(value).strip()
    lowered = raw.lower()
    if lowered in ONNX_PROVIDER_OPTIONS:
        return lowered
    for key, cfg in ONNX_PROVIDER_OPTIONS.items():
        if raw == cfg.get("provider"):
            return key
    return DEFAULT_ONNX_PROVIDER


def onnx_provider_label(value=None):
    return ONNX_PROVIDER_OPTIONS[onnx_provider_key(value)]["label"]


def available_onnx_providers():
    try:
        providers = list(ort.get_available_providers())
    except Exception:
        providers = []
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def resolve_onnx_providers(preference=None, available=None):
    key = onnx_provider_key(preference)
    available = list(available if available is not None else available_onnx_providers())
    if key == "auto":
        providers = [p for p in ONNX_AUTO_PROVIDER_ORDER if p in available]
        if not providers and available:
            providers = [available[0]]
        selected = providers[0] if providers else None
        reason = "auto selected first available provider" if selected else "no ONNX providers available"
    else:
        requested = ONNX_PROVIDER_OPTIONS[key]["provider"]
        if requested in available:
            providers = [requested]
            if requested != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            selected = requested
            reason = "override honored"
        elif "CPUExecutionProvider" in available:
            providers = ["CPUExecutionProvider"]
            selected = "CPUExecutionProvider"
            reason = f"{ONNX_PROVIDER_OPTIONS[key]['label']} unavailable; using CPUExecutionProvider"
        else:
            providers = available[:1]
            selected = providers[0] if providers else None
            reason = f"{ONNX_PROVIDER_OPTIONS[key]['label']} unavailable; using {selected or 'none'}"
    return {
        "preference": key,
        "preference_label": ONNX_PROVIDER_OPTIONS[key]["label"],
        "available": available,
        "providers": providers,
        "selected_provider": selected,
        "fallback_reason": reason,
    }


def create_onnx_session(model_path, provider_preference=None, model_label="ONNX model"):
    resolution = resolve_onnx_providers(provider_preference)
    providers = resolution["providers"] or ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        if providers != ["CPUExecutionProvider"] and "CPUExecutionProvider" in resolution["available"]:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            resolution = resolution.copy()
            resolution["providers"] = ["CPUExecutionProvider"]
            resolution["selected_provider"] = "CPUExecutionProvider"
            resolution["fallback_reason"] = (
                f"{model_label} failed on {providers[0]} ({e}); using CPUExecutionProvider"
            )
        else:
            raise
    actual = session.get_providers()[0] if session.get_providers() else resolution["selected_provider"]
    if actual != resolution["selected_provider"]:
        resolution = resolution.copy()
        resolution["selected_provider"] = actual
        resolution["fallback_reason"] = (
            f"ONNX Runtime selected {actual} instead of {providers[0]}"
        )
    return session, resolution


def benchmark_onnx_model(model_path, provider_preference=None, model_label="ONNX model", runs=3):
    session, resolution = create_onnx_session(model_path, provider_preference, model_label)
    input_name = session.get_inputs()[0].name
    sample = np.zeros((1, 3, 512, 512), dtype=np.float32)
    session.run(None, {input_name: sample})
    timings = []
    for _ in range(max(1, int(runs))):
        started = time.perf_counter()
        session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "model": model_label,
        "provider": resolution["selected_provider"],
        "preference": resolution["preference"],
        "fallback_reason": resolution["fallback_reason"],
        "ms_per_frame": sum(timings) / len(timings),
    }


def _model_ready_for_diagnostics(cfg):
    return validate_model_artifact(_model_path(cfg), cfg)[0]


def provider_diagnostics_lines(provider_preference=None, parser_model=None,
                               run_benchmark=False, ensure_parser=False):
    pref = onnx_provider_key(provider_preference)
    parser_key = parser_model_key(parser_model)
    resolution = resolve_onnx_providers(pref)
    lines = [
        "Available ONNX providers: " + (", ".join(resolution["available"]) or "none"),
        f"Provider preference: {resolution['preference_label']}",
    ]

    parser_label = parser_model_label(parser_key)
    parser_ready = parser_model_ready(parser_key)
    if ensure_parser and not parser_ready:
        parser_ready = ensure_parsing_model(parser_key)
    if run_benchmark and parser_ready:
        try:
            bench = benchmark_onnx_model(parser_model_path(parser_key), pref, parser_label)
            lines.append(
                f"Face parsing ({parser_label}): {bench['provider']}, "
                f"{bench['ms_per_frame']:.1f} ms/frame, {bench['fallback_reason']}"
            )
        except Exception as e:
            fallback = resolve_onnx_providers(pref)
            lines.append(
                f"Face parsing ({parser_label}): benchmark failed ({e}); "
                f"would select {fallback['selected_provider']}, {fallback['fallback_reason']}"
            )
    else:
        lines.append(
            f"Face parsing ({parser_label}): "
            f"{'model ready' if parser_ready else 'model not cached'}, "
            f"would select {resolution['selected_provider']}, {resolution['fallback_reason']}"
        )

    matting_ready = _model_ready_for_diagnostics(MATTE_MODEL)
    if run_benchmark and matting_ready:
        try:
            bench = benchmark_onnx_model(MODNET_PATH, pref, MATTE_MODEL["label"])
            lines.append(
                f"Matting ({MATTE_MODEL['label']}): {bench['provider']}, "
                f"{bench['ms_per_frame']:.1f} ms/frame, {bench['fallback_reason']}"
            )
        except Exception as e:
            lines.append(
                f"Matting ({MATTE_MODEL['label']}): benchmark failed ({e}); "
                f"would select {resolution['selected_provider']}, {resolution['fallback_reason']}"
            )
    else:
        lines.append(
            f"Matting ({MATTE_MODEL['label']}): "
            f"{'model ready' if matting_ready else 'model not cached'}, "
            f"would select {resolution['selected_provider']}, {resolution['fallback_reason']}"
        )
    return lines


def provider_diagnostics_text(provider_preference=None, parser_model=None,
                              run_benchmark=False, ensure_parser=False):
    return "\n".join(provider_diagnostics_lines(
        provider_preference, parser_model, run_benchmark, ensure_parser))


def all_model_configs():
    return (
        [LANDMARK_MODEL]
        + [PARSER_MODELS[key] for key in sorted(PARSER_MODELS)]
        + [MATTE_MODEL]
        + [RESTORATION_MODELS[key] for key in sorted(RESTORATION_MODELS)]
    )


def model_config_by_key(model_key):
    key = str(model_key or "").strip()
    for cfg in all_model_configs():
        if cfg["key"] == key:
            return cfg
    raise KeyError(f"Unknown model: {model_key}")


def model_runtime_provider(cfg, provider_preference=None):
    if cfg.get("kind") == "landmark":
        return "MediaPipe Tasks / TFLite"
    return resolve_onnx_providers(provider_preference)["selected_provider"] or "unavailable"


def model_inventory(provider_preference=None):
    items = []
    for cfg in all_model_configs():
        path = _model_path(cfg)
        verified, reason = validate_model_artifact(path, cfg)
        if verified:
            status = "verified"
        elif reason == "missing":
            status = "downloadable"
        else:
            status = f"invalid ({reason})"
        items.append({
            "key": cfg["key"],
            "label": cfg["label"],
            "kind": cfg.get("kind", "model"),
            "filename": cfg["filename"],
            "source": cfg["source"],
            "source_url": cfg["source_url"],
            "download_url": cfg["url"],
            "license": cfg["license"],
            "license_url": cfg["license_url"],
            "expected_size": cfg["size_bytes"],
            "sha256": cfg["sha256"],
            "cache_path": path,
            "status": status,
            "verified": verified,
            "provider": model_runtime_provider(cfg, provider_preference),
        })
    return items


def model_inventory_lines(provider_preference=None):
    lines = []
    for item in model_inventory(provider_preference):
        lines.append(
            f"{item['label']} [{item['key']}]\n"
            f"  status: {item['status']} | provider: {item['provider']}\n"
            f"  source: {item['source']} ({item['source_url']})\n"
            f"  license: {item['license']} ({item['license_url']})\n"
            f"  expected: {item['expected_size']} bytes | sha256: {item['sha256'][:12]}...\n"
            f"  cache: {item['cache_path']}\n"
            f"  download: {item['download_url']}"
        )
    return lines


def model_inventory_text(provider_preference=None):
    return "\n\n".join(model_inventory_lines(provider_preference))


def redownload_model(model_key):
    cfg = model_config_by_key(model_key)
    _remove_quietly(_model_path(cfg))
    return _download_verified_model(cfg)


def redownload_models(model_key="all"):
    keys = [cfg["key"] for cfg in all_model_configs()] if model_key == "all" else [model_key]
    results = {}
    for key in keys:
        results[key] = redownload_model(key)
    return results

# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════════════
