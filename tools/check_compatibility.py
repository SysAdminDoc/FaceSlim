#!/usr/bin/env python3
"""Compatibility lane checker for FaceSlim runtime dependencies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REQUIREMENTS = ROOT / "requirements.txt"
DOCKER_REQUIREMENTS = ROOT / "requirements-docker.txt"
CANARY_REQUIREMENTS = ROOT / "requirements-upgrade-canary.txt"

PYTHON_MATRIX = [
    ("3.11", "Release/build host", "Verified by normal test and PyInstaller build path."),
    ("3.12", "Upgrade lane", "Verify with --canary before bumping runtime pins."),
    ("3.13", "Watch", "Do not claim support until MediaPipe, PyQt, ONNX, and PyInstaller smokes pass."),
    ("3.14", "Experimental", "Interpreter is present locally, but package ABI coverage is still moving."),
]

SMOKE_IMPORTS = [
    ("PyQt5.QtCore", "PyQt5"),
    ("cv2", "OpenCV"),
    ("mediapipe", "MediaPipe"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("PIL", "Pillow"),
    ("onnxruntime", "ONNX Runtime"),
]


def parse_requirement_line(line: str) -> tuple[str, str] | None:
    clean = line.split("#", 1)[0].strip()
    if not clean or clean.startswith("-"):
        return None
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in clean:
            name, version = clean.split(marker, 1)
            return name.strip(), marker + version.strip()
    return clean, ""


def read_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_requirement_line(line)
        if parsed:
            requirements[parsed[0]] = parsed[1]
    return requirements


def fetch_pypi_latest(package: str) -> dict[str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as response:
        info = json.load(response)["info"]
    return {
        "version": info.get("version", ""),
        "requires_python": info.get("requires_python") or "",
    }


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (
            "Command failed: " + " ".join(cmd),
            "stdout:\n" + result.stdout.strip() if result.stdout.strip() else "",
            "stderr:\n" + result.stderr.strip() if result.stderr.strip() else "",
        ) if part)
        raise RuntimeError(detail)
    return result


def python_launcher(version: str | None) -> list[str]:
    if not version:
        return [sys.executable]
    if os.name == "nt":
        return ["py", f"-{version}"]
    exe = shutil.which(f"python{version}") or shutil.which("python3") or sys.executable
    return [exe]


def canary_smoke_code() -> str:
    imports = "\n".join(
        f"import {module} as _module_{idx}" if "." not in module
        else f"__import__({module!r})"
        for idx, (module, _label) in enumerate(SMOKE_IMPORTS)
    )
    return (
        "import json, platform\n"
        f"{imports}\n"
        "import cv2, mediapipe, numpy, onnxruntime, scipy\n"
        "from PIL import Image\n"
        "print(json.dumps({\n"
        "  'python': platform.python_version(),\n"
        "  'opencv': cv2.__version__,\n"
        "  'mediapipe': mediapipe.__version__,\n"
        "  'numpy': numpy.__version__,\n"
        "  'scipy': scipy.__version__,\n"
        "  'onnxruntime': onnxruntime.__version__,\n"
        "  'pillow': Image.__version__,\n"
        "}, sort_keys=True))\n"
    )


def run_canary(requirements: Path, version: str | None, dry_run: bool) -> list[str]:
    launcher = python_launcher(version)
    lines = [f"Canary Python launcher: {' '.join(launcher)}"]
    if dry_run:
        result = run(
            launcher + [
                "-m", "pip", "install", "--dry-run", "--ignore-installed",
                "-r", str(requirements),
            ],
            timeout=900,
        )
        lines.append(result.stdout.strip())
        return lines

    with tempfile.TemporaryDirectory(prefix="faceslim-compat-") as tmp:
        venv_dir = Path(tmp) / "venv"
        run(launcher + ["-m", "venv", str(venv_dir)], timeout=300)
        py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"], timeout=300)
        run([str(py), "-m", "pip", "install", "-r", str(requirements)], timeout=1200)
        smoke = run([str(py), "-c", canary_smoke_code()], timeout=120)
        lines.append("Canary import smoke: " + smoke.stdout.strip())
    return lines


def print_matrix(online: bool) -> None:
    print("Python compatibility lanes")
    for version, status, note in PYTHON_MATRIX:
        print(f"  {version:4s} {status:18s} {note}")
    print()

    for label, path in (("runtime", RUNTIME_REQUIREMENTS), ("docker", DOCKER_REQUIREMENTS),
                        ("canary", CANARY_REQUIREMENTS)):
        print(f"{label} requirements: {path.name}")
        requirements = read_requirements(path)
        for name, spec in requirements.items():
            suffix = ""
            if online:
                latest = fetch_pypi_latest(name)
                suffix = (
                    f" latest={latest['version']}"
                    + (f" requires_python={latest['requires_python']}" if latest["requires_python"] else "")
                )
            print(f"  {name:24s} {spec:14s}{suffix}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FaceSlim dependency upgrade lanes.")
    parser.add_argument("--online", action="store_true", help="Fetch current PyPI versions.")
    parser.add_argument("--canary", action="store_true", help="Run the upgrade canary requirements.")
    parser.add_argument("--allow-blocked", action="store_true",
                        help="Return success when the canary records a known blocker.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve canary requirements without installing.")
    parser.add_argument("--python-version", default=None, help="Python version for the canary, e.g. 3.12.")
    parser.add_argument("--requirements", type=Path, default=CANARY_REQUIREMENTS,
                        help="Canary requirements file.")
    args = parser.parse_args()

    print_matrix(args.online)
    if args.canary:
        try:
            for line in run_canary(args.requirements.resolve(), args.python_version, args.dry_run):
                if line:
                    print(line)
            print("Canary status: PASS")
        except RuntimeError as e:
            print("Canary status: BLOCKED")
            print(str(e))
            if not args.allow_blocked:
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
