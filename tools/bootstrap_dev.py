#!/usr/bin/env python3
"""Create or repair the local FaceSlim development environment."""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


def venv_python_path(venv_dir=VENV_DIR):
    if os.name == "nt":
        return Path(venv_dir) / "Scripts" / "python.exe"
    return Path(venv_dir) / "bin" / "python"


def pyvenv_home(venv_dir=VENV_DIR):
    cfg = Path(venv_dir) / "pyvenv.cfg"
    if not cfg.exists():
        return None
    for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("home"):
            _key, _sep, value = line.partition("=")
            return Path(value.strip()) if value.strip() else None
    return None


def run(cmd, **kwargs):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=ROOT, **kwargs)


def venv_needs_rebuild(venv_dir=VENV_DIR):
    python = venv_python_path(venv_dir)
    if not python.exists():
        return True, f"missing interpreter: {python}"
    home = pyvenv_home(venv_dir)
    if home is not None and not home.exists():
        return True, f"stale base interpreter path: {home}"
    try:
        subprocess.run([str(python), "--version"], check=True, cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return True, f"interpreter check failed: {exc}"
    return False, "usable"


def remove_venv(venv_dir=VENV_DIR):
    target = Path(venv_dir).resolve()
    root = ROOT.resolve()
    if root not in target.parents and target != root / ".venv":
        raise RuntimeError(f"refusing to remove outside repo: {target}")
    if target.exists():
        shutil.rmtree(target)


def ensure_venv():
    needs_rebuild, reason = venv_needs_rebuild()
    if needs_rebuild:
        print(f"Rebuilding .venv: {reason}")
        remove_venv()
        venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)
    else:
        print(".venv is usable")
    return venv_python_path()


def install_requirements(python):
    run([python, "-m", "pip", "install", "-r", REQUIREMENTS])


def verify_environment(python):
    run([python, "-m", "compileall", "-q", "FaceSlim.py", "FaceSlim_v1.py", "runtime_hook_mp.py"])
    run([python, "FaceSlim_v1.py", "--list-presets"])


def main():
    parser = argparse.ArgumentParser(description="Bootstrap and verify FaceSlim's local Python environment.")
    parser.add_argument("--skip-install", action="store_true", help="Only repair/check the venv and run verification.")
    parser.add_argument("--skip-verify", action="store_true", help="Install requirements but skip compile/CLI smoke.")
    args = parser.parse_args()

    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9+ is required to create the FaceSlim environment.")
    python = ensure_venv()
    if not args.skip_install:
        install_requirements(python)
    if not args.skip_verify:
        verify_environment(python)
    print(f"Ready: {python}")


if __name__ == "__main__":
    main()
