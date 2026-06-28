import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bootstrap_dev.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_dev", SCRIPT)
bootstrap_dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap_dev)


class BootstrapDevTests(unittest.TestCase):
    def test_missing_venv_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / ".venv"

            needs_rebuild, reason = bootstrap_dev.venv_needs_rebuild(missing)

            self.assertTrue(needs_rebuild)
            self.assertIn("missing interpreter", reason)

    def test_stale_pyvenv_home_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv_dir = Path(tmp) / ".venv"
            python_path = bootstrap_dev.venv_python_path(venv_dir)
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")
            stale_home = Path(tmp) / "missing-python"
            (venv_dir / "pyvenv.cfg").write_text(f"home = {stale_home}{os.linesep}", encoding="utf-8")

            needs_rebuild, reason = bootstrap_dev.venv_needs_rebuild(venv_dir)

            self.assertTrue(needs_rebuild)
            self.assertIn("stale base interpreter path", reason)


if __name__ == "__main__":
    unittest.main()
