import importlib.util
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_compatibility.py"
SPEC = importlib.util.spec_from_file_location("check_compatibility", SCRIPT)
check_compatibility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_compatibility)


class CompatibilityToolTests(unittest.TestCase):
    def test_parse_requirement_line_extracts_operator_and_version(self):
        self.assertEqual(
            check_compatibility.parse_requirement_line("onnxruntime==1.27.0 # canary"),
            ("onnxruntime", "==1.27.0"),
        )
        self.assertEqual(
            check_compatibility.parse_requirement_line("numpy>=2,<3"),
            ("numpy", ">=2,<3"),
        )
        self.assertIsNone(check_compatibility.parse_requirement_line("# comment"))

    def test_canary_requirements_are_runtime_subset(self):
        canary = check_compatibility.read_requirements(check_compatibility.CANARY_REQUIREMENTS)

        self.assertEqual(canary["onnxruntime"], "==1.27.0")
        self.assertEqual(canary["numpy"], "==2.5.0")
        self.assertIn("mediapipe", canary)

    def test_python_launcher_uses_windows_py_selector_when_requested(self):
        launcher = check_compatibility.python_launcher("3.12")

        if os.name == "nt":
            self.assertEqual(launcher, ["py", "-3.12"])
        else:
            self.assertTrue(launcher[0])


if __name__ == "__main__":
    unittest.main()
