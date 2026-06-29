import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

import FaceSlim_v1 as faceslim


class AccessibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_contrast_pairs_pass(self):
        failures = [
            f"{item['name']} ratio={item['ratio']:.2f} minimum={item['minimum']:.2f}"
            for item in faceslim.accessibility_contrast_report()
            if item["ratio"] < item["minimum"]
        ]

        self.assertEqual([], failures)

    def test_main_window_controls_have_accessible_metadata(self):
        window = faceslim.FaceSlimApp(show_responsible_gate=False)
        try:
            missing = faceslim.accessibility_audit_window(window)
        finally:
            window.close()

        self.assertEqual([], missing)

    def test_pseudo_locale_main_controls_fit_guardrails(self):
        self.assertEqual([], faceslim.pseudo_locale_overflow_report())

    def test_pseudo_locale_reaches_main_window_controls(self):
        previous = faceslim.current_locale()
        faceslim.set_locale("pseudo")
        window = faceslim.FaceSlimApp(show_responsible_gate=False)
        try:
            self.assertTrue(window.btn_webcam.text().startswith("[!!"))
            self.assertTrue(window.btn_exp_video.text().startswith("[!!"))
        finally:
            window.close()
            faceslim.set_locale(previous)


if __name__ == "__main__":
    unittest.main()
