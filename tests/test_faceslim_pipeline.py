import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np

import FaceSlim_v1 as faceslim


ROOT = Path(__file__).resolve().parents[1]


class DummyEngine:
    def __init__(self, *args, **kwargs):
        pass

    def warp_single_image(self, rgb, params):
        return rgb.copy(), []

    def close(self):
        pass


class ModelManifestTests(unittest.TestCase):
    def test_validates_exact_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            payload = b"verified model bytes"
            path.write_bytes(payload)
            cfg = {
                "label": "Fixture Model",
                "filename": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

            ok, reason = faceslim.validate_model_artifact(str(path), cfg)

            self.assertTrue(ok)
            self.assertEqual(reason, "verified")

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.bin"
            path.write_bytes(b"corrupt")
            cfg = {
                "label": "Fixture Model",
                "filename": path.name,
                "size_bytes": len(b"corrupt"),
                "sha256": "0" * 64,
            }

            ok, reason = faceslim.validate_model_artifact(str(path), cfg)

            self.assertFalse(ok)
            self.assertIn("sha256", reason)


class CliAndManifestTests(unittest.TestCase):
    def test_list_presets_cli_exits_without_model_download(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "FaceSlim_v1.py"), "--list-presets"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FaceSlim v", result.stdout)
        self.assertIn("Glamour", result.stdout)

    def test_manifest_parses_defaults_and_per_file_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = base / "batch.json"
            manifest.write_text(
                json.dumps({
                    "preset": "Beauty",
                    "parser_model": "bisenet_resnet34",
                    "faces": 3,
                    "watermark": True,
                    "files": [{
                        "input": "input.jpg",
                        "output": "out/result.png",
                        "params": {"jaw": 22},
                        "face_params": {"2": {"cheeks": 18}},
                    }],
                }),
                encoding="utf-8",
            )

            jobs = faceslim.load_batch_manifest(str(manifest), str(base / "fallback"))

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["max_faces"], 3)
            self.assertTrue(jobs[0]["watermark"])
            self.assertEqual(jobs[0]["parser_model"], "bisenet_resnet34")
            self.assertEqual(jobs[0]["params"]["jaw"], 22)
            self.assertEqual(jobs[0]["params"]["face_params"][1]["cheeks"], 18)
            self.assertTrue(jobs[0]["input"].endswith("input.jpg"))
            self.assertTrue(jobs[0]["output"].endswith(os.path.join("out", "result.png")))

    def test_face_override_validation(self):
        overrides = faceslim.parse_face_overrides(
            face_presets=["2=Beauty"],
            face_params=["1:jaw=35", "3:skin-smooth=20"],
        )

        self.assertEqual(overrides[0]["jaw"], 35)
        self.assertGreater(overrides[1]["skin_smooth"], 0)
        self.assertEqual(overrides[2]["skin_smooth"], 20)

        with self.assertRaises(ValueError):
            faceslim.parse_face_overrides(face_params=["1:not-a-param=5"])


class BatchPipelineTests(unittest.TestCase):
    def _with_temp_render_log(self, tmp_path):
        old_path = faceslim.RENDER_LOG_PATH
        faceslim.RENDER_LOG_PATH = str(Path(tmp_path) / "render.log")
        return old_path

    def test_unsupported_media_job_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_log = self._with_temp_render_log(tmp)
            text_path = Path(tmp) / "notes.txt"
            text_path.write_text("not media", encoding="utf-8")
            done = []
            updates = []
            try:
                thread = faceslim.BatchThread(
                    [str(text_path)],
                    str(Path(tmp) / "out"),
                    faceslim.DEFAULT_PARAMS.copy(),
                    jobs=[{
                        "input": str(text_path),
                        "output_dir": str(Path(tmp) / "out"),
                        "params": faceslim.DEFAULT_PARAMS.copy(),
                        "max_faces": 1,
                        "watermark": False,
                        "preserve_metadata": True,
                        "compare_mode": "none",
                        "parser_model": faceslim.DEFAULT_PARSER_MODEL,
                    }],
                )
                thread.job_update.connect(lambda *args: updates.append(args))
                thread.finished.connect(lambda *args: done.append(args))

                thread.run()
            finally:
                faceslim.RENDER_LOG_PATH = old_log

            self.assertEqual(done[-1], (0, 1, 0))
            self.assertIn("Unsupported file type", updates[-1][3])
            self.assertIn("batch_unsupported_media", (Path(tmp) / "render.log").read_text(encoding="utf-8"))

    def test_cli_failure_logs_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_log = self._with_temp_render_log(tmp)
            old_ensure_model = faceslim.ensure_model
            old_ensure_parsing = faceslim.ensure_parsing_model
            text_path = Path(tmp) / "notes.txt"
            text_path.write_text("not media", encoding="utf-8")
            args = SimpleNamespace(
                input=[str(text_path)],
                output=str(Path(tmp) / "out"),
                parser_model=faceslim.DEFAULT_PARSER_MODEL,
                preset=None,
                face_preset=[],
                face_param=[],
                manifest=None,
                faces=1,
                watermark=False,
                strip_metadata=False,
                video_compare="none",
            )
            try:
                faceslim.ensure_model = lambda: True
                faceslim.ensure_parsing_model = lambda _model: True
                with self.assertRaises(SystemExit) as raised:
                    faceslim.cli_process(args)
            finally:
                faceslim.ensure_model = old_ensure_model
                faceslim.ensure_parsing_model = old_ensure_parsing
                faceslim.RENDER_LOG_PATH = old_log

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("cli_unsupported_media", (Path(tmp) / "render.log").read_text(encoding="utf-8"))

    def test_image_export_smoke_uses_pipeline_save_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "input.png"
            output_path = tmp_path / "out" / "slimmed.png"
            cv2.imwrite(str(image_path), np.full((16, 16, 3), 160, dtype=np.uint8))
            original_engine = faceslim.FaceWarpEngine
            faceslim.FaceWarpEngine = DummyEngine
            try:
                thread = faceslim.BatchThread(
                    [str(image_path)],
                    str(output_path.parent),
                    faceslim.DEFAULT_PARAMS.copy(),
                )
                ok = thread._process_image({
                    "input": str(image_path),
                    "output": str(output_path),
                    "params": faceslim.DEFAULT_PARAMS.copy(),
                    "max_faces": 1,
                    "watermark": False,
                    "preserve_metadata": True,
                    "parser_model": faceslim.DEFAULT_PARSER_MODEL,
                }, 0)
            finally:
                faceslim.FaceWarpEngine = original_engine

            self.assertTrue(ok)
            self.assertTrue(output_path.exists())
            self.assertIsNotNone(cv2.imread(str(output_path)))

    def test_worker_job_cancellation_skips_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "input.png"
            output_path = tmp_path / "out" / "slimmed.png"
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 120, dtype=np.uint8))
            done = []
            updates = []
            thread = faceslim.BatchThread(
                [str(image_path)],
                str(output_path.parent),
                faceslim.DEFAULT_PARAMS.copy(),
                jobs=[{
                    "input": str(image_path),
                    "output": str(output_path),
                    "params": faceslim.DEFAULT_PARAMS.copy(),
                    "max_faces": 1,
                    "watermark": False,
                    "preserve_metadata": True,
                    "compare_mode": "none",
                    "parser_model": faceslim.DEFAULT_PARSER_MODEL,
                }],
            )
            thread.job_update.connect(lambda *args: updates.append(args))
            thread.finished.connect(lambda *args: done.append(args))
            thread.cancel_job(0)

            thread.run()

            self.assertEqual(done[-1], (0, 0, 1))
            self.assertIn("Skipped before start", updates[-1][3])
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
