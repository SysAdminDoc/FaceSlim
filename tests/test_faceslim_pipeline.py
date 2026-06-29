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
from PIL import Image as PILImage

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


class ModelInventoryTests(unittest.TestCase):
    def test_model_paths_use_configured_model_directory(self):
        self.assertEqual(
            Path(faceslim.MODEL_PATH).parent,
            Path(faceslim.MODEL_DIR),
        )
        self.assertEqual(
            Path(faceslim.parser_model_path("bisenet_resnet34")).parent,
            Path(faceslim.MODEL_DIR),
        )

    def test_model_inventory_includes_source_license_hash_and_provider(self):
        inventory = {item["key"]: item for item in faceslim.model_inventory("cpu")}

        self.assertIn("face_landmarker", inventory)
        self.assertEqual(inventory["face_landmarker"]["license"], "Apache-2.0")
        self.assertIn("mediapipe", inventory["face_landmarker"]["source_url"])
        self.assertEqual(inventory["bisenet_resnet18"]["license"], "MIT")
        self.assertEqual(inventory["bisenet_resnet18"]["provider"], "CPUExecutionProvider")
        self.assertIn("sha256", inventory["modnet_photographic"])
        self.assertEqual(inventory["gfpgan_1.4"]["license"], "Apache-2.0")
        self.assertEqual(inventory["real_esrgan_x2"]["license"], "BSD-3-Clause")

    def test_list_models_cli_exits_without_landmark_startup(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "FaceSlim_v1.py"), "--list-models", "--onnx-provider", "cpu"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Model Inventory", result.stdout)
        self.assertIn("Apache-2.0", result.stdout)
        self.assertIn("CPUExecutionProvider", result.stdout)


class RuntimeProviderTests(unittest.TestCase):
    def test_provider_override_falls_back_to_cpu_when_unavailable(self):
        resolution = faceslim.resolve_onnx_providers(
            "cuda",
            available=["CPUExecutionProvider"],
        )

        self.assertEqual(resolution["preference"], "cuda")
        self.assertEqual(resolution["providers"], ["CPUExecutionProvider"])
        self.assertEqual(resolution["selected_provider"], "CPUExecutionProvider")
        self.assertIn("unavailable", resolution["fallback_reason"])

    def test_directml_override_places_cpu_fallback_second(self):
        resolution = faceslim.resolve_onnx_providers(
            "directml",
            available=["DmlExecutionProvider", "CPUExecutionProvider"],
        )

        self.assertEqual(resolution["providers"], ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(resolution["selected_provider"], "DmlExecutionProvider")
        self.assertEqual(resolution["fallback_reason"], "override honored")

    def test_provider_diagnostics_lists_models_without_benchmark(self):
        text = faceslim.provider_diagnostics_text(
            "cpu",
            faceslim.DEFAULT_PARSER_MODEL,
            run_benchmark=False,
            ensure_parser=False,
        )

        self.assertIn("Available ONNX providers", text)
        self.assertIn("Provider preference: CPU", text)
        self.assertIn("Face parsing", text)
        self.assertIn("Matting", text)


class PostStageTests(unittest.TestCase):
    def test_post_stage_output_dimensions_and_compare_resize(self):
        params = {"post_stage_model": faceslim.POST_STAGE_REALESRGAN}
        self.assertEqual(faceslim.post_stage_output_dimensions(320, 240, params), (640, 480))
        self.assertEqual(faceslim.video_output_size(320, 240, "side_by_side", params), (1280, 480))

        original = np.zeros((4, 4, 3), dtype=np.uint8)
        processed = np.full((8, 8, 3), 255, dtype=np.uint8)
        compared = faceslim.compose_compare_frame(original, processed, "split")

        self.assertEqual(compared.shape, processed.shape)

    def test_tiled_upscale_covers_last_row_and_column(self):
        class FakeUpscaler(faceslim.UpscalePostStage):
            def __init__(self):
                pass

            def _run_tile(self, tile_rgb):
                return np.repeat(np.repeat(tile_rgb, 2, axis=0), 2, axis=1)

        image = np.zeros((130, 131, 3), dtype=np.uint8)
        image[-1, -1] = [10, 20, 30]

        upscaled = FakeUpscaler().upscale(image, tile_size=128)

        self.assertEqual(upscaled.shape, (260, 262, 3))
        self.assertTrue(np.all(upscaled[-1, -1] == [10, 20, 30]))


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
                    "onnx_provider": "cpu",
                    "post_stage_model": faceslim.POST_STAGE_GFPGAN,
                    "faces": 3,
                    "watermark": True,
                    "files": [{
                        "input": "input.jpg",
                        "output": "out/result.png",
                        "post_stage_model": faceslim.POST_STAGE_GFPGAN_REALESRGAN,
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
            self.assertEqual(jobs[0]["onnx_provider"], "cpu")
            self.assertEqual(jobs[0]["params"]["post_stage_model"], faceslim.POST_STAGE_GFPGAN_REALESRGAN)
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
            with PILImage.open(output_path) as saved:
                self.assertIn("XML:com.adobe.xmp", saved.info)
                self.assertEqual(saved.info.get("FaceSlim:DisclosureWatermark"), "false")

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


class MediaPreflightTests(unittest.TestCase):
    def test_image_preflight_refuses_directory_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "input.png"
            output_dir = tmp_path / "already_a_dir"
            output_dir.mkdir()
            cv2.imwrite(str(image_path), np.full((12, 12, 3), 160, dtype=np.uint8))

            report = faceslim.preflight_media_job(str(image_path), str(output_dir))

            self.assertFalse(report["ok"])
            self.assertEqual(report["kind"], "image")
            self.assertIn("directory", faceslim.preflight_failure_message(report))
            self.assertIn("est output", faceslim.format_preflight_summary(report))

    def test_video_preflight_reports_metrics_disk_and_codec(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "input.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0,
                (16, 16),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV mp4v writer unavailable")
            writer.write(np.full((16, 16, 3), 80, dtype=np.uint8))
            writer.write(np.full((16, 16, 3), 120, dtype=np.uint8))
            writer.release()

            report = faceslim.preflight_media_job(
                str(video_path),
                str(tmp_path / "out" / "processed.mp4"),
                "side_by_side",
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["kind"], "video")
            self.assertEqual(report["width"], 16)
            self.assertEqual(report["output_width"], 32)
            self.assertGreaterEqual(report["frame_count"], 1)
            self.assertGreater(report["estimated_output_bytes"], 0)
            self.assertGreater(report["estimated_memory_bytes"], 0)
            self.assertGreater(report["available_disk_bytes"], 0)
            self.assertTrue(report["codec_ok"])
            self.assertIn("frames", faceslim.format_preflight_summary(report))

    def test_batch_image_preflight_fails_before_engine_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "input.png"
            output_dir = tmp_path / "blocked_output.png"
            output_dir.mkdir()
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 120, dtype=np.uint8))
            original_engine = faceslim.FaceWarpEngine
            faceslim.FaceWarpEngine = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("engine should not start after failed preflight")
            )
            try:
                thread = faceslim.BatchThread(
                    [str(image_path)],
                    str(tmp_path),
                    faceslim.DEFAULT_PARAMS.copy(),
                )
                with self.assertRaises(ValueError) as raised:
                    thread._process_image({
                        "input": str(image_path),
                        "output": str(output_dir),
                        "params": faceslim.DEFAULT_PARAMS.copy(),
                        "max_faces": 1,
                        "watermark": False,
                        "preserve_metadata": True,
                        "parser_model": faceslim.DEFAULT_PARSER_MODEL,
                    }, 0)
            finally:
                faceslim.FaceWarpEngine = original_engine

            self.assertIn("Preflight failed", str(raised.exception))


class ProvenanceMetadataTests(unittest.TestCase):
    def test_png_export_contains_xmp_and_iptc_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.png"
            rgb = np.full((12, 12, 3), 180, dtype=np.uint8)

            faceslim.save_rgb_image(str(output_path), rgb, None, preserve_metadata=False, watermark=True)

            with PILImage.open(output_path) as saved:
                xmp = saved.info["XML:com.adobe.xmp"]
                self.assertIn(faceslim.IPTC_DIGITAL_SOURCE_TYPE, xmp)
                self.assertIn("faceslim:DisclosureWatermark=\"true\"", xmp)
                self.assertEqual(saved.info["IPTC:DigitalSourceType"], faceslim.IPTC_DIGITAL_SOURCE_TYPE)
                self.assertEqual(saved.info["FaceSlim:SourceMetadataPreserved"], "false")

    def test_jpeg_export_contains_exif_and_xmp_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.jpg"
            rgb = np.full((12, 12, 3), 90, dtype=np.uint8)

            faceslim.save_rgb_image(str(output_path), rgb, None, preserve_metadata=True, watermark=False)

            data = output_path.read_bytes()
            self.assertIn(b"http://ns.adobe.com/xap/1.0/", data)
            self.assertIn(faceslim.IPTC_DIGITAL_SOURCE_TYPE.encode("utf-8"), data)
            with PILImage.open(output_path) as saved:
                exif = saved.getexif()
                self.assertIn("FaceSlim", exif.get(305))
                self.assertIn("AI modified by FaceSlim", exif.get(270))


if __name__ == "__main__":
    unittest.main()
