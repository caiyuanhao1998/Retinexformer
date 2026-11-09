"""CUDA integration regressions for the benchmark CLI; no dataset download needed."""

import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "Benchmark evaluation requires CUDA")
class BenchmarkCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

        cls.temp = tempfile.TemporaryDirectory(prefix="retinexformer-cli-test-")
        cls.folder = Path(cls.temp.name)
        cls.weights = cls.folder / "weights.pth"
        torch.manual_seed(0)
        model = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2])
        torch.save({"params": model.state_dict()}, cls.weights)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.folder = self.__class__.folder / self._testMethodName
        self.input = self.folder / "input"
        self.target = self.folder / "target"
        self.input.mkdir(parents=True)
        self.target.mkdir()
        self.output = self.folder / "output"

    def evaluate(self, ensemble=False):
        config = yaml.safe_load((ROOT / "Options/RetinexFormer_LOL_v1.yml").read_text())
        config["datasets"]["val"]["dataroot_lq"] = str(self.input)
        config["datasets"]["val"]["dataroot_gt"] = str(self.target)
        path = self.folder / "config.yml"
        path.write_text(yaml.safe_dump(config))
        command = [
            sys.executable, str(ROOT / "Enhancement/test_from_dataset.py"),
            "--opt", str(path), "--weights", str(self.weights), "--dataset", "LOL_v1",
            "--gpus", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "--output_dir", str(self.output), "--result_dir", str(self.folder / "results"),
        ]
        if ensemble:
            command.append("--self_ensemble")
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90, check=False)

    def test_missing_images_fail_instead_of_nan_success(self):
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No input images found", result.stderr)
        self.assertNotIn("PSNR: nan", result.stdout)

    def test_mismatched_pairs_are_not_silently_truncated(self):
        for name in ["one.png", "two.png"]:
            Image.new("RGB", (32, 32), (25, 25, 25)).save(self.input / name)
        Image.new("RGB", (32, 32), (80, 80, 80)).save(self.target / "one.png")
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("matching input/target counts", result.stderr)

    def test_odd_width_high_resolution_keeps_dimensions(self):
        size = (65, 3001)
        Image.new("RGB", size, (25, 25, 25)).save(self.input / "odd.png")
        Image.new("RGB", size, (80, 80, 80)).save(self.target / "odd.png")
        for ensemble in [False, True]:
            with self.subTest(self_ensemble=ensemble):
                result = self.evaluate(ensemble=ensemble)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with Image.open(self.output / "odd.png") as image:
                    self.assertEqual(image.size, size)
                metrics = re.findall(r"(?:PSNR|SSIM):\s*(\S+)", result.stdout)
                self.assertEqual(len(metrics), 2)
                self.assertTrue(all(math.isfinite(float(value)) for value in metrics))


if __name__ == "__main__":
    unittest.main()
