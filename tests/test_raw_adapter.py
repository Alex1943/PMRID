import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from raw_dataset import RawDataset, datasetgen_api
from run_raw import checkpoint_steps, evaluate, forward_padded, forward_tiled


DATASETGEN = Path(__file__).resolve().parents[2] / "DatasetGen"


class RawAdapterTests(unittest.TestCase):
    def test_manifest_scale_split_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for split, name in (("train", "a"), ("test", "b")):
                folder = root / "noise" / name
                folder.mkdir(parents=True)
                np.save(folder / "raw_noise_16.npy", np.full((20, 22, 4), 512, np.uint16))
                np.save(folder / "raw_gt.npy", np.full((20, 22, 4), 2.0, np.float32))
                np.save(folder / "alignratio.npy", np.array(3.0, np.float32))
                np.save(folder / "metadata.npy", {
                    "red_gain": np.array([2.0], np.float32),
                    "blue_gain": np.array([1.5], np.float32),
                    "cam2rgb": np.eye(3, dtype=np.float32)[None],
                })
                rows.append({"sample_id": name, "split": split, "subset": "noise",
                             "source_video": name, "input_path": str(folder.relative_to(root) / "raw_noise_16.npy"),
                             "gt_path": str(folder.relative_to(root) / "raw_gt.npy")})
            with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            train = RawDataset(root, "manifest.csv", DATASETGEN, "train", patch_size=16)
            test = RawDataset(root, "manifest.csv", DATASETGEN, "test")
            self.assertEqual((len(train), len(test)), (1, 1))
            self.assertEqual(train[0]["input"].shape, (4, 16, 16))
            self.assertAlmostEqual(float(train[0]["input"].mean()), 512 / 1023)
            self.assertAlmostEqual(float(train[0]["gt"].mean()), 2.0)
            data = test[0]
            protocol = datasetgen_api(DATASETGEN)[4](metric_max=65535, demosaic="net")
            raw = data["gt"].unsqueeze(0)
            metrics = protocol.calculate(raw, raw, data["metadata"])[0]
            self.assertEqual(set(("psnr", "ssim", "mu_psnr", "mu_ssim")) - metrics.keys(), set())
            self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)
            self.assertAlmostEqual(metrics["mu_ssim"], 1.0, places=5)
            class Identity(torch.nn.Module):
                def forward(self, x):
                    return x

            config = {"name": "PMRID-test", "experiment_note": "测试",
                      "metrics": {"hdr_scale": 16, "mu": 5000,
                                  "crop_border": 0, "demosaic": "net",
                                  "metric_max": 65535},
                      "test": {"print_freq": 1, "tile_size": 0,
                               "save_rgb": True, "save_rgb_interval": 1,
                               "save_rgb_max": 1}}
            report_dir = root / "report"
            report = evaluate(Identity(), test, config, DATASETGEN,
                              torch.device("cpu"), report_dir)
            self.assertEqual(report["summary"]["overall"]["count"], 1)
            self.assertEqual(len(list((report_dir / "rgb" / "noise").glob("*.png"))), 1)
            self.assertTrue((report_dir / "metrics_per_sample.csv").is_file())
            self.assertTrue((report_dir / "reproducibility.json").is_file())

    def test_padding_tiling_and_save_schedule(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                self.last_shape = x.shape
                return x

        model = Identity()
        raw = torch.rand(1, 4, 21, 29)
        self.assertTrue(torch.equal(forward_padded(model, raw), raw))
        self.assertEqual(model.last_shape[-2:], (32, 32))
        self.assertTrue(torch.allclose(forward_tiled(model, raw, 16, 4), raw, atol=1e-7))
        self.assertEqual(sorted(checkpoint_steps(80, 8)), list(range(10, 81, 10)))


if __name__ == "__main__":
    unittest.main()
