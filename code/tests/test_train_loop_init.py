import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.train_util import TrainLoop


class TrainLoopInitializationTests(unittest.TestCase):
    def test_device_is_available_when_precision_mode_is_resolved(self):
        cfg = SimpleNamespace(
            datetime="test",
            lr=1e-4,
            gradient_clipping=1.0,
            epochs=2,
            num_class=4,
            precision="bf16",
            dataset_cls="pku",
            experiment_name="constructor_test",
        )
        diffusion = SimpleNamespace(model=torch.nn.Linear(2, 2))
        with patch("scripts.train_util.SummaryWriter"):
            loop = TrainLoop(
                cfg,
                diffusion_model=diffusion,
                training_dl=None,
                testing_dl=None,
                evaling_dl=None,
                device=torch.device("cuda"),
            )
        self.assertEqual(loop.device.type, "cuda")
        self.assertTrue(loop.autocast_enabled)


if __name__ == "__main__":
    unittest.main()
