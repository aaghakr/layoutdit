import re
import tempfile
import unittest
from pathlib import Path

import torch

from data_process.dataloader import save_image_order


CODE_ROOT = Path(__file__).resolve().parents[1]


class SourceInvariantTests(unittest.TestCase):
    def test_image_order_save_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "image_order.pt"
            save_image_order(["b.png", "a.png"], output)
            self.assertEqual(torch.load(output), ["b.png", "a.png"])

    def test_box_normalization_never_slices_alternating_rows(self):
        bad = re.compile(r"(?:sal_box|intent_box)\[(?:::2|1::2)\]")
        offenders = []
        for path in CODE_ROOT.rglob("*.py"):
            if bad.search(path.read_text()):
                offenders.append(str(path.relative_to(CODE_ROOT)))
        self.assertEqual(offenders, [])

    def test_high_place_loss_config_is_actually_high(self):
        path = CODE_ROOT / "configs" / "pku_lambda_high_place.yaml"
        self.assertIn("lambda2: 0.10", path.read_text())

    def test_normalized_placement_padding_is_not_zero(self):
        loader = (CODE_ROOT / "data_process" / "dataloader.py").read_text()
        self.assertNotIn("padded = torch.zeros(self.max_elem, 4", loader)

    def test_placement_padding_is_masked_in_attention(self):
        model = (CODE_ROOT / "cgbdm" / "layout_model.py").read_text()
        self.assertIn("intent_box[..., 2] <= -1.0 + 1e-6", model)

    def test_training_accepts_independent_seed(self):
        train = (CODE_ROOT / "scripts" / "train.py").read_text()
        self.assertIn("seed = opt.seed", train)
        self.assertIn("'--seed'", train)

    def test_a100_training_path_does_not_retain_epoch_graphs(self):
        loop = (CODE_ROOT / "scripts" / "train_util.py").read_text()
        self.assertIn("total_loss = 0.0", loop)
        self.assertIn("loss_value = float(loss.detach())", loop)
        self.assertIn("self.diffusion_model.model.train()", loop)
        self.assertIn("torch.autocast", loop)
        self.assertIn("epoch == self.epochs", loop)

    def test_training_exposes_acceleration_controls(self):
        train = (CODE_ROOT / "scripts" / "train.py").read_text()
        for flag in (
            "--precision",
            "--allow-tf32",
            "--skip-training-validation",
            "--train-batch-size",
        ):
            self.assertIn(flag, train)
        common = (CODE_ROOT.parent / "scripts" / "paper" / "00_common.sh").read_text()
        self.assertIn("JOBS_PER_GPU", common)
        self.assertIn('gpu_array+=("$gpu")', common)

    def test_evaluation_can_save_raw_outputs(self):
        test = (CODE_ROOT / "scripts" / "test.py").read_text()
        self.assertIn("'--save-test-output'", test)
        self.assertIn("'test_output': test_output.cpu()", test)
        self.assertIn("'ddim_schedule': ddim_schedule", test)
        self.assertIn("'--prompts-csv'", test)
        self.assertIn("'--spatial-metrics'", test)
        self.assertIn("'--prompt-subset-only'", test)
        self.assertIn("create_text_spatial_boxes_batch", test)
        self.assertIn("'--protocol'", test)
        self.assertIn("_per_image.csv", test)

    def test_ddim_default_matches_training_schedule(self):
        diffusion = (CODE_ROOT / "cgbdm" / "diffusion.py").read_text()
        self.assertIn("ddim_schedule: str = 'cosine'", diffusion)
        self.assertIn("self.alphas_cumprod_ddim = self.alphas_cumprod", diffusion)
        self.assertIn("schedule in {'cosine', 'linear'}", diffusion)
        common = (CODE_ROOT.parent / "scripts" / "paper" / "00_common.sh").read_text()
        self.assertIn("DDIM_SCHEDULE=${DDIM_SCHEDULE:-cosine}", common)
        self.assertIn('actual_schedule = evidence.get("ddim_schedule")', common)

    def test_training_seed_aggregator_distinguishes_seed_types(self):
        script = (CODE_ROOT / "scripts" / "aggregate_training_seeds.py").read_text()
        self.assertIn("_trainseed", script)
        self.assertIn("_inferseed", script)


if __name__ == "__main__":
    unittest.main()
