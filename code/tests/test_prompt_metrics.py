import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.metric import tla_cal


class PromptMetricTests(unittest.TestCase):
    def test_zero_request_nonempty_layout_is_pla_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            prompt_path = Path(temporary) / "prompts.csv"
            prompt_path.write_text(
                "poster_path,text_prompt\n"
                "zero.png,No layout elements requested.\n"
            )
            cfg = SimpleNamespace(
                text_control=True,
                num_class=4,
                paths=SimpleNamespace(
                    test=SimpleNamespace(all_prompts=str(prompt_path))
                ),
            )
            classes = torch.tensor([[[1], [0], [0]]], dtype=torch.long)
            self.assertEqual(tla_cal(["zero.png"], classes, cfg), 0.0)

    def test_zero_request_empty_layout_is_pla_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            prompt_path = Path(temporary) / "prompts.csv"
            prompt_path.write_text(
                "poster_path,text_prompt\n"
                "zero.png,No layout elements requested.\n"
            )
            cfg = SimpleNamespace(
                text_control=True,
                num_class=4,
                paths=SimpleNamespace(
                    test=SimpleNamespace(all_prompts=str(prompt_path))
                ),
            )
            classes = torch.tensor([[[0], [0], [0]]], dtype=torch.long)
            self.assertEqual(tla_cal(["zero.png"], classes, cfg), 1.0)


if __name__ == "__main__":
    unittest.main()
