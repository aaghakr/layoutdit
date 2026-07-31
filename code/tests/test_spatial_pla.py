import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from cgbdm.text_spatial import parse_positions_from_prompt
from utils.metric import _parse_prompt_counts
from utils.spatial_pla import parse_relations_from_prompt, spatial_pla_cal


class SpatialPlaTests(unittest.TestCase):
    def test_perfect_and_mismatched_cells(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            path = handle.name
        try:
            pd.DataFrame(
                {
                    "poster_path": ["a.png", "b.png"],
                    "text_prompt": [
                        "1 Logo at top-left, 1 Text at bottom-center.",
                        "1 Logo at bottom-right.",
                    ],
                }
            ).to_csv(path, index=False)
            cfg = SimpleNamespace(
                num_class=3,
                paths=SimpleNamespace(test=SimpleNamespace(all_prompts=path)),
            )
            classes = torch.tensor([[[2], [1]], [[2], [0]]])
            boxes = torch.tensor(
                [
                    [[0.0, 0.0, 0.2, 0.2], [0.4, 0.8, 0.6, 1.0]],
                    [[0.0, 0.0, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]],
                ]
            )
            result = spatial_pla_cal(["a.png", "b.png"], classes, boxes, cfg)
            self.assertAlmostEqual(result["spla"], 2 / 3)
            self.assertAlmostEqual(result["spla_text"], 1.0)
            self.assertAlmostEqual(result["spla_logo"], 0.5)
            self.assertEqual(result["spla_n"], 2.0)
        finally:
            os.unlink(path)

    def test_relation_parser(self):
        relations = parse_relations_from_prompt(
            "text_0 should be below logo_0 and underlay_0 contains text_0"
        )
        self.assertEqual(
            relations,
            [
                ("Text", 0, "below", "Logo", 0),
                ("Underlay", 0, "contains", "Text", 0),
            ],
        )

    def test_grouped_spatial_prompt_parser(self):
        parsed = parse_positions_from_prompt(
            "Place two Logos with 1 at top-center and 1 at middle-left, "
            "3 Texts with 2 at top-center and 1 at bottom-center."
        )
        self.assertEqual(parsed["Logo"], ["top-center", "middle-left"])
        self.assertEqual(parsed["Text"], ["top-center", "top-center", "bottom-center"])

    def test_freeform_descriptive_spatial_prompt_parser(self):
        prompt = (
            "Create a poster with large title text at top-center, "
            "two text boxes at middle-center, a small logo at bottom-left, "
            "and circle underlays at top-left, middle-center, and bottom-right."
        )
        parsed = parse_positions_from_prompt(prompt)
        self.assertEqual(
            parsed["Text"],
            ["top-center", "middle-center", "middle-center"],
        )
        self.assertEqual(parsed["Logo"], ["bottom-left"])
        self.assertEqual(
            parsed["Underlay"],
            ["top-left", "middle-center", "bottom-right"],
        )

    def test_count_parser_uses_freeform_spatial_assignments(self):
        counts = _parse_prompt_counts(
            "Create a clean poster with title text at top-right, "
            "short caption text at middle-right, and a small logo at top-left."
        )
        self.assertEqual(counts["Text"], 2)
        self.assertEqual(counts["Logo"], 1)


if __name__ == "__main__":
    unittest.main()
