import math
import unittest

import torch

from utils.benchmark_metrics import aggregate_records, diagnostic_layout_fd, geometry_records


class BenchmarkMetricTests(unittest.TestCase):
    def test_geometry_and_paired_metrics_on_perfect_layout(self):
        prediction = torch.tensor(
            [[
                [1, 0.25, 0.25, 0.20, 0.20],
                [2, 0.75, 0.75, 0.20, 0.20],
                [0, 0.00, 0.00, 0.00, 0.00],
            ]],
            dtype=torch.float32,
        )
        ground_truth = [{
            "classes": [1, 2],
            "boxes": [[0.15, 0.15, 0.35, 0.35], [0.65, 0.65, 0.85, 0.85]],
        }]
        record = geometry_records(["sample.png"], prediction, ground_truth)[0]
        self.assertAlmostEqual(record["val"], 1.0)
        self.assertAlmostEqual(record["oob"], 0.0)
        self.assertAlmostEqual(record["sma"], 0.0)
        self.assertAlmostEqual(record["ove"], 0.0)
        self.assertAlmostEqual(record["paired_iou"], 1.0, places=5)
        self.assertAlmostEqual(record["max_iou"], 1.0, places=5)
        self.assertAlmostEqual(record["type_f1"], 1.0)
        self.assertTrue(math.isnan(record["undl"]))

    def test_invalid_small_and_oob_are_recorded_before_clamping(self):
        prediction = torch.tensor(
            [[[1, 1.05, 0.50, 0.20, 0.20], [2, 0.50, 0.50, 0.01, 0.01]]],
            dtype=torch.float32,
        )
        record = geometry_records(["sample.png"], prediction)[0]
        self.assertGreater(record["oob"], 0.0)
        self.assertGreater(record["sma"], 0.0)
        self.assertLess(record["val"], 1.0)

    def test_empty_underlay_values_do_not_poison_other_aggregates(self):
        aggregate = aggregate_records([
            {"image": "a", "ove": 0.1, "undl": float("nan")},
            {"image": "b", "ove": 0.3, "undl": 0.8},
        ])
        self.assertAlmostEqual(aggregate["ove"], 0.2)
        self.assertAlmostEqual(aggregate["undl"], 0.8)

    def test_handcrafted_fd_is_zero_for_identical_sets(self):
        prediction = torch.tensor(
            [
                [[1, 0.25, 0.25, 0.2, 0.2], [0, 0, 0, 0, 0]],
                [[2, 0.75, 0.75, 0.2, 0.2], [0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        ground_truth = [
            {"classes": [1], "boxes": [[0.15, 0.15, 0.35, 0.35]]},
            {"classes": [2], "boxes": [[0.65, 0.65, 0.85, 0.85]]},
        ]
        self.assertAlmostEqual(diagnostic_layout_fd(prediction, ground_truth, 4), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
