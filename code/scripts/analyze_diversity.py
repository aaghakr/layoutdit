"""Quality-aware diversity analysis over repeated samples of identical inputs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_metrics import decode_layout_tensor


def sample_distance(classes_a, boxes_a, classes_b, boxes_b) -> tuple[float, float]:
    distances = []
    matched = 0
    union_count = max(np.sum(classes_a > 0), np.sum(classes_b > 0), 1)
    for class_id in sorted(set(classes_a.tolist()) | set(classes_b.tolist())):
        if class_id <= 0:
            continue
        a = boxes_a[classes_a == class_id]
        b = boxes_b[classes_b == class_id]
        if len(a) == 0 or len(b) == 0:
            continue
        center_a = (a[:, :2] + a[:, 2:]) / 2
        center_b = (b[:, :2] + b[:, 2:]) / 2
        cost = np.linalg.norm(center_a[:, None] - center_b[None, :], axis=2)
        row, col = linear_sum_assignment(cost)
        distances.extend(cost[row, col].tolist())
        matched += len(row)
    displacement = float(np.mean(distances)) if distances else 0.0
    count_disagreement = 1.0 - matched / union_count
    return displacement, float(count_disagreement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payloads = [torch.load(path, map_location="cpu") for path in args.predictions]
    names = [list(payload["img_names"]) for payload in payloads]
    if any(items != names[0] for items in names[1:]):
        raise ValueError("All prediction files must use identical image ordering")
    decoded = [decode_layout_tensor(payload["test_output"]) for payload in payloads]
    per_image = []
    for image_index, image_name in enumerate(names[0]):
        displacements, count_disagreements = [], []
        for left, right in itertools.combinations(range(len(decoded)), 2):
            cls_a, _, box_a = decoded[left]
            cls_b, _, box_b = decoded[right]
            displacement, disagreement = sample_distance(
                cls_a[image_index], box_a[image_index], cls_b[image_index], box_b[image_index]
            )
            displacements.append(displacement)
            count_disagreements.append(disagreement)
        mean_displacement = float(np.mean(displacements))
        mean_count_disagreement = float(np.mean(count_disagreements))
        per_image.append(
            {
                "image": str(image_name),
                "matched_center_displacement": mean_displacement,
                "count_disagreement": mean_count_disagreement,
                "duplicate": bool(
                    mean_displacement <= args.duplicate_tolerance
                    and mean_count_disagreement == 0.0
                ),
            }
        )
    result = {
        "num_samples_per_condition": len(payloads),
        "num_conditions": len(per_image),
        "matched_center_displacement": float(np.mean([x["matched_center_displacement"] for x in per_image])),
        "count_disagreement": float(np.mean([x["count_disagreement"] for x in per_image])),
        "duplicate_rate": float(np.mean([x["duplicate"] for x in per_image])),
        "per_image": per_image,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote diversity analysis to {output}")


if __name__ == "__main__":
    main()
