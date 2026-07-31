"""Measure target response and non-target stability after a one-class prompt edit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark_metrics import decode_layout_tensor


def non_target_displacement(classes_a, boxes_a, classes_b, boxes_b, target_class):
    distances = []
    for class_id in sorted(set(classes_a.tolist()) | set(classes_b.tolist())):
        if class_id <= 0 or class_id == target_class:
            continue
        a = boxes_a[classes_a == class_id]
        b = boxes_b[classes_b == class_id]
        if not len(a) or not len(b):
            continue
        centers_a = (a[:, :2] + a[:, 2:]) / 2
        centers_b = (b[:, :2] + b[:, 2:]) / 2
        cost = np.linalg.norm(centers_a[:, None] - centers_b[None, :], axis=2)
        row, col = linear_sum_assignment(cost)
        distances.extend(cost[row, col].tolist())
    return float(np.mean(distances)) if distances else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--edited", required=True)
    parser.add_argument("--target-class", type=int, default=1)
    parser.add_argument("--target-delta", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu")
    edited = torch.load(args.edited, map_location="cpu")
    if list(base["img_names"]) != list(edited["img_names"]):
        raise ValueError("Prompt-edit predictions must have identical image ordering")
    base_cls, _, base_box = decode_layout_tensor(base["test_output"])
    edit_cls, _, edit_box = decode_layout_tensor(edited["test_output"])
    responses, locality = [], []
    for index in range(len(base_cls)):
        delta = int(np.sum(edit_cls[index] == args.target_class) - np.sum(base_cls[index] == args.target_class))
        responses.append(delta)
        locality.append(non_target_displacement(
            base_cls[index], base_box[index], edit_cls[index], edit_box[index], args.target_class
        ))
    result = {
        "n": len(responses),
        "target_class": args.target_class,
        "requested_delta": args.target_delta,
        "mean_observed_delta": float(np.mean(responses)),
        "exact_response_rate": float(np.mean(np.asarray(responses) == args.target_delta)),
        "directional_response_rate": float(np.mean(np.asarray(responses) * args.target_delta > 0)),
        "non_target_center_displacement": float(np.mean(locality)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote prompt-edit locality to {output}")


if __name__ == "__main__":
    main()
