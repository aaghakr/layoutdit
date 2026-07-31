"""Count and box-density controls for occlusion/readability claims.

This script computes layout-density summaries directly from saved prediction
tensors. If a method has lower occlusion, these controls help determine whether
the gain coincides with fewer predicted elements or smaller predicted boxes.

Example:
    python scripts/analyze_density_controls.py \
      --prediction "PKU LayoutDiT" pku ../other_baselines/layoutidit/pku_anno_uncond_test_output.pt \
      --prediction "PKU IntentDiT" pku ../experiments/paper_figures/ivc_pku_vit_both_trainseed1_inferseed1_test_output.pt \
      --output-json ../experiments/paper_results/density_controls.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = Path(__file__).resolve().parents[1]

from utils.benchmark_metrics import decode_layout_tensor, load_ground_truth_layouts
from utils.util import load_config, rebase_project_path


def _areas(boxes: np.ndarray) -> np.ndarray:
    wh = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2])
    return wh[:, 0] * wh[:, 1]


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(finite)) if finite else float("nan")


def _load_prediction(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "img_names" not in payload or "test_output" not in payload:
        raise ValueError(f"{path} must contain a dict with img_names and test_output")
    return list(payload["img_names"]), payload["test_output"]


def summarize_prediction(path: Path, dataset: str, path_profile: str) -> dict[str, Any]:
    cfg = load_config(
        str(CODE_ROOT / "configs" / f"{dataset}_anno_test.yaml"),
        path_profile=path_profile,
    )
    img_names, test_output = _load_prediction(path)
    ground_truth = load_ground_truth_layouts(img_names, cfg)
    classes, raw_boxes, boxes = decode_layout_tensor(test_output)

    per_image: list[dict[str, float | str]] = []
    all_pred_areas: list[float] = []
    all_gt_areas: list[float] = []

    for image_name, pred_cls, _pred_raw_box, pred_box, gt in zip(
        img_names, classes, raw_boxes, boxes, ground_truth
    ):
        present = pred_cls > 0
        pred_boxes = pred_box[present]
        pred_areas = _areas(pred_boxes) if len(pred_boxes) else np.array([], dtype=np.float64)
        widths = pred_boxes[:, 2] - pred_boxes[:, 0] if len(pred_boxes) else np.array([])
        heights = pred_boxes[:, 3] - pred_boxes[:, 1] if len(pred_boxes) else np.array([])
        small = (
            (pred_areas < 1e-3) | (widths < 0.02) | (heights < 0.02)
            if len(pred_areas)
            else np.array([], dtype=bool)
        )
        gt_boxes = np.asarray(gt["boxes"], dtype=np.float64).reshape(-1, 4)
        gt_areas = _areas(gt_boxes) if len(gt_boxes) else np.array([], dtype=np.float64)
        n_pred = int(present.sum())
        n_gt = int(len(gt["classes"]))
        all_pred_areas.extend(float(value) for value in pred_areas)
        all_gt_areas.extend(float(value) for value in gt_areas)
        per_image.append(
            {
                "image": os.path.basename(str(image_name)),
                "n_pred": float(n_pred),
                "n_gt": float(n_gt),
                "count_error": float(n_pred - n_gt),
                "abs_count_error": float(abs(n_pred - n_gt)),
                "empty_layout": float(n_pred == 0),
                "small_element_rate": float(small.mean()) if len(small) else 0.0,
                "undercount": float(n_pred < n_gt),
                "severe_undercount": float(n_gt >= 2 and n_pred <= n_gt - 2),
                "total_pred_area": float(pred_areas.sum()) if len(pred_areas) else 0.0,
                "mean_pred_area_per_image": float(pred_areas.mean()) if len(pred_areas) else 0.0,
                "median_pred_area_per_image": float(np.median(pred_areas)) if len(pred_areas) else 0.0,
                "total_gt_area": float(gt_areas.sum()) if len(gt_areas) else 0.0,
            }
        )

    result: dict[str, Any] = {
        "source": str(path),
        "dataset": dataset,
        "n": len(per_image),
        "metrics": {},
        "distributions": {
            "n_pred": [record["n_pred"] for record in per_image],
            "total_pred_area": [record["total_pred_area"] for record in per_image],
            "count_error": [record["count_error"] for record in per_image],
            "small_element_rate": [record["small_element_rate"] for record in per_image],
        },
    }

    for key in [
        "n_pred",
        "n_gt",
        "count_error",
        "abs_count_error",
        "empty_layout",
        "small_element_rate",
        "undercount",
        "severe_undercount",
        "total_pred_area",
        "mean_pred_area_per_image",
        "median_pred_area_per_image",
        "total_gt_area",
    ]:
        values = [float(record[key]) for record in per_image]
        result["metrics"][f"{key}_mean"] = _finite_mean(values)
        result["metrics"][f"{key}_std"] = _finite_std(values)
        result["metrics"][f"{key}_median"] = float(np.median(values)) if values else float("nan")

    result["metrics"]["mean_pred_box_area"] = _finite_mean(all_pred_areas)
    result["metrics"]["median_pred_box_area"] = float(np.median(all_pred_areas)) if all_pred_areas else 0.0
    result["metrics"]["mean_gt_box_area"] = _finite_mean(all_gt_areas)
    result["metrics"]["median_gt_box_area"] = float(np.median(all_gt_areas)) if all_gt_areas else 0.0
    return result


def aggregate_labeled_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted({key for run in runs for key in run["metrics"]})
    result = {
        "n_runs": len(runs),
        "sources": [run["source"] for run in runs],
        "dataset": runs[0]["dataset"],
        "n_images": [run["n"] for run in runs],
        "metrics": {},
    }
    for metric in metrics:
        values = [float(run["metrics"][metric]) for run in runs if metric in run["metrics"]]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            continue
        result["metrics"][metric] = {
            "mean": float(np.mean(finite)),
            "std_across_runs": float(np.std(finite)),
            "values": finite,
        }
    return result


def format_value(item: dict[str, float], digits: int = 3) -> str:
    mean = item["mean"]
    std = item.get("std_across_runs", 0.0)
    if std > 0:
        return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"
    return f"{mean:.{digits}f}"


def write_tex(summary: dict[str, Any], output: Path) -> None:
    rows = [
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "Dataset & Method & $n_{pred}$ & Count err. & Empty$\\downarrow$ & Small$\\downarrow$ & Severe under$\\downarrow$ & Total area & Box area \\\\",
        "\\midrule",
    ]
    for label, item in summary["methods"].items():
        metrics = item["metrics"]
        dataset = item["dataset"].upper()
        rows.append(
            f"{dataset} & {label} & "
            f"{format_value(metrics['n_pred_mean'])} & "
            f"{format_value(metrics['count_error_mean'])} & "
            f"{format_value(metrics['empty_layout_mean'])} & "
            f"{format_value(metrics['small_element_rate_mean'])} & "
            f"{format_value(metrics['severe_undercount_mean'])} & "
            f"{format_value(metrics['total_pred_area_mean'])} & "
            f"{format_value(metrics['mean_pred_box_area'])} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        nargs=3,
        action="append",
        metavar=("LABEL", "DATASET", "PATH"),
        required=True,
        help="Prediction entry. Repeat labels to aggregate multiple training seeds.",
    )
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", default="")
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label, dataset, raw_path in args.prediction:
        if dataset not in {"pku", "cgl"}:
            raise ValueError(f"Unsupported dataset for {label}: {dataset}")
        path = Path(rebase_project_path(raw_path, args.path_profile))
        grouped[label].append(summarize_prediction(path, dataset, args.path_profile))

    summary = {
        "definition": {
            "severe_undercount": "1 when predicted count is at least two fewer than ground-truth count",
            "areas": "normalized xyxy box area after clamping to [0,1]",
            "box_area": "mean/median over all predicted boxes in a run",
            "small_element_rate": "fraction of predicted elements with area < 1e-3 or width/height < 0.02 after clamping",
        },
        "methods": {
            label: aggregate_labeled_runs(runs)
            for label, runs in grouped.items()
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_tex:
        write_tex(summary, Path(args.output_tex))
    print(f"Wrote density controls for {len(summary['methods'])} methods to {output_json}")


if __name__ == "__main__":
    main()
