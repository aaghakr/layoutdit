"""Aggregate metric JSON files from independently trained checkpoints.

Expected filename convention:
    <experiment>_trainseed<N>_inferseed<M>_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="experiments/paper_figures")
    parser.add_argument("--inference-seed", type=int, default=1)
    parser.add_argument("--include-prefix", default="")
    parser.add_argument("--exclude-prefix", default="")
    parser.add_argument("--include-pattern", default="", help="Regex applied to metric filename")
    parser.add_argument(
        "--output", default="experiments/multi_seed_results/training_seed_summary.json"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pattern = re.compile(
        rf"^(?P<name>.+)_trainseed(?P<train>\d+)_inferseed{args.inference_seed}_metrics\.json$"
    )
    grouped: dict[str, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*_metrics.json")):
        if args.include_prefix and not path.name.startswith(args.include_prefix):
            continue
        if args.exclude_prefix and path.name.startswith(args.exclude_prefix):
            continue
        if args.include_pattern and not re.search(args.include_pattern, path.name):
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        with path.open() as f:
            metrics = json.load(f)
        grouped[match.group("name")].append((int(match.group("train")), metrics))

    summary = {}
    for name, runs in sorted(grouped.items()):
        metric_names = sorted({key for _, run in runs for key in run})
        result = {"training_seeds": sorted(seed for seed, _ in runs), "metrics": {}}
        for metric in metric_names:
            values = [float(run[metric]) for _, run in runs if metric in run]
            values = [value for value in values if math.isfinite(value)]
            if values:
                result["metrics"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
        summary[name] = result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    markdown = output.with_suffix(".md")
    metric_order = [
        "val", "oob", "sma", "ali", "ove", "vb", "spacing_cv", "uti", "occ", "rea",
        "undl", "unds", "paired_iou", "max_iou", "format_failure", "type_f1",
        "n_pred", "n_gt", "empty_layout", "undercount", "severe_undercount",
        "total_pred_area", "mean_pred_area", "median_pred_area", "pla_count",
        "exact_count_match", "spla", "hfd"
    ]
    lines = [
        "| experiment | training seeds | " + " | ".join(metric_order) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(metric_order)) + " |",
    ]
    for name, result in summary.items():
        cells = [name, ",".join(map(str, result["training_seeds"]))]
        for metric in metric_order:
            item = result["metrics"].get(metric)
            cells.append("---" if item is None else f"{item['mean']:.4f}±{item['std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output} and {markdown} ({len(summary)} experiment groups).")


if __name__ == "__main__":
    main()
