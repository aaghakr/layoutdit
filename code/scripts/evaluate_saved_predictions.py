"""Evaluate IntentDiT or third-party predictions with one shared evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = Path(__file__).resolve().parents[1]

from utils.benchmark_metrics import load_ground_truth_layouts
from utils.metric import metric
from utils.util import load_config, rebase_project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--anno", choices=("anno", "unanno"), default="anno")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--protocol",
        choices=("image_only", "oracle_prompt", "freeform_prompt", "text_baseline", "cross_dataset"),
        required=True,
    )
    parser.add_argument("--prompts-csv", default="")
    parser.add_argument("--text-control", action="store_true")
    parser.add_argument("--spatial-metrics", action="store_true")
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    cfg = load_config(
        str(CODE_ROOT / "configs" / f"{args.dataset}_{args.anno}_test.yaml"),
        path_profile=args.path_profile,
    )
    cfg.text_control = args.text_control
    cfg.spatial_metrics = args.spatial_metrics
    cfg.protocol = args.protocol
    if args.prompts_csv:
        cfg.paths.test.all_prompts = rebase_project_path(args.prompts_csv, args.path_profile)

    source_path = Path(rebase_project_path(args.predictions, args.path_profile))
    payload = torch.load(source_path, map_location="cpu")
    if isinstance(payload, dict):
        if "img_names" not in payload or "test_output" not in payload:
            raise ValueError("Prediction dictionary must contain img_names and test_output")
        img_names = list(payload["img_names"])
        test_output = payload["test_output"]
    elif torch.is_tensor(payload):
        raise ValueError(
            "A bare tensor has no image ordering. Save {'img_names', 'test_output'} instead."
        )
    else:
        raise TypeError(f"Unsupported prediction payload: {type(payload)!r}")

    ground_truth = (
        load_ground_truth_layouts(img_names, cfg) if args.anno == "anno" else None
    )
    metrics, records = metric(
        img_names,
        test_output,
        cfg,
        ground_truth=ground_truth,
        return_records=True,
    )
    invalid_names = {
        os.path.basename(str(name)) for name in payload.get("invalid_image_names", [])
    }
    if invalid_names:
        for record in records:
            record["format_failure"] = float(record["image"] in invalid_names)
        metrics["format_failure_rate"] = float(
            sum(record["format_failure"] for record in records) / len(records)
        )
    else:
        metrics["format_failure_rate"] = 0.0

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.project_root) / "experiments" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.experiment_name
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(records).to_csv(f"{prefix}_per_image.csv", index=False)
    evidence = {
        "experiment": args.experiment_name,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "annotation_split": args.anno,
        "source_predictions": str(source_path),
        "num_samples": len(img_names),
        "shared_evaluator": True,
        "invalid_outputs": len(invalid_names),
    }
    Path(f"{prefix}_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Wrote {prefix}_metrics.json and per-image evidence for {len(img_names)} samples")


if __name__ == "__main__":
    main()
