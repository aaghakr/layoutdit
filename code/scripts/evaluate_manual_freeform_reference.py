"""Evaluate free-form predictions against manually audited prompt references.

The automatic free-form evaluator derives requested counts/positions from the
text-spatial parser.  Once parser outputs have been manually audited, the paper
needs a primary evaluation against the manual interpretation rather than the
parser output itself.  This script recomputes prompt-adherence metrics from
saved prediction tensors without rerunning model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgbdm.text_spatial import get_class_map
from utils.benchmark_metrics import decode_layout_tensor


GRID_ROWS = ("top", "middle", "bottom")
GRID_COLS = ("left", "center", "right")


def _cell(x: float, y: float) -> str:
    col = min(2, max(0, int(x * 3)))
    row = min(2, max(0, int(y * 3)))
    return f"{GRID_ROWS[row]}-{GRID_COLS[col]}"


def _json_loads(value: str, *, row: int, column: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in row {row}, column {column}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in row {row}, column {column}")
    return parsed


def _load_audit(path: Path, class_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "poster_path",
            "text_prompt",
            "audit_status",
            "manual_counts_json",
            "manual_positions_json",
            "parser_counts_json",
            "parser_positions_json",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            image = Path(str(row["poster_path"])).name
            status = str(row["audit_status"]).strip().lower()
            if status not in {"accepted", "corrected", "exclude"}:
                raise ValueError(f"{path}:{line_number} has invalid audit_status={status!r}")
            manual_counts = _json_loads(
                row["manual_counts_json"], row=line_number, column="manual_counts_json"
            )
            manual_positions = _json_loads(
                row["manual_positions_json"], row=line_number, column="manual_positions_json"
            )
            parser_counts = _json_loads(
                row["parser_counts_json"], row=line_number, column="parser_counts_json"
            )
            parser_positions = _json_loads(
                row["parser_positions_json"], row=line_number, column="parser_positions_json"
            )
            rows.append(
                {
                    "image": image,
                    "poster_path": row["poster_path"],
                    "text_prompt": row["text_prompt"],
                    "audit_status": status,
                    "is_retained": status != "exclude",
                    "parser_correct": status == "accepted",
                    "manual_counts": {
                        name: int(manual_counts.get(name, 0) or 0)
                        for name in class_names
                    },
                    "manual_positions": {
                        name: [str(item) for item in manual_positions.get(name, [])]
                        for name in class_names
                    },
                    "parser_counts": {
                        name: int(parser_counts.get(name, 0) or 0)
                        for name in class_names
                    },
                    "parser_positions": {
                        name: [str(item) for item in parser_positions.get(name, [])]
                        for name in class_names
                    },
                }
            )
    frame = pd.DataFrame(rows)
    if frame["image"].duplicated().any():
        duplicated = sorted(frame.loc[frame["image"].duplicated(), "image"].unique())
        raise ValueError(f"{path} contains duplicate image rows: {duplicated[:10]}")
    return frame


def _load_prediction(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "img_names" not in payload or "test_output" not in payload:
        raise ValueError(f"{path} must contain a dict with img_names and test_output")
    names = [Path(str(name)).name for name in payload["img_names"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate image names")
    classes, raw_boxes, boxes = decode_layout_tensor(payload["test_output"])
    if len(names) != len(classes):
        raise ValueError(f"{path} image count and tensor batch differ")
    return {
        "path": str(path),
        "names": names,
        "index": {name: idx for idx, name in enumerate(names)},
        "classes": classes,
        "raw_boxes": raw_boxes,
        "boxes": boxes,
    }


def _load_existing_per_image(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "image" not in frame.columns:
        raise ValueError(f"{path} must contain an image column")
    frame = frame.copy()
    frame["image"] = frame["image"].astype(str).map(lambda value: Path(value).name)
    if frame["image"].duplicated().any():
        duplicated = sorted(frame.loc[frame["image"].duplicated(), "image"].unique())
        raise ValueError(f"{path} contains duplicate image rows: {duplicated[:10]}")
    return frame


def _count_scores(
    predicted_counts: dict[str, int],
    expected_counts: dict[str, int],
) -> dict[str, float]:
    tp = sum(
        min(int(predicted_counts.get(name, 0)), int(expected_counts.get(name, 0)))
        for name in expected_counts
    )
    n_pred = sum(int(value) for value in predicted_counts.values())
    n_req = sum(int(value) for value in expected_counts.values())
    precision = tp / n_pred if n_pred else (1.0 if n_req == 0 else 0.0)
    recall = tp / n_req if n_req else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = all(
        int(predicted_counts.get(name, 0)) == int(expected_counts.get(name, 0))
        for name in expected_counts
    )
    return {
        "manual_count_precision": float(precision),
        "manual_count_recall": float(recall),
        "manual_count_f1": float(f1),
        "count_precision": float(precision),
        "count_recall": float(recall),
        "count_f1": float(f1),
        "pla_count": float(f1),
        "exact_count_match": float(exact),
        "n_req": float(n_req),
        "manual_count_error": float(n_pred - n_req),
        "manual_abs_count_error": float(abs(n_pred - n_req)),
        "manual_undercount": float(n_pred < n_req),
        "manual_severe_undercount": float(n_req >= 2 and n_pred <= n_req - 2),
    }


def _position_scores(
    classes: np.ndarray,
    boxes: np.ndarray,
    expected_positions: dict[str, list[str]],
    id_to_name: dict[int, str],
    class_names: list[str],
) -> dict[str, float]:
    predicted_positions: dict[str, list[str]] = {name: [] for name in class_names}
    for class_id, box in zip(classes.tolist(), boxes.tolist()):
        class_id = int(class_id)
        class_name = id_to_name.get(class_id)
        if class_name is None:
            continue
        x = float((box[0] + box[2]) / 2)
        y = float((box[1] + box[3]) / 2)
        predicted_positions[class_name].append(_cell(x, y))

    matched = requested = 0
    exact = True
    result: dict[str, float] = {}
    for class_name in class_names:
        expected_counter = Counter(expected_positions.get(class_name, []))
        predicted_counter = Counter(predicted_positions.get(class_name, []))
        class_requested = sum(expected_counter.values())
        class_matched = sum(
            min(count, predicted_counter.get(cell, 0))
            for cell, count in expected_counter.items()
        )
        if expected_counter != predicted_counter:
            exact = False
        requested += class_requested
        matched += class_matched
        key = class_name.lower()
        result[f"spla_{key}_matched"] = float(class_matched)
        result[f"spla_{key}_requested"] = float(class_requested)

    result.update(
        {
            "spla_matched": float(matched),
            "spla_requested": float(requested),
            "spla": float(matched / requested) if requested else float("nan"),
            "manual_position_exact_match": float(exact),
        }
    )
    return result


def _prediction_counts(classes: np.ndarray, id_to_name: dict[int, str], class_names: list[str]) -> dict[str, int]:
    counter = Counter(int(value) for value in classes.tolist() if int(value) > 0)
    return {
        name: int(counter.get(class_id, 0))
        for class_id, name in sorted(id_to_name.items())
        if name in class_names
    }


def _manual_records(
    audit: pd.DataFrame,
    prediction: dict[str, Any],
    existing: pd.DataFrame,
    id_to_name: dict[int, str],
    class_names: list[str],
    *,
    method: str,
    seed: int | None,
) -> pd.DataFrame:
    existing_by_image = existing.set_index("image")
    records: list[dict[str, Any]] = []
    for row in audit.to_dict("records"):
        image = row["image"]
        if image not in prediction["index"]:
            raise ValueError(f"{prediction['path']} has no image {image}")
        if image not in existing_by_image.index:
            raise ValueError(f"Existing per-image metrics for {method} have no image {image}")
        index = prediction["index"][image]
        classes = prediction["classes"][index]
        boxes = prediction["boxes"][index]
        predicted_counts = _prediction_counts(classes, id_to_name, class_names)
        record = dict(existing_by_image.loc[image].to_dict())
        record.update(
            {
                "image": image,
                "method": method,
                "train_seed": int(seed) if seed is not None else "",
                "audit_status": row["audit_status"],
                "is_retained": float(bool(row["is_retained"])),
                "parser_correct": float(bool(row["parser_correct"])),
            }
        )
        record.update(_count_scores(predicted_counts, row["manual_counts"]))
        record.update(
            _position_scores(
                classes,
                boxes,
                row["manual_positions"],
                id_to_name,
                class_names,
            )
        )
        records.append(record)
    return pd.DataFrame(records)


def _finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else float("nan")


def _weighted_ratio(frame: pd.DataFrame, matched: str, requested: str) -> float:
    m = pd.to_numeric(frame[matched], errors="coerce")
    r = pd.to_numeric(frame[requested], errors="coerce")
    valid = np.isfinite(m) & np.isfinite(r) & (r > 0)
    if not valid.any():
        return float("nan")
    return float(m[valid].sum() / r[valid].sum())


def _aggregate_one(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    metric_names = [
        "val",
        "oob",
        "occ",
        "rea",
        "empty_layout",
        "sma",
        "manual_undercount",
        "manual_severe_undercount",
        "n_req",
        "n_pred",
        "manual_count_error",
        "manual_abs_count_error",
        "total_pred_area",
        "mean_pred_area",
        "median_pred_area",
        "exact_count_match",
        "pla_count",
        "manual_count_precision",
        "manual_count_recall",
        "manual_count_f1",
        "manual_position_exact_match",
        "type_f1",
    ]
    for metric in metric_names:
        if metric in frame.columns:
            result[metric] = _finite_mean(frame[metric])
    if {"spla_matched", "spla_requested"} <= set(frame.columns):
        result["spla"] = _weighted_ratio(frame, "spla_matched", "spla_requested")
        result["spla_requests"] = float(pd.to_numeric(frame["spla_requested"], errors="coerce").sum())

    count_matched = frame[pd.to_numeric(frame["exact_count_match"], errors="coerce") == 1.0]
    result["count_matched_n"] = float(len(count_matched))
    if len(count_matched):
        result["occ_count_matched"] = _finite_mean(count_matched["occ"])
        result["rea_count_matched"] = _finite_mean(count_matched["rea"])
    else:
        result["occ_count_matched"] = float("nan")
        result["rea_count_matched"] = float("nan")
    return result


def _aggregate_seeds(seed_frames: list[pd.DataFrame]) -> dict[str, dict[str, Any]]:
    per_seed = [_aggregate_one(frame) for frame in seed_frames]
    metrics = sorted({metric for item in per_seed for metric in item})
    result: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        values = [float(item[metric]) for item in per_seed if metric in item and math.isfinite(float(item[metric]))]
        if values:
            result[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return result


def _format_metric(value: dict[str, Any] | float | None) -> str:
    if value is None:
        return "--"
    if isinstance(value, dict):
        return f"{value['mean']:.3f}±{value['std']:.3f}"
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.3f}"
    return "--"


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = [
        "n_req",
        "n_pred",
        "empty_layout",
        "sma",
        "manual_severe_undercount",
        "manual_count_error",
        "manual_abs_count_error",
        "total_pred_area",
        "mean_pred_area",
        "median_pred_area",
        "occ",
        "rea",
        "occ_count_matched",
        "rea_count_matched",
        "exact_count_match",
        "pla_count",
        "spla",
        "type_f1",
    ]
    lines = [
        "| subset | method | " + " | ".join(metrics) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(metrics)) + " |",
    ]
    for subset, subset_data in summary["subsets"].items():
        for method, item in subset_data["methods"].items():
            metric_values = item["metrics"]
            cells = [subset, method]
            for metric in metrics:
                cells.append(_format_metric(metric_values.get(metric)))
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_summary_tex(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Subset & Method & $n_{pred}$ & Count err. & Small & Severe under & Area & Occ & Exact & PLA/SPLA \\\\",
        "\\midrule",
    ]
    for subset, subset_data in summary["subsets"].items():
        label = "All 120" if subset == "all" else "Retained"
        for method, item in subset_data["methods"].items():
            metrics = item["metrics"]
            cells = [
                label,
                "IntentDiT" if method == "intentdit" else "External reference",
                _format_metric(metrics.get("n_pred")),
                _format_metric(metrics.get("manual_count_error")),
                _format_metric(metrics.get("sma")),
                _format_metric(metrics.get("manual_severe_undercount")),
                _format_metric(metrics.get("total_pred_area")),
                _format_metric(metrics.get("occ")),
                _format_metric(metrics.get("exact_count_match")),
                f"{_format_metric(metrics.get('pla_count'))}/{_format_metric(metrics.get('spla'))}",
            ]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--metric-dir", default="experiments/paper_figures")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-per-image", default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--inference-seed", type=int, default=1)
    parser.add_argument("--output-dir", default="experiments/paper_results")
    args = parser.parse_args()

    num_class = 5 if args.dataset == "cgl" else 4
    name_to_id = get_class_map(num_class)
    class_names = [name for name, _ in sorted(name_to_id.items(), key=lambda item: item[1])]
    id_to_name = {class_id: name for name, class_id in name_to_id.items()}

    audit = _load_audit(Path(args.audit_csv), class_names)
    metric_dir = Path(args.metric_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_per_image = (
        Path(args.baseline_per_image)
        if args.baseline_per_image
        else metric_dir / f"baseline_text_{args.dataset}_freeform_per_image.csv"
    )
    baseline_frame = _manual_records(
        audit,
        _load_prediction(Path(args.baseline_predictions)),
        _load_existing_per_image(baseline_per_image),
        id_to_name,
        class_names,
        method="external_text_baseline",
        seed=None,
    )

    intent_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        prefix = f"ivc_prompt_{args.dataset}_vit_both_text_freeform_trainseed{seed}_inferseed{args.inference_seed}"
        prediction_path = metric_dir / f"{prefix}_test_output.pt"
        per_image_path = metric_dir / f"{prefix}_per_image.csv"
        intent_frames.append(
            _manual_records(
                audit,
                _load_prediction(prediction_path),
                _load_existing_per_image(per_image_path),
                id_to_name,
                class_names,
                method="intentdit",
                seed=seed,
            )
        )

    def write_subset(name: str, mask: pd.Series) -> dict[str, Any]:
        subset_dir = output_dir
        base_subset = baseline_frame[mask.to_numpy()].copy()
        base_path = subset_dir / f"manual_freeform_{args.dataset}_external_text_baseline_{name}_per_image.csv"
        base_subset.to_csv(base_path, index=False)
        intent_paths = []
        intent_subsets = []
        for seed, frame in zip(args.seeds, intent_frames):
            subset = frame[mask.to_numpy()].copy()
            path = subset_dir / f"manual_freeform_{args.dataset}_intentdit_seed{seed}_{name}_per_image.csv"
            subset.to_csv(path, index=False)
            intent_paths.append(str(path))
            intent_subsets.append(subset)
        return {
            "n": int(mask.sum()),
            "baseline_per_image": str(base_path),
            "intentdit_per_image": intent_paths,
            "methods": {
                "external_text_baseline": {
                    "metrics": _aggregate_one(base_subset),
                },
                "intentdit": {
                    "training_seeds": args.seeds,
                    "metrics": _aggregate_seeds(intent_subsets),
                },
            },
        }

    all_mask = pd.Series([True] * len(audit))
    retained_mask = audit["is_retained"].astype(bool)
    summary = {
        "dataset": args.dataset,
        "audit_csv": str(Path(args.audit_csv)),
        "class_names": class_names,
        "n_total": int(len(audit)),
        "n_retained": int(retained_mask.sum()),
        "n_excluded": int((~retained_mask).sum()),
        "status_counts": {
            key: int(value)
            for key, value in audit["audit_status"].value_counts().sort_index().items()
        },
        "subsets": {
            "all": write_subset("all", all_mask),
            "retained": write_subset("retained", retained_mask),
        },
        "status_strata": {},
    }
    for status in ("accepted", "corrected", "exclude"):
        mask = audit["audit_status"].eq(status)
        if not mask.any():
            continue
        summary["status_strata"][status] = {
            "n": int(mask.sum()),
            "external_text_baseline": _aggregate_one(baseline_frame[mask.to_numpy()]),
            "intentdit": _aggregate_seeds([frame[mask.to_numpy()] for frame in intent_frames]),
        }

    output_json = output_dir / f"manual_freeform_{args.dataset}_summary.json"
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary_markdown(summary, output_json.with_suffix(".md"))
    _write_summary_tex(summary, output_json.with_suffix(".tex"))
    print(f"Wrote manual free-form reference summary to {output_json}")


if __name__ == "__main__":
    main()
