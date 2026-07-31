"""Build an auditable free-form prompt evaluation manifest.

The manuscript uses independently authored free-form prompts.  This script
freezes the exact prompt, parser interpretation, baseline output, IntentDiT
seed outputs, and per-image metrics into JSONL so the reported free-form
numbers can be traced sample-by-sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent

from cgbdm.text_spatial import get_class_map, parse_positions_from_prompt
from utils.benchmark_metrics import decode_layout_tensor
from utils.metric import _parse_prompt_counts


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit or "unknown",
        "dirty": bool(status),
        "status_short": status or "",
    }


def _load_prompts(path: Path) -> pd.DataFrame:
    prompts = pd.read_csv(path, keep_default_na=False)
    prompt_col = "text_prompt" if "text_prompt" in prompts.columns else "prompt"
    required = {"poster_path", prompt_col}
    missing = required - set(prompts.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    prompts = prompts.copy()
    prompts.rename(columns={prompt_col: "text_prompt"}, inplace=True)
    prompts["image"] = prompts["poster_path"].astype(str).map(lambda value: Path(value).name)
    prompts["text_prompt"] = prompts["text_prompt"].astype(str)
    if prompts["image"].duplicated().any():
        duplicated = sorted(prompts.loc[prompts["image"].duplicated(), "image"].unique())
        raise ValueError(f"{path} contains duplicate images: {duplicated[:10]}")
    return prompts


def _load_prediction_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a dictionary with img_names/test_output")
    if "img_names" not in payload or "test_output" not in payload:
        raise ValueError(f"{path} must contain img_names and test_output")
    names = [Path(str(name)).name for name in payload["img_names"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path} contains duplicate image names")
    classes, raw_boxes, boxes = decode_layout_tensor(payload["test_output"])
    if len(names) != classes.shape[0]:
        raise ValueError(f"{path} image-name count does not match tensor batch")
    return {
        "path": str(path),
        "names": names,
        "classes": classes,
        "raw_boxes": raw_boxes,
        "boxes": boxes,
        "metadata": {
            key: _jsonable(value)
            for key, value in payload.items()
            if key not in {"test_output", "img_names"}
        },
    }


def _load_per_image(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    if "image" not in frame.columns:
        raise ValueError(f"{path} must contain an image column")
    frame = frame.copy()
    frame["image"] = frame["image"].astype(str).map(lambda value: Path(value).name)
    if frame["image"].duplicated().any():
        duplicated = sorted(frame.loc[frame["image"].duplicated(), "image"].unique())
        raise ValueError(f"{path} contains duplicate image metrics: {duplicated[:10]}")
    records: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        image = str(record.pop("image"))
        records[image] = _jsonable(record)
    return records


def _dataset_class_maps(dataset: str) -> tuple[dict[str, int], dict[int, str]]:
    num_class = 5 if dataset == "cgl" else 4
    name_to_id = get_class_map(num_class)
    id_to_name = {value: key for key, value in name_to_id.items()}
    return name_to_id, id_to_name


def _prediction_counts(classes: np.ndarray, id_to_name: dict[int, str]) -> dict[str, int]:
    counter = Counter(int(value) for value in classes.tolist() if int(value) > 0)
    return {
        name: int(counter.get(class_id, 0))
        for class_id, name in sorted(id_to_name.items())
    }


def _elements(
    classes: np.ndarray,
    raw_boxes: np.ndarray,
    boxes: np.ndarray,
    id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for class_id, raw_box, box in zip(classes.tolist(), raw_boxes.tolist(), boxes.tolist()):
        class_id = int(class_id)
        if class_id <= 0:
            continue
        items.append(
            {
                "class_id": class_id,
                "class_name": id_to_name.get(class_id, f"Class{class_id}"),
                "box_xyxy": [round(float(value), 6) for value in box],
                "box_xyxy_unclamped": [round(float(value), 6) for value in raw_box],
            }
        )
    return items


def _expected_counts(prompt: str, class_names: list[str]) -> dict[str, int]:
    parsed = _parse_prompt_counts(prompt)
    return {name: int(parsed.get(name, 0)) for name in class_names}


def _exact_count_match(expected: dict[str, int], predicted: dict[str, int]) -> bool:
    return all(int(expected.get(name, 0)) == int(predicted.get(name, 0)) for name in expected)


def _method_record(
    image: str,
    payload: dict[str, Any],
    per_image: dict[str, dict[str, Any]],
    expected: dict[str, int],
    id_to_name: dict[int, str],
    *,
    label: str,
    seed: int | None = None,
) -> dict[str, Any]:
    if image not in payload["names"]:
        raise ValueError(f"{payload['path']} has no prediction for {image}")
    if image not in per_image:
        raise ValueError(f"Per-image metrics for {label} have no row for {image}")
    index = payload["names"].index(image)
    classes = payload["classes"][index]
    predicted_counts = _prediction_counts(classes, id_to_name)
    record = {
        "label": label,
        "source_predictions": payload["path"],
        "per_image_metrics": per_image[image],
        "predicted_counts": predicted_counts,
        "exact_count_match": _exact_count_match(expected, predicted_counts),
        "elements": _elements(
            classes,
            payload["raw_boxes"][index],
            payload["boxes"][index],
            id_to_name,
        ),
    }
    if seed is not None:
        record["seed"] = int(seed)
    if payload.get("metadata"):
        record["metadata"] = payload["metadata"]
    return record


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _reported_prompt_aggregates(method_records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Reproduce aggregate prompt metrics that are weighted by request counts.

    Most reported metrics are simple per-image means.  SPLA and relation-style
    prompt scores are exceptions: the evaluator aggregates matched/requested
    counts globally.  The manifest keeps both views so the per-sample audit and
    manuscript tables remain traceable.
    """
    per_image_mean: dict[str, float | None] = {}
    metric_keys = sorted(
        {
            metric
            for method in method_records
            for metric in method.get("per_image_metrics", {}).keys()
        }
    )
    for metric in metric_keys:
        values = []
        for method in method_records:
            value = _finite_number(method.get("per_image_metrics", {}).get(metric))
            if value is not None:
                values.append(value)
        if values:
            per_image_mean[metric] = _mean(values)

    reported = dict(per_image_mean)
    requested = sum(
        _finite_number(method.get("per_image_metrics", {}).get("spla_requested")) or 0.0
        for method in method_records
    )
    matched = sum(
        _finite_number(method.get("per_image_metrics", {}).get("spla_matched")) or 0.0
        for method in method_records
    )
    if requested > 0:
        reported["spla"] = matched / requested

    class_prefixes = sorted(
        key[: -len("_requested")]
        for key in metric_keys
        if key.startswith("spla_") and key.endswith("_requested")
    )
    for prefix in class_prefixes:
        requested = sum(
            _finite_number(method.get("per_image_metrics", {}).get(f"{prefix}_requested")) or 0.0
            for method in method_records
        )
        matched = sum(
            _finite_number(method.get("per_image_metrics", {}).get(f"{prefix}_matched")) or 0.0
            for method in method_records
        )
        if requested > 0:
            reported[prefix] = matched / requested

    relation_requested = sum(
        _finite_number(method.get("per_image_metrics", {}).get("relation_evaluable")) or 0.0
        for method in method_records
    )
    relation_matched = sum(
        _finite_number(method.get("per_image_metrics", {}).get("relation_matched")) or 0.0
        for method in method_records
    )
    if relation_requested > 0:
        reported["relation_satisfaction"] = relation_matched / relation_requested

    hierarchy_requested = sum(
        _finite_number(method.get("per_image_metrics", {}).get("hierarchy_requested")) or 0.0
        for method in method_records
    )
    if hierarchy_requested > 0:
        hierarchy_score = 0.0
        for method in method_records:
            metrics = method.get("per_image_metrics", {})
            score = _finite_number(metrics.get("hierarchy_satisfaction"))
            count = _finite_number(metrics.get("hierarchy_requested")) or 0.0
            if score is not None:
                hierarchy_score += score * count
        reported["hierarchy_satisfaction"] = hierarchy_score / hierarchy_requested

    return reported


def _summarize_method(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    method_records = [record[key] for record in records]
    per_image_mean_metrics: dict[str, float | None] = {}
    metric_keys = sorted(
        {
            metric
            for method in method_records
            for metric in method.get("per_image_metrics", {}).keys()
        }
    )
    for metric in metric_keys:
        values = []
        for method in method_records:
            value = method.get("per_image_metrics", {}).get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            per_image_mean_metrics[metric] = _mean(values)
    reported_metrics = _reported_prompt_aggregates(method_records)
    return {
        "n": len(method_records),
        "exact_count_match_rate": _mean(
            [float(method["exact_count_match"]) for method in method_records]
        ),
        "mean_metrics": reported_metrics,
        "reported_aggregate_metrics": reported_metrics,
        "per_image_mean_metrics": per_image_mean_metrics,
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "values": finite,
    }


def _summarize_intent_seed_means(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = sorted(
        {
            metric
            for row in seed_rows
            for metric in row["summary"].get("mean_metrics", {}).keys()
        }
    )
    return {
        "seeds": [int(row["seed"]) for row in seed_rows],
        "exact_count_match_rate": _mean_std(
            [
                float(row["summary"].get("exact_count_match_rate"))
                for row in seed_rows
                if row["summary"].get("exact_count_match_rate") is not None
            ]
        ),
        "mean_metrics": {
            metric: _mean_std(
                [
                    float(row["summary"]["mean_metrics"][metric])
                    for row in seed_rows
                    if row["summary"].get("mean_metrics", {}).get(metric) is not None
                ]
            )
            for metric in metric_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--baseline-label", default="External text baseline")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-per-image", required=True)
    parser.add_argument(
        "--intentdit",
        action="append",
        nargs=3,
        metavar=("SEED", "PREDICTIONS", "PER_IMAGE"),
        default=[],
        help="IntentDiT seed, prediction .pt, and per-image metric CSV.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    prompts = _load_prompts(prompts_path)
    name_to_id, id_to_name = _dataset_class_maps(args.dataset)
    class_names = list(name_to_id.keys())

    baseline_payload = _load_prediction_payload(Path(args.baseline_predictions))
    baseline_metrics = _load_per_image(Path(args.baseline_per_image))
    intent_payloads = []
    for seed_str, predictions, per_image in args.intentdit:
        intent_payloads.append(
            {
                "seed": int(seed_str),
                "payload": _load_prediction_payload(Path(predictions)),
                "per_image": _load_per_image(Path(per_image)),
            }
        )

    code_files = [
        CODE_ROOT / "utils" / "metric.py",
        CODE_ROOT / "utils" / "spatial_pla.py",
        CODE_ROOT / "utils" / "benchmark_metrics.py",
        CODE_ROOT / "cgbdm" / "text_spatial.py",
    ]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "num_prompts": int(len(prompts)),
        "class_map": name_to_id,
        "prompts_file": str(prompts_path),
        "baseline_label": args.baseline_label,
        "git": _git_metadata(),
        "evaluator_files": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in code_files
            if path.is_file()
        },
        "parser_note": (
            "parsed_counts and parsed_positions are deterministic outputs of "
            "utils.metric._parse_prompt_counts and cgbdm.text_spatial.parse_positions_from_prompt"
        ),
    }

    manifest_records: list[dict[str, Any]] = []
    for _, prompt_row in prompts.iterrows():
        image = str(prompt_row["image"])
        prompt = str(prompt_row["text_prompt"])
        expected = _expected_counts(prompt, class_names)
        parsed_positions = {
            key: values
            for key, values in parse_positions_from_prompt(prompt).items()
            if key in class_names
        }
        record = {
            "dataset": args.dataset,
            "image": image,
            "poster_path": str(prompt_row["poster_path"]),
            "text_prompt": prompt,
            "author_id": _jsonable(prompt_row.get("author_id", "")),
            "independent_of_ground_truth": _jsonable(
                prompt_row.get("independent_of_ground_truth", "")
            ),
            "parsed_prompt": {
                "counts": expected,
                "positions": parsed_positions,
                "num_count_elements": int(sum(expected.values())),
                "num_spatial_assignments": int(
                    sum(len(values) for values in parsed_positions.values())
                ),
            },
            "baseline": _method_record(
                image,
                baseline_payload,
                baseline_metrics,
                expected,
                id_to_name,
                label=args.baseline_label,
            ),
            "intentdit": [],
        }
        for item in intent_payloads:
            record["intentdit"].append(
                _method_record(
                    image,
                    item["payload"],
                    item["per_image"],
                    expected,
                    id_to_name,
                    label="IntentDiT",
                    seed=item["seed"],
                )
            )
        manifest_records.append(_jsonable(record))

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    flattened_intent_records = []
    for seed_item in intent_payloads:
        seed = seed_item["seed"]
        seed_records = []
        for record in manifest_records:
            method = next(item for item in record["intentdit"] if item["seed"] == seed)
            seed_records.append({"intentdit": method})
        flattened_intent_records.append(
            {
                "seed": seed,
                "source_predictions": seed_item["payload"]["path"],
                "summary": _summarize_method(seed_records, "intentdit"),
            }
        )

    summary = {
        **metadata,
        "manifest_jsonl": str(output_jsonl),
        "baseline": {
            "source_predictions": baseline_payload["path"],
            "summary": _summarize_method(manifest_records, "baseline"),
        },
        "intentdit": flattened_intent_records,
        "intentdit_mean_across_seeds": _summarize_intent_seed_means(flattened_intent_records),
    }
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(manifest_records)} records to {output_jsonl}")
    print(f"Wrote summary to {output_summary}")


if __name__ == "__main__":
    main()
