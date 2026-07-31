"""Bootstrap both independent training seeds and paired test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"oob", "sma", "ali", "ove", "vb", "spacing_cv", "occ", "rea", "hfd", "format_failure"}


WEIGHTED_RATIO_METRICS = {
    "spla": ("spla_matched", "spla_requested"),
    "relation_satisfaction": ("relation_matched", "relation_evaluable"),
}


def _weighted_metric_available(metric: str, baseline: pd.DataFrame, methods: list[pd.DataFrame]) -> bool:
    if metric not in WEIGHTED_RATIO_METRICS:
        return False
    matched, requested = WEIGHTED_RATIO_METRICS[metric]
    return (
        matched in baseline.columns
        and requested in baseline.columns
        and all(matched in frame.columns and requested in frame.columns for frame in methods)
    )


def _weighted_bootstrap_metric(
    metric: str,
    baseline: pd.DataFrame,
    methods: list[pd.DataFrame],
    common_images: list[str],
    rng: np.random.Generator,
    iterations: int,
) -> dict[str, object] | None:
    matched_key, requested_key = WEIGHTED_RATIO_METRICS[metric]
    b_matched = baseline.loc[common_images, matched_key].to_numpy(dtype=float)
    b_requested = baseline.loc[common_images, requested_key].to_numpy(dtype=float)
    a_matched = np.stack(
        [frame.loc[common_images, matched_key].to_numpy(dtype=float) for frame in methods]
    )
    a_requested = np.stack(
        [frame.loc[common_images, requested_key].to_numpy(dtype=float) for frame in methods]
    )
    valid = (
        np.isfinite(b_matched)
        & np.isfinite(b_requested)
        & np.isfinite(a_matched).all(axis=0)
        & np.isfinite(a_requested).all(axis=0)
        & (b_requested > 0)
        & (a_requested > 0).all(axis=0)
    )
    a_matched, a_requested = a_matched[:, valid], a_requested[:, valid]
    b_matched, b_requested = b_matched[valid], b_requested[valid]
    if len(b_matched) == 0:
        return None

    mean_a = float(a_matched.sum() / a_requested.sum())
    mean_b = float(b_matched.sum() / b_requested.sum())
    oriented_delta = mean_a - mean_b
    if metric in LOWER_IS_BETTER:
        oriented_delta = -oriented_delta

    samples = []
    for _ in range(iterations):
        seed_indices = rng.integers(0, len(methods), len(methods))
        image_indices = rng.integers(0, len(b_matched), len(b_matched))
        sample_a_matched = a_matched[seed_indices][:, image_indices].sum()
        sample_a_requested = a_requested[seed_indices][:, image_indices].sum()
        sample_b_matched = b_matched[image_indices].sum()
        sample_b_requested = b_requested[image_indices].sum()
        if sample_a_requested <= 0 or sample_b_requested <= 0:
            continue
        delta = sample_a_matched / sample_a_requested - sample_b_matched / sample_b_requested
        samples.append(float(-delta if metric in LOWER_IS_BETTER else delta))

    sample_array = np.asarray(samples)
    return {
        "n": int(len(b_matched)),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "oriented_delta_a_b": float(oriented_delta),
        "ci95": [
            float(np.quantile(sample_array, 0.025)),
            float(np.quantile(sample_array, 0.975)),
        ],
        "aggregation": f"request_weighted:{matched_key}/{requested_key}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-a", nargs="+", required=True, help="One per-image CSV per training seed")
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--name-a", default="intentdit")
    parser.add_argument("--name-b", default="baseline")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = pd.read_csv(args.method_b).set_index("image")
    methods = [pd.read_csv(path).set_index("image") for path in args.method_a]
    common_images = sorted(set(baseline.index).intersection(*(set(frame.index) for frame in methods)))
    if not common_images:
        raise ValueError("No images shared by all seeds and the baseline")
    common_metrics = sorted(
        set(baseline.select_dtypes(include="number").columns).intersection(
            *(set(frame.select_dtypes(include="number").columns) for frame in methods)
        )
    )
    rng = np.random.default_rng(args.seed)
    result = {
        "method_a": args.name_a,
        "method_b": args.name_b,
        "training_seeds": len(methods),
        "paired_images": len(common_images),
        "iterations": args.iterations,
        "metrics": {},
    }
    for metric in common_metrics:
        if _weighted_metric_available(metric, baseline, methods):
            weighted = _weighted_bootstrap_metric(
                metric, baseline, methods, common_images, rng, args.iterations
            )
            if weighted is not None:
                weighted["higher_oriented_delta_favors"] = args.name_a
                result["metrics"][metric] = weighted
                continue

        b = baseline.loc[common_images, metric].to_numpy(dtype=float)
        a = np.stack([frame.loc[common_images, metric].to_numpy(dtype=float) for frame in methods])
        valid = np.isfinite(b) & np.isfinite(a).all(axis=0)
        a, b = a[:, valid], b[valid]
        if not len(b):
            continue
        oriented = a - b[None, :]
        if metric in LOWER_IS_BETTER:
            oriented = -oriented
        samples = []
        for _ in range(args.iterations):
            seed_indices = rng.integers(0, len(methods), len(methods))
            image_indices = rng.integers(0, len(b), len(b))
            samples.append(float(oriented[seed_indices][:, image_indices].mean()))
        sample_array = np.asarray(samples)
        result["metrics"][metric] = {
            "n": int(len(b)),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "oriented_delta_a_b": float(oriented.mean()),
            "ci95": [float(np.quantile(sample_array, 0.025)), float(np.quantile(sample_array, 0.975))],
            "higher_oriented_delta_favors": args.name_a,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote seed×image bootstrap to {output}")


if __name__ == "__main__":
    main()
