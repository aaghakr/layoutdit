"""Paired image-level bootstrap comparisons for layout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOWER_IS_BETTER = {"oob", "sma", "ali", "ove", "occ", "rea", "hfd"}

WEIGHTED_RATIO_METRICS = {
    "spla": ("spla_matched", "spla_requested"),
    "relation_satisfaction": ("relation_matched", "relation_evaluable"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-a", required=True, help="Per-image CSV")
    parser.add_argument("--method-b", required=True, help="Per-image CSV")
    parser.add_argument("--name-a", default="intentdit")
    parser.add_argument("--name-b", default="baseline")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = pd.read_csv(args.method_a)
    b = pd.read_csv(args.method_b)
    if "image" not in a or "image" not in b:
        raise ValueError("Both inputs require an image column")
    merged = a.merge(b, on="image", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) == 0:
        raise ValueError("No shared images between methods")

    common_metrics = sorted(
        column[:-2]
        for column in merged.columns
        if column.endswith("_a") and f"{column[:-2]}_b" in merged
    )
    rng = np.random.default_rng(args.seed)
    result = {
        "method_a": args.name_a,
        "method_b": args.name_b,
        "n_paired_images": len(merged),
        "iterations": args.iterations,
        "metrics": {},
    }
    for metric in common_metrics:
        if metric in WEIGHTED_RATIO_METRICS:
            matched_key, requested_key = WEIGHTED_RATIO_METRICS[metric]
            required = {
                f"{matched_key}_a",
                f"{requested_key}_a",
                f"{matched_key}_b",
                f"{requested_key}_b",
            }
            if required <= set(merged.columns):
                a_matched = pd.to_numeric(merged[f"{matched_key}_a"], errors="coerce").to_numpy()
                a_requested = pd.to_numeric(merged[f"{requested_key}_a"], errors="coerce").to_numpy()
                b_matched = pd.to_numeric(merged[f"{matched_key}_b"], errors="coerce").to_numpy()
                b_requested = pd.to_numeric(merged[f"{requested_key}_b"], errors="coerce").to_numpy()
                valid = (
                    np.isfinite(a_matched)
                    & np.isfinite(a_requested)
                    & np.isfinite(b_matched)
                    & np.isfinite(b_requested)
                    & (a_requested > 0)
                    & (b_requested > 0)
                )
                if valid.any():
                    a_matched, a_requested = a_matched[valid], a_requested[valid]
                    b_matched, b_requested = b_matched[valid], b_requested[valid]
                    mean_a = float(a_matched.sum() / a_requested.sum())
                    mean_b = float(b_matched.sum() / b_requested.sum())
                    oriented_delta = mean_a - mean_b
                    if metric in LOWER_IS_BETTER:
                        oriented_delta = -oriented_delta
                    samples = []
                    for _ in range(args.iterations):
                        indices = rng.integers(0, len(a_matched), len(a_matched))
                        delta = (
                            a_matched[indices].sum() / a_requested[indices].sum()
                            - b_matched[indices].sum() / b_requested[indices].sum()
                        )
                        samples.append(float(-delta if metric in LOWER_IS_BETTER else delta))
                    sample_array = np.asarray(samples)
                    result["metrics"][metric] = {
                        "n": int(len(a_matched)),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "oriented_delta_a_b": float(oriented_delta),
                        "ci95": [
                            float(np.quantile(sample_array, 0.025)),
                            float(np.quantile(sample_array, 0.975)),
                        ],
                        "a_win_rate": float("nan"),
                        "tie_rate": float("nan"),
                        "aggregation": f"request_weighted:{matched_key}/{requested_key}",
                        "higher_oriented_delta_favors": args.name_a,
                    }
                    continue

        av = pd.to_numeric(merged[f"{metric}_a"], errors="coerce").to_numpy()
        bv = pd.to_numeric(merged[f"{metric}_b"], errors="coerce").to_numpy()
        valid = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[valid], bv[valid]
        if len(av) == 0:
            continue
        raw_delta = av - bv
        # Positive oriented delta always favors method A.
        oriented = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
        indices = rng.integers(0, len(oriented), size=(args.iterations, len(oriented)))
        samples = oriented[indices].mean(axis=1)
        ties = np.isclose(oriented, 0.0, atol=1e-12)
        result["metrics"][metric] = {
            "n": int(len(oriented)),
            "mean_a": float(av.mean()),
            "mean_b": float(bv.mean()),
            "oriented_delta_a_b": float(oriented.mean()),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "a_win_rate": float(np.mean(oriented > 0)),
            "tie_rate": float(np.mean(ties)),
            "higher_oriented_delta_favors": args.name_a,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote paired comparison for {len(merged)} images to {output}")


if __name__ == "__main__":
    main()
