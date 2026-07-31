"""Report quality by element count and saliency complexity strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(group: pd.DataFrame) -> dict:
    excluded = {"n_gt", "n_pred", "n_matched", "max_iou_covered", "saliency_mass"}
    numeric = [column for column in group.select_dtypes(include="number") if column not in excluded]
    return {
        "n": len(group),
        "metrics": {
            column: float(group[column].dropna().mean())
            for column in numeric
            if len(group[column].dropna())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.per_image)
    if "n_gt" not in frame or "saliency_mass" not in frame:
        raise ValueError("Per-image CSV needs n_gt and saliency_mass")
    frame["element_stratum"] = pd.cut(
        frame.n_gt, bins=[-1, 3, 6, float("inf")], labels=["1-3", "4-6", "7+"]
    )
    try:
        frame["saliency_stratum"] = pd.qcut(
            frame.saliency_mass, q=3, labels=["low", "medium", "high"], duplicates="drop"
        )
    except ValueError:
        frame["saliency_stratum"] = "single"
    result = {
        "element_count": {
            str(name): summarize(group) for name, group in frame.groupby("element_stratum", observed=True)
        },
        "saliency": {
            str(name): summarize(group) for name, group in frame.groupby("saliency_stratum", observed=True)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote metric strata to {output}")


if __name__ == "__main__":
    main()
