"""Aggregate per-image measurements by prompt category/conflict flag."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-image", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.per_image)
    prompts = pd.read_csv(args.prompts)
    prompts["image"] = prompts.poster_path.astype(str).map(lambda value: Path(value).name)
    columns = [column for column in ("image", "prompt_category", "is_conflicting") if column in prompts]
    merged = metrics.merge(prompts[columns].drop_duplicates("image"), on="image", how="inner")
    if "prompt_category" not in merged:
        merged["prompt_category"] = "all"
    numeric = [
        column for column in merged.select_dtypes(include="number").columns
        if column not in {"is_conflicting"}
    ]
    result = {"n": len(merged), "groups": {}}
    for category, group in merged.groupby("prompt_category"):
        result["groups"][str(category)] = {
            "n": len(group),
            "metrics": {
                column: float(group[column].dropna().mean())
                for column in numeric
                if len(group[column].dropna())
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['groups'])} prompt subgroups to {output}")


if __name__ == "__main__":
    main()
