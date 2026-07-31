"""Analyze free-form prompt parser coverage and parsed/unparsed performance.

The paper reports strong adherence on canonical Spatial prompts but much lower
SPLA on independently authored free-form prompts.  This script makes that gap
auditable by separating:

* prompts with broad spatial language,
* prompts successfully parsed by the current text-spatial grammar, and
* prompts with spatial language that the parser does not support.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgbdm.text_spatial import parse_positions_from_prompt
from utils.metric import _parse_prompt_counts


BROAD_SPATIAL_PATTERN = re.compile(
    r"\b("
    r"top|bottom|left|right|center|centre|middle|upper|lower|corner|side|"
    r"above|below|under|over|near|around|between|beside|next\s+to|"
    r"foreground|background"
    r")\b",
    re.IGNORECASE,
)


METRIC_COLUMNS = [
    "count_f1",
    "pla_count",
    "spla",
    "spla_requested",
    "spla_matched",
    "type_f1",
    "occ",
    "rea",
    "oob",
    "n_pred",
    "n_gt",
    "empty_layout",
]


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _metric_summary(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in METRIC_COLUMNS:
        if column in frame.columns:
            value = _safe_mean(frame[column])
            if math.isfinite(value):
                result[column] = value
    if {"spla_matched", "spla_requested"} <= set(frame.columns):
        requested = pd.to_numeric(frame["spla_requested"], errors="coerce").fillna(0.0).sum()
        matched = pd.to_numeric(frame["spla_matched"], errors="coerce").fillna(0.0).sum()
        if requested > 0:
            result["spla"] = float(matched / requested)
    if {"relation_matched", "relation_evaluable"} <= set(frame.columns):
        requested = pd.to_numeric(frame["relation_evaluable"], errors="coerce").fillna(0.0).sum()
        matched = pd.to_numeric(frame["relation_matched"], errors="coerce").fillna(0.0).sum()
        if requested > 0:
            result["relation_satisfaction"] = float(matched / requested)
    return result


def load_prompts(path: Path) -> pd.DataFrame:
    prompts = pd.read_csv(path, keep_default_na=False)
    prompt_col = "text_prompt" if "text_prompt" in prompts.columns else "prompt"
    required = {"poster_path", prompt_col}
    if not required <= set(prompts.columns):
        raise ValueError(f"{path} must contain poster_path and {prompt_col}")
    frame = prompts[["poster_path", prompt_col]].copy()
    frame.rename(columns={prompt_col: "text_prompt"}, inplace=True)
    frame["image"] = frame.poster_path.astype(str).map(lambda value: Path(value).name)
    frame["text_prompt"] = frame.text_prompt.astype(str)
    return frame


def annotate_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in prompts.iterrows():
        prompt = str(row.text_prompt)
        parsed = parse_positions_from_prompt(prompt)
        counts = _parse_prompt_counts(prompt)
        spatial_candidate = bool(BROAD_SPATIAL_PATTERN.search(prompt))
        parsed_total = sum(len(values) for values in parsed.values())
        rows.append(
            {
                "image": row.image,
                "poster_path": row.poster_path,
                "text_prompt": prompt,
                "count_parser_total": int(sum(counts.values())),
                "count_parser_success": bool(sum(counts.values()) > 0),
                "spatial_candidate": spatial_candidate,
                "spatial_parser_success": bool(parsed_total > 0),
                "parsed_spatial_assignments": int(parsed_total),
                "parsed_spatial_classes": sorted(parsed.keys()),
                "parser_group": (
                    "parsed_spatial"
                    if parsed_total > 0
                    else "candidate_unparsed"
                    if spatial_candidate
                    else "no_spatial_candidate"
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_prompt_coverage(annotated: pd.DataFrame) -> dict[str, Any]:
    n = len(annotated)
    candidates = annotated[annotated.spatial_candidate]
    parsed = annotated[annotated.spatial_parser_success]
    candidate_unparsed = annotated[
        annotated.spatial_candidate & ~annotated.spatial_parser_success
    ]
    return {
        "n": int(n),
        "unique_images": int(annotated.image.nunique()),
        "count_parser_coverage": _safe_rate(annotated.count_parser_success),
        "spatial_candidate_rate": _safe_rate(annotated.spatial_candidate),
        "spatial_parser_coverage_all": _safe_rate(annotated.spatial_parser_success),
        "spatial_parser_coverage_candidates": (
            _safe_rate(candidates.spatial_parser_success) if len(candidates) else float("nan")
        ),
        "spatial_parser_failure_rate_candidates": (
            float(len(candidate_unparsed) / len(candidates)) if len(candidates) else float("nan")
        ),
        "n_spatial_candidates": int(len(candidates)),
        "n_spatial_parsed": int(len(parsed)),
        "n_candidate_unparsed": int(len(candidate_unparsed)),
        "unsupported_spatial_examples": candidate_unparsed[
            ["poster_path", "text_prompt"]
        ].head(10).to_dict("records"),
    }


def summarize_metrics(annotated: pd.DataFrame, label: str, per_image_path: Path) -> dict[str, Any]:
    metrics = pd.read_csv(per_image_path)
    if "image" not in metrics.columns:
        raise ValueError(f"{per_image_path} must contain an image column")
    merged = metrics.merge(
        annotated[["image", "parser_group", "spatial_candidate", "spatial_parser_success"]],
        on="image",
        how="inner",
    )
    result: dict[str, Any] = {"label": label, "path": str(per_image_path), "n": int(len(merged)), "groups": {}}
    for group_name, group in merged.groupby("parser_group"):
        group_result = {"n": int(len(group)), "metrics": _metric_summary(group)}
        result["groups"][str(group_name)] = group_result
    all_result = {"n": int(len(merged)), "metrics": _metric_summary(merged)}
    result["groups"]["all"] = all_result
    return result


def write_tex(summary: dict[str, Any], output: Path) -> None:
    coverage = summary["coverage"]
    rows = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Dataset & Prompts & Spatial candidates & Parsed spatial & Candidate failure & Count parsed \\\\",
        "\\midrule",
        f"{summary['dataset'].upper()} & {coverage['n']} & "
        f"{coverage['n_spatial_candidates']} ({coverage['spatial_candidate_rate']:.3f}) & "
        f"{coverage['n_spatial_parsed']} ({coverage['spatial_parser_coverage_all']:.3f}) & "
        f"{coverage['spatial_parser_failure_rate_candidates']:.3f} & "
        f"{coverage['count_parser_coverage']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]

    if summary.get("metric_files"):
        rows.extend(
            [
                "",
                "\\begin{tabular}{llccccc}",
                "\\toprule",
                "Run & Group & $N$ & PLA/count-overlap & SPLA & Type-F1 & Occ \\\\",
                "\\midrule",
            ]
        )
        for metric_result in summary["metric_files"]:
            for group_name in ["parsed_spatial", "candidate_unparsed", "no_spatial_candidate", "all"]:
                group = metric_result["groups"].get(group_name)
                if not group:
                    continue
                metrics = group["metrics"]
                rows.append(
                    f"{metric_result['label']} & {group_name.replace('_', ' ')} & {group['n']} & "
                    f"{metrics.get('pla_count', float('nan')):.3f} & "
                    f"{metrics.get('spla', float('nan')):.3f} & "
                    f"{metrics.get('type_f1', float('nan')):.3f} & "
                    f"{metrics.get('occ', float('nan')):.3f} \\\\"
                )
        rows.extend(["\\bottomrule", "\\end{tabular}"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument(
        "--per-image",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        default=[],
        help="Optional per-image metric CSV to summarize by parser group.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", default="")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    annotated = annotate_prompts(prompts)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "prompts": str(args.prompts),
        "coverage": summarize_prompt_coverage(annotated),
        "parser_groups": annotated.parser_group.value_counts().to_dict(),
        "metric_files": [],
    }
    for label, path in args.per_image:
        summary["metric_files"].append(summarize_metrics(annotated, label, Path(path)))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_tex:
        write_tex(summary, Path(args.output_tex))
    print(f"Wrote free-form parser coverage for {args.dataset} to {output_json}")


if __name__ == "__main__":
    main()
