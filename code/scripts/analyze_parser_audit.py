"""Analyze author-reviewed parser-audit CSV files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


AUDITED_STATUSES = {"accepted", "corrected"}
EXCLUDED_STATUSES = {"exclude", "excluded"}


def _load_json_dict(value: Any, *, row_id: str, column: str) -> dict:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_id}: invalid JSON in {column}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{row_id}: {column} must be a JSON object")
    return parsed


def _normal_counts(value: dict) -> dict[str, int]:
    return {str(key): int(val) for key, val in value.items()}


def _normal_positions(value: dict) -> dict[str, list[str]]:
    result = {}
    for key, vals in value.items():
        if vals is None:
            result[str(key)] = []
        elif isinstance(vals, list):
            result[str(key)] = [str(item) for item in vals]
        else:
            raise ValueError(f"Position value for {key!r} must be a list")
    return result


def _position_counter(value: dict[str, list[str]]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for class_name, cells in value.items():
        for cell in cells:
            counter[(class_name, cell)] += 1
    return counter


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def analyze_frame(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    required = {
        "poster_path",
        "parser_counts_json",
        "parser_positions_json",
        "manual_counts_json",
        "manual_positions_json",
        "audit_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} audit CSV missing columns: {sorted(missing)}")

    audited = []
    excluded = 0
    pending = 0
    for index, row in frame.iterrows():
        status = str(row["audit_status"]).strip().lower()
        if status in AUDITED_STATUSES:
            audited.append(row)
        elif status in EXCLUDED_STATUSES:
            excluded += 1
        else:
            pending += 1

    count_exact = 0
    position_exact = 0
    count_abs_error = 0
    count_items = 0
    position_tp = position_fp = position_fn = 0
    per_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []

    for row in audited:
        row_id = str(row["poster_path"])
        parser_counts = _normal_counts(
            _load_json_dict(row["parser_counts_json"], row_id=row_id, column="parser_counts_json")
        )
        manual_counts = _normal_counts(
            _load_json_dict(row["manual_counts_json"], row_id=row_id, column="manual_counts_json")
        )
        parser_positions = _normal_positions(
            _load_json_dict(row["parser_positions_json"], row_id=row_id, column="parser_positions_json")
        )
        manual_positions = _normal_positions(
            _load_json_dict(row["manual_positions_json"], row_id=row_id, column="manual_positions_json")
        )

        class_names = sorted(set(parser_counts) | set(manual_counts))
        is_count_exact = all(parser_counts.get(name, 0) == manual_counts.get(name, 0) for name in class_names)
        count_exact += int(is_count_exact)
        for name in class_names:
            count_abs_error += abs(parser_counts.get(name, 0) - manual_counts.get(name, 0))
            count_items += 1

        parser_counter = _position_counter(parser_positions)
        manual_counter = _position_counter(manual_positions)
        pos_keys = set(parser_counter) | set(manual_counter)
        is_position_exact = all(parser_counter.get(key, 0) == manual_counter.get(key, 0) for key in pos_keys)
        position_exact += int(is_position_exact)
        for key in pos_keys:
            tp = min(parser_counter.get(key, 0), manual_counter.get(key, 0))
            fp = max(0, parser_counter.get(key, 0) - manual_counter.get(key, 0))
            fn = max(0, manual_counter.get(key, 0) - parser_counter.get(key, 0))
            position_tp += tp
            position_fp += fp
            position_fn += fn
            class_name = key[0]
            per_class_counts[class_name]["tp"] += tp
            per_class_counts[class_name]["fp"] += fp
            per_class_counts[class_name]["fn"] += fn

        if (not is_count_exact or not is_position_exact) and len(examples) < 10:
            examples.append(
                {
                    "poster_path": row_id,
                    "text_prompt": row.get("text_prompt", ""),
                    "parser_counts": parser_counts,
                    "manual_counts": manual_counts,
                    "parser_positions": parser_positions,
                    "manual_positions": manual_positions,
                    "notes": row.get("notes", ""),
                }
            )

    n = len(audited)
    precision = _safe_div(position_tp, position_tp + position_fp)
    recall = _safe_div(position_tp, position_tp + position_fn)
    per_class = {}
    for class_name, counts in sorted(per_class_counts.items()):
        p = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        r = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        per_class[class_name] = {
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
        }
    return {
        "label": label,
        "n_total": int(len(frame)),
        "n_audited": int(n),
        "n_excluded": int(excluded),
        "n_pending": int(pending),
        "count_exact_accuracy": _safe_div(count_exact, n),
        "mean_abs_count_error_per_class": _safe_div(count_abs_error, count_items),
        "position_exact_accuracy": _safe_div(position_exact, n),
        "position_precision": precision,
        "position_recall": recall,
        "position_f1": _f1(precision, recall),
        "position_tp": int(position_tp),
        "position_fp": int(position_fp),
        "position_fn": int(position_fn),
        "per_class_position": per_class,
        "error_examples": examples,
    }


def write_tex(results: list[dict[str, Any]], output: Path) -> None:
    rows = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Dataset & Audited & Count exact & Pos. exact & Pos. precision & Pos. recall & Pos. F1 \\\\",
        "\\midrule",
    ]
    for result in results:
        rows.append(
            f"{result['label'].upper()} & {result['n_audited']}/{result['n_total']} & "
            f"{result['count_exact_accuracy']:.3f} & "
            f"{result['position_exact_accuracy']:.3f} & "
            f"{result['position_precision']:.3f} & "
            f"{result['position_recall']:.3f} & "
            f"{result['position_f1']:.3f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        required=True,
        help="Dataset label and reviewed audit CSV.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tex", required=True)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any row is still needs_review.",
    )
    args = parser.parse_args()

    results = []
    for label, csv_path in args.audit:
        result = analyze_frame(pd.read_csv(csv_path, keep_default_na=False), label)
        if args.require_complete and result["n_pending"]:
            raise SystemExit(
                f"{label}: {result['n_pending']} rows are still pending parser audit"
            )
        if result["n_audited"] == 0:
            raise SystemExit(f"{label}: no audited rows found")
        results.append(result)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": results}, indent=2) + "\n")
    write_tex(results, Path(args.output_tex))
    print(f"Wrote parser audit summary to {output_json} and {args.output_tex}")


if __name__ == "__main__":
    main()
