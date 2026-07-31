"""Generate protocol-separated CSV/LaTeX tables from archived summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("val", "oob", "sma", "ali", "ove", "vb", "uti", "occ", "rea", "undl", "unds", "paired_iou", "max_iou", "type_f1", "pla_count", "spla")


def is_prompted(name: str) -> bool:
    return any(token in name for token in ("_text", "lambda_", "pooled_text", "pixel_map_only_text", "intent_boxes_only_text"))


def format_value(item: dict | None) -> str:
    if not item:
        return "--"
    return f"{item['mean']:.4f} $\\pm$ {item['std']:.4f}"


def write_table(name: str, rows: list[tuple[str, dict]], output_dir: Path) -> None:
    active_metrics = [metric for metric in METRICS if any(metric in row[1]["metrics"] for row in rows)]
    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "training_seeds", *active_metrics])
        for experiment, result in rows:
            writer.writerow([
                experiment,
                ",".join(map(str, result["training_seeds"])),
                *[format_value(result["metrics"].get(metric)).replace("$\\pm$", "±") for metric in active_metrics],
            ])
    latex_path = output_dir / f"{name}.tex"
    column_spec = "l" + "c" * len(active_metrics)
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        "Experiment & " + " & ".join(metric.replace("_", "\\_") for metric in active_metrics) + " \\\\",
        "\\midrule",
    ]
    for experiment, result in rows:
        label = experiment.replace("_", "\\_")
        values = [format_value(result["metrics"].get(metric)) for metric in active_metrics]
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    latex_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_rows = [(name, result) for name, result in summary.items() if not is_prompted(name)]
    prompt_rows = [(name, result) for name, result in summary.items() if is_prompted(name)]
    write_table("image_only_results", image_rows, output_dir)
    write_table("oracle_prompt_results", prompt_rows, output_dir)
    print(f"Wrote protocol-separated result tables to {output_dir}")


if __name__ == "__main__":
    main()
