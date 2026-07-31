"""Create paired prompts differing by one requested Text element."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


NAMES = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}


def prompt_from_counts(counts: Counter) -> str:
    fragments = [f"{counts[class_id]} {NAMES[class_id]}" for class_id in sorted(counts) if class_id in NAMES]
    return "Create a poster layout with " + ", ".join(fragments) + "."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--edited-output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations)
    base_rows, edited_rows = [], []
    for poster_path, group in frame.groupby("poster_path", sort=True):
        counts = Counter(int(value) for value in group.cls_elem if int(value) > 0)
        base_rows.append({"poster_path": poster_path, "text_prompt": prompt_from_counts(counts), "target_class": 1, "target_delta": 1})
        edited = counts.copy()
        edited[1] += 1
        edited_rows.append({"poster_path": poster_path, "text_prompt": prompt_from_counts(edited), "target_class": 1, "target_delta": 1})
        if len(base_rows) >= args.limit:
            break
    for path, rows in ((args.base_output, base_rows), (args.edited_output, edited_rows)):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(base_rows)} paired prompt edits")


if __name__ == "__main__":
    main()
