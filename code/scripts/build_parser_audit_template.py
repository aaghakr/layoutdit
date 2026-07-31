"""Create an author-checkable parser-audit CSV for free-form prompts.

The production parser output is prefilled for convenience, but rows are excluded
from the accuracy table until an author marks ``audit_status`` as ``accepted``
or ``corrected``.  This avoids circularly treating parser output as ground
truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgbdm.text_spatial import get_class_map, parse_positions_from_prompt
from utils.metric import _parse_prompt_counts


def _filtered_counts(prompt: str, dataset: str) -> dict[str, int]:
    num_class = 5 if dataset == "cgl" else 4
    valid = set(get_class_map(num_class))
    counts = _parse_prompt_counts(prompt)
    return {name: int(counts.get(name, 0)) for name in sorted(valid)}


def _filtered_positions(prompt: str, dataset: str) -> dict[str, list[str]]:
    num_class = 5 if dataset == "cgl" else 4
    valid = set(get_class_map(num_class))
    parsed = parse_positions_from_prompt(prompt)
    return {name: list(parsed.get(name, [])) for name in sorted(valid) if parsed.get(name)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing template. By default existing author edits are preserved.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"Preserving existing audit template: {output}")
        return

    frame = pd.read_csv(args.prompts, keep_default_na=False)
    prompt_col = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    rows = []
    for _, row in frame.iterrows():
        prompt = str(row[prompt_col])
        parsed_counts = _filtered_counts(prompt, args.dataset)
        parsed_positions = _filtered_positions(prompt, args.dataset)
        rows.append(
            {
                "dataset": args.dataset,
                "poster_path": Path(str(row["poster_path"])).name,
                "text_prompt": prompt,
                "parser_counts_json": json.dumps(parsed_counts, sort_keys=True),
                "parser_positions_json": json.dumps(parsed_positions, sort_keys=True),
                # Author workflow:
                # - accepted: manual_* fields equal parser_* fields.
                # - corrected: manual_* fields were edited by the author.
                # - exclude: row is ambiguous and excluded from audit denominator.
                # - needs_review: not counted yet.
                "audit_status": "needs_review",
                "manual_counts_json": json.dumps(parsed_counts, sort_keys=True),
                "manual_positions_json": json.dumps(parsed_positions, sort_keys=True),
                "notes": "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} audit rows to {output}")
    print("Review each row, edit manual_* JSON if needed, and set audit_status to accepted/corrected/exclude.")


if __name__ == "__main__":
    main()
