#!/usr/bin/env python3
"""Verify that saved paper evidence uses the intended DDIM schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing *_evidence.json files")
    parser.add_argument(
        "--expected-schedule",
        choices=("training", "cosine", "linear"),
        default="cosine",
        help="Required DDIM schedule recorded in each evidence file",
    )
    parser.add_argument(
        "--allowed-steps",
        type=int,
        nargs="+",
        default=[100],
        help="Allowed DDIM step counts; include efficiency-sweep values if needed",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only validate evidence whose experiment starts with this prefix. Can be repeated.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    prefixes = tuple(args.include_prefix)
    allowed_steps = set(args.allowed_steps)
    mismatches: list[str] = []
    checked = 0

    for path in sorted(input_dir.glob("*_evidence.json")):
        try:
            evidence = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            mismatches.append(f"{path}: unreadable evidence JSON ({exc})")
            continue

        experiment = str(evidence.get("experiment") or path.name.removesuffix("_evidence.json"))
        if prefixes and not experiment.startswith(prefixes):
            continue

        checked += 1
        schedule = evidence.get("ddim_schedule")
        steps = int(evidence.get("ddim_steps", -1))
        if schedule != args.expected_schedule or steps not in allowed_steps:
            mismatches.append(
                f"{path}: experiment={experiment}, ddim_schedule={schedule!r}, "
                f"ddim_steps={steps}; expected schedule={args.expected_schedule!r}, "
                f"steps in {sorted(allowed_steps)}"
            )

    if mismatches:
        print("DDIM evidence mismatches:")
        for item in mismatches:
            print(f"  {item}")
        raise SystemExit(1)

    print(
        f"Validated {checked} evidence files with DDIM schedule "
        f"{args.expected_schedule!r} and allowed steps {sorted(allowed_steps)}."
    )


if __name__ == "__main__":
    main()
