"""Fail when shared metrics disagree with an official evaluator export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_METRICS = ("val", "ove", "ali", "undl", "unds", "uti", "occ", "rea", "sma")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--mapping-json", default="{}", help="ours key -> reference key")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ours = json.loads(Path(args.ours).read_text())
    reference = json.loads(Path(args.reference).read_text())
    mapping = json.loads(args.mapping_json)
    comparisons = {}
    failures = []
    for key in args.metrics:
        reference_key = mapping.get(key, key)
        if key not in ours or reference_key not in reference:
            continue
        left, right = float(ours[key]), float(reference[reference_key])
        difference = abs(left - right)
        tolerance = args.atol + args.rtol * abs(right)
        passed = difference <= tolerance
        comparisons[key] = {
            "ours": left,
            "reference": right,
            "reference_key": reference_key,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": passed,
        }
        if not passed:
            failures.append(key)
    if not comparisons:
        raise SystemExit("No requested metric keys were shared with the reference JSON")
    result = {"passed": not failures, "failed_metrics": failures, "comparisons": comparisons}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise SystemExit(f"Metric parity failed for: {', '.join(failures)}")
    print(f"Metric parity passed for {len(comparisons)} metrics")


if __name__ == "__main__":
    main()
