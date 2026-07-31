"""Crossed participant/item bootstrap for blinded pairwise preferences."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


def intentdit_outcome(row: sqlite3.Row, method_a: str) -> int | None:
    if row["choice"] == "tie":
        return None
    if row["left_method"] == method_a:
        return int(row["choice"] == "left")
    if row["right_method"] == method_a:
        return int(row["choice"] == "right")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="user_study/data/study.sqlite")
    parser.add_argument("--baseline", default="layoutdit")
    parser.add_argument("--method-a", default="intentdit_image")
    parser.add_argument("--criterion", choices=("quality", "instruction"), default="quality")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-participants", type=int, default=20)
    parser.add_argument(
        "--output", default="experiments/user_study/hierarchical_bootstrap.json"
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT participant,image_id,left_method,right_method,criterion,choice FROM responses_a"
    ).fetchall()
    usable = []
    for row in rows:
        methods = {row["left_method"], row["right_method"]}
        if row["criterion"] != args.criterion or methods != {args.method_a, args.baseline}:
            continue
        outcome = intentdit_outcome(row, args.method_a)
        if outcome is not None:
            usable.append((row["participant"], row["image_id"], outcome))

    participants = sorted({row[0] for row in usable})
    items = sorted({row[1] for row in usable})
    if len(participants) < args.min_participants:
        raise SystemExit(
            f"Need at least {args.min_participants} participants; found {len(participants)}"
        )
    if not usable:
        raise SystemExit("No non-tie IntentDiT/baseline comparisons found")

    point = float(np.mean([row[2] for row in usable]))
    rng = np.random.default_rng(args.seed)
    samples = []
    for _ in range(args.iterations):
        participant_weights = Counter(rng.choice(participants, len(participants), replace=True))
        item_weights = Counter(rng.choice(items, len(items), replace=True))
        weighted_wins = 0
        weighted_total = 0
        for participant, item, outcome in usable:
            weight = participant_weights[participant] * item_weights[item]
            weighted_wins += weight * outcome
            weighted_total += weight
        if weighted_total:
            samples.append(weighted_wins / weighted_total)

    sample_array = np.asarray(samples)
    result = {
        "baseline": args.baseline,
        "method_a": args.method_a,
        "criterion": args.criterion,
        "participants": len(participants),
        "items": len(items),
        "non_tie_ratings": len(usable),
        "preference": point,
        "ci95": [
            float(np.quantile(sample_array, 0.025)),
            float(np.quantile(sample_array, 0.975)),
        ],
        "two_sided_p_vs_0.5": float(
            min(1.0, 2 * min(np.mean(sample_array <= 0.5), np.mean(sample_array >= 0.5)))
        ),
        "iterations": len(samples),
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
