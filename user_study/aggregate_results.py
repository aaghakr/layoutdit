"""
Aggregate user-study responses into descriptive statistics.

Inputs
------
user_study/data/study.sqlite   (Flask app writes to this)

Outputs
-------
experiments/user_study/tables/user_study_partA.md
experiments/user_study/tables/user_study_partB.md
experiments/user_study/figures/user_study_partA.csv
experiments/user_study/figures/user_study_partB.csv

Tests
-----
- Descriptive pairwise wins with a Wilson 95% confidence interval.
- Participant counts are exported explicitly so ratings are not mistaken for subjects.

Final inferential analysis should use a participant/item repeated-measures model or
hierarchical bootstrap. Trial-level binomial or McNemar p-values are not valid
for this crossed repeated-measures design and are intentionally not emitted.
"""
import csv
import math
import os
import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "user_study" / "data" / "study.sqlite"
TABLES = PROJECT_ROOT / "experiments" / "user_study" / "tables"
FIGS = PROJECT_ROOT / "experiments" / "user_study" / "figures"


def wilson_interval(successes: int, trials: int, z: float = 1.96):
    """Wilson score interval for a descriptive binary proportion."""
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return center - margin, center + margin


def part_a_stats(conn) -> dict:
    rows = conn.execute(
        "SELECT participant, image_id, left_method, right_method, criterion, choice FROM responses_a"
    ).fetchall()

    # Reduce to "did IntentDiT win, lose, tie" per row.
    wins = Counter()  # method-vs-intentdit -> {win, loss, tie}
    pairs = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})
    for r in rows:
        intent_method = "intentdit_text" if r["criterion"] == "instruction" else "intentdit_image"
        if r["left_method"] == intent_method:
            other = r["right_method"]
            if r["choice"] == "left":
                pairs[other]["win"] += 1
            elif r["choice"] == "right":
                pairs[other]["loss"] += 1
            else:
                pairs[other]["tie"] += 1
        elif r["right_method"] == intent_method:
            other = r["left_method"]
            if r["choice"] == "right":
                pairs[other]["win"] += 1
            elif r["choice"] == "left":
                pairs[other]["loss"] += 1
            else:
                pairs[other]["tie"] += 1

    summary = {}
    for other, c in pairs.items():
        b, n = c["win"], c["win"] + c["loss"]
        winrate = b / n if n else float("nan")
        ci_low, ci_high = wilson_interval(b, n)
        participant_n = conn.execute(
            "SELECT COUNT(DISTINCT participant) FROM responses_a "
            "WHERE left_method = ? OR right_method = ?", (other, other)
        ).fetchone()[0]
        criterion = "instruction" if other == "textbaseline" else "quality"
        summary[f"{criterion}:{other}"] = {
            "criterion": criterion,
            "baseline": other,
            "win_rate_excl_ties": winrate,
            "wins": c["win"], "losses": c["loss"], "ties": c["tie"],
            "ci_low": ci_low, "ci_high": ci_high,
            "participants": participant_n,
            "n_total": c["win"] + c["loss"] + c["tie"],
        }
    return summary


def part_b_stats(conn) -> dict:
    rows = conn.execute(
        "SELECT participant, category, severity FROM responses_b"
    ).fetchall()
    by_cat = defaultdict(list)
    participants_by_cat = defaultdict(set)
    for r in rows:
        by_cat[r["category"]].append(r["severity"])
        participants_by_cat[r["category"]].add(r["participant"])
    out = {}
    for cat, vals in by_cat.items():
        out[cat] = {
            "mean": sum(vals) / len(vals) if vals else float("nan"),
            "median": sorted(vals)[len(vals) // 2] if vals else None,
            "n": len(vals),
            "participants": len(participants_by_cat[cat]),
        }
    return out


def main():
    if not DB.exists():
        print(f"[aggregate] {DB} missing -- run the study first.", file=sys.stderr)
        sys.exit(0)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    a = part_a_stats(conn)
    b = part_b_stats(conn)

    # Part A markdown
    md_a = ["### Part A &mdash; pairwise preference", "",
            "Win rate of IntentDiT against each baseline (ties excluded).",
            "Wilson intervals are descriptive and do not adjust for repeated ratings.",
            "",
            "| Baseline | Participants | Wins | Losses | Ties | Win rate | 95% CI |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    md_a[5] = "| Criterion | Baseline | Participants | Wins | Losses | Ties | Win rate | 95% CI |"
    md_a[6] = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    for name, s in a.items():
        md_a.append(f"| {s['criterion']} | {s['baseline']} | {s['participants']} | {s['wins']} | {s['losses']} | "
                    f"{s['ties']} | {s['win_rate_excl_ties']:.2%} | "
                    f"[{s['ci_low']:.2%}, {s['ci_high']:.2%}] |")
    (TABLES / "user_study_partA.md").write_text("\n".join(md_a))

    md_b = ["### Part B &mdash; failure-severity ratings", "",
            "Mean severity rating (1-7) per failure category. Higher = more design-critical.",
            "",
            "| Failure category | Mean severity | Median | Participants | Ratings |",
            "| --- | --- | --- | --- | --- |"]
    # Order by mean severity, descending
    for cat, s in sorted(b.items(), key=lambda kv: -kv[1]["mean"]):
        md_b.append(f"| {cat.replace('_', ' ')} | {s['mean']:.2f} | {s['median']} | "
                    f"{s['participants']} | {s['n']} |")
    (TABLES / "user_study_partB.md").write_text("\n".join(md_b))

    # CSVs for plotting
    with open(FIGS / "user_study_partA.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "baseline", "participants", "wins", "losses", "ties", "win_rate", "ci_low", "ci_high"])
        for name, s in a.items():
            w.writerow([s["criterion"], s["baseline"], s["participants"], s["wins"], s["losses"], s["ties"],
                        f"{s['win_rate_excl_ties']:.4f}",
                        f"{s['ci_low']:.4f}", f"{s['ci_high']:.4f}"])
    with open(FIGS / "user_study_partB.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "mean", "median", "participants", "ratings"])
        for cat, s in b.items():
            w.writerow([cat, f"{s['mean']:.4f}", s["median"], s["participants"], s["n"]])

    print("Wrote", TABLES / "user_study_partA.md",
          "and", TABLES / "user_study_partB.md")


if __name__ == "__main__":
    main()
