#!/usr/bin/env python3
"""
Run evaluation with a fixed protocol for fair comparison across all experiments.

Usage:
  cd code
  python scripts/run_fair_eval_all.py --experiments experiments_fair_eval.yaml [--anno unanno] [--ddim_num_steps 100] [--ddim_schedule cosine] [--gpuid 0]

The YAML file lists each experiment with:
  - experiment_name: used for saving images and metrics
  - check_path: path to Epoch*_cgbdm_weights.pth
  - dataset: pku or cgl
  - v_encoder: vit or swin
  - spatial_guidance: 0, 1, or 2
  - text_control: true/false

All runs use the same: seed (from test.py), --anno, --task uncond, --ddim_num_steps, --ddim_schedule.
Outputs: runs test for each row, then writes experiments/paper_figures/fair_eval_results.csv (and .md).
"""

import argparse
import json
import os
import subprocess
import sys

try:
    import yaml
except ImportError:
    yaml = None


def load_experiments(path):
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("experiments", data) if isinstance(data, dict) else data


def run_test(exp, code_dir, anno, ddim_num_steps, ddim_schedule, gpuid, path_profile):
    cmd = [
        sys.executable,
        "scripts/test.py",
        "--dataset", str(exp["dataset"]),
        "--anno", anno,
        "--task", "uncond",
        "--check_path", os.path.expanduser(exp["check_path"]),
        "--v_encoder", str(exp.get("v_encoder", "vit")),
        "--spatial_guidance", str(exp.get("spatial_guidance", 0)),
        "--experiment_name", str(exp["experiment_name"]),
        "--ddim_num_steps", str(ddim_num_steps),
        "--ddim_schedule", ddim_schedule,
        "--gpuid", str(gpuid),
        "--path-profile", path_profile,
    ]
    if exp.get("text_control"):
        cmd.append("--text_control")
    return subprocess.run(cmd, cwd=code_dir)


def collect_metrics(experiments, project_root):
    paper_figures = os.path.join(project_root, "experiments", "paper_figures")
    rows = []
    for exp in experiments:
        name = exp["experiment_name"]
        path = os.path.join(paper_figures, f"{name}_metrics.json")
        if not os.path.isfile(path):
            rows.append({"experiment": name, "error": "no metrics file"})
            continue
        with open(path) as f:
            m = json.load(f)
        row = {"experiment": name, "dataset": exp.get("dataset", "")}
        for k in ["val", "ove", "undl", "unds", "occ", "rea", "tla"]:
            row[k] = m.get(k, "")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fair evaluation: same protocol for all experiments")
    parser.add_argument("--experiments", type=str, required=True, help="YAML file listing experiments (see docstring)")
    parser.add_argument("--anno", type=str, default="unanno", choices=["anno", "unanno"], help="Test set (default: unanno)")
    parser.add_argument("--ddim_num_steps", type=int, default=100, help="DDIM steps (default: 100)")
    parser.add_argument(
        "--ddim-schedule", "--ddim_schedule",
        dest="ddim_schedule",
        choices=["training", "cosine", "linear"],
        default="cosine",
        help="DDIM alpha schedule; default matches the cosine DDPM training schedule.",
    )
    parser.add_argument("--gpuid", type=int, default=0, help="GPU id")
    parser.add_argument("--skip_run", action="store_true", help="Only aggregate existing metrics, do not run test")
    parser.add_argument("--path-profile", choices=["local", "server"], default="local")
    opt = parser.parse_args()

    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(code_dir)
    experiments = load_experiments(opt.experiments)

    if not opt.skip_run:
        for i, exp in enumerate(experiments):
            print(f"[{i+1}/{len(experiments)}] {exp['experiment_name']} ...")
            ret = run_test(
                exp, code_dir, opt.anno, opt.ddim_num_steps, opt.ddim_schedule,
                opt.gpuid, opt.path_profile,
            )
            if ret.returncode != 0:
                print(f"  WARNING: test.py exited with {ret.returncode}")

    rows = collect_metrics(experiments, project_root)
    out_dir = os.path.join(project_root, "experiments", "paper_figures")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "fair_eval_results.csv")
    cols = ["experiment", "dataset", "val", "ove", "undl", "unds", "occ", "rea", "tla"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"Wrote {csv_path}")

    # Markdown table
    md_path = os.path.join(out_dir, "fair_eval_results.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
