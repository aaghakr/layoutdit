"""
Multi-seed evaluation script for generating mean +/- std results.

Must be run from the project root (intent_aware_layout_generation/):
    conda activate cgbdm
    python code/scripts/run_multi_seed_eval.py \
        --config code/configs/experiments_fair_eval.yaml \
        --seeds 1 2 3 --anno anno
"""

import sys
import os

import argparse
import json
import subprocess
import numpy as np
import yaml
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
PAPER_FIGURES_DIR = os.path.join(PROJECT_ROOT, "experiments", "paper_figures")


def run_single_experiment(experiment, anno, seed, gpuid, path_profile, ddim_schedule):
    """Run a single test experiment from the code/ directory."""
    exp_name = f"{experiment['experiment_name']}_seed{seed}"
    cmd = [
        sys.executable, "scripts/test.py",
        "--gpuid", str(gpuid),
        "--dataset", experiment["dataset"],
        "--anno", anno,
        "--task", "uncond",
        "--check_path", experiment["check_path"],
        "--v_encoder", experiment["v_encoder"],
        "--spatial_guidance", str(experiment["spatial_guidance"]),
        "--experiment_name", exp_name,
        "--seed", str(seed),
        "--ddim_num_steps", "100",
        "--ddim_schedule", ddim_schedule,
        "--path-profile", path_profile,
    ]
    if experiment.get("text_control", False):
        cmd.append("--text_control")

    print(f"\n{'='*60}")
    print(f"Running: {exp_name} (seed={seed})")
    print(f"CWD: {CODE_DIR}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=CODE_DIR)
    if result.returncode != 0:
        print(f"WARNING: {exp_name} failed with exit code {result.returncode}")
        return None

    metrics_path = os.path.join(PAPER_FIGURES_DIR, f"{exp_name}_metrics.json")
    if os.path.exists(metrics_path):
        return metrics_path
    print(f"WARNING: metrics file not found at {metrics_path}")
    return None


def aggregate_results(all_metrics):
    """Compute mean and std across seeds for each metric."""
    if not all_metrics:
        return {}

    metric_keys = all_metrics[0].keys()
    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiments YAML config")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Random seeds to evaluate (default: 1 2 3)")
    parser.add_argument("--anno", type=str, default="unanno",
                        choices=["anno", "unanno"])
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="Filter to specific experiment names")
    parser.add_argument("--path-profile", choices=["local", "server"], default="local")
    parser.add_argument(
        "--ddim-schedule", "--ddim_schedule",
        dest="ddim_schedule",
        choices=["training", "cosine", "linear"],
        default="cosine",
        help="DDIM alpha schedule; default matches the cosine DDPM training schedule.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]
    if args.experiments:
        experiments = [e for e in experiments if e["experiment_name"] in args.experiments]

    output_dir = os.path.join(PROJECT_ROOT, "experiments", "multi_seed_results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for experiment in experiments:
        exp_name = experiment["experiment_name"]
        seed_metrics = []

        for seed in args.seeds:
            metrics_path = run_single_experiment(
                experiment, args.anno, seed, args.gpuid, args.path_profile,
                args.ddim_schedule,
            )
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    seed_metrics.append(json.load(f))

        if seed_metrics:
            aggregated = aggregate_results(seed_metrics)
            all_results[exp_name] = aggregated
            print(f"\n--- {exp_name} (n={len(seed_metrics)} seeds) ---")
            for metric, vals in aggregated.items():
                print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")

    summary_path = os.path.join(output_dir, f"multi_seed_summary_{args.anno}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    md_path = os.path.join(output_dir, f"multi_seed_results_{args.anno}.md")
    with open(md_path, "w") as f:
        f.write("| experiment | dataset | val | ove | undl | unds | occ | rea |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for exp_name, metrics in all_results.items():
            dataset = exp_name.split("_")[0]
            row = [exp_name, dataset]
            for m in ["val", "ove", "undl", "unds", "occ", "rea"]:
                if m in metrics:
                    row.append(f"{metrics[m]['mean']:.4f}±{metrics[m]['std']:.4f}")
                else:
                    row.append("---")
            f.write("| " + " | ".join(row) + " |\n")
    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
