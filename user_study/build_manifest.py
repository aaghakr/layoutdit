"""
Build the user-study manifest (Part A pairwise + Part B severity exemplars).

Inputs
------
user_study/data/renders/<method>/<image>.png
    Pre-rendered posters per method. Methods expected:
        - intentdit
        - layoutdit
        - postero  (optional)

user_study/data/failures/<category>/<example_id>.png
    Pre-rendered failure exemplars per category.
    Categories:
        subject_occlusion, illegible_text, element_overlap,
        underlay_misalignment, count_mismatch

Outputs
-------
user_study/data/manifest.json with this shape:
    {
        "part_a": [
            {"image_id": ..., "left_method": ..., "right_method": ...,
             "left_render": <relative path under data/renders>,
             "right_render": <relative path under data/renders>}
            ...
        ],
        "part_b": [
            {"category": ..., "example_id": ..., "render": <relative path under data/renders>}
            ...
        ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def list_renders(method: str) -> list[str]:
    folder = DATA / "renders" / method
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".png")


def build_pairs(
    rng: random.Random,
    n: int,
    left_name: str,
    right_name: str,
    criterion: str,
    prompts: dict[str, str] | None = None,
) -> list[dict]:
    common = set(list_renders(left_name)) & set(list_renders(right_name))
    if not common:
        return []
    common = sorted(common)
    rng.shuffle(common)
    pairs = []
    for image_id in common[:n]:
        if rng.random() < 0.5:
            left, right = left_name, right_name
        else:
            left, right = right_name, left_name
        pairs.append({
            "image_id": image_id,
            "criterion": criterion,
            "prompt": (prompts or {}).get(image_id, ""),
            "left_method": left, "right_method": right,
            "left_render":  f"{left}/{image_id}",
            "right_render": f"{right}/{image_id}",
        })
    return pairs


def build_part_b(per_category: int, rng: random.Random) -> list[dict]:
    out = []
    base = DATA / "failures"
    if not base.exists():
        return out
    for cat in sorted(p.name for p in base.iterdir() if p.is_dir()):
        files = sorted(p.name for p in (base / cat).iterdir() if p.suffix.lower() == ".png")
        rng.shuffle(files)
        for fname in files[:per_category]:
            out.append({
                "category": cat,
                "example_id": fname,
                "render": f"failures/{cat}/{fname}",
            })
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quality-n", type=int, default=30)
    p.add_argument("--instruction-n", type=int, default=30)
    p.add_argument("--quality-method-a", default="intentdit_image")
    p.add_argument("--quality-method-b", default="layoutdit")
    p.add_argument("--instruction-method-a", default="intentdit_text")
    p.add_argument("--instruction-method-b", default="textbaseline")
    p.add_argument("--prompts-csv", default="")
    p.add_argument("--per-category", type=int, default=5, help="examples per failure category for Part B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="user_study/data/manifest.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    prompts = {}
    if args.instruction_n:
        if not args.prompts_csv:
            raise SystemExit("--prompts-csv is required for instruction-adherence trials")
        import pandas as pd
        frame = pd.read_csv(args.prompts_csv)
        prompt_col = "text_prompt" if "text_prompt" in frame.columns else "prompt"
        prompts = {
            os.path.basename(str(row.poster_path)): str(getattr(row, prompt_col))
            for row in frame.itertuples()
        }
    part_a = build_pairs(
        rng, args.quality_n, args.quality_method_a, args.quality_method_b, "quality"
    )
    part_a += build_pairs(
        rng, args.instruction_n, args.instruction_method_a,
        args.instruction_method_b, "instruction", prompts
    )
    rng.shuffle(part_a)
    manifest = {
        "part_a": part_a,
        "part_b": build_part_b(args.per_category, rng),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}: part_a={len(manifest['part_a'])}, part_b={len(manifest['part_b'])}")


if __name__ == "__main__":
    main()
