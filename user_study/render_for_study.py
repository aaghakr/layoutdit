"""
Render posters for the user study from existing layout outputs.

Reads layout boxes saved by code/scripts/test.py for each method+seed
(under experiments/paper_figures/<exp>_seed<seed>/...) and overlays them
on the corresponding background image at PKU/CGL native resolution.

Usage
-----
    python user_study/render_for_study.py \
        --pku-images data/dataset/pku/split/test/inpaint \
        --cgl-images data/dataset/cgl/split/test/inpaint \
        --out user_study/data/renders \
        --seed 1

Protocol-specific folders are `intentdit_image`, `layoutdit`, `intentdit_text`, and
`textbaseline`. Baseline folders must come from the actual external methods; this
utility must never relabel an IntentDiT ablation as LayoutDiT or as a text baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RENDERS_DIR = PROJECT_ROOT / "user_study" / "data" / "renders"

CLASS_COLORS = {1: ("Text", (30, 100, 220)), 2: ("Logo", (210, 40, 40)),
                3: ("Underlay", (40, 160, 60)), 4: ("Embellishment", (220, 140, 30))}


def render_layout(image_path: Path, boxes: list[tuple[int, int, int, int, int]],
                  out_path: Path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for cls, x1, y1, x2, y2 in boxes:
        name, color = CLASS_COLORS.get(int(cls), ("?", (128, 128, 128)))
        draw.rectangle([x1, y1, x2, y2], outline=color + (220,), width=4)
        if name == "Underlay":
            draw.rectangle([x1, y1, x2, y2], fill=color + (40,))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pku-images", default="data/dataset/pku/split/test/inpaint")
    p.add_argument("--cgl-images", default="data/dataset/cgl/split/test/inpaint")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--method-name", default="intentdit_image",
                   help="Subfolder under user_study/data/renders/")
    p.add_argument("--exp-name", default="pku_vit_both_text",
                   help="Experiment name prefix used by code/scripts/test.py")
    p.add_argument("--exp-dir", default="experiments/paper_figures",
                   help="Folder with rendered outputs from code/scripts/test.py")
    p.add_argument("--out", default=str(RENDERS_DIR))
    args = p.parse_args()

    # The visualization script in code/utils/visualize.py renders posters
    # into experiments/paper_figures/<exp_name>_seed<seed>/. We re-render
    # them here at full resolution so the user study sees consistent sizes.
    src = Path(args.exp_dir) / f"{args.exp_name}_seed{args.seed}"
    if not src.exists():
        raise SystemExit(f"Source folder {src} not found. "
                         "Run code/scripts/test.py for this variant first.")

    out_root = Path(args.out) / args.method_name
    out_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(src.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        # The renders saved by visualize_images already have boxes drawn,
        # so we just copy them with downscaling for the study.
        img = Image.open(f).convert("RGB")
        img.thumbnail((720, 720))
        img.save(out_root / f.name, format="PNG", optimize=True)
        n += 1
    print(f"Copied {n} renders into {out_root}")


if __name__ == "__main__":
    main()
