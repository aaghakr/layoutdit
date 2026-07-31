#!/usr/bin/env python3
"""
Export Part B exemplars from *real* model outputs (layout tensors), not synthetic overlays.

Pipeline
--------
1. Run inference with saved tensors (one-time):

   cd code && python scripts/test.py ... --experiment_name pku_vit_intent_text \\
       --save-test-output auto

   This writes experiments/paper_figures/<experiment_name>_test_output.pt

2. Build failure folders + thumbnails:

   python user_study/export_real_failures.py \\
       --config code/configs/pku_unanno_test.yaml \\
       --test-output experiments/paper_figures/pku_vit_intent_text_seed1_test_output.pt \\
       --experiment-name pku_vit_intent_text_seed1 \\
       --text-control

   (--experiment-name selects the paper_figures folder to resolve img names if needed;
    rendering always uses cfg.paths.test.inp_dir + tensors.)

3. Rebuild manifest:

   python user_study/build_manifest.py --n 30 --per-category 5

Categories use the same automated signals as the paper failure taxonomy:
  subject_occlusion   — high mean saliency under *all* layout boxes (per-image occlusion)
  illegible_text      — high image-gradient under text boxes (unreadability proxy)
  element_overlap     — high mean pairwise IoU among non-underlay elements
  underlay_misalignment — low max overlap between underlay panels and other elements
  count_mismatch      — high |expected − predicted| class counts vs prompt (text runs only)

By default, picks are de-duplicated across categories (worst-first per category).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
import yaml

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from torchvision.transforms.functional import to_tensor  # noqa: E402
from einops import rearrange  # noqa: E402

from utils.util import load_config, process_paths, Config, box_cxcywh_to_xyxy  # noqa: E402
from utils.metric import (  # noqa: E402
    getRidOfInvalid,
    _parse_prompt_counts,
    CLASS_INDEX_TO_NAME,
    _list_all_pair_indices,
    _compute_iou_group,
    _extract_grad,
)
from utils.visualize import draw_image  # noqa: E402

def _load_cfg(cfg_path: Path, paths_base: str = "") -> object:
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    if paths_base:
        raw["paths"]["base"] = str(Path(paths_base).resolve())
    raw = process_paths(raw)
    raw["datetime"] = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%m_%d_%H%M")
    return Config(raw)


FAILURE_CATEGORIES = [
    "subject_occlusion",
    "illegible_text",
    "element_overlap",
    "underlay_misalignment",
    "count_mismatch",
]


def _prepare_outputs(test_output: torch.Tensor, cfg):
    clses, boxes = test_output[:, :, :1].clone(), test_output[:, :, 1:].clone()
    boxes = torch.clamp(box_cxcywh_to_xyxy(boxes), 0, 1)
    clses = getRidOfInvalid(clses, boxes)
    boxes_px = boxes.clone()
    boxes_px[:, :, ::2] *= cfg.width
    boxes_px[:, :, 1::2] *= cfg.height
    boxes_px = boxes_px.round().int()
    return clses, boxes_px


def _per_image_occlusion(name: str, cls: torch.Tensor, box: torch.Tensor, cfg) -> float:
    import cv2  # noqa: WPS433 — matches metric.py

    sal_1 = np.array(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L"))
    sal_2 = np.array(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L"))
    sal_map = Image.fromarray(np.maximum(sal_1, sal_2))
    sal_map = to_tensor(sal_map.resize((cfg.width, cfg.height)))
    sal_map = rearrange(sal_map, "1 h w ->h w")

    cls_np = np.array(cls.cpu(), dtype=int)
    box_np = np.array(box.cpu(), dtype=int)
    mask = (cls_np > 0).reshape(-1)
    mask_box = box_np[mask]
    cal_mask = torch.zeros_like(sal_map)
    for mb in mask_box:
        xl, yl, xr, yr = mb
        cal_mask[yl:yr, xl:xr] = 1
    occ = sal_map[cal_mask.bool()]
    return float(occ.mean().item()) if occ.numel() else 0.0


def _per_image_unreadability(name: str, cls: torch.Tensor, box: torch.Tensor, cfg) -> float:
    image = to_tensor(
        Image.open(os.path.join(cfg.paths.test.inp_dir, name))
        .convert("RGB")
        .resize((cfg.width, cfg.height))
    )
    image = rearrange(image, "c h w ->h w c")

    cls_np = np.array(cls.cpu(), dtype=int)
    box_np = np.array(box.cpu(), dtype=int)

    bbox_mask_special = torch.zeros(cfg.height, cfg.width)
    text_mask = (cls_np == 1).reshape(-1)
    text_boxes = box_np[text_mask]
    for mb in text_boxes:
        xl, yl, xr, yr = mb
        bbox_mask_special[yl:yr, xl:xr] = 1
    underlay_mask = (cls_np == 3).reshape(-1)
    underlay_boxes = box_np[underlay_mask]
    for mb in underlay_boxes:
        xl, yl, xr, yr = mb
        bbox_mask_special[yl:yr, xl:xr] = 0
    g_xy = _extract_grad(image)
    ur = g_xy[bbox_mask_special.bool()]
    return float(ur.mean().item()) if ur.numel() else 0.0


def _per_image_overlap(cls: torch.Tensor, box: torch.Tensor) -> float:
    cls_np = np.array(cls.cpu(), dtype=int).reshape(-1)
    box_np = np.array(box.cpu(), dtype=int)
    mask = (cls_np > 0) & (cls_np != 3)
    mask_box = box_np[mask]
    n = len(mask_box)
    if n <= 1:
        return 0.0
    ii, jj = _list_all_pair_indices(mask_box)
    iou = _compute_iou_group(mask_box[ii], mask_box[jj], method="iou", transform=True)
    return float(np.mean(iou))


def _per_image_underlay_misalignment(cls: torch.Tensor, box: torch.Tensor) -> Optional[float]:
    """Returns (1 - mean max IoU) so higher = more misaligned. None if no underlay."""
    cls_np = np.array(cls.cpu(), dtype=int).reshape(-1)
    box_np = np.array(box.cpu(), dtype=int)
    mask_und = cls_np == 3
    mask_other = (cls_np > 0) & (cls_np != 3)
    box_und = box_np[mask_und]
    box_other = box_np[mask_other]
    if len(box_und) == 0 or len(box_other) == 0:
        return None
    max_ious = []
    for i in range(len(box_und)):
        bb1 = box_und[i]
        best = 0.0
        for j in range(len(box_other)):
            bb2 = box_other[j]
            ios = _compute_iou_group(bb1, bb2, method="ai/a2", transform=False)
            best = max(best, float(ios))
        max_ious.append(best)
    align = float(np.mean(max_ious))
    return 1.0 - align


def _prompt_count_mismatch(name: str, cls: torch.Tensor, poster_to_prompt: dict, num_class: int) -> Optional[float]:
    key = name if name in poster_to_prompt else os.path.basename(name)
    prompt = poster_to_prompt.get(key, "")
    expected = _parse_prompt_counts(prompt)
    row = np.array(cls.cpu(), dtype=int).reshape(-1)
    valid = row > 0
    pred_counts: dict[str, int] = {}
    for c in range(1, num_class + 1):
        cname = CLASS_INDEX_TO_NAME.get(c, f"Class{c}")
        pred_counts[cname] = int(np.sum((row == c) & valid))
    for cname in CLASS_INDEX_TO_NAME.values():
        pred_counts.setdefault(cname, 0)

    total_exp = sum(expected.values())
    if total_exp == 0:
        return None
    diff = sum(abs(expected.get(k, 0) - pred_counts.get(k, 0)) for k in CLASS_INDEX_TO_NAME.values())
    return float(diff)


def _score_table(
    img_names: list[str],
    clses: torch.Tensor,
    boxes: torch.Tensor,
    cfg,
    poster_to_prompt: Optional[dict],
) -> dict[str, list[tuple[str, float]]]:
    num_class = getattr(cfg, "num_class", 4)
    occ, rea, ove, und_mis, cnt = [], [], [], [], []
    for i, name in enumerate(img_names):
        c, b = clses[i], boxes[i]
        occ.append((name, _per_image_occlusion(name, c, b, cfg)))
        rea.append((name, _per_image_unreadability(name, c, b, cfg)))
        ove.append((name, _per_image_overlap(c, b)))
        u = _per_image_underlay_misalignment(c, b)
        if u is not None:
            und_mis.append((name, u))
        if poster_to_prompt is not None:
            cm = _prompt_count_mismatch(name, c, poster_to_prompt, num_class)
            if cm is not None:
                cnt.append((name, cm))

    return {
        "subject_occlusion": occ,
        "illegible_text": rea,
        "element_overlap": ove,
        "underlay_misalignment": und_mis,
        "count_mismatch": cnt,
    }


def _pick_top(
    pairs: list[tuple[str, float]],
    k: int,
    used: set[str],
    *,
    high_is_worse: bool,
) -> list[str]:
    pairs = sorted(pairs, key=lambda x: x[1], reverse=high_is_worse)
    out = []
    for name, _ in pairs:
        if name in used:
            continue
        out.append(name)
        used.add(name)
        if len(out) >= k:
            break
    # allow reuse if pool is small
    if len(out) < k:
        for name, _ in pairs:
            if name not in out:
                out.append(name)
            if len(out) >= k:
                break
    return out[:k]


def render_study_png(
    cfg,
    test_output: torch.Tensor,
    idx: int,
    poster_filename: str,
    dest: Path,
    max_side: int,
) -> None:
    """Match user_study/render_for_study.py thumbnail size."""
    clses, boxes = test_output[:, :, :1], test_output[:, :, 1:]
    box, cls = boxes[idx], clses[idx]
    image_path = os.path.join(cfg.paths.test.inp_dir, poster_filename)
    img = Image.open(image_path).convert("RGB")
    tmp_dir = dest.parent / "_tmp_render"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        draw_image(box, cls, img, poster_filename, cfg.width, cfg.height, cfg.num_class, str(tmp_dir))
        rendered = Image.open(tmp_dir / poster_filename).convert("RGB")
        rendered.thumbnail((max_side, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(dest, format="PNG", optimize=True)
    finally:
        tfile = tmp_dir / poster_filename
        if tfile.exists():
            tfile.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="code/configs/pku_unanno_test.yaml")
    p.add_argument("--test-output", type=str, required=True, help=".pt from test.py --save-test-output")
    p.add_argument("--out", type=str, default="user_study/data/failures")
    p.add_argument("--per-category", type=int, default=5)
    p.add_argument("--max-side", type=int, default=720)
    p.add_argument("--text-control", action="store_true", help="Set if run used text prompts (for count_mismatch)")
    p.add_argument("--dedup", action="store_true", default=True)
    p.add_argument("--no-dedup", action="store_false", dest="dedup")
    p.add_argument("--selection-json", type=str, default="", help="Write chosen filenames + scores for records")
    p.add_argument(
        "--paths-base",
        type=str,
        default="",
        help="Override YAML paths.base (e.g. .../data/dataset/pku/split) when metrics fail to find images",
    )
    args = p.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.is_file():
        cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path, args.paths_base) if args.paths_base else load_config(str(cfg_path))
    if not args.paths_base and not os.path.isdir(getattr(cfg.paths.test, "inp_dir", "")):
        guess = PROJECT_ROOT / "data/dataset/pku/split"
        if guess.is_dir():
            cfg = _load_cfg(cfg_path, str(guess))
            print(f"[export_real_failures] Using paths.base fallback: {guess}")
    if args.text_control:
        cfg.text_control = True

    blob = torch.load(Path(args.test_output).expanduser(), map_location="cpu")
    if isinstance(blob, dict):
        img_names = blob["img_names"]
        test_output = blob["test_output"]
    else:
        raise SystemExit("test-output .pt must be a dict with keys img_names, test_output")

    img_names = [str(x) for x in img_names]
    n = test_output.shape[0]
    if len(img_names) < n:
        raise SystemExit("img_names shorter than test_output")
    img_names = img_names[:n]

    poster_to_prompt: Optional[dict] = None
    import pandas as pd  # noqa: WPS433

    path = getattr(cfg, "paths", None)
    prompts_path = getattr(path.test, "all_prompts", None) if path is not None else None
    if getattr(cfg, "prompts_csv_override", None):
        prompts_path = cfg.prompts_csv_override
    if prompts_path and os.path.isfile(prompts_path):
        df = pd.read_csv(prompts_path)
        col = "text_prompt" if "text_prompt" in df.columns else "prompt"
        poster_to_prompt = df.groupby("poster_path")[col].first().to_dict()

    _prepare_outputs(test_output, cfg)  # validate shapes
    clses, boxes_px = _prepare_outputs(test_output, cfg)
    scores = _score_table(img_names, clses, boxes_px, cfg, poster_to_prompt)

    used: set[str] = set()
    plan: dict[str, list[str]] = {}
    for cat in FAILURE_CATEGORIES:
        pairs = scores[cat]
        if not pairs:
            raise SystemExit(f"No scorable samples for category {cat!r} (check data paths / prompts).")
        bucket = used if args.dedup else set()
        picked = _pick_top(pairs, args.per_category, bucket, high_is_worse=True)
        plan[cat] = picked

    out_root = PROJECT_ROOT / args.out
    name_to_idx = {img_names[i]: i for i in range(len(img_names))}
    manifest_scores: dict[str, list[dict]] = {c: [] for c in FAILURE_CATEGORIES}
    score_by_poster = {c: dict(scores[c]) for c in FAILURE_CATEGORIES}

    for cat, names in plan.items():
        for j, poster in enumerate(names):
            idx = name_to_idx[poster]
            dest = out_root / cat / f"{cat}_{j + 1:02d}.png"
            render_study_png(cfg, test_output, idx, poster, dest, args.max_side)
            manifest_scores[cat].append({
                "file": dest.name,
                "poster": poster,
                "score": score_by_poster[cat].get(poster),
            })

    if args.selection_json:
        outp = Path(args.selection_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(manifest_scores, f, indent=2)
        print("Wrote", outp)

    print(f"Wrote {args.per_category * len(FAILURE_CATEGORIES)} PNGs under {out_root}")


if __name__ == "__main__":
    main()
