from __future__ import annotations

import sys
import re
from difflib import get_close_matches

import torch
import os
import copy
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageDraw
from math import log
from einops import rearrange, reduce, repeat
from utils import logger
from utils.util import box_cxcywh_to_xyxy
from torch import Tensor
from typing import Callable, Optional, Union, Any
from torchvision.transforms.functional import to_tensor
from utils.spatial_pla import spatial_pla_cal, spatial_prompt_records
from utils.benchmark_metrics import (
    aggregate_records,
    content_records,
    diagnostic_layout_fd,
    geometry_records,
    merge_records,
)
from cgbdm.text_spatial import parse_positions_from_prompt

# Class index to name for TLA (Text-Layout Alignment). Must match prompt vocabulary.
CLASS_INDEX_TO_NAME = {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"}
WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "single": 1,
    # Common misspellings / variants from augmentation
    "too": 2, "to": 2, "tree": 3, "for": 4, "fiv": 5, "fife": 5, "to": 2,
}
# Remove duplicate key "to" (keep last); "to" as number is ambiguous so prefer 2 only when context is clear
if "to" in WORD_TO_NUM:
    WORD_TO_NUM["to"] = 2

# Synonyms and variants for class names (incl. from augmentation / paraphrasing).
# Map lowercase token or phrase -> canonical class name.
CLASS_SYNONYMS = {
    "text": "Text", "texts": "Text", "text box": "Text", "text boxes": "Text",
    "text field": "Text", "text fields": "Text", "caption": "Text", "captions": "Text",
    "logo": "Logo", "logos": "Logo", "icon": "Logo", "icons": "Logo",
    "brand mark": "Logo", "brand marks": "Logo",
    "underlay": "Underlay", "underlays": "Underlay", "panel": "Underlay", "panels": "Underlay",
    "background panel": "Underlay", "background panels": "Underlay",
    "embellishment": "Embellishment", "embellishments": "Embellishment",
    "decoration": "Embellishment", "decorations": "Embellishment",
    "graphic element": "Embellishment", "graphic elements": "Embellishment",
}
# For fuzzy matching: all unique lowercase forms we accept
_CLASS_TOKENS = list(set(CLASS_SYNONYMS.keys()))
# Avoid matching "to" as number when it's part of "top" or "two"
WORD_TO_NUM.pop("to", None)
WORD_TO_NUM.pop("too", None)
WORD_TO_NUM["two"] = 2


def _normalize_token(t: str) -> str:
    return (t or "").strip().lower()


def _token_to_class(token: str, use_fuzzy: bool = True) -> Optional[str]:
    """Map a single word or phrase to canonical class name. If use_fuzzy, allow close spelling matches."""
    t = _normalize_token(token)
    if not t:
        return None
    if t in CLASS_SYNONYMS:
        return CLASS_SYNONYMS[t]
    if use_fuzzy and len(t) > 2:
        matches = get_close_matches(t, _CLASS_TOKENS, n=1, cutoff=0.75)
        if matches:
            return CLASS_SYNONYMS.get(matches[0])
    return None


def _parse_prompt_counts(prompt: str, use_fuzzy: bool = True) -> dict[str, int]:
    """
    Parse a prompt string into expected counts per class. Tolerates:
    - Synonyms (e.g. icon, text box, panel) and paraphrases from augmentation.
    - Number words and digits; 'a'/'an'/'single' as 1.
    - Small spelling errors via fuzzy matching when use_fuzzy=True.
    """
    prompt = (prompt or "").strip()
    counts = {name: 0 for name in CLASS_INDEX_TO_NAME.values()}
    if not prompt:
        return counts

    # Normalize and split into words (keep boundaries for phrase matching)
    prompt_lower = prompt.lower()
    words = re.findall(r"\b[\w']+\b", prompt_lower)

    i = 0
    while i < len(words):
        w = words[i]
        n_val = None
        if w.isdigit():
            n_val = int(w)
        elif w in WORD_TO_NUM:
            n_val = WORD_TO_NUM[w]
        if n_val is not None and i + 1 < len(words):
            # Next token(s) as class: single word or two-word phrase
            c = _token_to_class(words[i + 1], use_fuzzy)
            if c is None and i + 2 <= len(words):
                c = _token_to_class(" ".join(words[i + 1 : i + 3]), use_fuzzy)
            if c is not None:
                counts[c] += n_val
                # Skip number + class token(s). If class was two-word phrase, skip one more.
                if i + 2 <= len(words) and _token_to_class(" ".join(words[i + 1 : i + 3]), use_fuzzy) == c:
                    i += 3
                else:
                    i += 2
                continue
        i += 1

    # Free-form prompts often omit explicit counts but attach a class phrase to
    # a spatial location, e.g. "large title text at top-center" or "small logo
    # at bottom-left".  The text-spatial parser can recover these assignments;
    # use them as a lower-bound count without double-counting stricter
    # count-class matches already found above.
    for class_name, positions in parse_positions_from_prompt(prompt).items():
        if class_name in counts:
            counts[class_name] = max(counts[class_name], len(positions))

    return counts


def tla_cal(img_names, clses, cfg) -> float:
    """
    Text-Layout Alignment (TLA): measures how well generated layout counts match
    the counts described in the text prompt. Uses prompt CSV at cfg.paths.test.all_prompts;
    column 'prompt' or 'text_prompt' per poster_path. Higher is better (0-1).
    """
    if not getattr(cfg, "text_control", False):
        return float("nan")
    all_prompts_path = getattr(getattr(cfg, "paths", None), "test", None)
    if all_prompts_path is None or not hasattr(all_prompts_path, "all_prompts"):
        return float("nan")
    path = getattr(all_prompts_path, "all_prompts", None)
    if not path or not os.path.isfile(path):
        logger.log("TLA: all_prompts CSV not found, skipping TLA.")
        return float("nan")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.log(f"TLA: failed to load prompts CSV: {e}")
        return float("nan")

    prompt_col = "text_prompt" if "text_prompt" in df.columns else "prompt"
    if prompt_col not in df.columns:
        logger.log("TLA: no prompt/text_prompt column in CSV, skipping TLA.")
        return float("nan")

    # First prompt per image (poster_path = image filename)
    poster_to_prompt = df.groupby("poster_path")[prompt_col].first().to_dict()
    num_class = getattr(cfg, "num_class", 4)

    scores = []
    clses_np = clses.cpu().numpy()  # (N, max_elem, 1) or (N, max_elem)
    if clses_np.ndim == 3:
        clses_np = clses_np.squeeze(-1)

    for idx, name in enumerate(img_names):
        # Match image name (with or without path)
        key = name if name in poster_to_prompt else os.path.basename(name)
        prompt = poster_to_prompt.get(key)
        if prompt is None or not str(prompt).strip():
            # Missing/blank prompts are not interpretable requests and should
            # not enter the prompt-alignment denominator.
            continue
        prompt = str(prompt)
        expected = _parse_prompt_counts(prompt)
        # Predicted counts from layout (class indices 1..num_class)
        row = clses_np[idx]
        valid = row > 0
        pred_counts = {}
        for c in range(1, num_class + 1):
            cname = CLASS_INDEX_TO_NAME.get(c, f"Class{c}")
            pred_counts[cname] = int(np.sum((row == c) & valid))
        for cname in CLASS_INDEX_TO_NAME.values():
            if cname not in pred_counts:
                pred_counts[cname] = 0

        total_exp = sum(expected.values())
        total_pred = sum(pred_counts.values())
        if total_exp == 0 and total_pred == 0:
            scores.append(1.0)
            continue
        if total_exp == 0:
            # A parsed zero-element request must count as a failure when the
            # model generates anything.  Earlier code skipped this case, which
            # could inflate PLA on zero-request prompts.
            scores.append(0.0)
            continue
        if total_pred == 0:
            scores.append(0.0)
            continue
        # 1 - normalized L1 over counts; 1 when perfect match
        diff = sum(abs(expected.get(k, 0) - pred_counts.get(k, 0)) for k in CLASS_INDEX_TO_NAME.values())
        denom = total_exp + total_pred
        scores.append(1.0 - (diff / denom) if denom > 0 else 0.0)

    return float(np.mean(scores)) if scores else float("nan")


def _mean(values: list[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    else:
        return sum(values) / len(values)

def cvt_pilcv(img, req='pil2cv', color_code=None):
    if req == 'pil2cv':
        if color_code == None:
            color_code = cv2.COLOR_RGB2BGR
        dst = cv2.cvtColor(np.asarray(img), color_code)
    elif req == 'cv2pil':
        if color_code == None:
            color_code = cv2.COLOR_BGR2RGB
        dst = Image.fromarray(cv2.cvtColor(img, color_code))
    return dst

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    grad_xy = grad_xy / np.max(grad_xy) * 255
    img_g_xy = Image.fromarray(grad_xy).convert('L')
    return img_g_xy

def _extract_grad(image):
    image_npy = np.array(image * 255)
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)


def _list_all_pair_indices(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all pairs
    """
    N = bbox.shape[0]
    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]
    return ii, jj


def metrics_inter_oneside(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2

    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0

    l_max = np.maximum(xl_1, xl_2)
    r_min = np.minimum(xr_1, xr_2)
    t_max = np.maximum(yl_1, yr_1)
    b_min = np.minimum(yl_2, yr_2)
    cond = (l_max < r_min) & (t_max < b_min)

    a_i = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a_1))

    return a_inter / a_2

def _compute_iou_group(
    box_1: Union[np.ndarray, Tensor],
    box_2: Union[np.ndarray, Tensor],
    method: str = "iou",
    transform: bool = True,
) -> np.ndarray:
    assert method in ["iou", "giou", "ai/a1", "ai/a2"]

    if isinstance(box_1, Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    if transform:
        l1, t1, r1, b1 = box_1.T
        l2, t2, r2, b2 = box_2.T
    else:
        l1, t1, r1, b1 = box_1
        l2, t2, r2, b2 = box_2
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    if transform:
        ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))
    else:
        ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1))

    au = a1 + a2 - ai
    iou = ai / au

    if method == "iou":
        return iou
    elif method == "ai/a1":
        return ai / a1
    elif method == "ai/a2":
        return ai / a2

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou: np.ndarray = iou - (ac - au) / ac

    return giou

def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    c3 = xr_1 >= xr_2
    c4 = yr_1 >= yr_2
    return c1 and c2 and c3 and c4


def validity_cal(clses, boxes):
    mask = clses > 0
    valid_boxes = boxes[mask.squeeze(-1)]

    if valid_boxes.numel() == 0:
        return 0

    clamped_boxes = torch.clamp(valid_boxes, 0, 1)
    areas = (clamped_boxes[:, 2] - clamped_boxes[:, 0]) * (clamped_boxes[:, 3] - clamped_boxes[:, 1])

    empty_count = torch.sum(areas < 1e-3)
    return 1 - empty_count.float() / valid_boxes.shape[0]

# def validity_cal(clses, boxes):
#     total_elem = 0
#     empty_elem = 0
#     for cls, box in zip(clses, boxes):
#         mask = (cls > 0).reshape(-1)
#         mask_box = box[mask]
#         total_elem += len(mask_box)
#         for mb in mask_box:
#             xl, yl, xr, yr = mb
#             xl = max(0, xl)
#             yl = max(0, yl)
#             xr = min(1, xr)
#             yr = min(1, yr)
#             if abs((xr - xl) * (yr - yl)) < (1 / 1000):
#                 empty_elem += 1
#     if total_elem:
#         return 1 - empty_elem / total_elem
#     else:
#         return 0

def getRidOfInvalid(clses, boxes):
    clamped_boxes = torch.clamp(boxes, 0, 1)
    areas = (clamped_boxes[..., 2] - clamped_boxes[..., 0]) * (clamped_boxes[..., 3] - clamped_boxes[..., 1])
    invalid_mask = areas < 1e-3
    clses[invalid_mask] = 0
    return clses
# def getRidOfInvalid(clses, boxes):
#     for i, (cls, box) in enumerate(zip(clses, boxes)):
#         for j, b in enumerate(box):
#             xl, yl, xr, yr = b
#             xl = max(0, xl)
#             yl = max(0, yl)
#             xr = min(1, xr)
#             yr = min(1, yr)
#             if abs((xr - xl) * (yr - yl)) < (1 / 1000):
#                 if clses[i, j]:
#                     clses[i, j] = 0
#     return clses


def overlap_cal(clses, boxes):
    """
    Ratio of overlapping area.
    Lower is better.
    """
    metrics = []
    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        if n in [0, 1]:
            continue
        ii, jj = _list_all_pair_indices(mask_box)
        iou: np.ndarray = _compute_iou_group(mask_box[ii], mask_box[jj], method="iou", transform=True)
        result: float = iou.mean().item()
        metrics.append(result)
    return np.mean(np.array(metrics))


def underlay_cal(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metric_l = []
    metric_s = []
    thresh = 1.0 - np.finfo(np.float32).eps

    for cls, box in zip(clses, boxes):
        mask_und = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_und = box[mask_und]
        box_other = box[mask_other]
        n1 = len(box_und)
        n2 = len(box_other)
        if n1:
            for i in range(n1):
                max_iou = 0
                bb1 = box_und[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    # ios = metrics_inter_oneside(bb1, bb2)
                    ios = _compute_iou_group(bb1, bb2, method="ai/a2", transform=False)
                    max_iou = max(max_iou, ios)
                strict_score = (max_iou >= thresh).any().astype(np.float32)
                metric_l.append(max_iou)
                metric_s.append(strict_score)

    return np.mean(np.array(metric_l)), np.mean(np.array(metric_s))


def utilization_cal(img_names, clses, boxes, cfg):
    metric = 0
    img_size = (cfg.width, cfg.height)
    for idx, name in enumerate(img_names):
        sal_1 = np.array(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L"))
        sal_2 = np.array(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L"))
        sal_map = Image.fromarray(np.maximum(sal_1, sal_2))
        sal_map = to_tensor(sal_map.resize(img_size))
        sal_map = rearrange(sal_map, "1 h w ->h w")
        inv_saliency = 1.0 - sal_map

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        cal_mask = torch.zeros_like(sal_map)
        # cal_mask[mask_box[:, 1]:mask_box[:, 3], mask_box[:, 0]:mask_box[:, 2]] = True
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        numerator = torch.sum(inv_saliency * cal_mask)
        denominator = torch.sum(inv_saliency)
        assert denominator > 0.0
        metric += (numerator / denominator).item()
    return metric / len(img_names)

def occlusion_cal(img_names, clses, boxes, cfg):
    '''
    Average saliency of the pixels covered.
    Lower is better.
    '''
    metric = 0
    img_size = (cfg.width, cfg.height)

    for idx, name in enumerate(img_names):
        sal_1 = np.array(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L"))
        sal_2 = np.array(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L"))
        sal_map = Image.fromarray(np.maximum(sal_1, sal_2))
        sal_map = to_tensor(sal_map.resize(img_size))
        sal_map = rearrange(sal_map, "1 h w ->h w")

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        cal_mask = torch.zeros_like(sal_map)

        # cal_mask[mask_box[:, 1]:mask_box[:, 3], mask_box[:, 0]:mask_box[:, 2]] = True
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        occlusion = sal_map[cal_mask.bool()]
        if len(occlusion) != 0:
            metric += occlusion.mean().item()

    return metric / len(img_names)

def unreadability_cal(img_names, clses, boxes, cfg):
    metrics = []
    img_size = (cfg.width, cfg.height)

    for idx, name in enumerate(img_names):
        image = to_tensor(Image.open(os.path.join(cfg.paths.test.inp_dir, name)).convert("RGB").resize(img_size))
        image = rearrange(image, "c h w ->h w c")

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        bbox_mask_special = torch.zeros(cfg.height, cfg.width)
        text_mask = (cls == 1).reshape(-1)
        text_boxes = box[text_mask]
        # if text_boxes.numel() > 0:
            # bbox_mask_special[text_boxes[:, 1]:text_boxes[:, 3], text_boxes[:, 0]:text_boxes[:, 2]] = True
        for mb in text_boxes:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 1
        underlay_mask = (cls == 3).reshape(-1)
        underlay_boxes = box[underlay_mask]
        # if underlay_boxes.numel() > 0:
        #     bbox_mask_special[underlay_boxes[:, 1]:underlay_boxes[:, 3],
        #     underlay_boxes[:, 0]:underlay_boxes[:, 2]] = False
        for mb in underlay_boxes:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 0
        g_xy = _extract_grad(image)
        unreadability = g_xy[bbox_mask_special.bool()]

        metrics.append(unreadability.mean().item() if unreadability.numel() > 0 else 0.0)

    return np.mean(np.array(metrics))

def _prompt_count_records(img_names, clses, cfg):
    """Per-image count precision/recall/F1 against the active prompt file."""
    prompt_path = getattr(getattr(cfg.paths, "test", None), "all_prompts", "")
    if not prompt_path or not os.path.isfile(prompt_path):
        return []
    frame = pd.read_csv(prompt_path)
    prompt_col = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    if "poster_path" not in frame.columns or prompt_col not in frame.columns:
        return []
    prompts = frame.groupby("poster_path")[prompt_col].first().to_dict()
    class_array = clses.squeeze(-1).detach().cpu().numpy()
    records = []
    for image_name, predicted in zip(img_names, class_array):
        key = image_name if image_name in prompts else os.path.basename(str(image_name))
        prompt = prompts.get(key)
        if prompt is None or not str(prompt).strip():
            continue
        expected = _parse_prompt_counts(str(prompt))
        expected_total = sum(expected.values())
        pred_counts = {
            name: int(np.sum(predicted == class_id))
            for class_id, name in CLASS_INDEX_TO_NAME.items()
            if class_id < int(cfg.num_class)
        }
        predicted_total = sum(pred_counts.values())
        true_positive = sum(
            min(expected.get(name, 0), pred_counts.get(name, 0))
            for name in CLASS_INDEX_TO_NAME.values()
        )
        precision = true_positive / predicted_total if predicted_total else (
            1.0 if expected_total == 0 else 0.0
        )
        recall = true_positive / expected_total if expected_total else (
            1.0 if predicted_total == 0 else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        diff = sum(
            abs(expected.get(name, 0) - pred_counts.get(name, 0))
            for name in CLASS_INDEX_TO_NAME.values()
        )
        denominator = expected_total + predicted_total
        similarity = 1.0 - diff / denominator if denominator else 1.0
        exact_count_match = all(
            expected.get(name, 0) == pred_counts.get(name, 0)
            for name in CLASS_INDEX_TO_NAME.values()
        )
        records.append(
            {
                "image": os.path.basename(str(image_name)),
                "count_precision": float(precision),
                "count_recall": float(recall),
                "count_f1": float(f1),
                "pla_count": float(similarity),
                "exact_count_match": float(exact_count_match),
            }
        )
    return records


def metric(img_names, test_output, cfg, ground_truth=None, return_records=False):
    """Compute aggregate and optionally per-image benchmark measurements."""
    logger.log("Calculating protocol-aware metrics...")
    geometry = geometry_records(img_names, test_output, ground_truth=ground_truth)
    content = content_records(img_names, test_output, cfg)
    records = merge_records(geometry, content)
    clses = test_output[:, :, :1]

    if getattr(cfg, 'text_control', False):
        count_records = _prompt_count_records(img_names, clses, cfg)
        if count_records:
            records = merge_records(records, count_records)

    # Backward-compatible alias. The explicit name prevents it being mistaken for
    # complete natural-language alignment.
    if getattr(cfg, 'text_control', False):
        if getattr(cfg, 'spatial_metrics', False):
            boxes = torch.clamp(
                box_cxcywh_to_xyxy(test_output[:, :, 1:]), 0.0, 1.0
            )
            spatial_records = spatial_prompt_records(img_names, clses, boxes, cfg)
            if spatial_records:
                records = merge_records(records, spatial_records)

    metrics = aggregate_records(records)
    if ground_truth is not None and len(ground_truth) >= 2:
        metrics['hfd'] = diagnostic_layout_fd(
            test_output, ground_truth, int(cfg.num_class)
        )
    if getattr(cfg, 'text_control', False):
        tla_val = tla_cal(img_names, clses, cfg)
        if not np.isnan(tla_val):
            metrics['tla'] = tla_val
            metrics['pla_count'] = tla_val
        if getattr(cfg, 'spatial_metrics', False):
            metrics.update(spatial_pla_cal(img_names, clses, boxes, cfg))

    for key, value in metrics.items():
        logger.log(f"{key}:{value:.6f}")

    if return_records:
        return metrics, records
    return metrics
