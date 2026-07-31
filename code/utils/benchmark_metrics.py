"""Protocol-aware metrics for content-aware poster layout generation.

The functions in this module are deterministic and operate on normalized XYXY
boxes.  They intentionally distinguish paired ground-truth IoU from the
distributional MaxIoU used by generic layout papers.  The latter requires an
official feature extractor/matching protocol and must not be approximated under
the same name.
"""

from __future__ import annotations

import ast
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment

from utils.metric_other import Alignment_ralf
from utils.util import box_cxcywh_to_xyxy


UNDERLAY_CLASS = 3


def safe_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def decode_layout_tensor(test_output: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return classes, unclamped XYXY boxes, and clamped XYXY boxes."""
    classes = test_output[..., 0].detach().cpu().numpy().astype(np.int64)
    boxes_cxcywh = test_output[..., 1:].detach().cpu()
    boxes_raw = box_cxcywh_to_xyxy(boxes_cxcywh).numpy().astype(np.float64)
    boxes = np.clip(boxes_raw, 0.0, 1.0)
    return classes, boxes_raw, boxes


def _areas(boxes: np.ndarray) -> np.ndarray:
    wh = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2])
    return wh[:, 0] * wh[:, 1]


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    union = _areas(a)[:, None] + _areas(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _alignment(classes: np.ndarray, boxes: np.ndarray) -> float:
    valid = classes > 0
    if valid.sum() <= 1:
        return 0.0
    cls_t = torch.from_numpy(classes[None, :, None])
    box_t = torch.from_numpy(boxes[None]).float()
    return float(Alignment_ralf((1, 1), cls_t, box_t))


def _overlap(classes: np.ndarray, boxes: np.ndarray) -> float:
    selected = boxes[(classes > 0) & (classes != UNDERLAY_CLASS)]
    if len(selected) <= 1:
        return 0.0
    matrix = _pairwise_iou(selected, selected)
    return float(matrix[~np.eye(len(selected), dtype=bool)].mean())


def _underlay(classes: np.ndarray, boxes: np.ndarray) -> tuple[float, float]:
    underlays = boxes[classes == UNDERLAY_CLASS]
    foreground = boxes[(classes > 0) & (classes != UNDERLAY_CLASS)]
    if len(underlays) == 0:
        return float("nan"), float("nan")
    scores = []
    for underlay in underlays:
        if len(foreground) == 0:
            scores.append(0.0)
            continue
        lt = np.maximum(underlay[None, :2], foreground[:, :2])
        rb = np.minimum(underlay[None, 2:], foreground[:, 2:])
        wh = np.maximum(0.0, rb - lt)
        intersection = wh[:, 0] * wh[:, 1]
        foreground_area = _areas(foreground)
        containment = np.divide(
            intersection,
            foreground_area,
            out=np.zeros_like(intersection),
            where=foreground_area > 0,
        )
        scores.append(float(containment.max(initial=0.0)))
    return float(np.mean(scores)), float(np.mean(np.asarray(scores) >= 1.0 - 1e-6))


def _visual_balance(classes: np.ndarray, boxes: np.ndarray) -> float:
    selected = boxes[classes > 0]
    if len(selected) == 0:
        return 1.0
    area = _areas(selected)
    centers = (selected[:, :2] + selected[:, 2:]) / 2
    if area.sum() <= 0:
        return 1.0
    center_of_mass = np.average(centers, axis=0, weights=area)
    return float(np.linalg.norm(center_of_mass - 0.5) / math.sqrt(0.5))


def _spacing_cv(classes: np.ndarray, boxes: np.ndarray) -> float:
    selected = boxes[(classes > 0) & (classes != UNDERLAY_CLASS)]
    if len(selected) <= 2:
        return 0.0
    centers = (selected[:, :2] + selected[:, 2:]) / 2
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    mean = nearest.mean()
    return float(nearest.std() / mean) if mean > 0 else 0.0


def _matched_iou(
    pred_classes: np.ndarray,
    pred_boxes: np.ndarray,
    gt_classes: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[float, int]:
    """Hungarian IoU for paired samples, matched only within the same class."""
    matched = []
    for class_id in sorted(set(gt_classes.tolist()) | set(pred_classes.tolist())):
        if class_id <= 0:
            continue
        pred = pred_boxes[pred_classes == class_id]
        gt = gt_boxes[gt_classes == class_id]
        if len(pred) == 0 or len(gt) == 0:
            continue
        iou = _pairwise_iou(pred, gt)
        row, col = linear_sum_assignment(1.0 - iou)
        matched.extend(iou[row, col].tolist())
    return (float(np.mean(matched)), len(matched)) if matched else (0.0, 0)


def _count_scores(pred_classes: np.ndarray, gt_classes: np.ndarray) -> tuple[float, float, float]:
    pred = Counter(int(x) for x in pred_classes if x > 0)
    gt = Counter(int(x) for x in gt_classes if x > 0)
    true_positive = sum(min(pred[key], gt[key]) for key in set(pred) | set(gt))
    n_pred = sum(pred.values())
    n_gt = sum(gt.values())
    precision = true_positive / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    recall = true_positive / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def geometry_records(
    img_names: Sequence[str],
    test_output: torch.Tensor,
    ground_truth: Sequence[dict] | None = None,
) -> list[dict[str, float | str]]:
    classes, raw_boxes, boxes = decode_layout_tensor(test_output)
    records: list[dict[str, float | str]] = []
    for index, image_name in enumerate(img_names):
        cls = classes[index]
        raw = raw_boxes[index]
        box = boxes[index]
        present = cls > 0
        present_raw = raw[present]
        present_boxes = box[present]
        present_classes = cls[present]
        areas = _areas(present_boxes)
        total_pred_area = float(areas.sum()) if len(areas) else 0.0
        mean_pred_area = float(areas.mean()) if len(areas) else 0.0
        median_pred_area = float(np.median(areas)) if len(areas) else 0.0

        valid = areas >= 1e-3
        oob = np.any((present_raw < 0.0) | (present_raw > 1.0), axis=1) if len(present_raw) else np.array([])
        widths = present_boxes[:, 2] - present_boxes[:, 0] if len(present_boxes) else np.array([])
        heights = present_boxes[:, 3] - present_boxes[:, 1] if len(present_boxes) else np.array([])
        small = (areas < 1e-3) | (widths < 0.02) | (heights < 0.02) if len(areas) else np.array([])
        undl, unds = _underlay(present_classes, present_boxes)
        record: dict[str, float | str] = {
            "image": os.path.basename(str(image_name)),
            "val": float(valid.mean()) if len(valid) else 0.0,
            "oob": float(oob.mean()) if len(oob) else 0.0,
            "sma": float(small.mean()) if len(small) else 0.0,
            "ali": _alignment(cls, box),
            "ove": _overlap(cls, box),
            "undl": undl,
            "unds": unds,
            "vb": _visual_balance(cls, box),
            "spacing_cv": _spacing_cv(cls, box),
            "empty_layout": float(present.sum() == 0),
            "format_failure": 0.0,
            "n_pred": float(present.sum()),
            "total_pred_area": total_pred_area,
            "mean_pred_area": mean_pred_area,
            "median_pred_area": median_pred_area,
        }
        if ground_truth is not None:
            gt = ground_truth[index]
            gt_cls = np.asarray(gt["classes"], dtype=np.int64)
            gt_box = np.asarray(gt["boxes"], dtype=np.float64).reshape(-1, 4)
            gt_areas = _areas(gt_box) if len(gt_box) else np.array([])
            paired_iou, n_matched = _matched_iou(
                present_classes, present_boxes, gt_cls, gt_box
            )
            precision, recall, f1 = _count_scores(present_classes, gt_cls)
            n_pred = int(present.sum())
            n_gt = int(len(gt_cls))
            record.update(
                {
                    "paired_iou": paired_iou,
                    "n_matched": float(n_matched),
                    "type_precision": precision,
                    "type_recall": recall,
                    "type_f1": f1,
                    "n_gt": float(n_gt),
                    "undercount": float(n_pred < n_gt),
                    "severe_undercount": float(n_gt >= 2 and n_pred <= n_gt - 2),
                    "count_error": float(n_pred - n_gt),
                    "abs_count_error": float(abs(n_pred - n_gt)),
                    "total_gt_area": float(gt_areas.sum()) if len(gt_areas) else 0.0,
                    "mean_gt_area": float(gt_areas.mean()) if len(gt_areas) else 0.0,
                    "median_gt_area": float(np.median(gt_areas)) if len(gt_areas) else 0.0,
                }
            )
        records.append(record)
    if ground_truth is not None:
        # LayoutDM/LayoutGAN++ style conditional maximum IoU: compare only layouts
        # with an identical class multiset and retain the most similar real layout.
        candidates: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
        for item in ground_truth:
            gt_cls = np.asarray(item["classes"], dtype=np.int64)
            gt_box = np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 4)
            signature = tuple(sorted(int(value) for value in gt_cls if value > 0))
            candidates.setdefault(signature, []).append((gt_cls, gt_box))
        for index, record in enumerate(records):
            pred_cls = classes[index][classes[index] > 0]
            pred_box = boxes[index][classes[index] > 0]
            signature = tuple(sorted(int(value) for value in pred_cls))
            scores = [
                _matched_iou(pred_cls, pred_box, gt_cls, gt_box)[0]
                for gt_cls, gt_box in candidates.get(signature, [])
            ]
            record["max_iou"] = float(max(scores)) if scores else float("nan")
            record["max_iou_covered"] = float(bool(scores))
    return records


def load_ground_truth_layouts(img_names: Sequence[str], cfg) -> list[dict]:
    """Load paired annotated layouts in normalized XYXY form."""
    annotation_path = Path(cfg.paths.test.annotated_dir)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Ground-truth annotations not found: {annotation_path}")
    frame = pd.read_csv(annotation_path)
    required = {"poster_path", "box_elem", "cls_elem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{annotation_path} is missing columns: {sorted(missing)}")
    groups = frame.groupby(frame.poster_path)
    result = []
    for image_name in img_names:
        key = os.path.basename(str(image_name))
        if key not in groups.groups:
            result.append({"classes": np.empty(0, dtype=np.int64), "boxes": np.empty((0, 4))})
            continue
        rows = groups.get_group(key).iloc[: int(cfg.max_elem)]
        classes = rows.cls_elem.astype(int).to_numpy()
        parsed = np.asarray([ast.literal_eval(str(value)) for value in rows.box_elem], dtype=np.float64)
        if parsed.size == 0:
            boxes = np.empty((0, 4), dtype=np.float64)
        else:
            boxes = parsed.reshape(-1, 4)
            boxes[:, [0, 2]] /= float(cfg.width)
            boxes[:, [1, 3]] /= float(cfg.height)
            boxes[:, [0, 2]] = np.sort(boxes[:, [0, 2]], axis=1)
            boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)
            boxes = np.clip(boxes, 0.0, 1.0)
        result.append({"classes": classes, "boxes": boxes})
    return result


def content_records(
    img_names: Sequence[str],
    test_output: torch.Tensor,
    cfg,
) -> list[dict[str, float | str]]:
    """Per-image Uti/Occ/Rea with stable handling of empty and flat regions."""
    classes, _, boxes = decode_layout_tensor(test_output)
    records = []
    target_size = (int(cfg.width), int(cfg.height))
    for index, image_name in enumerate(img_names):
        name = os.path.basename(str(image_name))
        sal_a = np.asarray(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L").resize(target_size), dtype=np.float64) / 255.0
        sal_b = np.asarray(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L").resize(target_size), dtype=np.float64) / 255.0
        saliency = np.maximum(sal_a, sal_b)
        image = np.asarray(Image.open(os.path.join(cfg.paths.test.inp_dir, name)).convert("RGB").resize(target_size), dtype=np.uint8)
        gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float64)
        gy, gx = np.gradient(gray)
        gradient = np.sqrt(gx * gx + gy * gy)
        max_gradient = float(gradient.max(initial=0.0))
        if max_gradient > 0:
            gradient /= max_gradient

        cls = classes[index]
        pixel_boxes = boxes[index].copy()
        pixel_boxes[:, [0, 2]] *= target_size[0]
        pixel_boxes[:, [1, 3]] *= target_size[1]
        pixel_boxes = np.rint(pixel_boxes).astype(int)
        layout_mask = np.zeros(saliency.shape, dtype=bool)
        text_mask = np.zeros(saliency.shape, dtype=bool)
        for class_id, (x1, y1, x2, y2) in zip(cls, pixel_boxes):
            if class_id <= 0 or x2 <= x1 or y2 <= y1:
                continue
            layout_mask[y1:y2, x1:x2] = True
            if class_id == 1:
                text_mask[y1:y2, x1:x2] = True
        for class_id, (x1, y1, x2, y2) in zip(cls, pixel_boxes):
            if class_id == UNDERLAY_CLASS and x2 > x1 and y2 > y1:
                text_mask[y1:y2, x1:x2] = False

        occ = float(saliency[layout_mask].mean()) if layout_mask.any() else 0.0
        rea = float(gradient[text_mask].mean()) if text_mask.any() else 0.0
        non_salient = 1.0 - saliency
        denominator = float(non_salient.sum())
        uti = float((non_salient * layout_mask).sum() / denominator) if denominator > 0 else 0.0
        records.append({"image": name, "uti": uti, "occ": occ, "rea": rea})
        records[-1]["saliency_mass"] = float(saliency.mean())
    return records


def merge_records(*record_sets: Sequence[dict]) -> list[dict]:
    if not record_sets:
        return []
    merged = [dict(record) for record in record_sets[0]]
    for records in record_sets[1:]:
        if len(records) != len(merged):
            raise ValueError("Per-image metric record lengths differ")
        for destination, source in zip(merged, records):
            if destination.get("image") != source.get("image"):
                raise ValueError("Per-image metric order differs")
            destination.update({key: value for key, value in source.items() if key != "image"})
    return merged


def aggregate_records(records: Sequence[dict]) -> dict[str, float]:
    keys = sorted({key for record in records for key in record if key != "image"})
    result = {}
    for key in keys:
        value = safe_mean(record[key] for record in records if key in record)
        if math.isfinite(value):
            result[key] = value
    return result


def handcrafted_layout_features(classes: np.ndarray, boxes: np.ndarray, num_class: int) -> np.ndarray:
    """Permutation-invariant diagnostic features; not the standard layout-FID encoder."""
    features = []
    for class_id in range(1, num_class):
        selected = boxes[classes == class_id]
        if len(selected):
            cx = (selected[:, 0] + selected[:, 2]) / 2
            cy = (selected[:, 1] + selected[:, 3]) / 2
            width = selected[:, 2] - selected[:, 0]
            height = selected[:, 3] - selected[:, 1]
            values = np.stack([cx, cy, width, height], axis=1)
            features.extend([len(selected) / 16.0, *values.mean(0), *values.std(0)])
        else:
            features.extend([0.0] * 9)
    return np.asarray(features, dtype=np.float64)


def diagnostic_layout_fd(
    test_output: torch.Tensor,
    ground_truth: Sequence[dict],
    num_class: int,
) -> float:
    """Fréchet distance over documented handcrafted features.

    This is reported as ``hfd`` and must never be presented as standard layout FID.
    """
    classes, _, boxes = decode_layout_tensor(test_output)
    pred_features = np.stack(
        [handcrafted_layout_features(cls, box, num_class) for cls, box in zip(classes, boxes)]
    )
    gt_features = np.stack(
        [
            handcrafted_layout_features(
                np.asarray(item["classes"], dtype=np.int64),
                np.asarray(item["boxes"], dtype=np.float64).reshape(-1, 4),
                num_class,
            )
            for item in ground_truth
        ]
    )
    mu_pred, mu_gt = pred_features.mean(0), gt_features.mean(0)
    cov_pred = np.cov(pred_features, rowvar=False)
    cov_gt = np.cov(gt_features, rowvar=False)
    epsilon = np.eye(cov_pred.shape[0]) * 1e-8
    cov_pred = cov_pred + epsilon
    cov_gt = cov_gt + epsilon
    covariance_mean = sqrtm(cov_pred @ cov_gt)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    delta = mu_pred - mu_gt
    value = delta @ delta + np.trace(cov_pred + cov_gt - 2 * covariance_mean)
    return float(max(value, 0.0))
