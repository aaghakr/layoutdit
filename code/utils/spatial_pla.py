"""Spatial prompt-adherence metrics on a 3x3 canvas grid."""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from cgbdm.text_spatial import get_class_map, parse_positions_from_prompt


GRID_ROWS = ("top", "middle", "bottom")
GRID_COLS = ("left", "center", "right")

_ENTITY = r"(text|logo|underlay|embellishment|decoration|panel|brand mark)(?:[_\s-]?(\d+))?"
_RELATION = r"(above|below|left of|right of|inside|within|contains|overlaps)"
_RELATION_PATTERN = re.compile(
    rf"{_ENTITY}\s+(?:is\s+|should\s+be\s+|needs?\s+to\s+be\s+)?{_RELATION}\s+{_ENTITY}",
    re.IGNORECASE,
)
_CANONICAL = {
    "text": "Text", "logo": "Logo", "brand mark": "Logo",
    "underlay": "Underlay", "panel": "Underlay",
    "embellishment": "Embellishment", "decoration": "Embellishment",
}


def _cell(x: float, y: float) -> str:
    col = min(2, max(0, int(x * 3)))
    row = min(2, max(0, int(y * 3)))
    return f"{GRID_ROWS[row]}-{GRID_COLS[col]}"


def parse_relations_from_prompt(prompt: str) -> list[tuple[str, int, str, str, int]]:
    relations = []
    for match in _RELATION_PATTERN.finditer(prompt or ""):
        left_name, left_index, relation, right_name, right_index = match.groups()
        relation = "inside" if relation.lower() == "within" else relation.lower()
        relations.append(
            (
                _CANONICAL[left_name.lower()], int(left_index or 0), relation,
                _CANONICAL[right_name.lower()], int(right_index or 0),
            )
        )
    return relations


def _relation_holds(left: np.ndarray, relation: str, right: np.ndarray) -> bool:
    left_center = (left[:2] + left[2:]) / 2
    right_center = (right[:2] + right[2:]) / 2
    if relation == "above":
        return bool(left_center[1] < right_center[1])
    if relation == "below":
        return bool(left_center[1] > right_center[1])
    if relation == "left of":
        return bool(left_center[0] < right_center[0])
    if relation == "right of":
        return bool(left_center[0] > right_center[0])
    if relation == "inside":
        return bool(np.all(left[:2] >= right[:2]) and np.all(left[2:] <= right[2:]))
    if relation == "contains":
        return bool(np.all(right[:2] >= left[:2]) and np.all(right[2:] <= left[2:]))
    if relation == "overlaps":
        intersection = np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2])
        return bool(np.all(intersection > 0))
    return False


def spatial_prompt_records(img_names, clses, boxes_xyxy, cfg) -> list[dict]:
    """Return per-image absolute-cell and pairwise-relation adherence."""
    prompt_path = getattr(getattr(cfg.paths, "test", None), "all_prompts", "")
    if not prompt_path or not os.path.isfile(prompt_path):
        return []
    frame = pd.read_csv(prompt_path)
    prompt_col = "text_prompt" if "text_prompt" in frame.columns else "prompt"
    if "poster_path" not in frame.columns or prompt_col not in frame.columns:
        return []
    prompts = frame.groupby("poster_path")[prompt_col].first().to_dict()
    index_to_name = {index: name for name, index in get_class_map(int(cfg.num_class)).items()}
    class_array = clses.squeeze(-1).detach().cpu().numpy()
    box_array = boxes_xyxy.detach().cpu().numpy()
    records = []
    for sample_index, image_name in enumerate(img_names):
        key = image_name if image_name in prompts else os.path.basename(str(image_name))
        prompt = str(prompts.get(key, ""))
        expected = parse_positions_from_prompt(prompt)
        relations = parse_relations_from_prompt(prompt)
        predicted = defaultdict(list)
        for class_index, box in zip(class_array[sample_index], box_array[sample_index]):
            class_name = index_to_name.get(int(class_index))
            if class_name is not None:
                predicted[class_name].append(np.asarray(box, dtype=np.float64))

        absolute_requested = absolute_matched = 0
        class_requested = defaultdict(int)
        class_matched = defaultdict(int)
        for class_name, expected_cells in expected.items():
            expected_counts = Counter(expected_cells)
            predicted_counts = Counter(
                _cell(float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                for box in predicted.get(class_name, [])
            )
            hits = sum(
                min(count, predicted_counts.get(cell, 0))
                for cell, count in expected_counts.items()
            )
            absolute_matched += hits
            absolute_requested += len(expected_cells)
            class_matched[class_name] += hits
            class_requested[class_name] += len(expected_cells)

        relation_hits = hierarchy_hits = hierarchy_total = 0
        evaluable_relations = 0
        seen = set()
        conflict = False
        opposite = {"above": "below", "below": "above", "left of": "right of", "right of": "left of"}
        for left_name, left_index, relation, right_name, right_index in relations:
            signature = (left_name, left_index, right_name, right_index)
            if (signature, opposite.get(relation)) in seen:
                conflict = True
            seen.add((signature, relation))
            left_boxes = predicted.get(left_name, [])
            right_boxes = predicted.get(right_name, [])
            if left_index >= len(left_boxes) or right_index >= len(right_boxes):
                continue
            holds = _relation_holds(left_boxes[left_index], relation, right_boxes[right_index])
            relation_hits += int(holds)
            evaluable_relations += 1
            if relation in {"inside", "contains"}:
                hierarchy_hits += int(holds)
                hierarchy_total += 1

        record = {
                "image": os.path.basename(str(image_name)),
                "spla": absolute_matched / absolute_requested if absolute_requested else float("nan"),
                "spla_matched": float(absolute_matched),
                "spla_requested": float(absolute_requested),
                "relation_satisfaction": relation_hits / evaluable_relations if evaluable_relations else float("nan"),
                "relation_matched": float(relation_hits),
                "relation_requested": float(len(relations)),
                "relation_evaluable": float(evaluable_relations),
                "hierarchy_satisfaction": hierarchy_hits / hierarchy_total if hierarchy_total else float("nan"),
                "hierarchy_requested": float(hierarchy_total),
                "prompt_conflict": float(conflict),
            }
        for class_name in get_class_map(int(cfg.num_class)):
            key = class_name.lower()
            record[f"spla_{key}_matched"] = float(class_matched[class_name])
            record[f"spla_{key}_requested"] = float(class_requested[class_name])
        records.append(record)
    return records


def spatial_pla_cal(
    img_names: list[str],
    clses: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    cfg,
) -> dict[str, float]:
    """Return requested-position recall overall and per element class.

    Count/type mismatches are reported separately by PLA. Here, each requested
    class/cell occurrence is matched against generated elements of that class.
    """
    records = spatial_prompt_records(img_names, clses, boxes_xyxy, cfg)
    if not records:
        return {}
    absolute_requested = sum(r["spla_requested"] for r in records)
    relation_evaluable = sum(r["relation_evaluable"] for r in records)
    hierarchy_requested = sum(r["hierarchy_requested"] for r in records)
    result = {
        "prompt_conflict_rate": float(np.mean([r["prompt_conflict"] for r in records])),
    }
    if absolute_requested:
        result.update({
            "spla": float(sum(r["spla_matched"] for r in records) / absolute_requested),
            "spla_n": float(sum(r["spla_requested"] > 0 for r in records)),
            "spla_requests": float(absolute_requested),
        })
    if relation_evaluable:
        result.update({
            "relation_satisfaction": float(sum(r["relation_matched"] for r in records) / relation_evaluable),
            "relation_n": float(relation_evaluable),
        })
    if hierarchy_requested:
        result.update({
            "hierarchy_satisfaction": float(sum(r["hierarchy_satisfaction"] * r["hierarchy_requested"] for r in records if np.isfinite(r["hierarchy_satisfaction"])) / hierarchy_requested),
            "hierarchy_n": float(hierarchy_requested),
        })
    for class_name in get_class_map(int(cfg.num_class)):
        key = class_name.lower()
        requested = sum(r[f"spla_{key}_requested"] for r in records)
        if requested:
            result[f"spla_{key}"] = float(
                sum(r[f"spla_{key}_matched"] for r in records) / requested
            )
    return result
