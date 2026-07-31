"""Convert PosterO result .pt files to the shared baseline tensor format.

PosterO stores multiple SVG/layout candidates per image.  This script selects
one candidate per image, maps class names to this project's class IDs, maps CGL
original filenames back to numeric ``poster_path`` IDs when needed, and writes:

    {"img_names": [...], "test_output": Tensor[N, max_elem, 5], ...}

The output can be passed directly to ``scripts/paper/11_evaluate_external_baselines.sh``
as ``TEXT_BASELINE_*_PREDICTIONS``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch


CLASS_ID = {
    "text": 1,
    "logo": 2,
    "underlay": 3,
    "embellishment": 4,
}


def basename(value: str) -> str:
    return os.path.basename(str(value))


def stem(value: str) -> str:
    return Path(basename(value)).stem


def load_name_mapping(path: Optional[str]) -> dict[str, str]:
    """Return file-name stem -> poster_path mapping from a dataset CSV."""
    if not path:
        return {}
    frame_path = Path(path)
    if not frame_path.is_file():
        raise FileNotFoundError(f"Mapping CSV not found: {frame_path}")

    mapping: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    with frame_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "file_name" not in reader.fieldnames or "poster_path" not in reader.fieldnames:
            return {}
        for row in reader:
            file_name = row.get("file_name")
            poster_path = row.get("poster_path")
            if not file_name or not poster_path:
                continue
            key = stem(file_name)
            value = basename(poster_path)
            if key in mapping and mapping[key] != value:
                duplicates.setdefault(key, {mapping[key]}).add(value)
            mapping[key] = value
    if duplicates:
        examples = {k: sorted(v) for k, v in list(duplicates.items())[:5]}
        raise ValueError(f"Ambiguous file_name->poster_path mapping: {examples}")
    return mapping


def load_subset(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    subset_path = Path(path)
    if not subset_path.is_file():
        raise FileNotFoundError(f"Subset CSV not found: {subset_path}")
    with subset_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "poster_path" not in reader.fieldnames:
            raise ValueError(f"{subset_path} must contain a poster_path column")
        names = {basename(row["poster_path"]) for row in reader if row.get("poster_path")}
    if not names:
        raise ValueError(f"{subset_path} contains no poster_path values")
    return names


def choose_split(payload: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        if requested not in payload:
            raise KeyError(f"Split {requested!r} not found. Available: {sorted(payload)}")
        return requested
    # PosterO uses "valid" for annotated test in its released code.  Prefer it
    # when present, otherwise fall back to test.
    for candidate in ("valid", "test"):
        if candidate in payload:
            return candidate
    raise KeyError(f"No valid/test split found. Available: {sorted(payload)}")


def official_select(
    entries: list[dict[str, Any]],
    dataset: str,
    postero_code_root: Path,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Use PosterO's own ranker over its 10 generated candidates."""
    sys.path.insert(0, str(postero_code_root.resolve()))
    try:
        from eval import gather_pSplit_N  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external baseline env
        raise RuntimeError(
            "Could not import PosterO eval.py for official candidate selection. "
            "Use --selection first or install PosterO dependencies."
        ) from exc

    label_id = dict(CLASS_ID)
    if dataset == "pku":
        label_id.pop("embellishment", None)
    weight = {"overlay": 50, "alignment-LayoutGAN++": 20, "density_score": -1}
    return gather_pSplit_N(copy.deepcopy(entries), label_id, (width, height), weight, N=10)


def first_select(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    selected = []
    indices = []
    for entry in entries:
        layouts = entry.get("generated", {}).get("layout", [])
        indices.append(0)
        selected.append(layout_to_record(layouts[0] if layouts else {}))
    return selected, indices


def layout_to_record(layout: dict[str, Any]) -> dict[str, torch.Tensor]:
    labels = []
    boxes = []
    for cls_name, box in zip(layout.get("cls_elem", []), layout.get("box_elem", [])):
        cls_key = str(cls_name).strip().lower()
        if cls_key == "canvas":
            continue
        if cls_key not in CLASS_ID or len(box) != 4:
            continue
        x, y, w, h = [float(value) for value in box]
        labels.append(float(CLASS_ID[cls_key]))
        boxes.append([x, y, w, h])
    if not labels:
        labels = [0.0]
        boxes = [[0.0, 0.0, 0.0, 0.0]]
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    box_tensor[:, 0] = box_tensor[:, 0] + 0.5 * box_tensor[:, 2]
    box_tensor[:, 1] = box_tensor[:, 1] + 0.5 * box_tensor[:, 3]
    box_tensor[:, [0, 2]] /= 513.0
    box_tensor[:, [1, 3]] /= 750.0
    return {
        "label": torch.tensor(labels, dtype=torch.float32),
        "center_x": box_tensor[:, 0],
        "center_y": box_tensor[:, 1],
        "width": box_tensor[:, 2],
        "height": box_tensor[:, 3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PosterO .pt result file")
    parser.add_argument("--output", required=True, help="Shared-format .pt output")
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--split", default="auto", help="PosterO split: auto, valid, or test")
    parser.add_argument("--selection", choices=("official", "first"), default="official")
    parser.add_argument("--postero-code-root", default="external_baselines/PosterO-CVPR2025")
    parser.add_argument("--name-map-csv", default="", help="CSV with file_name and poster_path columns")
    parser.add_argument("--subset-csv", default="", help="Optional CSV of poster_path values to keep")
    parser.add_argument("--max-elements", type=int, default=16)
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--height", type=int, default=750)
    args = parser.parse_args()

    source = Path(args.input)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected PosterO result dict, got {type(payload)!r}")
    split = choose_split(payload, args.split)
    entries = list(payload[split])

    name_mapping = load_name_mapping(args.name_map_csv)
    subset = load_subset(args.subset_csv)

    if args.selection == "official":
        selected, selects = official_select(
            entries,
            args.dataset,
            Path(args.postero_code_root),
            args.width,
            args.height,
        )
    else:
        selected, selects = first_select(entries)

    rows = []
    skipped = []
    invalid_mapping = []
    for entry, record, select_index in zip(entries, selected, selects):
        original_name = basename(entry.get("poster_path", ""))
        mapped_name = name_mapping.get(stem(original_name), original_name)
        if name_mapping and stem(original_name) not in name_mapping:
            invalid_mapping.append(original_name)
        if subset is not None and mapped_name not in subset:
            skipped.append(mapped_name)
            continue
        rows.append((mapped_name, record, int(select_index)))

    output = torch.zeros((len(rows), args.max_elements, 5), dtype=torch.float32)
    img_names: list[str] = []
    selected_indices: list[int] = []
    for image_index, (image_name, record, select_index) in enumerate(rows):
        img_names.append(image_name)
        selected_indices.append(select_index)
        n = min(args.max_elements, int(record["label"].numel()))
        output[image_index, :n, 0] = record["label"][:n].float()
        output[image_index, :n, 1] = record["center_x"][:n].float()
        output[image_index, :n, 2] = record["center_y"][:n].float()
        output[image_index, :n, 3] = record["width"][:n].float()
        output[image_index, :n, 4] = record["height"][:n].float()

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "img_names": img_names,
            "test_output": output,
            "invalid_image_names": sorted(set(invalid_mapping)),
            "source": str(source),
            "source_split": split,
            "selection": args.selection,
            "selected_candidate_indices": selected_indices,
            "skipped_by_subset": len(skipped),
            "class_mapping": CLASS_ID,
        },
        destination,
    )
    print(
        f"Wrote {len(img_names)} layouts to {destination} "
        f"(split={split}, selection={args.selection}, skipped={len(skipped)}, "
        f"invalid_mappings={len(set(invalid_mapping))})"
    )


if __name__ == "__main__":
    main()
