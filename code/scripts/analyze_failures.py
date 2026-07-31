"""Generate an auditable heuristic failure taxonomy from saved predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.metric import CLASS_INDEX_TO_NAME, _parse_prompt_counts
from utils.util import box_cxcywh_to_xyxy, load_config


def iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def crop_mean(array: np.ndarray, box: np.ndarray) -> float:
    height, width = array.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, int(x1 * width)), min(width, int(x2 * width))))
    y1, y2 = sorted((max(0, int(y1 * height)), min(height, int(y2 * height))))
    crop = array[y1:y2, x1:x2]
    return float(crop.mean()) if crop.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", choices=("pku", "cgl"), default="pku")
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--overlap-threshold", type=float, default=0.10)
    parser.add_argument("--occlusion-threshold", type=float, default=0.25)
    parser.add_argument("--readability-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        str(code_root / "configs" / f"{args.dataset}_anno_test.yaml"),
        args.path_profile,
    )
    blob = torch.load(args.predictions, map_location="cpu")
    names = list(blob["img_names"])[: args.limit]
    predictions = blob["test_output"][: len(names)]
    classes = predictions[:, :, 0].round().long().numpy()
    boxes = torch.clamp(box_cxcywh_to_xyxy(predictions[:, :, 1:]), 0, 1).numpy()

    prompt_frame = pd.read_csv(args.prompts_csv)
    prompt_col = "text_prompt" if "text_prompt" in prompt_frame.columns else "prompt"
    prompts = prompt_frame.groupby("poster_path")[prompt_col].first().to_dict()
    counts = Counter()

    for sample_index, name in enumerate(names):
        row_classes = classes[sample_index]
        row_boxes = boxes[sample_index]
        valid = row_classes > 0
        active_classes = row_classes[valid]
        active_boxes = row_boxes[valid]

        overlap_failure = any(
            iou(active_boxes[i], active_boxes[j]) > args.overlap_threshold
            for i in range(len(active_boxes))
            for j in range(i + 1, len(active_boxes))
            if active_classes[i] != 3 and active_classes[j] != 3
        )
        counts["element_overlap"] += int(overlap_failure)

        saliency = np.asarray(
            Image.open(Path(cfg.paths.test.sal_dir) / name).convert("L"), dtype=np.float32
        ) / 255.0
        counts["subject_occlusion"] += int(
            any(
                crop_mean(saliency, box) > args.occlusion_threshold
                for cls, box in zip(active_classes, active_boxes)
                if cls != 3
            )
        )

        image = np.asarray(
            Image.open(Path(cfg.paths.test.inp_dir) / name).convert("RGB"), dtype=np.float32
        ) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        gradient = np.sqrt(gx * gx + gy * gy)
        gradient /= gradient.max() + 1e-8
        counts["illegible_text"] += int(
            any(
                crop_mean(gradient, box) > args.readability_threshold
                for cls, box in zip(active_classes, active_boxes)
                if cls == 1
            )
        )

        underlays = active_boxes[active_classes == 3]
        content = active_boxes[active_classes != 3]
        underlay_failure = bool(len(underlays)) and any(
            not any(
                underlay[0] <= item[0]
                and underlay[1] <= item[1]
                and underlay[2] >= item[2]
                and underlay[3] >= item[3]
                for item in content
            )
            for underlay in underlays
        )
        counts["underlay_misalignment"] += int(underlay_failure)

        prompt = prompts.get(name, prompts.get(os.path.basename(name), ""))
        expected = _parse_prompt_counts(str(prompt))
        predicted = Counter(
            CLASS_INDEX_TO_NAME.get(int(cls), f"Class{int(cls)}") for cls in active_classes
        )
        counts["count_mismatch"] += int(
            any(expected.get(key, 0) != predicted.get(key, 0) for key in expected)
        )

    n_samples = len(names)
    result = {
        "dataset": args.dataset,
        "n_samples": n_samples,
        "thresholds": {
            "overlap": args.overlap_threshold,
            "occlusion": args.occlusion_threshold,
            "readability": args.readability_threshold,
        },
        "failures": {
            key: {"count": int(value), "rate": float(value / n_samples)}
            for key, value in sorted(counts.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "| failure | count | rate |",
        "| --- | ---: | ---: |",
        *[
            f"| {key} | {item['count']} | {item['rate']:.4f} |"
            for key, item in result["failures"].items()
        ],
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
