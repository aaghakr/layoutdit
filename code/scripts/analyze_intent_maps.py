"""Compare saliency, intent, and simple spatial priors with layout density."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.util import load_config


def safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() < 1e-8 or second.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(first.reshape(-1), second.reshape(-1))[0, 1])


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = first > first.mean()
    second_mask = second > second.mean()
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_mask, second_mask).sum() / union)


def normalize_map(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    value = value - float(value.min(initial=0.0))
    scale = float(value.max(initial=0.0))
    return value / scale if scale > 1e-8 else value


def rasterize_density(
    rows: pd.DataFrame,
    width: int,
    height: int,
    sigma: float,
    *,
    accumulate: bool = False,
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    for value in rows["box_elem"]:
        x1, y1, x2, y2 = map(int, ast.literal_eval(value))
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        if x2 > x1 and y2 > y1:
            if accumulate:
                density[y1:y2, x1:x2] += 1.0
            else:
                density[y1:y2, x1:x2] = 1.0
    if sigma > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigma)
    return normalize_map(density)


def center_prior(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sx = max(width * 0.25, 1.0)
    sy = max(height * 0.25, 1.0)
    value = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    return normalize_map(value)


def saliency_distance_prior(saliency: np.ndarray) -> np.ndarray:
    salient = saliency > float(saliency.mean())
    non_salient = (~salient).astype(np.uint8)
    distance = cv2.distanceTransform(non_salient, cv2.DIST_L2, 5)
    return normalize_map(distance)


def resize_like(value: np.ndarray, width: int, height: int) -> np.ndarray:
    if value.shape == (height, width):
        return value
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pku", "cgl"), required=True)
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum annotated test images to evaluate; use <=0 for all available annotated images.",
    )
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-image-output",
        default="",
        help="Optional CSV with per-image correlations/IoUs and map checksums.",
    )
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        str(code_root / "configs" / f"{args.dataset}_anno_test.yaml"),
        args.path_profile,
    )
    annotations = pd.read_csv(cfg.paths.test.annotated_dir)
    groups = annotations.groupby("poster_path")
    names = sorted(name for name in os.listdir(cfg.paths.test.inp_dir) if name in groups.groups)
    if args.limit > 0:
        names = names[: args.limit]

    train_csv = Path(cfg.paths.base) / "csv" / "train.csv"
    train_density = None
    if train_csv.is_file():
        train_annotations = pd.read_csv(train_csv)
        train_density = rasterize_density(
            train_annotations,
            int(cfg.width),
            int(cfg.height),
            args.sigma,
            accumulate=True,
        )

    method_names = [
        "saliency",
        "inverse_saliency",
        "saliency_distance",
        "center_prior",
        "train_density",
        "intent",
        "oracle_density",
    ]
    values: dict[str, list[float]] = {
        f"{method}_{metric}": []
        for method in method_names
        for metric in ("corr", "iou")
    }
    per_image_records: list[dict[str, object]] = []
    for name in names:
        image_path = Path(cfg.paths.test.inp_dir) / name
        saliency_path = Path(cfg.paths.test.sal_dir) / name
        saliency_sub_path = Path(cfg.paths.test.sal_sub_dir) / name
        intent_path = Path(cfg.paths.test.intent_map_dir) / name
        image = Image.open(image_path)
        width, height = image.size
        density = rasterize_density(groups.get_group(name), width, height, args.sigma)

        saliency = np.asarray(
            Image.open(saliency_path).convert("L").resize((width, height)),
            dtype=np.float32,
        ) / 255.0
        if saliency_sub_path.is_file():
            saliency_sub = np.asarray(
                Image.open(saliency_sub_path).convert("L").resize((width, height)),
                dtype=np.float32,
            ) / 255.0
            saliency = np.maximum(saliency, saliency_sub)
        intent = np.asarray(
            Image.open(intent_path).convert("L").resize((width, height)),
            dtype=np.float32,
        ) / 255.0

        maps = {
            "saliency": saliency,
            "inverse_saliency": 1.0 - saliency,
            "saliency_distance": saliency_distance_prior(saliency),
            "center_prior": center_prior(width, height),
            "intent": intent,
            "oracle_density": density,
        }
        if train_density is not None:
            maps["train_density"] = resize_like(train_density, width, height)
        else:
            maps["train_density"] = np.zeros_like(density)

        for method, value in maps.items():
            corr = safe_corr(value, density)
            iou = binary_iou(value, density)
            values[f"{method}_corr"].append(corr)
            values[f"{method}_iou"].append(iou)
        per_image_records.append(
            {
                "image": name,
                "width": width,
                "height": height,
                "image_sha256": sha256_file(image_path),
                "saliency_sha256": sha256_file(saliency_path),
                "saliency_sub_sha256": sha256_file(saliency_sub_path)
                if saliency_sub_path.is_file()
                else "",
                "intent_sha256": sha256_file(intent_path),
                **{
                    f"{method}_{metric}": values[f"{method}_{metric}"][-1]
                    for method in method_names
                    for metric in ("corr", "iou")
                },
            }
        )

    result = {
        "dataset": args.dataset,
        "n_samples": len(names),
        "sigma": args.sigma,
        "protocol": "Each map is compared with the blurred ground-truth layout density; IoU thresholds each map by its own mean.",
        "train_density_source": str(train_csv) if train_csv.is_file() else "",
        **{key: float(np.mean(value)) for key, value in values.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.per_image_output:
        pd.DataFrame(per_image_records).to_csv(args.per_image_output, index=False)
    lines = [
        "| dataset | method | n | corr | IoU |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    labels = {
        "saliency": "Saliency",
        "inverse_saliency": "Inverse saliency",
        "saliency_distance": "Saliency distance",
        "center_prior": "Center prior",
        "train_density": "Train density",
        "intent": "Learned intent",
        "oracle_density": "Oracle density",
    }
    for method in method_names:
        lines.append(
            f"| {args.dataset} | {labels[method]} | {len(names)} | "
            f"{result[f'{method}_corr']:.4f} | {result[f'{method}_iou']:.4f} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
