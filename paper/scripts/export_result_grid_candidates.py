from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/paper_figures"
BASELINE_DIR = ROOT / "other_baselines"
OUT_ROOT = ROOT / "paper/fig/result_grid_candidates"
SYNC_OUT_ROOT = ROOT / "paper/fig/result_grid_candidates_synced"

CLASS_COLORS = {
    "Text": (96, 176, 72),
    "Logo": (239, 120, 148),
    "Underlay": (64, 142, 206),
    "Embellishment": (230, 154, 55),
}

DATASET_CLASSES = {
    "pku": {1: "Text", 2: "Logo", 3: "Underlay"},
    "cgl": {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"},
}

IMAGE_DIRS = {
    "pku": ROOT / "data/dataset/pku/split/test_anno/inpaint",
    "cgl": ROOT / "data/dataset/cgl/split/test_anno/inpaint",
}

PROMPT_CSVS = {
    ("pku", "basic"): ROOT / "data/dataset/pku/split/csv/test_with_prompts_basic.csv",
    ("pku", "enhanced"): ROOT / "data/dataset/pku/split/csv/test_with_prompts_enhanced.csv",
    ("pku", "advanced"): ROOT / "data/dataset/pku/split/csv/test_with_prompts_advanced.csv",
    ("pku", "spatial"): ROOT / "data/dataset/pku/split/csv/test_with_prompts_spatial.csv",
    ("pku", "rich"): ROOT / "data/dataset/pku/split/csv/test_with_rich_prompts.csv",
    ("pku", "freeform"): ROOT / "data/prompts/free_form_pku.csv",
    ("cgl", "basic"): ROOT / "data/dataset/cgl/split/csv/test_with_prompts_basic.csv",
    ("cgl", "enhanced"): ROOT / "data/dataset/cgl/split/csv/test_with_prompts_enhanced.csv",
    ("cgl", "advanced"): ROOT / "data/dataset/cgl/split/csv/test_with_prompts_advanced.csv",
    ("cgl", "spatial"): ROOT / "data/dataset/cgl/split/csv/test_with_prompts_spatial.csv",
    ("cgl", "rich"): ROOT / "data/dataset/cgl/split/csv/test_with_rich_prompts.csv",
    ("cgl", "freeform"): ROOT / "data/prompts/free_form_cgl.csv",
}

TEMPLATE_FAMILIES = ("basic", "enhanced", "advanced", "spatial", "rich")


@dataclass(frozen=True)
class PredictionSet:
    name: str
    output_pt: Path
    per_image_csv: Path | None = None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_TITLE = font(22, True)
FONT_SECTION = font(18, True)
FONT_BODY = font(16)
FONT_BODY_BOLD = font(16, True)
FONT_SMALL = font(14)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def ensure_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def load_prompts(path: Path) -> dict[str, str]:
    if not ensure_file(path):
        return {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        prompt_col = "text_prompt" if "text_prompt" in rows.fieldnames else "prompt"
        result: dict[str, str] = {}
        for row in rows:
            image_name = row.get("poster_path", "")
            if image_name and image_name not in result:
                result[image_name] = row.get(prompt_col, "")
        return result


def read_metrics(path: Path | None) -> pd.DataFrame:
    if path is None or not ensure_file(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "image" not in df.columns:
        return pd.DataFrame()
    df["image"] = df["image"].astype(str).map(lambda x: Path(x).name)
    return df.drop_duplicates("image").set_index("image", drop=False)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def load_prediction_elements(path: Path, class_names: dict[int, str]) -> dict[str, list[dict]]:
    payload = torch.load(path, map_location="cpu")
    img_names = [Path(str(x)).name for x in payload["img_names"]]
    output = payload["test_output"].detach().cpu()
    results: dict[str, list[dict]] = {}
    for image_name, one_output in zip(img_names, output):
        classes = one_output[:, 0].round().to(torch.int64)
        boxes = torch.clamp(cxcywh_to_xyxy(one_output[:, 1:]), 0.0, 1.0)
        elements = []
        for class_id, box in zip(classes.tolist(), boxes.tolist()):
            if class_id <= 0 or class_id not in class_names:
                continue
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            if area <= 1e-4:
                continue
            elements.append(
                {
                    "class_name": class_names[class_id],
                    "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                }
            )
        results[image_name] = elements
    return results


def image_names_from_prediction(path: Path) -> list[str]:
    payload = torch.load(path, map_location="cpu")
    return [Path(str(x)).name for x in payload["img_names"]]


def box_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(round(x0 * width))))
    x1 = max(0, min(width, int(round(x1 * width))))
    y0 = max(0, min(height, int(round(y0 * height))))
    y1 = max(0, min(height, int(round(y1 * height))))
    return x0, y0, x1, y1


def draw_box_overlay(image: Image.Image, elements: list[dict]) -> Image.Image:
    out = image.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = out.size
    ordered = sorted(elements, key=lambda e: 0 if e["class_name"] == "Underlay" else 1)
    for element in ordered:
        color = CLASS_COLORS.get(element["class_name"], (180, 180, 180))
        box = box_pixels(element["box_xyxy"], width, height)
        draw.rectangle(box, fill=(*color, 115), outline=(*color, 235), width=2)
    return Image.alpha_composite(out, overlay).convert("RGB")


def wrap_text(text: str, draw: ImageDraw.ImageDraw, max_width: int, font_obj, max_lines: int = 5) -> list[str]:
    words = str(text).replace("Create ", "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font_obj) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,") + "..."
    return lines


def metric_value(metrics: pd.DataFrame, image_name: str, key: str, default: float = 0.0) -> float:
    if metrics.empty or image_name not in metrics.index or key not in metrics.columns:
        return default
    return safe_float(metrics.loc[image_name, key], default)


def count_elements(elements: list[dict]) -> int:
    return len(elements)


def total_area(elements: list[dict]) -> float:
    total = 0.0
    for element in elements:
        x0, y0, x1, y1 = element["box_xyxy"]
        total += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return total


def method_caption(method: str, metrics: pd.DataFrame, image_name: str, elements: list[dict], kind: str) -> str:
    parts = [method]
    if kind == "prompt":
        parts.append(f"PLA {fmt(metric_value(metrics, image_name, 'pla_count'))}")
        if "spla" in metrics.columns:
            parts.append(f"SPLA {fmt(metric_value(metrics, image_name, 'spla'))}")
        parts.append(f"Exact {fmt(metric_value(metrics, image_name, 'exact_count_match'))}")
    else:
        parts.append(f"n {count_elements(elements)}")
        parts.append(f"area {fmt(total_area(elements))}")
        parts.append(f"Occ {fmt(metric_value(metrics, image_name, 'occ'))}")
        if "type_f1" in metrics.columns:
            parts.append(f"F1 {fmt(metric_value(metrics, image_name, 'type_f1'))}")
    return "   ".join(parts)


def make_prompt_panel(title: str, prompt: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    y = 8
    draw.text((10, y), title, fill=(0, 0, 0), font=FONT_BODY_BOLD)
    y += 24
    for line in wrap_text(prompt, draw, width - 20, FONT_BODY, max_lines=5):
        draw.text((10, y), line, fill=(0, 0, 0), font=FONT_BODY)
        y += 22
    return panel


def make_single_tile(
    title: str,
    image_name: str,
    prompt: str,
    image_path: Path,
    elements: list[dict],
    metrics: pd.DataFrame,
    kind: str,
) -> Image.Image:
    width = 360
    image_height = 526
    header_height = 64
    prompt_height = 150 if prompt else 0
    label_height = 42
    gap = 6

    base = Image.open(image_path).convert("RGB").resize((width, image_height), Image.LANCZOS)
    overlay = draw_box_overlay(base, elements)

    tile_height = header_height + prompt_height + label_height + image_height + gap + label_height + image_height
    tile = Image.new("RGB", (width, tile_height), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 4), image_name, fill=(0, 0, 0), font=FONT_TITLE)
    draw.text((8, 32), title, fill=(0, 0, 0), font=FONT_SMALL)
    y = header_height
    if prompt:
        tile.paste(make_prompt_panel("Prompt:", prompt, width, prompt_height), (0, y))
        y += prompt_height
    draw.text((8, y + 7), "Input image", fill=(0, 0, 0), font=FONT_SECTION)
    y += label_height
    tile.paste(base, (0, y))
    y += image_height + gap
    draw.text(
        (8, y + 7),
        method_caption("IntentDiT", metrics, image_name, elements, kind),
        fill=(0, 0, 0),
        font=FONT_SMALL,
    )
    y += label_height
    tile.paste(overlay, (0, y))
    return tile


def make_comparison_tile(
    title: str,
    image_name: str,
    prompt: str,
    image_path: Path,
    methods: list[tuple[str, list[dict], pd.DataFrame]],
    kind: str,
) -> Image.Image:
    panel_width = 320
    image_height = 468
    header_height = 76
    prompt_height = 140 if prompt else 0
    label_height = 50
    gap = 8
    width = panel_width * len(methods)
    tile_height = header_height + prompt_height + label_height + image_height + gap + label_height + image_height

    base_full = Image.open(image_path).convert("RGB")
    base = base_full.resize((panel_width, image_height), Image.LANCZOS)
    overlays = [draw_box_overlay(base, elements) for _, elements, _ in methods]

    tile = Image.new("RGB", (width, tile_height), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((10, 5), image_name, fill=(0, 0, 0), font=FONT_TITLE)
    draw.text((10, 35), title, fill=(0, 0, 0), font=FONT_BODY_BOLD)
    y = header_height
    if prompt:
        tile.paste(make_prompt_panel("Prompt:", prompt, width, prompt_height), (0, y))
        y += prompt_height

    for col in range(len(methods)):
        x = col * panel_width
        draw.text((x + 8, y + 7), "Input image", fill=(0, 0, 0), font=FONT_SECTION)
        tile.paste(base, (x, y + label_height))
    y += label_height + image_height + gap

    for col, ((method_name, elements, metrics), overlay) in enumerate(zip(methods, overlays)):
        x = col * panel_width
        caption = method_caption(method_name, metrics, image_name, elements, kind)
        for line_index, line in enumerate(wrap_text(caption, draw, panel_width - 16, FONT_SMALL, max_lines=2)):
            draw.text((x + 8, y + 5 + line_index * 18), line, fill=(0, 0, 0), font=FONT_SMALL)
        tile.paste(overlay, (x, y + label_height))
    return tile


def write_index(out_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)


def prepare_out_dir(path: Path, clean: bool) -> Path:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def common_image_names(*prediction_paths: Path) -> list[str]:
    common: set[str] | None = None
    first_order: list[str] = []
    for index, path in enumerate(prediction_paths):
        names = image_names_from_prediction(path)
        name_set = set(names)
        if index == 0:
            first_order = names
            common = name_set
        else:
            common = common.intersection(name_set) if common is not None else name_set
    common = common or set()
    return [name for name in first_order if name in common]


def export_comparison_gallery(
    out_dir: Path,
    dataset: str,
    title: str,
    prompt_map: dict[str, str],
    methods: list[PredictionSet],
    kind: str,
    score_fn,
    limit: int,
    clean: bool,
    fixed_images: list[str] | None = None,
) -> int:
    out_dir = prepare_out_dir(out_dir, clean)
    class_names = DATASET_CLASSES[dataset]
    loaded = []
    for method in methods:
        loaded.append(
            (
                method.name,
                load_prediction_elements(method.output_pt, class_names),
                read_metrics(method.per_image_csv),
            )
        )
    names = fixed_images if fixed_images is not None else common_image_names(*[m.output_pt for m in methods])
    rows = []
    for image_name in names:
        image_path = IMAGE_DIRS[dataset] / image_name
        if not image_path.exists():
            continue
        prompt = prompt_map.get(image_name, "")
        score = score_fn(image_name, loaded, prompt)
        if score is None and fixed_images is None:
            continue
        if score is None:
            score = 0.0
        rows.append((float(score), image_name, prompt))
    if fixed_images is None:
        rows.sort(key=lambda x: (-x[0], x[1]))
        rows = rows[:limit]

    index_rows = []
    for rank, (score, image_name, prompt) in enumerate(rows, start=1):
        method_payload = [
            (method_name, elements_by_image.get(image_name, []), metrics)
            for method_name, elements_by_image, metrics in loaded
        ]
        tile = make_comparison_tile(
            title=title,
            image_name=image_name,
            prompt=prompt,
            image_path=IMAGE_DIRS[dataset] / image_name,
            methods=method_payload,
            kind=kind,
        )
        tile_name = f"{rank:03d}_{Path(image_name).stem}.png"
        tile.save(out_dir / tile_name)
        record = {
            "rank": rank,
            "tile": tile_name,
            "image": image_name,
            "score": score,
            "prompt": prompt,
        }
        for method_name, elements_by_image, metrics in loaded:
            key = method_name.lower().replace(" ", "_").replace("+", "plus")
            elements = elements_by_image.get(image_name, [])
            record[f"{key}_n_pred"] = count_elements(elements)
            record[f"{key}_total_area"] = total_area(elements)
            for metric_name in ("pla_count", "spla", "exact_count_match", "occ", "rea", "type_f1", "uti", "max_iou"):
                if not metrics.empty and metric_name in metrics.columns:
                    record[f"{key}_{metric_name}"] = metric_value(metrics, image_name, metric_name)
        index_rows.append(record)
    write_index(out_dir, index_rows)
    print(f"Wrote {len(index_rows)} comparison candidates to {out_dir}")
    return len(index_rows)


def export_single_gallery(
    out_dir: Path,
    dataset: str,
    title: str,
    prompt_map: dict[str, str],
    prediction: PredictionSet,
    kind: str,
    score_fn,
    limit: int,
    clean: bool,
    fixed_images: list[str] | None = None,
) -> int:
    out_dir = prepare_out_dir(out_dir, clean)
    elements_by_image = load_prediction_elements(prediction.output_pt, DATASET_CLASSES[dataset])
    metrics = read_metrics(prediction.per_image_csv)
    rows = []
    names = fixed_images if fixed_images is not None else image_names_from_prediction(prediction.output_pt)
    for image_name in names:
        image_path = IMAGE_DIRS[dataset] / image_name
        if not image_path.exists():
            continue
        prompt = prompt_map.get(image_name, "")
        score = score_fn(image_name, elements_by_image, metrics, prompt)
        if score is None and fixed_images is None:
            continue
        if score is None:
            score = 0.0
        rows.append((float(score), image_name, prompt))
    if fixed_images is None:
        rows.sort(key=lambda x: (-x[0], x[1]))
        rows = rows[:limit]

    index_rows = []
    for rank, (score, image_name, prompt) in enumerate(rows, start=1):
        elements = elements_by_image.get(image_name, [])
        tile = make_single_tile(
            title=title,
            image_name=image_name,
            prompt=prompt,
            image_path=IMAGE_DIRS[dataset] / image_name,
            elements=elements,
            metrics=metrics,
            kind=kind,
        )
        tile_name = f"{rank:03d}_{Path(image_name).stem}.png"
        tile.save(out_dir / tile_name)
        record = {
            "rank": rank,
            "tile": tile_name,
            "image": image_name,
            "score": score,
            "prompt": prompt,
            "n_pred": count_elements(elements),
            "total_area": total_area(elements),
        }
        for metric_name in ("pla_count", "spla", "exact_count_match", "occ", "rea", "type_f1", "uti", "max_iou"):
            if not metrics.empty and metric_name in metrics.columns:
                record[metric_name] = metric_value(metrics, image_name, metric_name)
        index_rows.append(record)
    write_index(out_dir, index_rows)
    print(f"Wrote {len(index_rows)} single-model candidates to {out_dir}")
    return len(index_rows)


def image_only_score(image_name: str, loaded: list[tuple[str, dict, pd.DataFrame]], prompt: str) -> float | None:
    baseline_name, baseline_elements, baseline_metrics = loaded[0]
    intent_name, intent_elements, intent_metrics = loaded[1]
    base_elements = baseline_elements.get(image_name, [])
    ours_elements = intent_elements.get(image_name, [])
    delta_n = count_elements(ours_elements) - count_elements(base_elements)
    delta_area = total_area(ours_elements) - total_area(base_elements)
    delta_uti = metric_value(intent_metrics, image_name, "uti") - metric_value(baseline_metrics, image_name, "uti")
    occ_penalty = max(0.0, metric_value(intent_metrics, image_name, "occ") - 0.22)
    f1 = metric_value(intent_metrics, image_name, "type_f1")
    if delta_n < 1 or delta_area <= 0:
        return None
    return delta_n + 6.0 * delta_area + 2.0 * delta_uti + 0.4 * f1 - 2.0 * occ_penalty


def freeform_score(image_name: str, loaded: list[tuple[str, dict, pd.DataFrame]], prompt: str) -> float | None:
    if not prompt:
        return None
    baseline_metrics = loaded[0][2]
    intent_metrics = loaded[1][2]
    ours_pla = metric_value(intent_metrics, image_name, "pla_count")
    base_pla = metric_value(baseline_metrics, image_name, "pla_count")
    ours_spla = metric_value(intent_metrics, image_name, "spla")
    base_spla = metric_value(baseline_metrics, image_name, "spla")
    exact_gain = metric_value(intent_metrics, image_name, "exact_count_match") - metric_value(baseline_metrics, image_name, "exact_count_match")
    if ours_pla < 0.65:
        return None
    return 3.0 * (ours_pla - base_pla) + 2.0 * (ours_spla - base_spla) + exact_gain + ours_pla + 0.5 * ours_spla


def text_mode_score(image_name: str, loaded: list[tuple[str, dict, pd.DataFrame]], prompt: str) -> float | None:
    if not prompt:
        return None
    pooled_metrics = loaded[0][2]
    token_metrics = loaded[1][2]
    token_pla = metric_value(token_metrics, image_name, "pla_count")
    pooled_pla = metric_value(pooled_metrics, image_name, "pla_count")
    token_exact = metric_value(token_metrics, image_name, "exact_count_match")
    pooled_exact = metric_value(pooled_metrics, image_name, "exact_count_match")
    token_f1 = metric_value(token_metrics, image_name, "type_f1")
    if token_pla < 0.75:
        return None
    return 3.0 * (token_pla - pooled_pla) + (token_exact - pooled_exact) + token_pla + 0.3 * token_f1


def image_ablation_score(image_name: str, loaded: list[tuple[str, dict, pd.DataFrame]], prompt: str) -> float | None:
    saliency_metrics = loaded[0][2]
    intent_metrics = loaded[1][2]
    both_metrics = loaded[2][2]
    both_elements = loaded[2][1].get(image_name, [])
    both_occ = metric_value(both_metrics, image_name, "occ")
    both_rea = metric_value(both_metrics, image_name, "rea")
    best_single_occ = min(metric_value(saliency_metrics, image_name, "occ"), metric_value(intent_metrics, image_name, "occ"))
    best_single_rea = min(metric_value(saliency_metrics, image_name, "rea"), metric_value(intent_metrics, image_name, "rea"))
    n_pred = count_elements(both_elements)
    if n_pred < 2:
        return None
    return (best_single_occ - both_occ) * 4.0 + (best_single_rea - both_rea) * 8.0 + 0.2 * n_pred - 0.3 * max(0.0, both_occ - 0.22)


def template_score(image_name: str, elements_by_image: dict[str, list[dict]], metrics: pd.DataFrame, prompt: str) -> float | None:
    if not prompt:
        return None
    pla = metric_value(metrics, image_name, "pla_count")
    exact = metric_value(metrics, image_name, "exact_count_match")
    spla = metric_value(metrics, image_name, "spla")
    type_f1 = metric_value(metrics, image_name, "type_f1")
    occ = metric_value(metrics, image_name, "occ")
    if pla < 0.65:
        return None
    return exact + pla + spla + 0.25 * type_f1 - 0.25 * occ


def freeform_single_score(image_name: str, elements_by_image: dict[str, list[dict]], metrics: pd.DataFrame, prompt: str) -> float | None:
    return template_score(image_name, elements_by_image, metrics, prompt)


def sync_prediction_paths(dataset: str) -> list[Path]:
    paths = [
        EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
        BASELINE_DIR / f"standardized/postero_{dataset}_freeform_subset.pt",
        EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_test_output.pt",
        BASELINE_DIR / f"layoutidit/{dataset}_anno_uncond_test_output.pt",
    ]
    paths.extend(
        EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_test_output.pt"
        for family in TEMPLATE_FAMILIES
    )
    if dataset == "pku":
        paths.extend(
            [
                EXPERIMENT_DIR / "ivc_pku_vit_pooled_text_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_prompt_pku_vit_both_text_basic_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_saliency_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_intent_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_both_trainseed1_inferseed1_test_output.pt",
            ]
        )
    return paths


def rank_synchronized_images(dataset: str, limit: int) -> list[str]:
    common = common_image_names(*sync_prediction_paths(dataset))
    freeform_metrics = read_metrics(
        EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv"
    )
    text_ref_metrics = read_metrics(EXPERIMENT_DIR / f"baseline_text_{dataset}_freeform_per_image.csv")
    image_metrics = read_metrics(EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_per_image.csv")
    template_metrics = [
        read_metrics(EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_per_image.csv")
        for family in TEMPLATE_FAMILIES
    ]
    rows = []
    for image_name in common:
        if not (IMAGE_DIRS[dataset] / image_name).exists():
            continue
        score = 0.0
        free_pla = metric_value(freeform_metrics, image_name, "pla_count")
        free_spla = metric_value(freeform_metrics, image_name, "spla")
        free_exact = metric_value(freeform_metrics, image_name, "exact_count_match")
        ref_pla = metric_value(text_ref_metrics, image_name, "pla_count")
        ref_spla = metric_value(text_ref_metrics, image_name, "spla")
        score += 2.0 * free_pla + free_spla + free_exact
        score += max(0.0, free_pla - ref_pla) + 0.5 * max(0.0, free_spla - ref_spla)
        score += 0.25 * metric_value(image_metrics, image_name, "type_f1")
        score -= 0.25 * metric_value(image_metrics, image_name, "occ")
        for metrics in template_metrics:
            score += 0.5 * metric_value(metrics, image_name, "pla_count")
            score += 0.5 * metric_value(metrics, image_name, "exact_count_match")
            score += 0.35 * metric_value(metrics, image_name, "spla")
        rows.append((score, image_name))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [image_name for _, image_name in rows[:limit]]


def write_synced_selection_manifest(out_root: Path, selections: dict[str, list[str]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for dataset, image_names in selections.items():
        pd.DataFrame(
            [{"rank": rank, "image": image_name} for rank, image_name in enumerate(image_names, start=1)]
        ).to_csv(out_root / f"selected_{dataset}_images.csv", index=False)


def export_main_result_galleries(limit: int, clean: bool) -> None:
    for dataset in ("pku", "cgl"):
        export_comparison_gallery(
            OUT_ROOT / "01_image_only_tradeoff_vs_layoutdit" / dataset,
            dataset=dataset,
            title="Table 1 and Table 2: image-only density and structure trade-off",
            prompt_map={},
            methods=[
                PredictionSet(
                    "LayoutDiT",
                    BASELINE_DIR / f"layoutidit/{dataset}_anno_uncond_test_output.pt",
                    EXPERIMENT_DIR / f"baseline_layoutdit_{dataset}_image_only_per_image.csv",
                ),
                PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_per_image.csv",
                ),
            ],
            kind="image",
            score_fn=image_only_score,
            limit=limit,
            clean=clean,
        )

        export_comparison_gallery(
            OUT_ROOT / "02_freeform_controllability_vs_external_text" / dataset,
            dataset=dataset,
            title="Table 4 and Table 5: free-form prompt controllability",
            prompt_map=load_prompts(PROMPT_CSVS[(dataset, "freeform")]),
            methods=[
                PredictionSet(
                    "External text ref",
                    BASELINE_DIR / f"standardized/postero_{dataset}_freeform_subset.pt",
                    EXPERIMENT_DIR / f"baseline_text_{dataset}_freeform_per_image.csv",
                ),
                PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
                ),
            ],
            kind="prompt",
            score_fn=freeform_score,
            limit=limit,
            clean=clean,
        )

    export_comparison_gallery(
        OUT_ROOT / "03_text_conditioning_ablation_pku_basic" / "pku",
        dataset="pku",
        title="Table 6: pooled sentence vs token-level text conditioning",
        prompt_map=load_prompts(PROMPT_CSVS[("pku", "basic")]),
        methods=[
            PredictionSet(
                "Pooled sentence",
                EXPERIMENT_DIR / "ivc_pku_vit_pooled_text_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_pooled_text_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Token-level",
                EXPERIMENT_DIR / "ivc_prompt_pku_vit_both_text_basic_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_prompt_pku_vit_both_text_basic_trainseed1_inferseed1_per_image.csv",
            ),
        ],
        kind="prompt",
        score_fn=text_mode_score,
        limit=limit,
        clean=clean,
    )

    export_comparison_gallery(
        OUT_ROOT / "04_image_conditioning_ablation_pku" / "pku",
        dataset="pku",
        title="Table 6: saliency, intent, and saliency plus intent conditioning",
        prompt_map={},
        methods=[
            PredictionSet(
                "Saliency",
                EXPERIMENT_DIR / "ivc_pku_vit_saliency_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_saliency_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Intent",
                EXPERIMENT_DIR / "ivc_pku_vit_intent_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_intent_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Saliency+Intent",
                EXPERIMENT_DIR / "ivc_pku_vit_both_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_both_trainseed1_inferseed1_per_image.csv",
            ),
        ],
        kind="image",
        score_fn=image_ablation_score,
        limit=limit,
        clean=clean,
    )

    for dataset in ("pku", "cgl"):
        for family in TEMPLATE_FAMILIES:
            export_single_gallery(
                OUT_ROOT / "05_template_prompt_families" / family / dataset,
                dataset=dataset,
                title=f"Figure 3 candidate: {family} prompt family",
                prompt_map=load_prompts(PROMPT_CSVS[(dataset, family)]),
                prediction=PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_per_image.csv",
                ),
                kind="prompt",
                score_fn=template_score,
                limit=limit,
                clean=clean,
            )

        export_single_gallery(
            OUT_ROOT / "06_freeform_single_model_examples" / dataset,
            dataset=dataset,
            title="Figure 3 candidate: free-form prompt",
            prompt_map=load_prompts(PROMPT_CSVS[(dataset, "freeform")]),
            prediction=PredictionSet(
                "IntentDiT",
                EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
            ),
            kind="prompt",
            score_fn=freeform_single_score,
            limit=limit,
            clean=clean,
        )


def export_synchronized_result_galleries(limit: int, clean: bool) -> None:
    selections = {dataset: rank_synchronized_images(dataset, limit) for dataset in ("pku", "cgl")}
    write_synced_selection_manifest(SYNC_OUT_ROOT, selections)

    for dataset in ("pku", "cgl"):
        fixed_images = selections[dataset]
        export_comparison_gallery(
            SYNC_OUT_ROOT / "01_image_only_tradeoff_vs_layoutdit" / dataset,
            dataset=dataset,
            title="Table 1 and Table 2: image-only density and structure trade-off",
            prompt_map={},
            methods=[
                PredictionSet(
                    "LayoutDiT",
                    BASELINE_DIR / f"layoutidit/{dataset}_anno_uncond_test_output.pt",
                    EXPERIMENT_DIR / f"baseline_layoutdit_{dataset}_image_only_per_image.csv",
                ),
                PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_{dataset}_vit_both_trainseed1_inferseed1_per_image.csv",
                ),
            ],
            kind="image",
            score_fn=image_only_score,
            limit=limit,
            clean=clean,
            fixed_images=fixed_images,
        )

        export_comparison_gallery(
            SYNC_OUT_ROOT / "02_freeform_controllability_vs_external_text" / dataset,
            dataset=dataset,
            title="Table 4 and Table 5: free-form prompt controllability",
            prompt_map=load_prompts(PROMPT_CSVS[(dataset, "freeform")]),
            methods=[
                PredictionSet(
                    "External text ref",
                    BASELINE_DIR / f"standardized/postero_{dataset}_freeform_subset.pt",
                    EXPERIMENT_DIR / f"baseline_text_{dataset}_freeform_per_image.csv",
                ),
                PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
                ),
            ],
            kind="prompt",
            score_fn=freeform_score,
            limit=limit,
            clean=clean,
            fixed_images=fixed_images,
        )

    export_comparison_gallery(
        SYNC_OUT_ROOT / "03_text_conditioning_ablation_pku_basic" / "pku",
        dataset="pku",
        title="Table 6: pooled sentence vs token-level text conditioning",
        prompt_map=load_prompts(PROMPT_CSVS[("pku", "basic")]),
        methods=[
            PredictionSet(
                "Pooled sentence",
                EXPERIMENT_DIR / "ivc_pku_vit_pooled_text_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_pooled_text_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Token-level",
                EXPERIMENT_DIR / "ivc_prompt_pku_vit_both_text_basic_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_prompt_pku_vit_both_text_basic_trainseed1_inferseed1_per_image.csv",
            ),
        ],
        kind="prompt",
        score_fn=text_mode_score,
        limit=limit,
        clean=clean,
        fixed_images=selections["pku"],
    )

    export_comparison_gallery(
        SYNC_OUT_ROOT / "04_image_conditioning_ablation_pku" / "pku",
        dataset="pku",
        title="Table 6: saliency, intent, and saliency plus intent conditioning",
        prompt_map={},
        methods=[
            PredictionSet(
                "Saliency",
                EXPERIMENT_DIR / "ivc_pku_vit_saliency_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_saliency_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Intent",
                EXPERIMENT_DIR / "ivc_pku_vit_intent_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_intent_trainseed1_inferseed1_per_image.csv",
            ),
            PredictionSet(
                "Saliency+Intent",
                EXPERIMENT_DIR / "ivc_pku_vit_both_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / "ivc_pku_vit_both_trainseed1_inferseed1_per_image.csv",
            ),
        ],
        kind="image",
        score_fn=image_ablation_score,
        limit=limit,
        clean=clean,
        fixed_images=selections["pku"],
    )

    for dataset in ("pku", "cgl"):
        fixed_images = selections[dataset]
        for family in TEMPLATE_FAMILIES:
            export_single_gallery(
                SYNC_OUT_ROOT / "05_template_prompt_families" / family / dataset,
                dataset=dataset,
                title=f"Figure 3 candidate: {family} prompt family",
                prompt_map=load_prompts(PROMPT_CSVS[(dataset, family)]),
                prediction=PredictionSet(
                    "IntentDiT",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_test_output.pt",
                    EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_{family}_trainseed1_inferseed1_per_image.csv",
                ),
                kind="prompt",
                score_fn=template_score,
                limit=limit,
                clean=clean,
                fixed_images=fixed_images,
            )

        export_single_gallery(
            SYNC_OUT_ROOT / "06_freeform_single_model_examples" / dataset,
            dataset=dataset,
            title="Figure 3 candidate: free-form prompt",
            prompt_map=load_prompts(PROMPT_CSVS[(dataset, "freeform")]),
            prediction=PredictionSet(
                "IntentDiT",
                EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
                EXPERIMENT_DIR / f"ivc_prompt_{dataset}_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
            ),
            kind="prompt",
            score_fn=freeform_single_score,
            limit=limit,
            clean=clean,
            fixed_images=fixed_images,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument(
        "--synchronized",
        action="store_true",
        help="Use the same ranked PKU and CGL image IDs in every result-family folder.",
    )
    args = parser.parse_args()
    if args.synchronized:
        export_synchronized_result_galleries(limit=args.limit, clean=not args.no_clean)
    else:
        export_main_result_galleries(limit=args.limit, clean=not args.no_clean)


if __name__ == "__main__":
    main()
