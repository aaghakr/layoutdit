from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "paper/fig/grid_candidates"

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

CONFIGS = {
    ("free_form", "pku"): {
        "image_dir": ROOT / "data/dataset/pku/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/prompts/free_form_pku.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_pku_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
        "per_image": ROOT / "experiments/paper_figures/ivc_prompt_pku_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
    },
    ("free_form", "cgl"): {
        "image_dir": ROOT / "data/dataset/cgl/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/prompts/free_form_cgl.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_cgl_vit_both_text_freeform_trainseed1_inferseed1_test_output.pt",
        "per_image": ROOT / "experiments/paper_figures/ivc_prompt_cgl_vit_both_text_freeform_trainseed1_inferseed1_per_image.csv",
    },
    ("non_free_form", "pku"): {
        "image_dir": ROOT / "data/dataset/pku/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/dataset/pku/split/csv/test_with_prompts_spatial.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_pku_vit_both_text_spatial_trainseed1_inferseed1_test_output.pt",
        "per_image": ROOT / "experiments/paper_figures/ivc_prompt_pku_vit_both_text_spatial_trainseed1_inferseed1_per_image.csv",
    },
    ("non_free_form", "cgl"): {
        "image_dir": ROOT / "data/dataset/cgl/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/dataset/cgl/split/csv/test_with_prompts_spatial.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_cgl_vit_both_text_spatial_trainseed1_inferseed1_test_output.pt",
        "per_image": ROOT / "experiments/paper_figures/ivc_prompt_cgl_vit_both_text_spatial_trainseed1_inferseed1_per_image.csv",
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_TITLE = font(20, True)
FONT_LABEL = font(18, True)
FONT_PROMPT = font(16)
FONT_PROMPT_BOLD = font(16, True)


def load_prompts(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {row["poster_path"]: row["text_prompt"] for row in rows}


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def load_prediction_elements(path: Path, class_names: dict[int, str]) -> dict[str, list[dict]]:
    payload = torch.load(path, map_location="cpu")
    img_names = list(payload["img_names"])
    test_output = payload["test_output"].detach().cpu()
    results: dict[str, list[dict]] = {}
    for image_name, output in zip(img_names, test_output):
        classes = output[:, 0].round().to(torch.int64)
        boxes = torch.clamp(cxcywh_to_xyxy(output[:, 1:]), 0.0, 1.0)
        elements = []
        for class_id, box in zip(classes.tolist(), boxes.tolist()):
            if class_id <= 0 or class_id not in class_names:
                continue
            x0, y0, x1, y1 = box
            if (x1 - x0) * (y1 - y0) <= 1e-4:
                continue
            elements.append(
                {
                    "class_name": class_names[class_id],
                    "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                }
            )
        results[image_name] = elements
    return results


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
        draw.rectangle(box, fill=(*color, 125), outline=(*color, 230), width=2)
    return Image.alpha_composite(out, overlay).convert("RGB")


def wrap_prompt(prompt: str, draw: ImageDraw.ImageDraw, max_width: int, max_lines: int = 7) -> list[str]:
    prompt = prompt.replace("Create ", "")
    lines: list[str] = []
    current = ""
    for word in prompt.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=FONT_PROMPT) <= max_width:
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


def draw_prompt_panel(prompt: str, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    y = 8
    draw.text((10, y), "Prompt:", fill=(0, 0, 0), font=FONT_PROMPT_BOLD)
    y += 22
    for line in wrap_prompt(prompt, draw, width - 20):
        draw.text((10, y), line, fill=(0, 0, 0), font=FONT_PROMPT)
        y += 22
    return panel


def make_tile(
    image_name: str,
    prompt: str,
    image_path: Path,
    elements: list[dict],
    metrics: dict,
) -> Image.Image:
    width = 360
    img_h = 526
    prompt_h = 185
    header_h = 58
    gap = 6
    image = Image.open(image_path).convert("RGB").resize((width, img_h), Image.LANCZOS)
    overlay = draw_box_overlay(image, elements)
    tile_h = header_h + prompt_h + gap + header_h + img_h + gap + header_h + img_h
    tile = Image.new("RGB", (width, tile_h), "white")
    draw = ImageDraw.Draw(tile)
    pla = metrics.get("pla_count", "")
    spla = metrics.get("spla", "")
    exact = metrics.get("exact_count_match", "")
    draw.text((8, 4), image_name, fill=(0, 0, 0), font=FONT_TITLE)
    draw.text(
        (8, 29),
        f"PLA={pla:.3g}  SPLA={spla:.3g}  Exact={exact:.3g}",
        fill=(0, 0, 0),
        font=FONT_PROMPT_BOLD,
    )
    y = header_h
    tile.paste(draw_prompt_panel(prompt, width, prompt_h), (0, y))
    y += prompt_h + gap
    draw.text((8, y + 5), "Input image", fill=(0, 0, 0), font=FONT_LABEL)
    y += header_h
    tile.paste(image, (0, y))
    y += img_h + gap
    draw.text((8, y + 5), "Image with predicted elements", fill=(0, 0, 0), font=FONT_LABEL)
    y += header_h
    tile.paste(overlay, (0, y))
    return tile


def select_rows(per_image_path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(per_image_path)
    for column in ("exact_count_match", "pla_count", "spla", "type_f1"):
        if column not in df.columns:
            df[column] = 0.0
    if "occ" not in df.columns:
        df["occ"] = 1.0
    df = df.sort_values(
        ["exact_count_match", "pla_count", "spla", "type_f1", "occ"],
        ascending=[False, False, False, False, True],
    )
    return df.head(limit).copy()


def export_candidates(mode: str, dataset: str, limit: int, clean: bool) -> None:
    config = CONFIGS[(mode, dataset)]
    out_dir = OUT_ROOT / mode / dataset
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(config["prompt_csv"])
    predictions = load_prediction_elements(config["prediction"], DATASET_CLASSES[dataset])
    selected = select_rows(config["per_image"], limit)
    rows_for_csv = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        image_name = getattr(row, "image")
        prompt = prompts.get(image_name, "")
        elements = predictions.get(image_name, [])
        metrics = {
            "pla_count": float(getattr(row, "pla_count", 0.0)),
            "spla": float(getattr(row, "spla", 0.0)),
            "exact_count_match": float(getattr(row, "exact_count_match", 0.0)),
            "type_f1": float(getattr(row, "type_f1", 0.0)),
            "occ": float(getattr(row, "occ", 0.0)),
        }
        tile = make_tile(
            image_name,
            prompt,
            config["image_dir"] / image_name,
            elements,
            metrics,
        )
        stem = Path(image_name).stem
        tile_path = out_dir / f"{rank:03d}_{stem}.png"
        tile.save(tile_path)
        rows_for_csv.append(
            {
                "rank": rank,
                "tile": tile_path.name,
                "image": image_name,
                "prompt": prompt,
                **metrics,
            }
        )
    pd.DataFrame(rows_for_csv).to_csv(out_dir / "index.csv", index=False)
    print(f"Wrote {len(rows_for_csv)} candidates to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--mode", choices=("free_form", "non_free_form", "both"), default="both")
    parser.add_argument("--dataset", choices=("pku", "cgl", "both"), default="both")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    modes = ["free_form", "non_free_form"] if args.mode == "both" else [args.mode]
    datasets = ["pku", "cgl"] if args.dataset == "both" else [args.dataset]
    for mode in modes:
        for dataset in datasets:
            export_candidates(mode, dataset, args.limit, clean=not args.no_clean)


if __name__ == "__main__":
    main()
