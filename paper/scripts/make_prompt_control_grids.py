import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "pku": {
        "title": "PKU spatial prompt-control examples",
        "image_dir": ROOT / "data/dataset/pku/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/dataset/pku/split/csv/test_with_prompts_spatial.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_pku_vit_both_text_spatial_trainseed1_inferseed1_test_output.pt",
        "examples": ["1763.png", "2626.png", "8144.png", "4079.png", "3043.png"],
        "classes": {1: "Text", 2: "Logo", 3: "Underlay"},
    },
    "cgl": {
        "title": "CGL spatial prompt-control examples",
        "image_dir": ROOT / "data/dataset/cgl/split/test_anno/inpaint",
        "prompt_csv": ROOT / "data/dataset/cgl/split/csv/test_with_prompts_spatial.csv",
        "prediction": ROOT / "experiments/paper_figures/ivc_prompt_cgl_vit_both_text_spatial_trainseed1_inferseed1_test_output.pt",
        "examples": ["10453.png", "14373.png", "17211.png", "19039.png", "37455.png"],
        "classes": {1: "Text", 2: "Logo", 3: "Underlay", 4: "Embellishment"},
    },
}

CLASS_COLORS = {
    "Text": (96, 176, 72),
    "Logo": (239, 120, 148),
    "Underlay": (64, 142, 206),
    "Embellishment": (230, 154, 55),
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


FONT_TITLE = font(28, True)
FONT_LABEL = font(22, True)
FONT_PROMPT = font(17)
FONT_PROMPT_BOLD = font(17, True)


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


def prepare_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((width, height), Image.LANCZOS)


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


def wrap_prompt(
    prompt: str,
    draw: ImageDraw.ImageDraw,
    max_width: int,
    max_lines: int = 7,
) -> list[str]:
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


def prompt_panel(image: Image.Image, prompt: str, width: int, image_height: int, prompt_height: int) -> Image.Image:
    tile = Image.new("RGB", (width, prompt_height), "white")
    draw = ImageDraw.Draw(tile)
    y = 9
    draw.text((10, y), "Prompt:", fill=(0, 0, 0), font=FONT_PROMPT_BOLD)
    y += 23
    lines = wrap_prompt(prompt, draw, width - 20)
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=FONT_PROMPT)
        y += 23
    return tile


def build_grid(dataset: str, config: dict) -> Image.Image:
    prompts = load_prompts(config["prompt_csv"])
    predictions = load_prediction_elements(config["prediction"], config["classes"])
    col_w = 300
    img_h = 438
    prompt_h = 176
    gap = 8
    margin = 18
    title_h = 44
    cols = len(config["examples"])
    width = margin * 2 + cols * col_w + (cols - 1) * gap
    label_h = 26
    height = title_h + label_h + prompt_h + gap + label_h + img_h + gap + label_h + img_h + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 5), config["title"], fill=(0, 0, 0), font=FONT_TITLE)
    y_prompt_label = title_h
    y_prompt = y_prompt_label + label_h
    y_input_label = y_prompt + prompt_h + gap
    y_input = y_input_label + label_h
    y_elements_label = y_input + img_h + gap
    y_elements = y_elements_label + label_h
    draw.text((margin, y_prompt_label), "Prompt", fill=(0, 0, 0), font=FONT_LABEL)
    draw.text((margin, y_input_label), "Input image", fill=(0, 0, 0), font=FONT_LABEL)
    draw.text((margin, y_elements_label), "Image with predicted elements", fill=(0, 0, 0), font=FONT_LABEL)
    for idx, image_name in enumerate(config["examples"]):
        elements = predictions[image_name]
        image = prepare_image(config["image_dir"] / image_name, col_w, img_h)
        boxes = draw_box_overlay(image, elements)
        prompt = prompt_panel(image, prompts[image_name], col_w, img_h, prompt_h)
        x = margin + idx * (col_w + gap)
        canvas.paste(prompt, (x, y_prompt))
        canvas.paste(image, (x, y_input))
        canvas.paste(boxes, (x, y_elements))
    return canvas


def main() -> None:
    out_dir = ROOT / "paper/fig"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, config in DATASETS.items():
        grid = build_grid(dataset, config)
        png_path = out_dir / f"fig06_{dataset}.png"
        pdf_path = out_dir / f"fig06_{dataset}.pdf"
        grid.save(png_path)
        grid.save(pdf_path, resolution=300.0)
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
