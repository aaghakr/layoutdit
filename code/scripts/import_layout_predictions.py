"""Convert row-wise baseline CSV predictions to the shared tensor format."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coordinate-space", choices=("normalized", "pixels"), default="normalized")
    parser.add_argument("--width", type=float, default=513)
    parser.add_argument("--height", type=float, default=750)
    parser.add_argument("--max-elements", type=int, default=16)
    parser.add_argument("--box-format", choices=("xyxy", "cxcywh"), default="xyxy")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"poster_path", "cls_elem", "box_elem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    image_names = sorted(frame.poster_path.astype(str).unique().tolist())
    output = torch.zeros((len(image_names), args.max_elements, 5), dtype=torch.float32)
    groups = frame.groupby(frame.poster_path)
    invalid_image_names = []
    for image_index, image_name in enumerate(image_names):
        rows = groups.get_group(image_name).iloc[: args.max_elements]
        if "valid" in rows and not rows["valid"].astype(str).str.lower().isin(
            {"1", "true", "yes"}
        ).all():
            invalid_image_names.append(image_name)
            continue
        for element_index, (_, row) in enumerate(rows.iterrows()):
            try:
                box = np.asarray(ast.literal_eval(str(row.box_elem)), dtype=np.float64)
            except (ValueError, SyntaxError):
                invalid_image_names.append(image_name)
                output[image_index].zero_()
                break
            if args.coordinate_space == "pixels":
                box[[0, 2]] /= args.width
                box[[1, 3]] /= args.height
            if args.box_format == "xyxy":
                x1, y1, x2, y2 = box
                box = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            output[image_index, element_index, 0] = int(row.cls_elem)
            output[image_index, element_index, 1:] = torch.from_numpy(box.astype(np.float32))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "img_names": image_names,
        "test_output": output,
        "invalid_image_names": sorted(set(invalid_image_names)),
    }, destination)
    print(f"Wrote {len(image_names)} layouts to {destination}")


if __name__ == "__main__":
    main()
