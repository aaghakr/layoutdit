"""Build deterministic relation, OOD-count, synonym, and conflict prompts."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


CLASS_NAMES = {1: "text", 2: "logo", 3: "underlay", 4: "embellishment"}
SYNONYMS = {1: "caption", 2: "brand mark", 3: "background panel", 4: "decoration"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations)
    required = {"poster_path", "cls_elem", "box_elem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    rows = []
    for sample_index, (poster_path, group) in enumerate(frame.groupby("poster_path", sort=True)):
        classes = [int(value) for value in group.cls_elem]
        boxes = [ast.literal_eval(str(value)) for value in group.box_elem]
        counts = {class_id: classes.count(class_id) for class_id in sorted(set(classes))}
        category = sample_index % 4
        if category == 0 and len(classes) >= 2:
            left_name, right_name = CLASS_NAMES[classes[0]], CLASS_NAMES[classes[1]]
            left_box, right_box = boxes[0], boxes[1]
            relation = "above" if (left_box[1] + left_box[3]) < (right_box[1] + right_box[3]) else "below"
            prompt = f"{left_name}_0 should be {relation} {right_name}_0."
            prompt_category, conflicting = "relative_relation", False
        elif category == 1:
            class_id = 1 if 1 in CLASS_NAMES else classes[0]
            prompt = f"Place 10 {CLASS_NAMES[class_id]} elements while keeping all content readable."
            prompt_category, conflicting = "ood_count", False
        elif category == 2:
            fragments = [f"{count} {SYNONYMS[class_id]}" for class_id, count in counts.items() if class_id in SYNONYMS]
            prompt = "Create a design with " + ", ".join(fragments) + "."
            prompt_category, conflicting = "synonym", False
        elif len(classes) >= 2:
            left_name, right_name = CLASS_NAMES[classes[0]], CLASS_NAMES[classes[1]]
            prompt = (
                f"{left_name}_0 should be above {right_name}_0 and "
                f"{left_name}_0 should be below {right_name}_0."
            )
            prompt_category, conflicting = "contradictory", True
        else:
            prompt = "Place 10 text elements while keeping all content readable."
            prompt_category, conflicting = "ood_count", False
        rows.append(
            {
                "poster_path": poster_path,
                "text_prompt": prompt,
                "prompt_category": prompt_category,
                "is_conflicting": int(conflicting),
            }
        )
        if len(rows) >= args.limit:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} controlled stress prompts to {output}")


if __name__ == "__main__":
    main()
