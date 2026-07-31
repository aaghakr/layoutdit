#!/usr/bin/env python3
"""Generate placement-suitability maps for one prepared dataset split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object in {path}: {type(state).__name__}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def image_tensor(path: Path, preprocess) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = preprocess(image).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, default=513)
    parser.add_argument("--height", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {args.input_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    pending = [
        path for path in inputs if args.overwrite or not (args.output_dir / path.name).is_file()
    ]
    if not inputs:
        raise RuntimeError(f"No images found in {args.input_dir}")
    if not pending:
        print(f"READY {args.output_dir}: {len(inputs)} maps (nothing to generate)")
        return

    # Some timm releases probe an optional wandb install while importing.  The
    # predictor does not log to wandb, so keep a broken optional installation
    # from preventing deterministic offline inference.
    sys.modules.setdefault("wandb", None)
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    from model import design_intent_detector

    model = design_intent_detector(act="none", action="forward", encoder_weights=None)
    missing, unexpected = model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    device = torch.device(args.device)
    model.to(device).eval()
    preprocess = get_preprocessing_fn("mit_b1", pretrained="imagenet")

    print(f"Generating {len(pending)}/{len(inputs)} maps on {device}")
    with torch.inference_mode():
        for offset in range(0, len(pending), args.batch_size):
            paths = pending[offset : offset + args.batch_size]
            batch = torch.stack([image_tensor(path, preprocess) for path in paths]).to(device)
            predictions = model(batch).detach().float().cpu().numpy()
            for path, prediction in zip(paths, predictions):
                array = np.clip(prediction.squeeze(), 0.0, 1.0)
                array = cv2.resize(
                    array, (args.width, args.height), interpolation=cv2.INTER_LINEAR
                )
                output = np.rint(array * 255.0).astype(np.uint8)
                if not cv2.imwrite(str(args.output_dir / path.name), output):
                    raise OSError(f"Could not write {args.output_dir / path.name}")
            print(f"[{min(offset + len(paths), len(pending))}/{len(pending)}]")

    absent = [path.name for path in inputs if not (args.output_dir / path.name).is_file()]
    if absent:
        raise RuntimeError(f"Map generation incomplete; {len(absent)} outputs absent")
    print(f"READY {args.output_dir}: {len(inputs)} maps")


if __name__ == "__main__":
    main()
