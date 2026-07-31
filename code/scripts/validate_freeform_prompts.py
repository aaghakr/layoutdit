#!/usr/bin/env python3
"""Validate independently authored free-form prompt files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--minimum-images", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = pd.read_csv(args.prompts, keep_default_na=False)
    annotations = pd.read_csv(args.annotations, usecols=["poster_path"])
    prompt_column = "text_prompt" if "text_prompt" in prompts else "prompt"
    required = {"poster_path", prompt_column}
    if not required <= set(prompts.columns):
        raise ValueError(f"{args.prompts} must contain poster_path and text_prompt (or prompt)")

    paths = prompts.poster_path.astype(str).str.strip()
    texts = prompts[prompt_column].astype(str).str.strip()
    if (paths == "").any():
        raise ValueError(f"{args.prompts} contains blank poster_path values")
    if (texts == "").any():
        raise ValueError(
            f"{args.prompts} contains {(texts == '').sum()} blank prompts; "
            "fill the independent-author template before evaluation"
        )
    unique_images = paths.nunique()
    if unique_images < args.minimum_images:
        raise ValueError(
            f"{args.prompts} has {unique_images} unique images; need at least {args.minimum_images}"
        )

    valid_images = set(annotations.poster_path.astype(str))
    unknown = sorted(set(paths) - valid_images)
    if unknown:
        raise ValueError(
            f"{args.prompts} names {len(unknown)} images outside the official test split; "
            f"first examples: {unknown[:5]}"
        )
    print(
        f"OK {args.prompts}: {len(prompts)} prompts, {unique_images} unique held-out images"
    )


if __name__ == "__main__":
    main()
