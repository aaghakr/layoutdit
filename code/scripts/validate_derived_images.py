#!/usr/bin/env python3
"""Check that every source image has a corresponding derived image."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    parser.add_argument("--label", default="derived images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_dir.is_dir() or not args.derived_dir.is_dir():
        raise FileNotFoundError(
            f"Missing directory: source={args.source_dir}, derived={args.derived_dir}"
        )
    sources = {
        path.name for path in args.source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    }
    derived = {
        path.name for path in args.derived_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not sources:
        raise RuntimeError(f"No source images found in {args.source_dir}")
    missing = sorted(sources - derived)
    if missing:
        raise RuntimeError(
            f"{args.label} incomplete: {len(missing)}/{len(sources)} absent; "
            f"first examples: {missing[:5]}"
        )
    print(f"OK {args.label}: {len(sources)}/{len(sources)} images")


if __name__ == "__main__":
    main()
