#!/usr/bin/env python3
"""Create deterministic annotation-derived prompts and free-form templates.

The generated basic/enhanced/advanced/spatial/rich prompts are oracle
diagnostics because they are derived from test annotations.  Free-form prompts
must be authored independently, so this script only creates blank assignment
templates for them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from generate_prompts import (  # noqa: E402
    create_rich_text_prompts,
    create_text_prompts_from_csv,
)
from utils.util import resolve_project_root  # noqa: E402


STYLES = ("basic", "enhanced", "advanced", "spatial")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def merge_prompt_files(csv_dir: Path, split: str, overwrite: bool) -> Path:
    output = csv_dir / f"{split}_with_all_prompts.csv"
    if output.exists() and not overwrite:
        return output

    sources = [
        csv_dir / f"{split}_with_rich_prompts.csv",
        *(csv_dir / f"{split}_with_prompts_{style}.csv" for style in STYLES),
    ]
    frames = []
    for source in sources:
        if source.exists():
            frame = pd.read_csv(source)
            if {"poster_path", "text_prompt"} <= set(frame.columns):
                frames.append(frame[["poster_path", "text_prompt"]])
    if not frames:
        raise RuntimeError(f"No prompt files are available for {csv_dir.parent.parent.name}/{split}")

    output.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values("poster_path", kind="stable", inplace=True)
    merged.to_csv(output, index=False, quoting=csv.QUOTE_ALL)
    return output


def prepare_derived_prompts(
    project_root: Path,
    dataset: str,
    splits: tuple[str, ...],
    overwrite: bool,
    rich_variations: int,
    seed: int,
) -> list[Path]:
    csv_dir = project_root / "data" / "dataset" / dataset / "split" / "csv"
    generated = []
    for split in splits:
        annotation = csv_dir / f"{split}.csv"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing source annotation: {annotation}")

        for style in STYLES:
            output = csv_dir / f"{split}_with_prompts_{style}.csv"
            if output.exists() and not overwrite:
                continue
            random.seed(stable_seed(seed, dataset, split, style))
            output.parent.mkdir(parents=True, exist_ok=True)
            if not create_text_prompts_from_csv(
                str(annotation), str(output), dataset_name=dataset, prompt_style=style
            ):
                raise RuntimeError(f"Failed to generate {output}")
            generated.append(output)

        rich = csv_dir / f"{split}_with_rich_prompts.csv"
        if overwrite or not rich.exists():
            random.seed(stable_seed(seed, dataset, split, "rich"))
            if not create_rich_text_prompts(
                str(annotation),
                str(rich),
                dataset_name=dataset,
                num_variations=rich_variations,
                use_augmentation=False,
            ):
                raise RuntimeError(f"Failed to generate {rich}")
            generated.append(rich)

        merged = merge_prompt_files(csv_dir, split, overwrite=overwrite or bool(generated))
        if merged not in generated:
            generated.append(merged)
    return generated


def create_freeform_template(
    project_root: Path, dataset: str, count: int, seed: int, overwrite: bool
) -> Path:
    annotation = (
        project_root / "data" / "dataset" / dataset / "split" / "csv" / "test.csv"
    )
    frame = pd.read_csv(annotation, usecols=["poster_path"])
    names = sorted(frame.poster_path.astype(str).unique().tolist())
    if len(names) < count:
        raise ValueError(f"{dataset} has only {len(names)} unique test images; requested {count}")
    chosen = random.Random(seed).sample(names, count)

    output = project_root / "data" / "prompts" / f"free_form_{dataset}_template.csv"
    if output.exists() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "poster_path": chosen,
            "text_prompt": [""] * len(chosen),
            "author_id": [""] * len(chosen),
            "independent_of_ground_truth": ["yes"] * len(chosen),
        }
    ).to_csv(output, index=False, quoting=csv.QUOTE_ALL)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    parser.add_argument("--datasets", nargs="+", choices=("pku", "cgl"), default=["pku", "cgl"])
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rich-variations", type=int, default=3)
    parser.add_argument("--freeform-count", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, root = resolve_project_root(args.path_profile)
    project_root = Path(root)
    random.seed(args.seed)

    for dataset_index, dataset in enumerate(args.datasets):
        generated = prepare_derived_prompts(
            project_root,
            dataset,
            tuple(args.splits),
            args.overwrite,
            args.rich_variations,
            args.seed,
        )
        template = create_freeform_template(
            project_root,
            dataset,
            args.freeform_count,
            args.seed + dataset_index,
            args.overwrite,
        )
        for path in generated:
            print(f"READY {path} sha256={sha256(path)}")
        print(f"TEMPLATE {template} sha256={sha256(template)}")

    print(
        "Free-form templates contain blank prompts by design. Have independent "
        "annotators fill them, then save copies without '_template' in the name."
    )


if __name__ == "__main__":
    main()
