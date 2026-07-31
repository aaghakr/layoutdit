#!/usr/bin/env python3
"""Validate portable IntentDiT YAML configuration files."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.util import load_config


CODE_ROOT = Path(__file__).resolve().parent
CONFIG_FILES = (
    "pku.yaml",
    "cgl.yaml",
    "pku_anno_test.yaml",
    "pku_unanno_test.yaml",
    "cgl_anno_test.yaml",
    "cgl_unanno_test.yaml",
    "pku_lambda_default.yaml",
    "pku_lambda_high_place.yaml",
    "pku_lambda_high_text.yaml",
)

MODEL_FIELDS = (
    "num_class",
    "max_elem",
    "d_model",
    "n_head",
    "n_layers",
    "feature_dim",
)
DATA_FIELDS = (
    "inp_dir",
    "sal_dir",
    "sal_sub_dir",
    "intent_map_dir",
    "salbox_dir",
    "intentbox_dir",
)


def validate(path: str, profile: str) -> list[str]:
    errors: list[str] = []
    cfg = load_config(path, path_profile=profile)
    for field in MODEL_FIELDS:
        if not hasattr(cfg, field):
            errors.append(f"missing model field: {field}")
    if not hasattr(cfg, "paths"):
        return errors + ["missing paths section"]
    for split in ("train", "test"):
        if not hasattr(cfg.paths, split):
            continue
        section = getattr(cfg.paths, split)
        for field in DATA_FIELDS:
            if not hasattr(section, field):
                errors.append(f"missing paths.{split}.{field}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-profile", choices=("local", "server"), default="local")
    args = parser.parse_args()

    failed = 0
    for name in CONFIG_FILES:
        config = CODE_ROOT / "configs" / name
        if not config.is_file():
            print(f"FAIL {config}: file not found")
            failed += 1
            continue
        errors = validate(str(config), args.path_profile)
        if errors:
            print(f"FAIL {config}: {'; '.join(errors)}")
            failed += 1
        else:
            print(f"OK   {config}")
    print(f"Validated {len(CONFIG_FILES) - failed}/{len(CONFIG_FILES)} configurations.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
