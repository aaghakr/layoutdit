"""Compute layout FID from features produced by an official feature extractor.

This command deliberately accepts features, not boxes. It prevents a custom
encoder from being silently presented as the standard LayoutGAN++/LayoutDM FID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm


def load_features(path: str) -> np.ndarray:
    source = np.load(path)
    if isinstance(source, np.lib.npyio.NpzFile):
        if "features" not in source:
            raise ValueError(f"{path} must contain an array named 'features'")
        source = source["features"]
    features = np.asarray(source, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError(f"Expected [N,D] features with N>=2, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite feature values in {path}")
    return features


def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float:
    mu_real, mu_generated = real.mean(0), generated.mean(0)
    cov_real = np.cov(real, rowvar=False)
    cov_generated = np.cov(generated, rowvar=False)
    covariance_mean = sqrtm(cov_real @ cov_generated)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    delta = mu_real - mu_generated
    return float(max(delta @ delta + np.trace(cov_real + cov_generated - 2 * covariance_mean), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-features", required=True)
    parser.add_argument("--generated-features", required=True)
    parser.add_argument("--extractor-name", required=True)
    parser.add_argument("--extractor-checksum", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    real = load_features(args.real_features)
    generated = load_features(args.generated_features)
    if real.shape[1] != generated.shape[1]:
        raise ValueError(f"Feature dimensions differ: {real.shape} vs {generated.shape}")
    result = {
        "layout_fid": frechet_distance(real, generated),
        "extractor_name": args.extractor_name,
        "extractor_checksum": args.extractor_checksum,
        "num_real": len(real),
        "num_generated": len(generated),
        "feature_dim": real.shape[1],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote official-feature layout FID to {output}")


if __name__ == "__main__":
    main()
