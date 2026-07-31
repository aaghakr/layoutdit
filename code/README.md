# IntentDiT implementation

Run commands from this directory so imports resolve consistently.

## Main entry points

- `scripts/train.py`: train one model with an explicit independent seed.
- `scripts/test.py`: evaluate a checkpoint and optionally save raw tensors.
- `scripts/run_single_image.py`: prompt-conditioned inference on one image.
- `scripts/run_fair_eval_all.py`: evaluate a YAML list under one protocol.
- `scripts/run_multi_seed_eval.py`: repeated inference sampling.
- `scripts/aggregate_training_seeds.py`: aggregate independently trained runs.
- `scripts/evaluate_saved_predictions.py`: shared evaluation for external methods.
- `scripts/paired_bootstrap.py`: paired image-level confidence intervals.
- `scripts/analyze_diversity.py`: repeated-condition diversity analysis.
- `scripts/analyze_density_controls.py`: element-count and area diagnostics for
  occlusion/readability confounds.
- `scripts/analyze_freeform_parser_coverage.py`: parser coverage and free-form
  prompt subgroup diagnostics.
- `scripts/compute_layout_fid.py`: FID from official extractor features.
- `generate_prompts.py`: create structural prompt CSVs.

Evaluation writes aggregate JSON, per-image CSV, profiling/evidence JSON, and
standardized raw tensors. Metrics cover geometry, poster content, paired reference
similarity, category-multiset MaxIoU, prompt counts, absolute position, relations,
and hierarchy. `hfd` is only a deterministic diagnostic; standard layout FID must
use the published extractor and checkpoint.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Configuration

Portable YAML files live in `configs/`. Runtime paths are resolved by
`utils/util.py` using `--path-profile local|server`; no dataset or checkpoint
is stored in Git.

See the root [README](../README.md) and
[reproducibility protocol](../docs/REPRODUCIBILITY.md).
