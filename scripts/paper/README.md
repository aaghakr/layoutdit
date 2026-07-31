# Numbered paper experiment pipeline

Do not start training until stage 01 passes. Every evaluation records its protocol,
checkpoint, seed, DDIM steps, parameters, latency, memory, aggregate metrics, and
per-image metrics.

## Preparation after a fresh data restore

Run this once before stage 01. It generates missing deterministic oracle prompt
families, creates blank assignments for independent free-form prompt authors, and
resumes any missing test intent-map predictions:

```bash
PATH_PROFILE=server bash scripts/paper/prepare_missing_inputs.sh
```

The command deliberately does not author free-form prompts. Give
`data/prompts/free_form_{pku,cgl}_template.csv` to annotators who cannot see the
ground-truth boxes. After at least 100 rows per dataset are filled, save them as
`free_form_pku.csv` and `free_form_cgl.csv`. Stage 01 validates non-empty prompts,
held-out image membership, and unique-image count.

Training does not consume these independent files. To begin stages 02–06 while
annotators are working, validate the training inputs with:

```bash
SKIP_FREEFORM_CHECK=1 PATH_PROFILE=server bash scripts/paper/01_validate_environment.sh
bash scripts/paper/02_train_paper_models.sh
```

Do not use `SKIP_FREEFORM_CHECK` for the complete stage-13 pipeline; stages 07 and
09 independently enforce the finished prompt files.

## Exact order

1. `01_validate_environment.sh` — tests, configs, models, datasets, maps, and prompts.
2. `02_train_paper_models.sh` — 18 main variants × 3 seeds.
3. `03_train_loss_sweep.sh` — five loss settings × 3 seeds.
4. `04_train_architecture_ablations.sh` — pooled text, pixel-map-only, and
   intent-box-only × 3 seeds.
5. `05_evaluate_paper_models.sh` — 78 checkpoints with one fixed inference seed.
6. `06_aggregate_paper_results.sh` — independent-training-seed summaries.
7. `07_evaluate_prompt_robustness.sh` — seven prompt families, including independent
   free-form prompts and relation/OOD/conflict stress tests.
8. `08_evaluate_sampling_efficiency.sh` — 10/20/50/100 DDIM steps and profiling.
9. `09_evaluate_diversity.sh` — five samples for each identical free-form condition.
10. `10_evaluate_cross_dataset.sh` — PKU→CGL and CGL→PKU transfer.
11. `11_evaluate_external_baselines.sh` — real baselines, paired bootstrap CIs,
    and optional official metric-parity / official-feature LayoutFID evidence.
12. `12_run_diagnostics.sh` — intent diagnostics, failure taxonomy, and study stimuli.
13. `16_evaluate_text_parser_ablation.sh` — free-form no-parser and parser-only
    text-conditioning diagnostics.
14. `17_analyze_parser_audit.sh` — analyze manually reviewed parser-audit CSVs.
15. `18_evaluate_manual_freeform_reference.sh` — re-score free-form outputs against
    manual prompt references and generate paired bootstrap CIs.
16. `13_run_all_automated.sh` — automated stages 01–12 plus stage 16.
17. `14_prepare_user_study.sh` — separate quality and instruction-adherence trials.
18. `15_verify_paper_evidence.sh` — final gate after participant collection.

The complete matrix is 78 training runs and 235 internal inference runs before
third-party baseline evaluation, plus 12 parser-ablation inference runs for the
parser diagnostic. Preview it first.

## Server execution

```bash
cd /path/to/intentdit
conda activate <your-environment>

export PATH_PROFILE=server
export GPU_IDS=0,1,2
export SEEDS="1 2 3"
export PYTHON_BIN=python
export FINAL_EPOCH=500
export INFERENCE_SEED=1
export DDIM_STEPS=100
export DDIM_SCHEDULE=cosine

PATH_PROFILE=server bash scripts/paper/prepare_missing_inputs.sh
DRY_RUN=1 bash scripts/paper/13_run_all_automated.sh
bash scripts/paper/13_run_all_automated.sh
```

## Two-A100 training profile

Stages 02–04 distribute the full run matrix—not whole seeds—across the GPU list,
so three seeds remain balanced over two devices. The paper scripts default to
A100-native BF16, TF32 for residual FP32 matrix operations, pinned/persistent data
loading, and no repeated DDIM validation during training. The final epoch-500
checkpoint is always saved and stage 05 performs the standardized evaluation.

```bash
export GPU_IDS=0,1
export TRAIN_PRECISION=bf16
export TRAIN_TF32=1
export SKIP_TRAINING_VALIDATION=1
export TRAIN_NUM_WORKERS=16
export JOBS_PER_GPU=1

bash scripts/paper/02_train_paper_models.sh
bash scripts/paper/03_train_loss_sweep.sh
bash scripts/paper/04_train_architecture_ablations.sh
```

The benchmark protocol retains PKU batch size 32 and CGL batch size 128. Optional
`TRAIN_BATCH_SIZE_PKU` and `TRAIN_BATCH_SIZE_CGL` overrides exist for profiling,
but changing them changes optimization and must be applied to every comparable run
and reported in the manuscript. Restore the slower legacy compute path with
`TRAIN_PRECISION=fp32 TRAIN_TF32=0 SKIP_TRAINING_VALIDATION=0`.

If `nvidia-smi` shows low utilization and ample memory, keep the benchmark batch
sizes unchanged and schedule multiple independent runs per device instead:

```bash
export GPU_IDS=0,1
export JOBS_PER_GPU=2
export TRAIN_NUM_WORKERS=8
bash scripts/paper/02_train_paper_models.sh
```

Two jobs per A100 preserve each run's optimization protocol. Do not start a second
copy of the same stage concurrently: a run is considered complete only after its
final checkpoint exists, so concurrent stage copies could duplicate unfinished
experiments.

## DDIM schedule protocol

The trained denoiser uses a cosine DDPM noise schedule. For the main paper
tables, evaluation defaults to `DDIM_SCHEDULE=cosine`, using the same cumulative
alpha schedule during DDIM inference as during DDPM training. This is the
submission protocol; all affected main results must be regenerated under this
schedule.

`DDIM_SCHEDULE=training` is retained as an explicit alias for the training
cumulative-alpha schedule, and `DDIM_SCHEDULE=linear` is retained only for
legacy/sensitivity checks. Do not mix linear and cosine metrics in the same
paper tables.

After changing `DDIM_SCHEDULE`, rerun inference/aggregation because the scripts
check the schedule recorded in each evidence file. Evidence files that predate
explicit schedule recording are treated as stale and will not satisfy the final
verification gate. Use `FORCE=1` if you want to overwrite everything deliberately:

```bash
cd /path/to/intentdit
export PATH_PROFILE=server
export GPU_IDS=0,1
export PYTHON_BIN=python
export DDIM_SCHEDULE=cosine
export DDIM_STEPS=100

bash scripts/paper/05_evaluate_paper_models.sh
bash scripts/paper/06_aggregate_paper_results.sh
```

Aggregation writes both schedule-specific and current-summary outputs:

```text
experiments/paper_results/primary_training_seed_summary_linear.json
experiments/paper_results/primary_training_seed_summary_cosine.json
experiments/paper_results/primary_training_seed_summary_training.json
experiments/paper_results/primary_training_seed_summary.json
```

Only after the corrected main tables are inspected should you rerun the longer
prompt, diversity, cross-dataset, and diagnostic stages:

```bash
bash scripts/paper/07_evaluate_prompt_robustness.sh
bash scripts/paper/08_evaluate_sampling_efficiency.sh
bash scripts/paper/09_evaluate_diversity.sh
bash scripts/paper/10_evaluate_cross_dataset.sh
bash scripts/paper/12_run_diagnostics.sh
bash scripts/paper/16_evaluate_text_parser_ablation.sh
bash scripts/paper/17_analyze_parser_audit.sh
bash scripts/paper/18_evaluate_manual_freeform_reference.sh
bash scripts/paper/15_verify_paper_evidence.sh
```

Use `PATH_PROFILE=local` in this checkout. Local paths resolve from the current
repository; the server profile resolves from the fixed server path above.

## Evaluation protocols

- `image_only`: no prompt, count, position, or test annotation enters the model.
- `oracle_prompt`: controlled prompt derived from a test annotation; diagnostic only.
- `freeform_prompt`: independently authored instruction without GT boxes.
- `text_baseline`: a third-party method receives the identical free-form prompt.
- `cross_dataset`: source model tested without target fine-tuning.

Prompted and image-only rows must never be ranked as one protocol.

## Independent prompt files

```text
data/prompts/free_form_pku.csv
data/prompts/free_form_cgl.csv
```

Each needs `poster_path` and `text_prompt` (or `prompt`), at least 100 unique held-out
images, and prompts authored without viewing ground-truth boxes.

## External evidence for stage 11

Supply actual baseline predictions in standardized
`{'img_names', 'test_output'}` `.pt` files:

```bash
export LAYOUTDIT_PKU_PREDICTIONS=/evidence/layoutdit_pku.pt
export LAYOUTDIT_CGL_PREDICTIONS=/evidence/layoutdit_cgl.pt
export TEXT_BASELINE_PKU_PREDICTIONS=/evidence/text_baseline_pku.pt
export TEXT_BASELINE_CGL_PREDICTIONS=/evidence/text_baseline_cgl.pt
```

Convert row-wise CSV predictions if necessary:

```bash
python code/scripts/import_layout_predictions.py \
  --input predictions.csv --output predictions.pt \
  --coordinate-space pixels --box-format xyxy
```

Optional official LayoutFID requires features from the same published
extractor/checkpoint. If these are unavailable, do not report LayoutFID; the
repository's `hfd` number remains only a diagnostic:

```bash
export OFFICIAL_LAYOUT_EXTRACTOR_NAME=<published-extractor>
export OFFICIAL_LAYOUT_EXTRACTOR_CHECKSUM=<sha256>
export OFFICIAL_REAL_LAYOUT_FEATURES_PKU=/evidence/pku_real_features.npz
export OFFICIAL_INTENTDIT_LAYOUT_FEATURES_PKU=/evidence/pku_intentdit_features.npz
export OFFICIAL_REAL_LAYOUT_FEATURES_CGL=/evidence/cgl_real_features.npz
export OFFICIAL_INTENTDIT_LAYOUT_FEATURES_CGL=/evidence/cgl_intentdit_features.npz
```

For optional metric-parity validation, independently run the official
PosterLayout/LayoutDiT evaluator on the saved IntentDiT outputs and provide its
aggregate JSON:

```bash
export OFFICIAL_METRIC_REFERENCE_PKU=/evidence/official_metrics_pku.json
export OFFICIAL_METRIC_REFERENCE_CGL=/evidence/official_metrics_cgl.json
```

Set `REQUIRE_BASELINES=1` to fail stage 11 immediately when external prediction
evidence is missing. Set `REQUIRE_OFFICIAL_LAYOUT_FID=1` or
`REQUIRE_METRIC_PARITY=1` only if you intend to make those official claims.

## Free-form parser ablation

After any change to `data/prompts/free_form_{pku,cgl}.csv` or
`code/cgbdm/text_spatial.py`, force-refresh the free-form evidence:

```bash
FORCE=1 PROMPT_STYLES=freeform bash scripts/paper/07_evaluate_prompt_robustness.sh
FORCE=1 bash scripts/paper/11_evaluate_external_baselines.sh
bash scripts/paper/16_evaluate_text_parser_ablation.sh
bash scripts/paper/12_run_diagnostics.sh
bash scripts/paper/15_verify_paper_evidence.sh
```

Stage 16 evaluates two diagnostics with the same trained full text model:
`no_parser` keeps BERT token conditioning but disables parsed text-spatial boxes;
`parser_only` keeps parsed text-spatial boxes but sets BERT text-token guidance
to zero. This separates whether free-form gains come mainly from language
tokens, parser-derived spatial tokens, or their combination.

## User study

Quality trials compare image-only IntentDiT with actual LayoutDiT. Instruction
trials compare text IntentDiT with a genuine text-conditioned baseline.

```bash
LAYOUTDIT_RENDER_DIR=/evidence/renders/layoutdit \
TEXT_BASELINE_RENDER_DIR=/evidence/renders/textbaseline \
  bash scripts/paper/14_prepare_user_study.sh

export INTENTDIT_USER_STUDY_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python user_study/app.py --host 0.0.0.0 --port 5000
bash scripts/paper/15_verify_paper_evidence.sh
```

## Output contract

For experiment `<name>`, evaluation writes:

```text
experiments/paper_figures/<name>_metrics.json
experiments/paper_figures/<name>_per_image.csv
experiments/paper_figures/<name>_evidence.json
experiments/paper_figures/<name>_test_output.pt
```

Every new evidence file records `ddim_schedule`; final verification fails if the
paper evidence does not match the active `DDIM_SCHEDULE` and DDIM step count.

`hfd` is a handcrafted-feature diagnostic and must not be called layout FID. The
reported layout-FID value comes only from the official extractor with its identity and
checksum recorded.
