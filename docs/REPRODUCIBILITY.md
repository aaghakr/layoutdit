# Reproducibility protocol

## Required paper runs

Train independent seeds 1, 2, and 3 for:

- PKU, ViT and Swin: Saliency; Intent; Saliency+Intent; Saliency+Text;
  Intent+Text; Saliency+Intent+Text.
- CGL, ViT: the same six conditioning variants.
- PKU loss isolation/sensitivity: `(0,0)`, `(0.1,0)`, `(0,0.05)`,
  `(0.1,0.05)`, `(0.2,0.05)`, and `(0.1,0.10)`.
- PKU architecture: token-level versus pooled text, and intent pixel maps with
  versus without intent-box tokens.

Use the same checkpoint rule and 100 DDIM sampling steps throughout. Evaluate
one fixed inference seed for the primary training-seed comparison; report
additional sampling seeds separately.

Also run controlled and independently authored prompts, relation/OOD/conflict
stress tests, quality-aware diversity, DDIM speed-quality profiling, cross-dataset
transfer, protocol-matched baselines, density-control diagnostics, free-form
parser coverage, and the parser-disabled/parser-only text diagnostic. Official
metric parity and official LayoutFID are optional evidence gates; report them
only if the official evaluator/extractor and provenance are available.

The executable source of truth is `scripts/paper/README.md` and its numbered
scripts.

## Required provenance

Archive, outside Git:

- exact command and resolved YAML configuration;
- Git commit hash;
- training and inference seeds;
- checkpoint and epoch-selection rule;
- raw per-image predictions and metric JSON;
- per-image metric CSV and evidence/profiling JSON;
- dataset/prompt split checksums;
- hardware and software environment.

## Reporting guardrails

- Do not combine inference seeds and call them independent training runs.
- Do not rank prompted and image-only methods as protocol-equivalent.
- Do not reuse the historical auxiliary-loss sweep after the loss correction.
- Human-study inference must account for repeated participant and item ratings.
- Generate paper tables from archived JSON/CSV outputs, not manual entry.
- `hfd` is diagnostic only; never report it as standard layout FID.
- If official LayoutFID or official metric-parity evidence is unavailable, do
  not claim those values; keep the limitation explicit.
- State the exact matching definitions for paired IoU and category-multiset MaxIoU.
- Quality and instruction-following studies must use separate matched methods.
