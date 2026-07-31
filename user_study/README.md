# IntentDiT user study

The Flask application runs a blinded pairwise study over rendered posters.
Stimuli and responses are local-only study artifacts.

## Prepare stimuli

Place matching filenames in four protocol-specific folders:

```text
user_study/data/renders/intentdit_image/
user_study/data/renders/layoutdit/
user_study/data/renders/intentdit_text/
user_study/data/renders/textbaseline/
```

`layoutdit/` and `textbaseline/` must contain actual baseline outputs, not renamed
IntentDiT ablations.

```bash
python user_study/build_manifest.py \
  --quality-n 30 --instruction-n 30 \
  --prompts-csv data/prompts/free_form_pku.csv --per-category 5 --seed 42
```

## Run

```bash
export INTENTDIT_USER_STUDY_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python user_study/app.py --host 0.0.0.0 --port 5000
```

## Aggregate

```bash
python user_study/aggregate_results.py
```

Quality and instruction adherence are separate randomized questions. Descriptive
outputs are written under `experiments/user_study/`. Final inference uses crossed
participant/item bootstraps:

```bash
python user_study/hierarchical_bootstrap.py \
  --criterion quality --method-a intentdit_image --baseline layoutdit \
  --min-participants 20
python user_study/hierarchical_bootstrap.py \
  --criterion instruction --method-a intentdit_text --baseline textbaseline \
  --min-participants 20
```

Pooled McNemar/binomial tests are invalid for repeated ratings.
