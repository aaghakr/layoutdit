#!/usr/bin/env bash
# Prepare a fresh blinded comparison manifest from final model renders.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"

QUALITY_TRIALS=${QUALITY_TRIALS:-30}
INSTRUCTION_TRIALS=${INSTRUCTION_TRIALS:-30}
STUDY_SEED=${STUDY_SEED:-42}
LAYOUTDIT_RENDER_DIR=${LAYOUTDIT_RENDER_DIR:-}
TEXT_BASELINE_RENDER_DIR=${TEXT_BASELINE_RENDER_DIR:-}
QUALITY_INTENTDIT_RENDER_DIR=${QUALITY_INTENTDIT_RENDER_DIR:-"$METRIC_DIR/ivc_pku_vit_both_trainseed1_inferseed${INFERENCE_SEED}"}
TEXT_INTENTDIT_RENDER_DIR=${TEXT_INTENTDIT_RENDER_DIR:-"$METRIC_DIR/ivc_prompt_pku_vit_both_text_freeform_trainseed1_inferseed${INFERENCE_SEED}"}
PROMPTS_CSV=${PROMPTS_CSV:-"$ACTIVE_ROOT/data/prompts/free_form_pku.csv"}
STUDY_DATA="$ACTIVE_ROOT/user_study/data"

[[ -d "$QUALITY_INTENTDIT_RENDER_DIR" ]] || die "Image-only IntentDiT renders not found: $QUALITY_INTENTDIT_RENDER_DIR"
[[ -d "$TEXT_INTENTDIT_RENDER_DIR" ]] || die "Text IntentDiT renders not found: $TEXT_INTENTDIT_RENDER_DIR"
[[ -d "$LAYOUTDIT_RENDER_DIR" ]] || die "Set LAYOUTDIT_RENDER_DIR to actual LayoutDiT renders"
[[ -d "$TEXT_BASELINE_RENDER_DIR" ]] || die "Set TEXT_BASELINE_RENDER_DIR to the genuine text baseline renders"
[[ -f "$PROMPTS_CSV" ]] || die "Prompt file not found: $PROMPTS_CSV"

mkdir -p "$STUDY_DATA/archive"
for method in intentdit_image layoutdit intentdit_text textbaseline; do
    mkdir -p "$STUDY_DATA/renders/$method"
    find "$STUDY_DATA/renders/$method" -maxdepth 1 -type f -name '*.png' -delete
done

if [[ -f "$STUDY_DATA/study.sqlite" ]]; then
    mv "$STUDY_DATA/study.sqlite" \
       "$STUDY_DATA/archive/study_$(date +%Y%m%d_%H%M%S).sqlite"
fi

find "$QUALITY_INTENTDIT_RENDER_DIR" -maxdepth 1 -type f -name '*.png' -exec cp {} "$STUDY_DATA/renders/intentdit_image/" \;
find "$LAYOUTDIT_RENDER_DIR" -maxdepth 1 -type f -name '*.png' -exec cp {} "$STUDY_DATA/renders/layoutdit/" \;
find "$TEXT_INTENTDIT_RENDER_DIR" -maxdepth 1 -type f -name '*.png' -exec cp {} "$STUDY_DATA/renders/intentdit_text/" \;
find "$TEXT_BASELINE_RENDER_DIR" -maxdepth 1 -type f -name '*.png' -exec cp {} "$STUDY_DATA/renders/textbaseline/" \;

run_command "$PYTHON_BIN" "$ACTIVE_ROOT/user_study/build_manifest.py" \
    --quality-n "$QUALITY_TRIALS" --instruction-n "$INSTRUCTION_TRIALS" \
    --prompts-csv "$PROMPTS_CSV" --per-category "${FAILURE_EXAMPLES:-5}" \
    --seed "$STUDY_SEED" \
    --out "$STUDY_DATA/manifest.json"

run_command "$PYTHON_BIN" -c \
    "import json; p='$STUDY_DATA/manifest.json'; m=json.load(open(p)); expected=$QUALITY_TRIALS+$INSTRUCTION_TRIALS; assert len(m['part_a'])==expected, (len(m['part_a']), expected); assert len(m['part_b']) >= ${FAILURE_EXAMPLES:-5}*5; print('Part A/B:', len(m['part_a']), len(m['part_b']))"

log "Manifest ready. Launch with:"
log "INTENTDIT_USER_STUDY_KEY=<secret> $PYTHON_BIN $ACTIVE_ROOT/user_study/app.py --host 0.0.0.0 --port 5000"
