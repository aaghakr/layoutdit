#!/usr/bin/env bash
# Reconstruct deterministic derived inputs without fabricating independent prompts.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/prepare_paper_inputs.py" \
    --path-profile "$PATH_PROFILE" \
    --datasets pku cgl \
    --splits train val test \
    --seed "${PROMPT_SEED:-2026}" \
    --freeform-count "${FREEFORM_COUNT:-120}"

for dataset in pku cgl; do
    input_dir="$ACTIVE_ROOT/data/dataset/$dataset/split/test_anno/inpaint"
    output_dir="$ACTIVE_ROOT/data/dataset/$dataset/split/test_anno/intent_map"
    if [[ "$dataset" == "pku" ]]; then
        checkpoint="$ACTIVE_ROOT/data/model_weights/intent_map/design_intent_pku_epoch100.pth"
    else
        checkpoint="$ACTIVE_ROOT/data/model_weights/intent_map/design_intent_cgl_epoch35.pth"
    fi
    run_command "$PYTHON_BIN" "$ACTIVE_ROOT/intent_detect/predict_split.py" \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --checkpoint "$checkpoint" \
        --device "${INTENT_DEVICE:-cuda:0}" \
        --batch-size "${INTENT_BATCH_SIZE:-64}"
done

log "Derived inputs are ready. Fill data/prompts/free_form_*_template.csv independently,"
log "then save them as free_form_pku.csv and free_form_cgl.csv before stage 01."
