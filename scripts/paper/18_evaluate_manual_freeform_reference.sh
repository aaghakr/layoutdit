#!/usr/bin/env bash
# Re-score free-form predictions against manually audited prompt references.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

run_manual_reference() {
    local dataset=$1
    local audit="$SUMMARY_DIR/fixed_parser_audit_${dataset}.csv"
    local baseline="$ACTIVE_ROOT/other_baselines/standardized/postero_${dataset}_freeform_subset.pt"
    [[ "$DRY_RUN" == "1" || -f "$audit" ]] || die "Missing manual parser audit: $audit"
    [[ "$DRY_RUN" == "1" || -f "$baseline" ]] || die "Missing external text baseline tensor: $baseline"

    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_manual_freeform_reference.py" \
        --dataset "$dataset" \
        --audit-csv "$audit" \
        --metric-dir "$METRIC_DIR" \
        --baseline-predictions "$baseline" \
        --seeds $SEEDS \
        --inference-seed "$INFERENCE_SEED" \
        --output-dir "$SUMMARY_DIR"

    local seed_array=() all_seed_paths=() retained_seed_paths=() seed
    read -r -a seed_array <<< "$SEEDS"
    for seed in "${seed_array[@]}"; do
        all_seed_paths+=("$SUMMARY_DIR/manual_freeform_${dataset}_intentdit_seed${seed}_all_per_image.csv")
        retained_seed_paths+=("$SUMMARY_DIR/manual_freeform_${dataset}_intentdit_seed${seed}_retained_per_image.csv")
    done

    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_seed_image_bootstrap.py" \
        --method-a "${all_seed_paths[@]}" \
        --method-b "$SUMMARY_DIR/manual_freeform_${dataset}_external_text_baseline_all_per_image.csv" \
        --name-a intentdit_text \
        --name-b external_text_baseline \
        --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
        --output "$SUMMARY_DIR/paired_manual_freeform_${dataset}_all.json"

    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_seed_image_bootstrap.py" \
        --method-a "${retained_seed_paths[@]}" \
        --method-b "$SUMMARY_DIR/manual_freeform_${dataset}_external_text_baseline_retained_per_image.csv" \
        --name-a intentdit_text \
        --name-b external_text_baseline \
        --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
        --output "$SUMMARY_DIR/paired_manual_freeform_${dataset}_retained.json"
}

run_manual_reference pku
run_manual_reference cgl

log "Manual-reference free-form evaluation complete"
