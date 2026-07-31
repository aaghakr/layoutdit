#!/usr/bin/env bash
# Aggregate independent training seeds and verify the primary result matrix.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

SCHEDULE_SUMMARY="$SUMMARY_DIR/primary_training_seed_summary_${DDIM_SCHEDULE}.json"
SCHEDULE_TABLE_DIR="$SUMMARY_DIR/tables_${DDIM_SCHEDULE}"

if [[ "$DRY_RUN" == "1" ]]; then
    print_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" \
        --input-dir "$METRIC_DIR" \
        --inference-seed "$INFERENCE_SEED" \
        --include-pattern '^ivc_(pku|cgl)_' \
        --output "$SCHEDULE_SUMMARY"
    print_command "$PYTHON_BIN" "$CODE_DIR/scripts/generate_result_tables.py" \
        --summary "$SCHEDULE_SUMMARY" \
        --output-dir "$SCHEDULE_TABLE_DIR"
    exit 0
fi

missing=0
read -r -a seed_array <<< "$SEEDS"
for entry in "${PAPER_VARIANTS[@]}"; do
    IFS='|' read -r dataset backbone guidance text slug <<< "$entry"
    for seed in "${seed_array[@]}"; do
        experiment=$(experiment_name "$dataset" "$backbone" "$slug" "$seed")
        output_name="${experiment}_inferseed${INFERENCE_SEED}"
        if ! result_complete_for "$output_name"; then
            printf 'MISSING_OR_STALE %s (expected DDIM schedule=%s, steps=%s)\n' \
                "$output_name" "$DDIM_SCHEDULE" "$DDIM_STEPS" >&2
            missing=1
        fi
    done
done

for slug in lambda_none lambda_text_only lambda_place_only lambda_high_text lambda_high_place; do
    for seed in "${seed_array[@]}"; do
        output_name="ivc_pku_vit_${slug}_trainseed${seed}_inferseed${INFERENCE_SEED}"
        if ! result_complete_for "$output_name"; then
            printf 'MISSING_OR_STALE %s (expected DDIM schedule=%s, steps=%s)\n' \
                "$output_name" "$DDIM_SCHEDULE" "$DDIM_STEPS" >&2
            missing=1
        fi
    done
done

for slug in pooled_text pixel_map_only_text intent_boxes_only_text; do
    for seed in "${seed_array[@]}"; do
        output_name="ivc_pku_vit_${slug}_trainseed${seed}_inferseed${INFERENCE_SEED}"
        if ! result_complete_for "$output_name"; then
            printf 'MISSING_OR_STALE %s (expected DDIM schedule=%s, steps=%s)\n' \
                "$output_name" "$DDIM_SCHEDULE" "$DDIM_STEPS" >&2
            missing=1
        fi
    done
done

[[ "$missing" == "0" ]] || die "Primary metric matrix is incomplete"

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" \
    --input-dir "$METRIC_DIR" \
    --inference-seed "$INFERENCE_SEED" \
    --include-pattern '^ivc_(pku|cgl)_' \
    --output "$SCHEDULE_SUMMARY"

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/generate_result_tables.py" \
    --summary "$SCHEDULE_SUMMARY" \
    --output-dir "$SCHEDULE_TABLE_DIR"

cp "$SCHEDULE_SUMMARY" "$SUMMARY_DIR/primary_training_seed_summary.json"
cp "${SCHEDULE_SUMMARY%.json}.md" "$SUMMARY_DIR/primary_training_seed_summary.md"
mkdir -p "$SUMMARY_DIR/tables"
cp "$SCHEDULE_TABLE_DIR"/* "$SUMMARY_DIR/tables"/

log "Schedule-specific summaries: ${SCHEDULE_SUMMARY%.json}.{json,md}"
log "Current summaries: $SUMMARY_DIR/primary_training_seed_summary.{json,md}"
