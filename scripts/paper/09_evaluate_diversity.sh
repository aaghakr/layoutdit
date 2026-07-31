#!/usr/bin/env bash
# Generate repeated samples for identical free-form conditions and quantify diversity.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

DIVERSITY_SEEDS=${DIVERSITY_SEEDS:-"11 12 13 14 15"}
train_seed=${DIVERSITY_TRAIN_SEED:-1}
gpu=${DIVERSITY_GPU:-${GPU_IDS%%,*}}

for dataset in pku cgl; do
    experiment=$(experiment_name "$dataset" vit both_text "$train_seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint: $experiment"
    prompts="$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
    [[ "$DRY_RUN" == "1" || -f "$prompts" ]] || die "Missing $prompts"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$PYTHON_BIN" "$CODE_DIR/scripts/validate_freeform_prompts.py" \
            --prompts "$prompts" \
            --annotations "$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv"
    fi
    predictions=()
    for inference_seed in $DIVERSITY_SEEDS; do
        name="ivc_diversity_${dataset}_trainseed${train_seed}_inferseed${inference_seed}"
        predictions+=("$METRIC_DIR/${name}_test_output.pt")
        args=(
            "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno
            --task uncond --check_path "$checkpoint" --v_encoder vit
            --spatial_guidance 2 --text_control --prompts-csv "$prompts"
            --prompt-subset-only --spatial-metrics --protocol freeform_prompt
            --seed "$inference_seed" --ddim_num_steps "$DDIM_STEPS"
            --ddim_schedule "$DDIM_SCHEDULE"
            --experiment_name "$name" --save-test-output auto --no-render
            --gpuid 0 --path-profile "$PATH_PROFILE"
        )
        log "DIVERSITY-SAMPLE $name on physical GPU $gpu"
        if [[ "$DRY_RUN" == "1" ]]; then
            print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
        elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name"; then
            (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
                2>&1 | tee "$LOG_DIR/${name}.log"
        fi
    done
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_diversity.py" \
        "${predictions[@]}" --output "$SUMMARY_DIR/diversity_${dataset}.json"
done

log "Diversity evaluation complete"
