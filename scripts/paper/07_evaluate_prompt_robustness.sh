#!/usr/bin/env bash
# Evaluate template complexity, spatial adherence, and independent free-form prompts.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

PROMPT_STYLES=${PROMPT_STYLES:-"basic enhanced advanced spatial rich freeform stress"}

for dataset in pku cgl; do
    stress="$ACTIVE_ROOT/data/prompts/stress_${dataset}.csv"
    annotations="$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv"
    if [[ "$DRY_RUN" == "1" || ! -f "$stress" ]]; then
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/build_prompt_stress_test.py" \
            --annotations "$annotations" --output "$stress" --limit "${STRESS_LIMIT:-400}"
    fi
done

if [[ " $PROMPT_STYLES " == *" freeform "* && "$DRY_RUN" != "1" ]]; then
    for dataset in pku cgl; do
        free_form="$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
        [[ -f "$free_form" ]] || die "Missing free-form prompt CSV: $free_form"
        "$PYTHON_BIN" "$CODE_DIR/scripts/validate_freeform_prompts.py" \
            --prompts "$free_form" \
            --annotations "$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv"
    done
fi

prompt_csv_for() {
    local dataset=$1 style=$2
    case "$style" in
        basic|enhanced|advanced|spatial)
            printf '%s/data/dataset/%s/split/csv/test_with_prompts_%s.csv' "$ACTIVE_ROOT" "$dataset" "$style"
            ;;
        rich)
            printf '%s/data/dataset/%s/split/csv/test_with_rich_prompts.csv' "$ACTIVE_ROOT" "$dataset"
            ;;
        freeform)
            printf '%s/data/prompts/free_form_%s.csv' "$ACTIVE_ROOT" "$dataset"
            ;;
        stress)
            printf '%s/data/prompts/stress_%s.csv' "$ACTIVE_ROOT" "$dataset"
            ;;
        *) die "Unknown prompt style: $style" ;;
    esac
}

evaluate_prompt_variant() {
    local dataset=$1 backbone=$2 guidance=$3 slug=$4 train_seed=$5 gpu=$6 style=$7
    local experiment checkpoint prompt_csv output_name log_file protocol
    experiment=$(experiment_name "$dataset" "$backbone" "$slug" "$train_seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint: $experiment"
    prompt_csv=$(prompt_csv_for "$dataset" "$style")
    if [[ "$DRY_RUN" != "1" ]]; then
        [[ -f "$prompt_csv" ]] || die "Missing prompt CSV: $prompt_csv"
    fi

    output_name="ivc_prompt_${dataset}_${backbone}_${slug}_${style}_trainseed${train_seed}_inferseed${INFERENCE_SEED}"
    log_file="$LOG_DIR/${output_name}.log"
    if [[ "${FORCE:-0}" != "1" ]] && result_complete_for "$output_name"; then
        log "SKIP $output_name (metrics exist for DDIM schedule=$DDIM_SCHEDULE, steps=$DDIM_STEPS)"
        return
    fi

    protocol=oracle_prompt
    [[ "$style" == "freeform" ]] && protocol=freeform_prompt
    args=(
        "$PYTHON_BIN" scripts/test.py
        --dataset "$dataset"
        --anno anno
        --task uncond
        --check_path "$checkpoint"
        --v_encoder "$backbone"
        --spatial_guidance "$guidance"
        --text_control
        --prompts-csv "$prompt_csv"
        --seed "$INFERENCE_SEED"
        --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$output_name"
        --save-test-output auto
        --gpuid 0
        --path-profile "$PATH_PROFILE"
        --protocol "$protocol"
    )
    [[ "$style" == "spatial" ]] && args+=(--spatial-metrics)
    [[ "$style" == "freeform" ]] && args+=(--prompt-subset-only --spatial-metrics)
    [[ "$style" == "stress" ]] && args+=(--prompt-subset-only --spatial-metrics)
    [[ "$dataset" == "pku" && "$backbone" == "vit" && "$slug" == "both_text" && "$style" == "freeform" && "$train_seed" == "1" ]] || args+=(--no-render)

    log "PROMPT-EVAL $output_name on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

evaluate_prompt_seed() {
    local seed=$1 gpu=$2 entry dataset backbone guidance slug style
    for entry in "${PROMPT_VARIANTS[@]}"; do
        IFS='|' read -r dataset backbone guidance slug <<< "$entry"
        for style in $PROMPT_STYLES; do
            evaluate_prompt_variant "$dataset" "$backbone" "$guidance" "$slug" "$seed" "$gpu" "$style"
        done
    done
}

run_seed_workers evaluate_prompt_seed

for dataset in pku cgl; do
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_prompt_subgroups.py" \
        --per-image "$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_stress_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv" \
        --prompts "$ACTIVE_ROOT/data/prompts/stress_${dataset}.csv" \
        --output "$SUMMARY_DIR/prompt_stress_${dataset}.json"
done

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" \
    --input-dir "$METRIC_DIR" \
    --inference-seed "$INFERENCE_SEED" \
    --include-prefix ivc_prompt_ \
    --output "$SUMMARY_DIR/prompt_training_seed_summary.json"

log "Prompt robustness evaluation complete"
