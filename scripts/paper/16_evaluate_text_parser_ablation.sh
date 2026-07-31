#!/usr/bin/env bash
# Diagnostic free-form ablation separating BERT text tokens from parsed spatial boxes.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

PARSER_ABLATION_DATASETS=${PARSER_ABLATION_DATASETS:-"pku cgl"}
PARSER_ABLATION_SEEDS=${PARSER_ABLATION_SEEDS:-$SEEDS}
PARSER_ABLATION_MODES=${PARSER_ABLATION_MODES:-"no_parser parser_only"}

PARSER_TASKS=()
for dataset in $PARSER_ABLATION_DATASETS; do
    for seed in $PARSER_ABLATION_SEEDS; do
        for mode in $PARSER_ABLATION_MODES; do
            PARSER_TASKS+=("$dataset|$seed|$mode")
        done
    done
done

run_parser_ablation_task() {
    local task_index=$1 gpu=$2
    local dataset seed mode experiment checkpoint prompts output_name log_file
    IFS='|' read -r dataset seed mode <<< "${PARSER_TASKS[$task_index]}"

    experiment=$(experiment_name "$dataset" vit both_text "$seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint for parser ablation: $experiment"

    prompts="$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
    if [[ "$DRY_RUN" != "1" ]]; then
        [[ -f "$prompts" ]] || die "Missing free-form prompt CSV: $prompts"
    fi

    output_name="ivc_prompt_${dataset}_vit_both_text_freeform_${mode}_trainseed${seed}_inferseed${INFERENCE_SEED}"
    log_file="$LOG_DIR/${output_name}.log"
    if [[ "${FORCE:-0}" != "1" ]] && result_complete_for "$output_name"; then
        log "SKIP $output_name (metrics exist for DDIM schedule=$DDIM_SCHEDULE, steps=$DDIM_STEPS)"
        return
    fi

    args=(
        "$PYTHON_BIN" scripts/test.py
        --dataset "$dataset"
        --anno anno
        --task uncond
        --check_path "$checkpoint"
        --v_encoder vit
        --spatial_guidance 2
        --text_control
        --prompts-csv "$prompts"
        --prompt-subset-only
        --spatial-metrics
        --protocol freeform_prompt
        --seed "$INFERENCE_SEED"
        --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$output_name"
        --save-test-output auto
        --no-render
        --gpuid 0
        --path-profile "$PATH_PROFILE"
    )

    case "$mode" in
        no_parser)
            args+=(--disable-text-spatial-parser)
            ;;
        parser_only)
            args+=(--text-guidance-scale 0)
            ;;
        *)
            die "Unknown parser ablation mode: $mode"
            ;;
    esac

    log "PARSER-ABLATION $output_name on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

run_task_workers run_parser_ablation_task "${#PARSER_TASKS[@]}"

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/aggregate_training_seeds.py" \
    --input-dir "$METRIC_DIR" \
    --inference-seed "$INFERENCE_SEED" \
    --include-prefix ivc_prompt_ \
    --include-pattern 'freeform_(no_parser|parser_only)_trainseed' \
    --output "$SUMMARY_DIR/text_parser_ablation_summary.json"

for dataset in $PARSER_ABLATION_DATASETS; do
    baseline="$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv"
    for mode in $PARSER_ABLATION_MODES; do
        ablation="$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_${mode}_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv"
        if [[ "$DRY_RUN" == "1" || ( -f "$baseline" && -f "$ablation" ) ]]; then
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_bootstrap.py" \
                --method-a "$baseline" \
                --method-b "$ablation" \
                --name-a freeform_with_parser \
                --name-b "$mode" \
                --output "$SUMMARY_DIR/paired_text_parser_${dataset}_${mode}.json"
        else
            log "SKIP paired parser bootstrap for $dataset/$mode; missing normal or ablated per-image CSV"
        fi
    done
done

log "Text parser ablation complete"
