#!/usr/bin/env bash
# Evaluate all independently trained checkpoints using one fixed inference seed.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

evaluate_variant() {
    local dataset=$1 backbone=$2 guidance=$3 text=$4 slug=$5 train_seed=$6 gpu=$7
    local text_mode=${8:-token} disable_boxes=${9:-0}
    local experiment checkpoint output_name prompt_csv log_file protocol
    experiment=$(experiment_name "$dataset" "$backbone" "$slug" "$train_seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing Epoch${FINAL_EPOCH}: $experiment"
    output_name="${experiment}_inferseed${INFERENCE_SEED}"
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
        --v_encoder "$backbone"
        --spatial_guidance "$guidance"
        --seed "$INFERENCE_SEED"
        --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$output_name"
        --gpuid 0
        --path-profile "$PATH_PROFILE"
        --save-test-output auto
        --text-conditioning-mode "$text_mode"
    )
    [[ "$disable_boxes" == "1" ]] && args+=(--disable-spatial-boxes)
    if [[ "$text" == "1" ]]; then
        prompt_csv="$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test_with_prompts_basic.csv"
        protocol=oracle_prompt
        args+=(--text_control --prompts-csv "$prompt_csv" --protocol "$protocol")
    else
        protocol=image_only
        args+=(--protocol "$protocol")
    fi
    [[ "${RENDER_ALL:-0}" == "1" || ( "$dataset" == "pku" && ( "$slug" == "both" || "$slug" == "both_text" ) && "$train_seed" == "1" ) ]] || args+=(--no-render)

    log "EVAL $output_name on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

evaluate_seed() {
    local seed=$1 gpu=$2 entry dataset backbone guidance text slug
    for entry in "${PAPER_VARIANTS[@]}"; do
        IFS='|' read -r dataset backbone guidance text slug <<< "$entry"
        evaluate_variant "$dataset" "$backbone" "$guidance" "$text" "$slug" "$seed" "$gpu"
    done

    for slug in lambda_none lambda_text_only lambda_place_only lambda_high_text lambda_high_place; do
        evaluate_variant pku vit 2 1 "$slug" "$seed" "$gpu"
    done
    evaluate_variant pku vit 2 1 pooled_text "$seed" "$gpu" pooled 0
    evaluate_variant pku vit 2 1 pixel_map_only_text "$seed" "$gpu" token 1
    evaluate_variant pku vit 3 1 intent_boxes_only_text "$seed" "$gpu" token 0
}

run_seed_workers evaluate_seed
log "Primary evaluation complete"
