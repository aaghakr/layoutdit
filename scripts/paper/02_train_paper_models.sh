#!/usr/bin/env bash
# Train the complete manuscript model/ablation matrix for independent seeds.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

train_variant() {
    local dataset=$1 backbone=$2 guidance=$3 text=$4 slug=$5 seed=$6 gpu=$7
    local experiment config log_file
    experiment=$(experiment_name "$dataset" "$backbone" "$slug" "$seed")
    config="configs/${dataset}.yaml"
    log_file="$LOG_DIR/${experiment}.log"

    if [[ "$DRY_RUN" != "1" ]] && has_checkpoint "$dataset" "$experiment"; then
        log "SKIP $experiment (Epoch${FINAL_EPOCH} exists)"
        return
    fi

    args=(
        "$PYTHON_BIN" scripts/train.py
        --dataset "$dataset"
        --config "$config"
        --task uncond
        --v_encoder "$backbone"
        --spatial_guidance "$guidance"
        --seed "$seed"
        --experiment_name "$experiment"
        --gpuid 0
        --path-profile "$PATH_PROFILE"
    )
    [[ "$text" == "1" ]] && args+=(--text_control)
    append_training_runtime_args args "$dataset"

    log "TRAIN $experiment on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

train_task() {
    local task_index=$1 gpu=$2
    local seed_array seed_index variant_index entry dataset backbone guidance text slug
    read -r -a seed_array <<< "$SEEDS"
    seed_index=$((task_index / ${#PAPER_VARIANTS[@]}))
    variant_index=$((task_index % ${#PAPER_VARIANTS[@]}))
    entry=${PAPER_VARIANTS[$variant_index]}
    IFS='|' read -r dataset backbone guidance text slug <<< "$entry"
    train_variant \
        "$dataset" "$backbone" "$guidance" "$text" "$slug" \
        "${seed_array[$seed_index]}" "$gpu"
}

log "Training paper matrix: ${#PAPER_VARIANTS[@]} variants x seeds {$SEEDS}"
read -r -a seed_array <<< "$SEEDS"
run_task_workers train_task "$(( ${#PAPER_VARIANTS[@]} * ${#seed_array[@]} ))"
log "Paper model training complete"
