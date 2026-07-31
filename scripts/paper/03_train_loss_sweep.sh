#!/usr/bin/env bash
# Train loss-isolation and high-weight PKU auxiliary-loss settings.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

LOSS_SETTINGS=(
    "lambda_none|0.0|0.0"
    "lambda_text_only|0.1|0.0"
    "lambda_place_only|0.0|0.05"
    "lambda_high_text|0.2|0.05"
    "lambda_high_place|0.1|0.10"
)

train_sweep_setting() {
    local seed=$1 gpu=$2 slug=$3 lambda1=$4 lambda2=$5
    local experiment="ivc_pku_vit_${slug}_trainseed${seed}"
    local log_file="$LOG_DIR/${experiment}.log"

    if [[ "$DRY_RUN" != "1" ]] && has_checkpoint pku "$experiment"; then
        log "SKIP $experiment (Epoch${FINAL_EPOCH} exists)"
        return
    fi

    args=(
        "$PYTHON_BIN" scripts/train.py
        --dataset pku
        --config configs/pku_lambda_default.yaml
        --task uncond
        --v_encoder vit
        --spatial_guidance 2
        --text_control
        --lambda1 "$lambda1"
        --lambda2 "$lambda2"
        --seed "$seed"
        --experiment_name "$experiment"
        --gpuid 0
        --path-profile "$PATH_PROFILE"
    )
    append_training_runtime_args args pku

    log "TRAIN $experiment on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

train_sweep_task() {
    local task_index=$1 gpu=$2
    local seed_array seed_index setting_index slug lambda1 lambda2
    read -r -a seed_array <<< "$SEEDS"
    seed_index=$((task_index / ${#LOSS_SETTINGS[@]}))
    setting_index=$((task_index % ${#LOSS_SETTINGS[@]}))
    IFS='|' read -r slug lambda1 lambda2 <<< "${LOSS_SETTINGS[$setting_index]}"
    train_sweep_setting \
        "${seed_array[$seed_index]}" "$gpu" "$slug" "$lambda1" "$lambda2"
}

log "Default loss setting is ivc_pku_vit_both_text from 02_train_paper_models.sh"
read -r -a seed_array <<< "$SEEDS"
run_task_workers train_sweep_task "$(( ${#LOSS_SETTINGS[@]} * ${#seed_array[@]} ))"
log "Loss sweep training complete"
