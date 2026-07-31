#!/usr/bin/env bash
# Train the two paper architecture ablations on PKU.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

ARCH_SETTINGS=(
    "pooled_text|pooled|0|2"
    "pixel_map_only_text|token|1|2"
    "intent_boxes_only_text|token|0|3"
)

train_architecture_setting() {
    local seed=$1 gpu=$2 slug=$3 text_mode=$4 disable_boxes=$5 guidance=$6
    local experiment="ivc_pku_vit_${slug}_trainseed${seed}"
    local log_file="$LOG_DIR/${experiment}.log"
    if [[ "$DRY_RUN" != "1" ]] && has_checkpoint pku "$experiment"; then
        log "SKIP $experiment (Epoch${FINAL_EPOCH} exists)"
        return
    fi
    args=(
        "$PYTHON_BIN" scripts/train.py
        --dataset pku --config configs/pku.yaml --task uncond
        --v_encoder vit --spatial_guidance "$guidance" --text_control
        --text-conditioning-mode "$text_mode"
        --seed "$seed" --experiment_name "$experiment" --gpuid 0
        --path-profile "$PATH_PROFILE"
    )
    [[ "$disable_boxes" == "1" ]] && args+=(--disable-spatial-boxes)
    append_training_runtime_args args pku
    log "TRAIN-ARCH $experiment on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    else
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$log_file"
    fi
}

train_architecture_task() {
    local task_index=$1 gpu=$2
    local seed_array seed_index setting_index slug text_mode disable_boxes guidance
    read -r -a seed_array <<< "$SEEDS"
    seed_index=$((task_index / ${#ARCH_SETTINGS[@]}))
    setting_index=$((task_index % ${#ARCH_SETTINGS[@]}))
    IFS='|' read -r slug text_mode disable_boxes guidance <<< "${ARCH_SETTINGS[$setting_index]}"
    train_architecture_setting \
        "${seed_array[$seed_index]}" "$gpu" "$slug" "$text_mode" \
        "$disable_boxes" "$guidance"
}

read -r -a seed_array <<< "$SEEDS"
run_task_workers train_architecture_task "$(( ${#ARCH_SETTINGS[@]} * ${#seed_array[@]} ))"
log "Architecture ablation training complete"
