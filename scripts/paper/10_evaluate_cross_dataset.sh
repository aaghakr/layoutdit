#!/usr/bin/env bash
# Evaluate image-only models across PKU/CGL using shared classes.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

train_seed=${CROSS_TRAIN_SEED:-1}
gpu=${CROSS_GPU:-${GPU_IDS%%,*}}

run_cross() {
    local source=$1 target=$2
    local experiment checkpoint name
    experiment=$(experiment_name "$source" vit both "$train_seed")
    checkpoint=$(checkpoint_for "$source" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint: $experiment"
    name="ivc_cross_${source}_to_${target}_trainseed${train_seed}_inferseed${INFERENCE_SEED}"
    args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$target" --model-dataset "$source"
        --anno anno --task uncond --check_path "$checkpoint" --v_encoder vit
        --spatial_guidance 2 --protocol cross_dataset --seed "$INFERENCE_SEED"
        --ddim_num_steps "$DDIM_STEPS" --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$name"
        --save-test-output auto --no-render --gpuid 0 --path-profile "$PATH_PROFILE"
    )
    log "CROSS-DATASET $name on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name"; then
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$LOG_DIR/${name}.log"
    fi
}

run_cross pku cgl
run_cross cgl pku
log "Cross-dataset evaluation complete; interpret only shared Text/Logo/Underlay classes"
