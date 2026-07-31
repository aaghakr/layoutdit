#!/usr/bin/env bash
# Measure quality, latency, throughput, and memory as DDIM steps vary.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

STEPS=${STEPS:-"10 20 50 100"}
seed=${EFFICIENCY_TRAIN_SEED:-1}
gpu=${EFFICIENCY_GPU:-${GPU_IDS%%,*}}

for dataset in pku cgl; do
    experiment=$(experiment_name "$dataset" vit both_text "$seed")
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint: $experiment"
    prompts="$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test_with_prompts_basic.csv"
    for steps in $STEPS; do
        name="ivc_efficiency_${dataset}_steps${steps}_trainseed${seed}_inferseed${INFERENCE_SEED}"
        args=(
            "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno
            --task uncond --check_path "$checkpoint" --v_encoder vit
            --spatial_guidance 2 --text_control --prompts-csv "$prompts"
            --protocol oracle_prompt --seed "$INFERENCE_SEED"
            --ddim_num_steps "$steps" --ddim_schedule "$DDIM_SCHEDULE"
            --experiment_name "$name"
            --save-test-output auto --no-render --gpuid 0
            --path-profile "$PATH_PROFILE"
        )
        log "EFFICIENCY $name on physical GPU $gpu"
        if [[ "$DRY_RUN" == "1" ]]; then
            print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
        elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name" "$steps"; then
            (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
                2>&1 | tee "$LOG_DIR/${name}.log"
        fi
    done
done

# Text-conditioning strength sensitivity on the PKU flagship checkpoint.
dataset=pku
experiment=$(experiment_name "$dataset" vit both_text "$seed")
checkpoint=$(checkpoint_for "$dataset" "$experiment")
prompts="$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test_with_prompts_basic.csv"
text_guidance_scales=${TEXT_GUIDANCE_SCALES:-0 0.5 1 1.5 2}
read -r -a text_guidance_scale_array <<< "$text_guidance_scales"
for scale in "${text_guidance_scale_array[@]}"; do
    scale_slug=${scale/./p}
    name="ivc_textscale_${dataset}_${scale_slug}_trainseed${seed}_inferseed${INFERENCE_SEED}"
    args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno --task uncond
        --check_path "$checkpoint" --v_encoder vit --spatial_guidance 2 --text_control
        --text-guidance-scale "$scale" --prompts-csv "$prompts" --protocol oracle_prompt
        --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$name" --save-test-output auto --no-render
        --gpuid 0 --path-profile "$PATH_PROFILE"
    )
    log "TEXT-SCALE $name on physical GPU $gpu"
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name"; then
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
            2>&1 | tee "$LOG_DIR/${name}.log"
    fi
done

log "Sampling-efficiency sweep complete"
