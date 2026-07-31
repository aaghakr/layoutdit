#!/usr/bin/env bash
# Shared definitions for the numbered paper experiment pipeline.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CODE_DIR="$PROJECT_ROOT/code"

PATH_PROFILE=${PATH_PROFILE:-server}
PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-"1 2 3"}
GPU_IDS=${GPU_IDS:-"0"}
FINAL_EPOCH=${FINAL_EPOCH:-500}
INFERENCE_SEED=${INFERENCE_SEED:-1}
DDIM_STEPS=${DDIM_STEPS:-100}
DDIM_SCHEDULE=${DDIM_SCHEDULE:-cosine}
DRY_RUN=${DRY_RUN:-0}
TRAIN_PRECISION=${TRAIN_PRECISION:-bf16}
TRAIN_TF32=${TRAIN_TF32:-1}
SKIP_TRAINING_VALIDATION=${SKIP_TRAINING_VALIDATION:-1}
TRAIN_BATCH_SIZE_PKU=${TRAIN_BATCH_SIZE_PKU:-}
TRAIN_BATCH_SIZE_CGL=${TRAIN_BATCH_SIZE_CGL:-}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}

if [[ "$PATH_PROFILE" == "server" ]]; then
    ACTIVE_ROOT=/home/viplab/Aagha/intent_aware_layout_generation
else
    ACTIVE_ROOT=$PROJECT_ROOT
fi

LOG_DIR="$ACTIVE_ROOT/logs/paper"
METRIC_DIR="$ACTIVE_ROOT/experiments/paper_figures"
SUMMARY_DIR="$ACTIVE_ROOT/experiments/paper_results"

# PKU uses both backbones in the manuscript; CGL uses the main ViT backbone.
# Format: dataset|backbone|spatial_guidance|text_control|slug
PAPER_VARIANTS=(
    "pku|vit|0|0|saliency"
    "pku|vit|1|0|intent"
    "pku|vit|2|0|both"
    "pku|vit|0|1|saliency_text"
    "pku|vit|1|1|intent_text"
    "pku|vit|2|1|both_text"
    "pku|swin|0|0|saliency"
    "pku|swin|1|0|intent"
    "pku|swin|2|0|both"
    "pku|swin|0|1|saliency_text"
    "pku|swin|1|1|intent_text"
    "pku|swin|2|1|both_text"
    "cgl|vit|0|0|saliency"
    "cgl|vit|1|0|intent"
    "cgl|vit|2|0|both"
    "cgl|vit|0|1|saliency_text"
    "cgl|vit|1|1|intent_text"
    "cgl|vit|2|1|both_text"
)

# Prompt analyses reported in the manuscript.
PROMPT_VARIANTS=(
    "pku|vit|0|saliency_text"
    "pku|vit|2|both_text"
    "pku|swin|0|saliency_text"
    "pku|swin|2|both_text"
    "cgl|vit|0|saliency_text"
    "cgl|vit|2|both_text"
)

log() {
    printf '[paper] %s\n' "$*"
}

die() {
    printf '[paper] ERROR: %s\n' "$*" >&2
    exit 1
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command "$@"
    else
        "$@"
    fi
}

experiment_name() {
    local dataset=$1 backbone=$2 slug=$3 seed=$4
    printf 'ivc_%s_%s_%s_trainseed%s' "$dataset" "$backbone" "$slug" "$seed"
}

checkpoint_for() {
    local dataset=$1 experiment=$2
    local root="$ACTIVE_ROOT/data/checkpoints/$dataset/$experiment"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%s/<timestamp>/Epoch%s_cgbdm_weights.pth' "$root" "$FINAL_EPOCH"
        return
    fi
    find "$root" -type f -name "Epoch${FINAL_EPOCH}_cgbdm_weights.pth" 2>/dev/null \
        | sort | tail -n 1
}

has_checkpoint() {
    [[ -n "$(checkpoint_for "$1" "$2")" && -f "$(checkpoint_for "$1" "$2")" ]]
}

result_complete_for() {
    local output_name=$1
    local expected_steps=${2:-$DDIM_STEPS}
    [[ -f "$METRIC_DIR/${output_name}_metrics.json" \
      && -f "$METRIC_DIR/${output_name}_per_image.csv" \
      && -f "$METRIC_DIR/${output_name}_evidence.json" \
      && -f "$METRIC_DIR/${output_name}_test_output.pt" ]] || return 1

    "$PYTHON_BIN" - "$METRIC_DIR/${output_name}_evidence.json" "$DDIM_SCHEDULE" "$expected_steps" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_schedule = sys.argv[2]
expected_steps = int(sys.argv[3])

try:
    evidence = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)

actual_schedule = evidence.get("ddim_schedule")
if actual_schedule is None:
    raise SystemExit(1)
actual_steps = int(evidence.get("ddim_steps", -1))
raise SystemExit(0 if actual_schedule == expected_schedule and actual_steps == expected_steps else 1)
PY
}

text_flag() {
    if [[ "$1" == "1" ]]; then
        printf '%s\n' '--text_control'
    fi
}

run_seed_workers() {
    local callback=$1
    local seed_array gpu_array
    read -r -a seed_array <<< "$SEEDS"
    IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
    [[ ${#gpu_array[@]} -gt 0 ]] || die "GPU_IDS is empty"

    local pids=() worker_index
    for worker_index in "${!gpu_array[@]}"; do
        (
            local seed_index
            for ((seed_index=worker_index; seed_index<${#seed_array[@]}; seed_index+=${#gpu_array[@]})); do
                "$callback" "${seed_array[$seed_index]}" "${gpu_array[$worker_index]}"
            done
        ) &
        pids+=("$!")
    done

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" == "0" ]] || die "One or more workers failed"
}

# Run an indexed task matrix evenly across all listed GPUs. Unlike seed-based
# assignment, this keeps both GPUs busy when the number of seeds is not a
# multiple of the number of GPUs.
run_task_workers() {
    local callback=$1 task_count=$2
    local physical_gpus gpu_array=() gpu replica
    IFS=',' read -r -a physical_gpus <<< "$GPU_IDS"
    [[ ${#physical_gpus[@]} -gt 0 ]] || die "GPU_IDS is empty"
    [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "JOBS_PER_GPU must be a positive integer"
    for gpu in "${physical_gpus[@]}"; do
        for ((replica=0; replica<JOBS_PER_GPU; replica++)); do
            gpu_array+=("$gpu")
        done
    done
    log "Task scheduler: ${#gpu_array[@]} workers over GPUs {$GPU_IDS} (${JOBS_PER_GPU} jobs/GPU)"

    local pids=() worker_index
    for worker_index in "${!gpu_array[@]}"; do
        (
            local task_index
            for ((task_index=worker_index; task_index<task_count; task_index+=${#gpu_array[@]})); do
                "$callback" "$task_index" "${gpu_array[$worker_index]}"
            done
        ) &
        pids+=("$!")
    done

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    [[ "$failed" == "0" ]] || die "One or more workers failed"
}

append_training_runtime_args() {
    local target_name=$1 dataset=$2
    local -n target="$target_name"
    target+=(--precision "$TRAIN_PRECISION" --epochs "$FINAL_EPOCH")
    [[ "$TRAIN_TF32" == "1" ]] && target+=(--allow-tf32)
    [[ "$SKIP_TRAINING_VALIDATION" == "1" ]] && target+=(--skip-training-validation)
    [[ -n "$TRAIN_NUM_WORKERS" ]] && target+=(--num-workers "$TRAIN_NUM_WORKERS")
    if [[ "$dataset" == "pku" && -n "$TRAIN_BATCH_SIZE_PKU" ]]; then
        target+=(--train-batch-size "$TRAIN_BATCH_SIZE_PKU")
    elif [[ "$dataset" == "cgl" && -n "$TRAIN_BATCH_SIZE_CGL" ]]; then
        target+=(--train-batch-size "$TRAIN_BATCH_SIZE_CGL")
    fi
}

mkdir_outputs() {
    if [[ "$DRY_RUN" != "1" ]]; then
        mkdir -p \
            "$LOG_DIR" \
            "$METRIC_DIR" \
            "$SUMMARY_DIR" \
            "$ACTIVE_ROOT/data/output/ptfile/image_name_order" \
            "$ACTIVE_ROOT/data/output/image" \
            "$ACTIVE_ROOT/data/checkpoints/pku" \
            "$ACTIVE_ROOT/data/checkpoints/cgl"
    fi
}
