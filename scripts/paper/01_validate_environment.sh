#!/usr/bin/env bash
# Validate code, configs, datasets, maps, annotations, and prompt files.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

log "Project: $PROJECT_ROOT"
log "Active root: $ACTIVE_ROOT"
log "Path profile: $PATH_PROFILE"
log "Python: $PYTHON_BIN"

run_command "$PYTHON_BIN" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

if command -v nvidia-smi >/dev/null 2>&1; then
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command nvidia-smi -L
    elif ! nvidia-smi -L; then
        log "nvidia-smi is present but no GPU is currently available"
    fi
else
    log "nvidia-smi not found; CPU-only validation continues"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    print_command bash -lc "cd '$CODE_DIR' && '$PYTHON_BIN' -m unittest discover -s tests -v"
else
    (cd "$CODE_DIR" && "$PYTHON_BIN" -m unittest discover -s tests -v)
fi
run_command "$PYTHON_BIN" "$CODE_DIR/validate_configs.py" --path-profile "$PATH_PROFILE"

if [[ "${SKIP_DATA_CHECK:-0}" == "1" ]]; then
    log "SKIP_DATA_CHECK=1; dataset checks skipped"
    exit 0
fi

required=()
required+=(
    "$ACTIVE_ROOT/data/model_weights/intent_map/design_intent_pku_epoch100.pth"
    "$ACTIVE_ROOT/data/model_weights/intent_map/design_intent_cgl_epoch35.pth"
    "$ACTIVE_ROOT/data/model_weights/saliency_detection/basnet.pth"
    "$ACTIVE_ROOT/data/model_weights/saliency_detection/isnet.pth"
)
for dataset in pku cgl; do
    base="$ACTIVE_ROOT/data/dataset/$dataset/split"
    for split in train val test_anno; do
        for folder in inpaint saliency saliency_sub intent_map; do
            required+=("$base/$split/$folder")
        done
    done
    for csv_name in \
        train.csv val.csv test.csv \
        train_sal.csv val_sal.csv test_anno_sal.csv \
        train_intent_mbbox.csv val_intent_mbbox.csv test_anno_intent_mbbox.csv \
        train_with_all_prompts.csv val_with_all_prompts.csv \
        test_with_prompts_basic.csv test_with_prompts_enhanced.csv \
        test_with_prompts_advanced.csv test_with_prompts_spatial.csv \
        test_with_rich_prompts.csv; do
        required+=("$base/csv/$csv_name")
    done
done

missing=0
for path in "${required[@]}"; do
    if [[ ! -e "$path" ]]; then
        printf 'MISSING %s\n' "$path" >&2
        missing=1
    fi
done

if [[ "$missing" == "0" ]]; then
    for dataset in pku cgl; do
        for split in train val test_anno; do
            if [[ "$DRY_RUN" == "1" ]]; then
                print_command "$PYTHON_BIN" "$CODE_DIR/scripts/validate_derived_images.py" \
                    --source-dir "$ACTIVE_ROOT/data/dataset/$dataset/split/$split/inpaint" \
                    --derived-dir "$ACTIVE_ROOT/data/dataset/$dataset/split/$split/intent_map" \
                    --label "$dataset/$split intent maps"
            elif ! "$PYTHON_BIN" "$CODE_DIR/scripts/validate_derived_images.py" \
                --source-dir "$ACTIVE_ROOT/data/dataset/$dataset/split/$split/inpaint" \
                --derived-dir "$ACTIVE_ROOT/data/dataset/$dataset/split/$split/intent_map" \
                --label "$dataset/$split intent maps"; then
                missing=1
            fi
        done
    done
fi

if [[ "${SKIP_FREEFORM_CHECK:-0}" == "1" ]]; then
    log "SKIP_FREEFORM_CHECK=1; training inputs can pass while independent prompts are authored"
else
    for dataset in pku cgl; do
        free_form="$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
        if [[ ! -f "$free_form" ]]; then
            printf 'MISSING %s (needed by stages 07 and 09; 100+ independent prompts)\n' "$free_form" >&2
            missing=1
        elif [[ "$DRY_RUN" == "1" ]]; then
            print_command "$PYTHON_BIN" "$CODE_DIR/scripts/validate_freeform_prompts.py" \
                --prompts "$free_form" \
                --annotations "$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv"
        elif ! "$PYTHON_BIN" "$CODE_DIR/scripts/validate_freeform_prompts.py" \
            --prompts "$free_form" \
            --annotations "$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv"; then
            missing=1
        fi
    done
fi

[[ "$missing" == "0" ]] || die "Required paper data is incomplete"
log "Environment and paper data validation passed"
