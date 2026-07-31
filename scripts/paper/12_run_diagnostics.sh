#!/usr/bin/env bash
# Reproduce intent-map diagnostics and the automated failure taxonomy.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs
gpu=${DIAGNOSTIC_GPU:-${GPU_IDS%%,*}}

for dataset in pku cgl; do
    intent_limit=${DIAGNOSTIC_LIMIT:-}
    if [[ -z "$intent_limit" ]]; then
        if [[ "$dataset" == "cgl" ]]; then
            intent_limit=6002
        else
            intent_limit=1000
        fi
    fi
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_intent_maps.py" \
        --dataset "$dataset" \
        --path-profile "$PATH_PROFILE" \
        --limit "$intent_limit" \
        --sigma "${INTENT_SIGMA:-10}" \
        --output "$SUMMARY_DIR/intent_map_${dataset}.json" \
        --per-image-output "$SUMMARY_DIR/intent_map_${dataset}_per_image.csv"
done

for dataset in pku cgl; do
    base_prompts="$ACTIVE_ROOT/data/prompts/edit_base_${dataset}.csv"
    edited_prompts="$ACTIVE_ROOT/data/prompts/edit_changed_${dataset}.csv"
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/build_prompt_edit_pairs.py" \
        --annotations "$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test.csv" \
        --base-output "$base_prompts" --edited-output "$edited_prompts" \
        --limit "${EDIT_LIMIT:-100}"
    experiment=$(experiment_name "$dataset" vit both_text 1)
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing prompt-edit checkpoint: $experiment"
    for version in base changed; do
        if [[ "$version" == "base" ]]; then
            prompt_file=$base_prompts
        else
            prompt_file=$edited_prompts
        fi
        name="ivc_edit_${dataset}_${version}_trainseed1_inferseed${INFERENCE_SEED}"
        args=(
            "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno --task uncond
            --check_path "$checkpoint" --v_encoder vit --spatial_guidance 2 --text_control
            --prompts-csv "$prompt_file" --prompt-subset-only --protocol oracle_prompt
            --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS"
            --ddim_schedule "$DDIM_SCHEDULE"
            --experiment_name "$name" --save-test-output auto --no-render
            --gpuid 0 --path-profile "$PATH_PROFILE"
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
        elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name"; then
            (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") \
                2>&1 | tee "$LOG_DIR/${name}.log"
        fi
    done
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_prompt_edit_locality.py" \
        --base "$METRIC_DIR/ivc_edit_${dataset}_base_trainseed1_inferseed${INFERENCE_SEED}_test_output.pt" \
        --edited "$METRIC_DIR/ivc_edit_${dataset}_changed_trainseed1_inferseed${INFERENCE_SEED}_test_output.pt" \
        --target-class 1 --target-delta 1 \
        --output "$SUMMARY_DIR/prompt_edit_locality_${dataset}.json"
done

for dataset in pku cgl; do
    run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_metric_strata.py" \
        --per-image "$METRIC_DIR/ivc_${dataset}_vit_both_text_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv" \
        --output "$SUMMARY_DIR/metric_strata_${dataset}.json"
done

density_args=(
    "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_density_controls.py"
    --path-profile "$PATH_PROFILE"
    --output-json "$SUMMARY_DIR/density_controls.json"
    --output-tex "$SUMMARY_DIR/density_controls.tex"
    --prediction "PKU LayoutDiT" pku "$ACTIVE_ROOT/other_baselines/layoutidit/pku_anno_uncond_test_output.pt"
    --prediction "CGL LayoutDiT" cgl "$ACTIVE_ROOT/other_baselines/layoutidit/cgl_anno_uncond_test_output.pt"
    --prediction "PKU External text baseline" pku "$ACTIVE_ROOT/other_baselines/standardized/postero_pku_freeform_subset.pt"
    --prediction "CGL External text baseline" cgl "$ACTIVE_ROOT/other_baselines/standardized/postero_cgl_freeform_subset.pt"
)
for seed in $SEEDS; do
    density_args+=(--prediction "PKU IntentDiT image-only" pku "$METRIC_DIR/ivc_pku_vit_both_trainseed${seed}_inferseed${INFERENCE_SEED}_test_output.pt")
    density_args+=(--prediction "CGL IntentDiT image-only" cgl "$METRIC_DIR/ivc_cgl_vit_both_trainseed${seed}_inferseed${INFERENCE_SEED}_test_output.pt")
    pku_freeform_raw="$METRIC_DIR/ivc_prompt_pku_vit_both_text_freeform_trainseed${seed}_inferseed${INFERENCE_SEED}_test_output.pt"
    cgl_freeform_raw="$METRIC_DIR/ivc_prompt_cgl_vit_both_text_freeform_trainseed${seed}_inferseed${INFERENCE_SEED}_test_output.pt"
    if [[ "$DRY_RUN" == "1" || -f "$pku_freeform_raw" ]]; then
        density_args+=(--prediction "PKU IntentDiT free-form" pku "$pku_freeform_raw")
    else
        log "SKIP free-form density raw tensor not found: $pku_freeform_raw"
    fi
    if [[ "$DRY_RUN" == "1" || -f "$cgl_freeform_raw" ]]; then
        density_args+=(--prediction "CGL IntentDiT free-form" cgl "$cgl_freeform_raw")
    else
        log "SKIP free-form density raw tensor not found: $cgl_freeform_raw"
    fi
done
run_command "${density_args[@]}"

for dataset in pku cgl; do
    parser_args=(
        "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_freeform_parser_coverage.py"
        --dataset "$dataset"
        --prompts "$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
        --per-image "IntentDiT" "$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv"
        --per-image "External text baseline" "$METRIC_DIR/baseline_text_${dataset}_freeform_per_image.csv"
        --output-json "$SUMMARY_DIR/freeform_parser_coverage_${dataset}.json"
        --output-tex "$SUMMARY_DIR/freeform_parser_coverage_${dataset}.tex"
    )
    run_command "${parser_args[@]}"
done

for dataset in pku cgl; do
    manifest_args=(
        "$PYTHON_BIN" "$CODE_DIR/scripts/build_freeform_evaluation_manifest.py"
        --dataset "$dataset"
        --prompts "$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"
        --baseline-label "External text baseline"
        --baseline-predictions "$ACTIVE_ROOT/other_baselines/standardized/postero_${dataset}_freeform_subset.pt"
        --baseline-per-image "$METRIC_DIR/baseline_text_${dataset}_freeform_per_image.csv"
        --output-jsonl "$SUMMARY_DIR/freeform_manifest_${dataset}.jsonl"
        --output-summary "$SUMMARY_DIR/freeform_manifest_${dataset}_summary.json"
    )
    for seed in $SEEDS; do
        manifest_args+=(
            --intentdit "$seed"
            "$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed${seed}_inferseed${INFERENCE_SEED}_test_output.pt"
            "$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed${seed}_inferseed${INFERENCE_SEED}_per_image.csv"
        )
    done
    run_command "${manifest_args[@]}"
done

for dataset in pku cgl; do
    experiment=$(experiment_name "$dataset" vit both_text 1)
    checkpoint=$(checkpoint_for "$dataset" "$experiment")
    [[ -n "$checkpoint" ]] || die "Missing checkpoint for oracle diagnostic: $experiment"
    prompts="$ACTIVE_ROOT/data/dataset/$dataset/split/csv/test_with_prompts_basic.csv"
    name="ivc_oracle_intent_${dataset}_trainseed1_inferseed${INFERENCE_SEED}"
    args=(
        "$PYTHON_BIN" scripts/test.py --dataset "$dataset" --anno anno --task uncond
        --check_path "$checkpoint" --v_encoder vit --spatial_guidance 2 --text_control
        --prompts-csv "$prompts" --protocol oracle_prompt --oracle-intent-map
        --seed "$INFERENCE_SEED" --ddim_num_steps "$DDIM_STEPS"
        --ddim_schedule "$DDIM_SCHEDULE"
        --experiment_name "$name" --save-test-output auto --no-render
        --gpuid 0 --path-profile "$PATH_PROFILE"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        print_command env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}"
    elif [[ "${FORCE:-0}" == "1" ]] || ! result_complete_for "$name"; then
        (cd "$CODE_DIR" && env CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}") 2>&1 | tee "$LOG_DIR/${name}.log"
    fi
done

raw="$METRIC_DIR/ivc_pku_vit_both_text_trainseed1_inferseed${INFERENCE_SEED}_test_output.pt"
prompts="$ACTIVE_ROOT/data/dataset/pku/split/csv/test_with_prompts_spatial.csv"

if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$raw" ]] || die "Missing raw predictions: $raw (rerun stage 05)"
    [[ -f "$prompts" ]] || die "Missing spatial prompts: $prompts"
fi

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_failures.py" \
    --predictions "$raw" \
    --dataset pku \
    --path-profile "$PATH_PROFILE" \
    --prompts-csv "$prompts" \
    --limit "${FAILURE_LIMIT:-1000}" \
    --overlap-threshold "${OVERLAP_THRESHOLD:-0.10}" \
    --occlusion-threshold "${OCCLUSION_THRESHOLD:-0.25}" \
    --readability-threshold "${READABILITY_THRESHOLD:-0.15}" \
    --output "$SUMMARY_DIR/failure_taxonomy_pku.json"

run_command "$PYTHON_BIN" "$ACTIVE_ROOT/user_study/export_real_failures.py" \
    --config "$CODE_DIR/configs/pku_anno_test.yaml" \
    --test-output "$raw" --text-control --per-category "${FAILURE_EXAMPLES:-5}" \
    --paths-base "$ACTIVE_ROOT/data/dataset/pku/split" \
    --out user_study/data/failures \
    --selection-json "$SUMMARY_DIR/user_study_failure_selection.json"

log "Diagnostics written under $SUMMARY_DIR"
