#!/usr/bin/env bash
# Evaluate real third-party predictions with the identical evaluator and pairwise CIs.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

missing=0
for dataset in pku cgl; do
    upper=${dataset^^}
    image_var="LAYOUTDIT_${upper}_PREDICTIONS"
    text_var="TEXT_BASELINE_${upper}_PREDICTIONS"
    real_feature_var="OFFICIAL_REAL_LAYOUT_FEATURES_${upper}"
    generated_feature_var="OFFICIAL_INTENTDIT_LAYOUT_FEATURES_${upper}"
    metric_reference_var="OFFICIAL_METRIC_REFERENCE_${upper}"
    image_predictions=${!image_var:-}
    text_predictions=${!text_var:-}
    real_features=${!real_feature_var:-}
    generated_features=${!generated_feature_var:-}
    metric_reference=${!metric_reference_var:-}
    prompts="$ACTIVE_ROOT/data/prompts/free_form_${dataset}.csv"

    if [[ -n "$image_predictions" ]]; then
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_saved_predictions.py" \
            --predictions "$image_predictions" --dataset "$dataset" --anno anno \
            --experiment-name "baseline_layoutdit_${dataset}_image_only" \
            --protocol image_only --path-profile "$PATH_PROFILE" --output-dir "$METRIC_DIR"
        ours="$METRIC_DIR/ivc_${dataset}_vit_both_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv"
        baseline="$METRIC_DIR/baseline_layoutdit_${dataset}_image_only_per_image.csv"
        if [[ "$DRY_RUN" == "1" || -f "$ours" ]]; then
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_bootstrap.py" \
                --method-a "$ours" --method-b "$baseline" \
                --name-a intentdit_image_only --name-b layoutdit \
                --output "$SUMMARY_DIR/paired_layoutdit_${dataset}.json"
            ours_seeds=()
            for train_seed in $SEEDS; do
                ours_seeds+=("$METRIC_DIR/ivc_${dataset}_vit_both_trainseed${train_seed}_inferseed${INFERENCE_SEED}_per_image.csv")
            done
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_seed_image_bootstrap.py" \
                --method-a "${ours_seeds[@]}" --method-b "$baseline" \
                --name-a intentdit_image_only --name-b layoutdit \
                --output "$SUMMARY_DIR/paired_layoutdit_${dataset}_multiseed.json"
        fi
    else
        log "MISSING $image_var (actual LayoutDiT standardized .pt)"
        missing=1
    fi

    if [[ -n "$text_predictions" ]]; then
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/evaluate_saved_predictions.py" \
            --predictions "$text_predictions" --dataset "$dataset" --anno anno \
            --experiment-name "baseline_text_${dataset}_freeform" \
            --protocol text_baseline --text-control --spatial-metrics \
            --prompts-csv "$prompts" --path-profile "$PATH_PROFILE" --output-dir "$METRIC_DIR"
        ours_text="$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed1_inferseed${INFERENCE_SEED}_per_image.csv"
        text_baseline="$METRIC_DIR/baseline_text_${dataset}_freeform_per_image.csv"
        if [[ "$DRY_RUN" == "1" || -f "$ours_text" ]]; then
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_bootstrap.py" \
                --method-a "$ours_text" --method-b "$text_baseline" \
                --name-a intentdit_text --name-b external_text_baseline \
                --output "$SUMMARY_DIR/paired_text_baseline_${dataset}.json"
            ours_text_seeds=()
            for train_seed in $SEEDS; do
                ours_text_seeds+=("$METRIC_DIR/ivc_prompt_${dataset}_vit_both_text_freeform_trainseed${train_seed}_inferseed${INFERENCE_SEED}_per_image.csv")
            done
            run_command "$PYTHON_BIN" "$CODE_DIR/scripts/paired_seed_image_bootstrap.py" \
                --method-a "${ours_text_seeds[@]}" --method-b "$text_baseline" \
                --name-a intentdit_text --name-b external_text_baseline \
                --output "$SUMMARY_DIR/paired_text_baseline_${dataset}_multiseed.json"
        fi
    else
        log "MISSING $text_var (actual text-conditioned baseline standardized .pt)"
        missing=1
    fi

    if [[ -n "$real_features" && -n "$generated_features" && -n "${OFFICIAL_LAYOUT_EXTRACTOR_NAME:-}" && -n "${OFFICIAL_LAYOUT_EXTRACTOR_CHECKSUM:-}" ]]; then
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/compute_layout_fid.py" \
            --real-features "$real_features" --generated-features "$generated_features" \
            --extractor-name "$OFFICIAL_LAYOUT_EXTRACTOR_NAME" \
            --extractor-checksum "$OFFICIAL_LAYOUT_EXTRACTOR_CHECKSUM" \
            --output "$SUMMARY_DIR/official_layout_fid_${dataset}.json"
    else
        log "MISSING official layout-FID features/identity for $dataset"
        [[ "${REQUIRE_OFFICIAL_LAYOUT_FID:-0}" == "0" ]] || missing=1
    fi

    ours_metrics="$METRIC_DIR/ivc_${dataset}_vit_both_trainseed1_inferseed${INFERENCE_SEED}_metrics.json"
    if [[ -n "$metric_reference" && ( "$DRY_RUN" == "1" || -f "$ours_metrics" ) ]]; then
        run_command "$PYTHON_BIN" "$CODE_DIR/scripts/validate_metric_parity.py" \
            --ours "$ours_metrics" --reference "$metric_reference" \
            --output "$SUMMARY_DIR/metric_parity_${dataset}.json"
    else
        log "MISSING $metric_reference_var or local metrics for parity validation"
        [[ "${REQUIRE_METRIC_PARITY:-0}" == "0" ]] || missing=1
    fi
done

if [[ "$missing" == "1" ]]; then
    log "Baseline stage incomplete. Use scripts/import_layout_predictions.py if conversion is needed."
    [[ "${REQUIRE_BASELINES:-0}" == "0" ]] || die "Required baseline predictions are missing"
fi
