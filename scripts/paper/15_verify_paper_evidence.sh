#!/usr/bin/env bash
# Final gate: verify code, primary results, prompt results, and study data presence.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"

if [[ "$DRY_RUN" == "1" ]]; then
    print_command bash -lc "cd '$CODE_DIR' && '$PYTHON_BIN' -m unittest discover -s tests -v"
else
    (cd "$CODE_DIR" && "$PYTHON_BIN" -m unittest discover -s tests -v)
fi
run_command "$PYTHON_BIN" "$CODE_DIR/validate_configs.py" --path-profile "$PATH_PROFILE"
run_command "$PYTHON_BIN" "$CODE_DIR/scripts/validate_ddim_schedule_evidence.py" \
    --input-dir "$METRIC_DIR" \
    --expected-schedule "$DDIM_SCHEDULE" \
    --allowed-steps 10 20 50 100 \
    --include-prefix ivc_

required=(
    "$SUMMARY_DIR/primary_training_seed_summary.json"
    "$SUMMARY_DIR/primary_training_seed_summary.md"
    "$SUMMARY_DIR/tables/image_only_results.tex"
    "$SUMMARY_DIR/tables/oracle_prompt_results.tex"
    "$SUMMARY_DIR/prompt_training_seed_summary.json"
    "$SUMMARY_DIR/prompt_training_seed_summary.md"
    "$SUMMARY_DIR/prompt_stress_pku.json"
    "$SUMMARY_DIR/prompt_stress_cgl.json"
    "$SUMMARY_DIR/intent_map_pku.json"
    "$SUMMARY_DIR/intent_map_cgl.json"
    "$SUMMARY_DIR/intent_map_pku_per_image.csv"
    "$SUMMARY_DIR/intent_map_cgl_per_image.csv"
    "$SUMMARY_DIR/failure_taxonomy_pku.json"
    "$SUMMARY_DIR/metric_strata_pku.json"
    "$SUMMARY_DIR/metric_strata_cgl.json"
    "$SUMMARY_DIR/density_controls.json"
    "$SUMMARY_DIR/density_controls.tex"
    "$SUMMARY_DIR/freeform_parser_coverage_pku.json"
    "$SUMMARY_DIR/freeform_parser_coverage_cgl.json"
    "$SUMMARY_DIR/freeform_parser_coverage_pku.tex"
    "$SUMMARY_DIR/freeform_parser_coverage_cgl.tex"
    "$SUMMARY_DIR/freeform_manifest_pku.jsonl"
    "$SUMMARY_DIR/freeform_manifest_cgl.jsonl"
    "$SUMMARY_DIR/freeform_manifest_pku_summary.json"
    "$SUMMARY_DIR/freeform_manifest_cgl_summary.json"
    "$SUMMARY_DIR/text_parser_ablation_summary.json"
    "$SUMMARY_DIR/text_parser_ablation_summary.md"
    "$SUMMARY_DIR/parser_audit_summary.json"
    "$SUMMARY_DIR/parser_audit_summary.tex"
    "$SUMMARY_DIR/manual_freeform_pku_summary.json"
    "$SUMMARY_DIR/manual_freeform_pku_summary.md"
    "$SUMMARY_DIR/manual_freeform_pku_summary.tex"
    "$SUMMARY_DIR/manual_freeform_cgl_summary.json"
    "$SUMMARY_DIR/manual_freeform_cgl_summary.md"
    "$SUMMARY_DIR/manual_freeform_cgl_summary.tex"
    "$SUMMARY_DIR/paired_manual_freeform_pku_all.json"
    "$SUMMARY_DIR/paired_manual_freeform_cgl_all.json"
    "$SUMMARY_DIR/paired_manual_freeform_pku_retained.json"
    "$SUMMARY_DIR/paired_manual_freeform_cgl_retained.json"
    "$SUMMARY_DIR/paired_text_parser_pku_no_parser.json"
    "$SUMMARY_DIR/paired_text_parser_pku_parser_only.json"
    "$SUMMARY_DIR/paired_text_parser_cgl_no_parser.json"
    "$SUMMARY_DIR/paired_text_parser_cgl_parser_only.json"
    "$SUMMARY_DIR/prompt_edit_locality_pku.json"
    "$SUMMARY_DIR/prompt_edit_locality_cgl.json"
    "$METRIC_DIR/ivc_oracle_intent_pku_trainseed1_inferseed${INFERENCE_SEED}_metrics.json"
    "$METRIC_DIR/ivc_oracle_intent_cgl_trainseed1_inferseed${INFERENCE_SEED}_metrics.json"
    "$SUMMARY_DIR/diversity_pku.json"
    "$SUMMARY_DIR/diversity_cgl.json"
    "$SUMMARY_DIR/paired_layoutdit_pku.json"
    "$SUMMARY_DIR/paired_layoutdit_cgl.json"
    "$SUMMARY_DIR/paired_layoutdit_pku_multiseed.json"
    "$SUMMARY_DIR/paired_layoutdit_cgl_multiseed.json"
    "$SUMMARY_DIR/paired_text_baseline_pku.json"
    "$SUMMARY_DIR/paired_text_baseline_pku_multiseed.json"
    "$SUMMARY_DIR/paired_text_baseline_cgl.json"
    "$SUMMARY_DIR/paired_text_baseline_cgl_multiseed.json"
    "$METRIC_DIR/ivc_cross_pku_to_cgl_trainseed1_inferseed${INFERENCE_SEED}_metrics.json"
    "$METRIC_DIR/ivc_cross_cgl_to_pku_trainseed1_inferseed${INFERENCE_SEED}_metrics.json"
    "$METRIC_DIR/baseline_text_pku_freeform_metrics.json"
    "$METRIC_DIR/baseline_text_cgl_freeform_metrics.json"
)

if [[ "${REQUIRE_OFFICIAL_LAYOUT_FID:-0}" == "1" ]]; then
    required+=(
        "$SUMMARY_DIR/official_layout_fid_pku.json"
        "$SUMMARY_DIR/official_layout_fid_cgl.json"
    )
else
    log "Skipping official LayoutFID gate (set REQUIRE_OFFICIAL_LAYOUT_FID=1 to require it)"
fi

if [[ "${REQUIRE_METRIC_PARITY:-0}" == "1" ]]; then
    required+=(
        "$SUMMARY_DIR/metric_parity_pku.json"
        "$SUMMARY_DIR/metric_parity_cgl.json"
    )
else
    log "Skipping official metric-parity gate (set REQUIRE_METRIC_PARITY=1 to require it)"
fi

for dataset in pku cgl; do
    for steps in 10 20 50 100; do
        required+=("$METRIC_DIR/ivc_efficiency_${dataset}_steps${steps}_trainseed1_inferseed${INFERENCE_SEED}_evidence.json")
    done
done
for scale in 0 0p5 1 1p5 2; do
    required+=("$METRIC_DIR/ivc_textscale_pku_${scale}_trainseed1_inferseed${INFERENCE_SEED}_metrics.json")
done

missing=0
for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
        printf 'MISSING %s\n' "$path" >&2
        missing=1
    fi
done

study_db="$ACTIVE_ROOT/user_study/data/study.sqlite"
if [[ -f "$study_db" ]]; then
    run_command "$PYTHON_BIN" "$ACTIVE_ROOT/user_study/aggregate_results.py"
    run_command "$PYTHON_BIN" "$ACTIVE_ROOT/user_study/hierarchical_bootstrap.py" \
        --database "$study_db" \
        --baseline layoutdit \
        --method-a intentdit_image --criterion quality \
        --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
        --min-participants "${MIN_PARTICIPANTS:-20}" \
        --output "$ACTIVE_ROOT/experiments/user_study/hierarchical_bootstrap_quality.json"
    run_command "$PYTHON_BIN" "$ACTIVE_ROOT/user_study/hierarchical_bootstrap.py" \
        --database "$study_db" --baseline textbaseline \
        --method-a intentdit_text --criterion instruction \
        --iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
        --min-participants "${MIN_PARTICIPANTS:-20}" \
        --output "$ACTIVE_ROOT/experiments/user_study/hierarchical_bootstrap_instruction.json"
else
    log "User study database not present yet: $study_db"
    missing=1
fi

[[ "$missing" == "0" ]] || die "Paper evidence is incomplete"
log "All automated paper evidence gates passed"
