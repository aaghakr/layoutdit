#!/usr/bin/env bash
# Execute all non-interactive stages. User-study collection remains manual.

set -Eeuo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

bash "$SCRIPT_DIR/01_validate_environment.sh"
bash "$SCRIPT_DIR/02_train_paper_models.sh"
bash "$SCRIPT_DIR/03_train_loss_sweep.sh"
bash "$SCRIPT_DIR/04_train_architecture_ablations.sh"
bash "$SCRIPT_DIR/05_evaluate_paper_models.sh"
bash "$SCRIPT_DIR/06_aggregate_paper_results.sh"
bash "$SCRIPT_DIR/07_evaluate_prompt_robustness.sh"
bash "$SCRIPT_DIR/08_evaluate_sampling_efficiency.sh"
bash "$SCRIPT_DIR/09_evaluate_diversity.sh"
bash "$SCRIPT_DIR/10_evaluate_cross_dataset.sh"
bash "$SCRIPT_DIR/11_evaluate_external_baselines.sh"
bash "$SCRIPT_DIR/16_evaluate_text_parser_ablation.sh"
bash "$SCRIPT_DIR/17_analyze_parser_audit.sh"
bash "$SCRIPT_DIR/18_evaluate_manual_freeform_reference.sh"
bash "$SCRIPT_DIR/12_run_diagnostics.sh"

printf '[paper] Automated stages complete. Run 14 to prepare the study, collect participants, then run 15.\n'
