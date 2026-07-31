#!/usr/bin/env bash
# Analyze author-reviewed free-form parser audit CSVs.

set -Eeuo pipefail
source "$(dirname "$0")/00_common.sh"
mkdir_outputs

pku_audit="$SUMMARY_DIR/fixed_parser_audit_pku.csv"
cgl_audit="$SUMMARY_DIR/fixed_parser_audit_cgl.csv"

if [[ ! -f "$pku_audit" ]]; then
    pku_audit="$SUMMARY_DIR/parser_audit_pku.csv"
fi
if [[ ! -f "$cgl_audit" ]]; then
    cgl_audit="$SUMMARY_DIR/parser_audit_cgl.csv"
fi

if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$pku_audit" ]] || die "Missing parser audit CSV; generate it with code/scripts/build_parser_audit_template.py and review it"
    [[ -f "$cgl_audit" ]] || die "Missing parser audit CSV; generate it with code/scripts/build_parser_audit_template.py and review it"
fi

run_command "$PYTHON_BIN" "$CODE_DIR/scripts/analyze_parser_audit.py" \
    --audit pku "$pku_audit" \
    --audit cgl "$cgl_audit" \
    --output-json "$SUMMARY_DIR/parser_audit_summary.json" \
    --output-tex "$SUMMARY_DIR/parser_audit_summary.tex" \
    --require-complete

log "Parser audit written under $SUMMARY_DIR"
