#!/bin/bash
# Run fair evaluation (same protocol: unanno, uncond, 100 DDIM steps).
# Activate your conda env first, e.g.:
#   conda activate cgbdm
# Then from the code/ directory:
#   cp configs/experiments_fair_eval_example.yaml configs/experiments_fair_eval.yaml
#   ./run_fair_eval.sh [gpu_id] [config]

set -e
cd "$(dirname "$0")"
GPUID=${1:-0}
CONFIG=${2:-configs/experiments_fair_eval.yaml}
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG. Copy configs/experiments_fair_eval_example.yaml and fill checkpoint paths." >&2
  exit 2
fi
echo "Running fair eval (gpuid=$GPUID, config=$CONFIG)."
python scripts/run_fair_eval_all.py \
  --experiments "$CONFIG" \
  --anno anno \
  --ddim_num_steps 100 \
  --ddim_schedule cosine \
  --gpuid "$GPUID"
echo "Done. Check experiments/paper_figures/fair_eval_results.csv and fair_eval_results.md"
