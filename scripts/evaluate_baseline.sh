#!/usr/bin/env bash
# Reproduce every Stage 1B result from a trained checkpoint. Run from the repository root.
# The order matters: settings are tuned on val and frozen before test or blind is decoded.
# Usage: bash scripts/evaluate_baseline.sh [checkpoint]
set -euo pipefail
export PYTHONUTF8=1
CHECKPOINT=${1:-data/checkpoints/baseline_rtx3050/best}
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python

"$PY" -m src.language_model --config configs/data.yaml --order 5
"$PY" -m src.logit_cache --split val --checkpoint "$CHECKPOINT"
"$PY" -m src.tune_lm --checkpoint "$CHECKPOINT" --write-config

for split in val test blind; do
  for decoder in greedy beam; do
    "$PY" -m src.evaluate --split "$split" --decoder "$decoder" --checkpoint "$CHECKPOINT"
  done
done

for decoder in greedy beam; do
  "$PY" -m src.analysis.error_analysis --predictions "results/evaluations/blind_$decoder.jsonl" \
    --output "results/error_analysis/blind_$decoder.json"
done
"$PY" -m src.check_inference --checkpoint "$CHECKPOINT"
"$PY" -m src.analysis.plots
