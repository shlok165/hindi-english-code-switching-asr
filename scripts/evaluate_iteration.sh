#!/usr/bin/env bash
# Evaluate one improvement iteration the way iteration 1 was, without touching
# iteration 1's files. Run from the repository root:
#   bash scripts/evaluate_iteration.sh NAME CHECKPOINT [LM_DIR] [VAL_MANIFEST_DIR]
# e.g.
#   bash scripts/evaluate_iteration.sh iteration3 data/checkpoints/iteration3/best \
#        data/language_model_latin data/processed/manifests_latin
# Order matters: the decoder is tuned on val and its config committed before
# test or blind is decoded. Outputs go to results/NAME/ and configs/decode_NAME.yaml.
set -euo pipefail
export PYTHONUTF8=1
NAME=$1
CHECKPOINT=$2
LM_DIR=${3:-data/language_model}
MANIFESTS=${4:-data/processed/manifests}
WORKERS=${WORKERS:-5}
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=python
OUT=results/$NAME
DECODE=configs/decode_$NAME.yaml

# Log-probabilities only; no references are read here.
for split in val test blind; do
  "$PY" -m src.logit_cache --split "$split" --checkpoint "$CHECKPOINT"
done

"$PY" -m src.tune_lm --checkpoint "$CHECKPOINT" --kenlm "$LM_DIR/train_5gram.klm" --manifest-dir "$MANIFESTS" \
  --workers "$WORKERS" --results "$OUT/lm_tuning.csv" --config-out "$DECODE" --write-config
git add "$DECODE" "$OUT/lm_tuning.csv"
git commit -q -m "Freeze ${NAME}'s decoder settings, tuned on val" -- "$DECODE" "$OUT/lm_tuning.csv"

for split in val test blind; do
  for decoder in greedy beam; do
    "$PY" -m src.evaluate --split "$split" --decoder "$decoder" --checkpoint "$CHECKPOINT" \
      --decode-config "$DECODE" --output-dir "$OUT/evaluations" --manifest-dir "$MANIFESTS" --workers "$WORKERS"
  done
done

for split in val blind; do
  for decoder in greedy beam; do
    "$PY" -m src.analysis.error_analysis --predictions "$OUT/evaluations/${split}_${decoder}.jsonl" \
      --output "$OUT/error_analysis/${split}_${decoder}.json"
  done
done
echo "evaluated $NAME"
