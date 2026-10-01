#!/usr/bin/env bash
# Walker-count ladder: one stage at a time on GPU 3; the init cache is PER N (parsed from the stage name N<N>_...).
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG=configs/alanine_nladder/alanine_N.yaml
ROOT=results/alanine_nladder
LOG="$ROOT/logs"; mkdir -p "$LOG"
for s in "$@"; do
  N=$(echo "$s" | sed -E 's/^N([0-9]+)_.*/\1/')
  INIT="$ROOT/init_c7eq_seeds0-15_N${N}.npz"
  echo "[$(date -Is)] start stage $s (init $INIT)" >> "$LOG/driver.log"
  python -u scripts/run_alanine_study.py --config "$CFG" --stage "$s" --init-cache "$INIT" --cuda-graph \
      > "$LOG/$s.log" 2>&1
  echo "[$(date -Is)] end   stage $s rc=$?" >> "$LOG/driver.log"
done
echo "[$(date -Is)] DONE: $*" >> "$LOG/driver.log"
