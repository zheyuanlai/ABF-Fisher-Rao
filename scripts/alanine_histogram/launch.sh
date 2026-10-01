#!/usr/bin/env bash
# Alanine adaptive-box histogram study: one stage at a time on GPU 3 (the graphed engine
# saturates the device).  Usage: launch.sh <stage> [<stage> ...]
# Every stage loads the FR-start-timing initial ensemble (bitwise the one the closed kernel
# arms started from) so histogram and kernel arms are paired by construction.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG=configs/alanine_histogram/alanine.yaml
INIT=results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz
LOG=results/alanine_histogram/logs
mkdir -p "$LOG"
for s in "$@"; do
  echo "[$(date -Is)] start stage $s" >> "$LOG/driver.log"
  python -u scripts/run_alanine_study.py --config "$CFG" --stage "$s" --init-cache "$INIT" --cuda-graph \
      > "$LOG/$s.log" 2>&1
  echo "[$(date -Is)] end   stage $s rc=$?" >> "$LOG/driver.log"
done
echo "[$(date -Is)] DONE: $*" >> "$LOG/driver.log"
