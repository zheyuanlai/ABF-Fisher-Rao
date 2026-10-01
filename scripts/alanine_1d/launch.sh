#!/usr/bin/env bash
# 1-D alanine study: one stage at a time on GPU 3.  Usage: launch.sh <stage> [<stage> ...]
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG=configs/alanine_1d/alanine_1d.yaml
INIT=results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz
LOG=results/alanine_1d/logs
mkdir -p "$LOG"
for s in "$@"; do
  echo "[$(date -Is)] start stage $s" >> "$LOG/driver.log"
  python -u scripts/run_alanine_1d_study.py --config "$CFG" --stage "$s" --init-cache "$INIT" --cuda-graph \
      > "$LOG/$s.log" 2>&1
  echo "[$(date -Is)] end   stage $s rc=$?" >> "$LOG/driver.log"
done
echo "[$(date -Is)] DONE: $*" >> "$LOG/driver.log"
