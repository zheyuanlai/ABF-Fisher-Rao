#!/usr/bin/env bash
# Low-T alanine arms: one stage at a time on GPU 3.  Usage: launch.sh <config.yaml> <stage> [<stage> ...]
# The init cache is built by the first stage at the run temperature and reused by the others.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG="$1"; shift
ROOT=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CFG'))['output_root'])")
INIT="$ROOT/init_c7eq_seeds0-15_N2048.npz"
LOG="$ROOT/logs"; mkdir -p "$LOG"
for s in "$@"; do
  echo "[$(date -Is)] start stage $s" >> "$LOG/driver.log"
  python -u scripts/run_alanine_study.py --config "$CFG" --stage "$s" --init-cache "$INIT" --cuda-graph \
      > "$LOG/$s.log" 2>&1
  echo "[$(date -Is)] end   stage $s rc=$?" >> "$LOG/driver.log"
done
echo "[$(date -Is)] DONE: $*" >> "$LOG/driver.log"
