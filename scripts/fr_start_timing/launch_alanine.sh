#!/usr/bin/env bash
# FR-start timing, alanine: one arm at a time on GPU 3 (the graphed engine saturates the
# device; concurrent arms only time-slice).  Priority order = primary contrast first.
# Stage abf builds the 20 ps init ensemble and caches it; every later arm loads it.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG=configs/fr_start_timing/alanine.yaml
INIT=results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz
LOG=results/fr_start_timing/alanine/logs
for s in abf u02_t5 u02_t20 u15_t5 u15_t20 u02_t10 u02_t2 o02_t5; do
  echo "[$(date -Is)] start stage $s" >> $LOG/driver.log
  python -u scripts/run_alanine_study.py --config $CFG --stage $s --init-cache $INIT --cuda-graph \
      > $LOG/$s.log 2>&1
  echo "[$(date -Is)] end   stage $s rc=$?" >> $LOG/driver.log
done
echo "[$(date -Is)] ALL DONE" >> $LOG/driver.log
