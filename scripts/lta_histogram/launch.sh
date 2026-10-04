#!/usr/bin/env bash
# Overnight chain of configs/lta_histogram/campaign.json, GPU 3 only, one process at a time.
# Order = campaign.json execution_order. Every stage is resumable (runners skip existing npz).
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python
LOG=results/lta_histogram/logs; mkdir -p "$LOG"
DRV="$LOG/driver.log"
stage () {  # name, command...
  local name=$1; shift
  echo "[$(date -u -Is)] START $name" >> "$DRV"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  echo "[$(date -u -Is)] END   $name rc=$rc" >> "$DRV"
  return $rc
}
echo "[$(date -u -Is)] CHAIN START pid $$ git $(git rev-parse --short HEAD)" >> "$DRV"
stage movie_T300        $PY -u scripts/run_lta_histogram.py movie --temperature 300
stage reference_T350    $PY -u scripts/run_lta_reference.py --temperature 350 --kappa 594 --n-steps 128571 --burn-in 25714 --seed 20260883 --unbiased-steps 100000
stage calibrate_T350    $PY -u scripts/run_lta_histogram.py calibrate --temperature 350
stage production_T300   $PY -u scripts/run_lta_histogram.py production --temperature 300
stage movie_T150        $PY -u scripts/run_lta_histogram.py movie --temperature 150
stage production_T150   $PY -u scripts/run_lta_histogram.py production --temperature 150
stage production_T350   $PY -u scripts/run_lta_histogram.py production --temperature 350
stage production_T225   $PY -u scripts/run_lta_histogram.py production --temperature 225
stage production_T80    $PY -u scripts/run_lta_histogram.py production --temperature 80
stage width_ladder_T300 $PY -u scripts/run_lta_histogram.py width-ladder
echo "[$(date -u -Is)] CHAIN DONE" >> "$DRV"
