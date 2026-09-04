#!/usr/bin/env bash
# FR-start timing, pentane R15: one process per cell (launch-bound engine, so the two cells
# and the alanine process share GPU 3 with little mutual slowdown).  Methods run in priority
# order (baseline, primary, closed re-run, dose arms, ladder fill-ins); each invocation
# resumes past valid .npz files.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
CFG=configs/fr_start_timing/r15.yaml
LOG=results/fr_start_timing/r15/logs
stage=$1
for m in abf u02_s5000 u02_s12000 u10_s5000 u10_s12000 u02_s8000 u02_s3000; do
  echo "[$(date -Is)] $stage start $m" >> $LOG/driver_$stage.log
  python -u scripts/run_alkanes_cv_extension.py --config $CFG --stage $stage --only-method $m \
      --require-single-gpu >> $LOG/$stage.log 2>&1
  echo "[$(date -Is)] $stage end   $m rc=$?" >> $LOG/driver_$stage.log
done
echo "[$(date -Is)] $stage ALL DONE" >> $LOG/driver_$stage.log
