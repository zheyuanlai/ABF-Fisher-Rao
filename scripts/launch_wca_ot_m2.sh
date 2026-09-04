#!/usr/bin/env bash
# M2: T0 + TR(c=0.5) first (all seeds), then TR(c=1).  GPU 3 only.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
LOG=results/ot_repair_campaign/wca/M2
echo "[$(date -Is)] start M2 c=0,0.5" >> $LOG/driver.log
python -u scripts/run_wca_ot_repair.py --stage M2 --seeds 820-823 --alpha 0.1 --c 0,0.5 >> $LOG/run.log 2>&1
echo "[$(date -Is)] end   M2 c=0,0.5 rc=$?" >> $LOG/driver.log
echo "[$(date -Is)] start M2 c=1" >> $LOG/driver.log
python -u scripts/run_wca_ot_repair.py --stage M2 --seeds 820-823 --alpha 0.1 --c 1 >> $LOG/run.log 2>&1
echo "[$(date -Is)] end   M2 c=1 rc=$?" >> $LOG/driver.log
echo "[$(date -Is)] M2 ALL DONE" >> $LOG/driver.log
