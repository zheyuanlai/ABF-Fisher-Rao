#!/usr/bin/env bash
# Detached, resumable launch of an LTA job file (single-threaded runs), registry line per launch.
#   bash scripts/adaptive_fr/launch_lta.sh configs/adaptive_fr/jobs_C3.json 32
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOBS=$1; PROCS=$2
STAGE=$(basename "$JOBS" .json); STAGE=${STAGE#jobs_}
LOG=$ROOT/results/adaptive_fr_campaign/logs/$STAGE.log
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup \
  /home/zheyuanlai/miniconda3/envs/abffr/bin/python scripts/adaptive_fr/run_lta_jobs.py "$JOBS" --procs "$PROCS" \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$STAGE" "$PID" "$PROCS" "$LOG" >> "$ROOT/results/adaptive_fr_campaign/logs/launches.tsv"
echo "$STAGE pid $PID log $LOG"
