#!/usr/bin/env bash
# Detached, resumable launch of one job file (single-threaded runs), with a registry line per launch.
#   bash scripts/convergence_fr/launch.sh configs/adaptive_fr/jobs_A.json 16
# Check: tail results/adaptive_fr_campaign/logs/<stage>.log ; grep ALLDONE ... ; registry: logs/launches.tsv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOBS=$1; PROCS=$2
STAGE=$(basename "$JOBS" .json); STAGE=${STAGE#jobs_}
LOG=$ROOT/results/adaptive_fr_campaign/logs/$STAGE.log
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" setsid nohup \
  /home/zheyuanlai/miniconda3/envs/abffr/bin/python scripts/eri_campaign/run_jobs.py "$JOBS" --procs "$PROCS" \
  >> "$LOG" 2>&1 < /dev/null &
PID=$!
printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$STAGE" "$PID" "$PROCS" "$LOG" >> "$ROOT/results/adaptive_fr_campaign/logs/launches.tsv"
echo "$STAGE pid $PID log $LOG"
