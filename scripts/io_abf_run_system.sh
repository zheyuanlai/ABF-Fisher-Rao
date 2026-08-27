#!/usr/bin/env bash
# Run one system's full preregistered phase chain, in order, stopping on failure.
set -euo pipefail
SYS="$1"; GPU="${2:-3}"   # GPU 3 only: 0/1 belong to another user, 2 to another session
LOG="results/io_abf_overnight/logs/${SYS}.log"
{
  echo "=== $SYS on GPU $GPU  $(date -u +%FT%TZ) ==="
  for PH in probe calibration pilot confirmatory; do
    echo "--- phase $PH ---"
    CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/run_io_abf_campaign.py \
        --system "$SYS" --phase "$PH"
  done
  echo "=== $SYS DONE $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
