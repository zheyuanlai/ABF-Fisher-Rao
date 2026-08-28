#!/usr/bin/env bash
set -uo pipefail
cd /home/zheyuanlai/ABF-Fisher-Rao
LOG=results/io_abf_overnight/logs/phase5_arms_v2.log
{
  sleep 90
  while pgrep -f "run_tau_[a]rms.py" > /dev/null; do sleep 60; done
  echo "--- verdict ---"
  OMP_NUM_THREADS=8 python -u scripts/analyze_tau_arms.py
} >> "$LOG" 2>&1
