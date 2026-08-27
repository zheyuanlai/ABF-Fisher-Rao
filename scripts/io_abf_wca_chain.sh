#!/usr/bin/env bash
# WCA phase chain on GPU 3 only. Idempotent: every phase skips records already on
# disk, so an interrupt costs at most the run in flight.
set -uo pipefail
cd /home/zheyuanlai/ABF-Fisher-Rao
LOG=results/io_abf_overnight/logs/wca_chain.log
export CUDA_VISIBLE_DEVICES=3
{
  echo "=== WCA chain start $(date -u +%FT%TZ) ==="
  # wait for the 6-seed screening already in flight
  while pgrep -f "io_abf_wca_[g]ate.py --mode screen" > /dev/null; do sleep 30; done
  echo "--- calibration: extend the A0 screening to 16 seeds ---"
  python -u scripts/io_abf_wca_gate.py --mode screen \
      --seeds 1006,1007,1008,1009,1010,1011,1012,1013,1014,1015
  echo "--- score the A0 calibration against cache/phase_hp_v3 ---"
  python -u scripts/analyze_io_abf_wca.py --phase screening
  echo "--- pilot: 8 paired seeds x 3 arms ---"
  python -u scripts/io_abf_wca_gate.py --mode pilot
  echo "=== WCA chain done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
