#!/usr/bin/env bash
# Mechanism-campaign chain on GPU 3, phases in the preregistered order.
set -uo pipefail
cd /home/zheyuanlai/ABF-Fisher-Rao
LOG=results/io_abf_overnight/logs/mechanism_chain.log
export CUDA_VISIBLE_DEVICES=3
{
  echo "=== mechanism chain start $(date -u +%FT%TZ) ==="
  while pgrep -f "run_qr_mechanism_[r]erun" > /dev/null; do sleep 30; done
  echo "--- phase 0 rerun done; decomposition ---"
  OMP_NUM_THREADS=8 python -u scripts/analyze_qr_mechanism_phase0.py
  echo "--- phase 1: prescribed-r family ---"
  python -u scripts/run_prescribed_r.py --phase 1
  echo "--- phase 2: h scaling ---"
  python -u scripts/run_prescribed_r.py --phase 2h
  echo "--- phase 2: m scaling ---"
  python -u scripts/run_prescribed_r.py --phase 2m
  echo "--- phase 4: beta sweep ---"
  python -u scripts/run_prescribed_r.py --phase 4
  echo "--- prescribed-r analysis (phases 1/2/4 gates) ---"
  OMP_NUM_THREADS=8 python -u scripts/analyze_prescribed_r.py
  echo "=== mechanism chain done $(date -u +%FT%TZ) ==="
} >> "$LOG" 2>&1
