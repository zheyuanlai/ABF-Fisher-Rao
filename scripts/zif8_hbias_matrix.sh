#!/usr/bin/env bash
# ABF-only online-bandwidth arms; readout swept offline in each.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
for hb in 0.10 0.05 0.025; do
  echo "[$(date +%H:%M:%S)] h_bias = $hb"
  python -u scripts/zif8_bandwidth_sweep.py --h-bias "$hb" \
    > "results/information_campaign/hbias_${hb}.log" 2>&1 || { echo "FAILED at $hb"; exit 1; }
  tail -3 "results/information_campaign/hbias_${hb}.log"
done
echo "[$(date +%H:%M:%S)] h_bias matrix complete"
