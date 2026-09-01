#!/usr/bin/env bash
# Instrumented ZIF-8 arms for the lineage mechanism experiment.
# Waits for the h_bias matrix so the two experiments never share the GPU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p results/information_campaign/lineage
while pgrep -f "zif8_bandwidth_sweep.py" >/dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] h_bias matrix done; starting instrumented arms"
python -u scripts/run_zif8_lineage.py > results/information_campaign/lineage/run.log 2>&1 \
  && python -u scripts/analyze_lineage_mechanism.py \
       > results/information_campaign/lineage/analysis.log 2>&1
echo "[$(date +%H:%M:%S)] lineage experiment complete"
tail -40 results/information_campaign/lineage/analysis.log
