#!/usr/bin/env bash
# Instrumented ZIF-8 arms for the lineage mechanism experiment.
# Waits for the h_bias matrix so the two experiments never share the GPU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p results/information_campaign/lineage
# Wait on a COMPLETION MARKER, not on pgrep.  `pgrep -f <pattern>` also matches
# the shell that launched THIS script, because that shell's command line
# contains this file's own text -- so the obvious version waits on itself
# forever.  A marker file cannot self-match.
MARKER="results/information_campaign/zif8_bandwidth_sweep_T300_hb0.025.json"
while [ ! -f "$MARKER" ]; do sleep 120; done
echo "[$(date +%H:%M:%S)] h_bias matrix done; starting instrumented arms"
python -u scripts/run_zif8_lineage.py > results/information_campaign/lineage/run.log 2>&1 \
  && python -u scripts/analyze_lineage_mechanism.py \
       > results/information_campaign/lineage/analysis.log 2>&1
echo "[$(date +%H:%M:%S)] lineage experiment complete"
tail -40 results/information_campaign/lineage/analysis.log
