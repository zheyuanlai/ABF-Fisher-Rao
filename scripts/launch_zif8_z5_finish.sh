#!/usr/bin/env bash
# Finisher (replaces the tail of launch_zif8_z4z5.sh after GPU 3 was granted on 2026-09-07): wait for the
# pilot arms (GPU 1) -> analyze pilot; wait for the confirmatory arms (GPU 3, launched speculatively) ->
# analyze confirmatory.  The confirmatory is REPORTED as confirmatory only if the pilot go rule fired.
set -u; cd "$(dirname "$0")/.."
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python; ROOT=results/ot_repair_campaign/zif8
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $ROOT/z4z5_driver.log; }
while pgrep -f 'run_zif8_o[t].py.*rng-seed 20260971' > /dev/null; do sleep 60; done
log "pilot: all arms finished"; $PY scripts/analyze_zif8_ot.py --stage pilot | tee -a $ROOT/z4z5_driver.log
log "pilot go: $($PY -c "import json; print(json.load(open('$ROOT/Z5/pilot/go_nogo.json'))['go'])")"
while pgrep -f 'run_zif8_o[t].py.*rng-seed 20260990' > /dev/null; do sleep 60; done
log "confirmatory (GPU 3, speculative): all arms finished"; $PY scripts/analyze_zif8_ot.py --stage confirmatory | tee -a $ROOT/z4z5_driver.log
log "chain complete"
