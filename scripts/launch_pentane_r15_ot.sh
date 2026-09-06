#!/usr/bin/env bash
# Unattended chain for the pentane R15 OT + repair campaign (docs/PENTANE_R15_OT_REPAIR.md).
#   calibration (may already be running/done) -> alpha* -> pilot (6 arms, 8 seeds, concurrent)
#   -> analysis -> go/no-go -> confirmatory (6 arms, 16 fresh seeds, concurrent) -> analysis.
# GPU 1 only.  Runs skip when their .npz exists.  Log: <root>/driver.log
set -u
cd "$(dirname "$0")/.."
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python
ROOT=results/ot_repair_campaign/pentane_r15
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $ROOT/driver.log; }

run_block () {   # stage n_seeds rng alpha
  local stage=$1 ns=$2 rng=$3 alpha=$4
  mkdir -p $ROOT/$stage/raw $ROOT/$stage/logs
  local pids=()
  for arm in abf fr ot abf_r fr_r ot_r; do
    $PY -u scripts/run_pentane_r15_ot.py --arm $arm --alpha $alpha --n-seeds $ns --rng-seed $rng --n-steps 80000 --save-every 4000 \
        --out $ROOT/$stage/raw > $ROOT/$stage/logs/$arm.log 2>&1 &
    pids+=($!)
  done
  log "$stage: launched 6 arms (alpha $alpha, $ns seeds, rng $rng) pids ${pids[*]}"
  wait "${pids[@]}"
  log "$stage: all arms finished"
}

# ---- calibration: wait for the 5 runs if they are still in flight, else launch ----
while pgrep -f 'run_pentane_r15_[o]t.py --arm .* --rng-seed 20260801' > /dev/null; do sleep 30; done
if [ "$(ls $ROOT/calibration/raw/*.npz 2>/dev/null | wc -l)" -lt 5 ]; then
  log "calibration: launching missing runs"
  mkdir -p $ROOT/calibration/raw $ROOT/calibration/logs; pids=()
  for spec in "abf 0" "fr 0" "ot 0.01" "ot 0.03" "ot 0.1"; do set -- $spec
    $PY -u scripts/run_pentane_r15_ot.py --arm $1 --alpha $2 --n-seeds 4 --rng-seed 20260801 --n-steps 40000 --save-every 2000 \
        --out $ROOT/calibration/raw > $ROOT/calibration/logs/${1}_a${2}.log 2>&1 & pids+=($!)
  done; wait "${pids[@]}"
fi
$PY scripts/analyze_pentane_r15_ot.py --stage calibration | tee -a $ROOT/driver.log
ALPHA=$($PY -c "import json; print(json.load(open('$ROOT/calibration/alpha_star.json'))['alpha_star'])")
log "alpha* = $ALPHA"

# ---- pilot ----
run_block pilot 8 20260719 $ALPHA
$PY scripts/analyze_pentane_r15_ot.py --stage pilot | tee -a $ROOT/driver.log
GO=$($PY -c "import json; print(json.load(open('$ROOT/pilot/go_nogo.json'))['go'])")
log "pilot go/no-go: $GO"

# ---- confirmatory (16 fresh seeds) ----
if [ "$GO" = "True" ] || [ "${FORCE_CONFIRMATORY:-0}" = "1" ]; then
  run_block confirmatory 16 20260906 $ALPHA
  $PY scripts/analyze_pentane_r15_ot.py --stage confirmatory | tee -a $ROOT/driver.log
else
  log "confirmatory NOT launched (pilot no-go; set FORCE_CONFIRMATORY=1 to override)"
fi
log "chain complete"
