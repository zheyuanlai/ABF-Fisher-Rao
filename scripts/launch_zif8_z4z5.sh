#!/usr/bin/env bash
# Unattended chain (docs/ZIF8_OT_Z4Z5.md): wait for the Z4 ladder -> frozen cell rule -> blind alpha
# calibration -> six-arm pilot -> analysis -> (go) confirmatory 16 fresh seeds -> analysis.  GPU 1 only.
set -u; cd "$(dirname "$0")/.."
PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python; ROOT=results/ot_repair_campaign/zif8; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $ROOT/z4z5_driver.log; }
while pgrep -f 'zif8_z4_budget_[l]adder.py' > /dev/null; do sleep 60; done
log "Z4 finished"; $PY scripts/zif8_z4_choose_cell.py | tee -a $ROOT/z4z5_driver.log
GO=$($PY -c "import json; print(json.load(open('$ROOT/Z4/cell_choice.json'))['go_z5'])"); NREP=$($PY -c "import json; print(json.load(open('$ROOT/Z4/cell_choice.json'))['n_replicas'])")
if [ "$GO" != "True" ]; then log "no establishment-limited/intermediate cell: Z5 NOT run"; exit 0; fi
log "Z5 cell: N = $NREP"
run_block () {   # stage n_seeds rng alpha n_steps [arms]
  local stage=$1 ns=$2 rng=$3 alpha=$4 nsteps=$5; shift 5; local arms="$*"; mkdir -p $ROOT/Z5/$stage/raw $ROOT/Z5/$stage/logs; local pids=()
  for arm in $arms; do
    case $arm in ot:*) a=${arm#ot:}; name=ot; al=$a;; *) name=$arm; al=$alpha;; esac
    $PY -u scripts/run_zif8_ot.py --arm $name --alpha $al --n-replicas $NREP --n-seeds $ns --rng-seed $rng --n-steps $nsteps --out $ROOT/Z5/$stage/raw > $ROOT/Z5/$stage/logs/${name}_a${al}.log 2>&1 & pids+=($!)
  done
  log "$stage: launched ${#pids[@]} arms ($arms) pids ${pids[*]}"; wait "${pids[@]}"; log "$stage: finished"
}
run_block calibration 2 20260980 0 150000 abf fr ot:0.03 ot:0.1 ot:0.3
$PY scripts/analyze_zif8_ot.py --stage calibration | tee -a $ROOT/z4z5_driver.log
ALPHA=$($PY -c "import json; print(json.load(open('$ROOT/Z5/calibration/alpha_star.json'))['alpha_star'])"); log "alpha* = $ALPHA"
run_block pilot 8 20260971 $ALPHA 300000 abf fr ot abf_r fr_r ot_r
$PY scripts/analyze_zif8_ot.py --stage pilot | tee -a $ROOT/z4z5_driver.log
GO2=$($PY -c "import json; print(json.load(open('$ROOT/Z5/pilot/go_nogo.json'))['go'])"); log "pilot go: $GO2"
if [ "$GO2" = "True" ] || [ "${FORCE_CONFIRMATORY:-0}" = "1" ]; then
  run_block confirmatory 16 20260990 $ALPHA 300000 abf fr ot abf_r fr_r ot_r
  $PY scripts/analyze_zif8_ot.py --stage confirmatory | tee -a $ROOT/z4z5_driver.log
fi
log "chain complete"
