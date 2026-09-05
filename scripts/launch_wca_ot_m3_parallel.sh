#!/usr/bin/env bash
# M3 core + repair as 3 concurrent processes on disjoint seed splits (the sampler is CPU/launch-
# bound: GPU time ~1 ms of ~2.8 ms per step), then the frozen analysis, go/no-go, and repair.
# Same protocol / seeds / analyzer as launch_wca_ot_m3.sh.  GPU 1 ONLY.  Each process resumes
# past valid .npz files.  WCA_DENSE_FORCE=1: compiled all-pairs force (force_impl recorded per run).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
export WCA_DENSE_FORCE=${WCA_DENSE_FORCE:-0}   # compiled force NOT deployed: it breaks the bit-identity tests (see engine docstring)
LOG=results/ot_repair_campaign/wca/M3
mkdir -p $LOG
step() { echo "[$(date -Is)] $1" >> $LOG/driver.log; }
SPLITS=("903-907" "908-911" "912-915")
run_stage() {   # $1 = stage
  local pids=()
  for i in 0 1 2; do
    python -u scripts/run_wca_ot_m3.py --stage "$1" --seeds "${SPLITS[$i]}" >> "$LOG/$1.p$i.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
step "start core x3 (splits ${SPLITS[*]}; seeds 900-902 already complete; WCA_DENSE_FORCE=$WCA_DENSE_FORCE)"
run_stage core || { step "core FAILED"; exit 1; }
step "analyze core"
python -u scripts/analyze_wca_ot_m3.py --stage core >> $LOG/analysis_core.log 2>&1 || { step "core analysis FAILED"; exit 1; }
if python -c "import json,sys; sys.exit(0 if json.load(open('$LOG/core/go_nogo.json'))['go'] else 1)"; then
  step "GO -> start repair x3"
  SPLITS=("900-905" "906-910" "911-915")
  run_stage repair || { step "repair FAILED"; exit 1; }
  step "analyze repair"
  python -u scripts/analyze_wca_ot_m3.py --stage repair >> $LOG/analysis_repair.log 2>&1 || step "repair analysis FAILED"
else
  step "NO-GO: M3-C not run"
fi
step "ALL DONE"
