#!/usr/bin/env bash
# M3 unattended chain (docs/WCA_OT_CONFIRMATORY_M3.md): calibration -> blind alpha* -> core -> go/no-go -> repair.
# GPU 1 ONLY (user instruction for this round).  Each stage resumes past valid .npz files.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
LOG=results/ot_repair_campaign/wca/M3
mkdir -p $LOG
step() { echo "[$(date -Is)] $1" >> $LOG/driver.log; }
step "start calibration"
python -u scripts/run_wca_ot_m3.py --stage calibration >> $LOG/calibration.log 2>&1 || { step "calibration FAILED rc=$?"; exit 1; }
step "analyze calibration (marginal-only)"
python -u scripts/analyze_wca_ot_m3.py --stage calibration >> $LOG/analysis_calibration.log 2>&1 || { step "calibration analysis FAILED rc=$?"; exit 1; }
step "start core (alpha* from calibration/alpha_star.json)"
python -u scripts/run_wca_ot_m3.py --stage core >> $LOG/core.log 2>&1 || { step "core FAILED rc=$?"; exit 1; }
step "analyze core"
python -u scripts/analyze_wca_ot_m3.py --stage core >> $LOG/analysis_core.log 2>&1 || { step "core analysis FAILED rc=$?"; exit 1; }
if python -c "import json,sys; sys.exit(0 if json.load(open('$LOG/core/go_nogo.json'))['go'] else 1)"; then
  step "GO -> start repair"
  python -u scripts/run_wca_ot_m3.py --stage repair >> $LOG/repair.log 2>&1 || { step "repair FAILED rc=$?"; exit 1; }
  step "analyze repair"
  python -u scripts/analyze_wca_ot_m3.py --stage repair >> $LOG/analysis_repair.log 2>&1 || step "repair analysis FAILED rc=$?"
else
  step "NO-GO: M3-C not run"
fi
step "ALL DONE"
