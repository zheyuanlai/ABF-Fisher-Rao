#!/usr/bin/env bash
# Overnight orchestration for the WCA FR + targeted solvent relaxation campaign.
# Runs the preregistered sequence W0-B -> W0 analysis -> (gate) -> W1 -> W1 analysis -> (gate) -> W2,
# committing each stage's frozen outputs before the next stage starts.  Every gate is enforced by the
# runner's own assertions (W1 refuses without tau_map.passed; W2 refuses without rho_selection.licensed).
# Detach with:  setsid nohup bash scripts/overnight_wca_targeted_relax.sh > results/targeted_relax_campaign/wca/overnight.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
START=${1:-W0}      # W0 (default) runs everything; W1 skips the W0 stages (used after amendment A1)
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:scripts
W=results/targeted_relax_campaign/wca
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
commit() { git add "$@" >/dev/null 2>&1; git commit -q -m "$COMMIT_MSG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && echo "[$(stamp)] committed: $(git log --oneline -1)"; }

if [ "$START" = "W0" ]; then
echo "[$(stamp)] === W0-A: instrument runs (completed runs are skipped) ==="
mkdir -p $W/W0 $W/W1
python -u scripts/run_wca_targeted_relax.py --stage W0A > $W/W0/run_W0A.log 2>&1 || { echo "[$(stamp)] W0-A FAILED"; exit 1; }
tail -3 $W/W0/run_W0A.log
echo "[$(stamp)] === W0-B: constrained correlations ==="
python -u scripts/run_wca_targeted_relax.py --stage W0B > $W/W0/run_W0B.log 2>&1 || { echo "[$(stamp)] W0-B FAILED"; exit 1; }
python -u scripts/analyze_wca_targeted_relax.py --stage W0 > $W/W0/analysis.log 2>&1 || { echo "[$(stamp)] W0 analysis FAILED"; exit 1; }
tail -5 $W/W0/analysis.log
if grep -q "TAU_MAP_UNRESOLVED" $W/W0/analysis.log; then
  echo "[$(stamp)] tau unresolved -> the ONE preregistered 2x extension"
  python -u scripts/run_wca_targeted_relax.py --stage W0B --extend > $W/W0/run_W0B_extended.log 2>&1 || { echo "[$(stamp)] W0-B extension FAILED"; exit 1; }
  python -u scripts/analyze_wca_targeted_relax.py --stage W0 > $W/W0/analysis_extended.log 2>&1
  tail -5 $W/W0/analysis_extended.log
fi
COMMIT_MSG="WCA targeted-relax W0 closed: $(python -c "import json;d=json.load(open('$W/W0/tau_map.json'));print(d['outcome'], 'spearman %.3f' % d['spearman'], 'tau map', d['sha256'][:12])")"
commit $W/W0/tau_map.json $W/W0/analysis.json $W/W0/selection.json $W/W0/provenance_W0A.json $W/W0/figures
if ! python -c "import json,sys;sys.exit(0 if json.load(open('$W/W0/tau_map.json'))['passed'] else 1)"; then
  echo "[$(stamp)] STOP at W0 (gate failed). W1 not started."; exit 0
fi
fi   # START=W0

echo "[$(stamp)] === W1: cost ladder (seeds 820-823) ==="
T1=$(date +%s)
python -u scripts/run_wca_targeted_relax.py --stage W1 > $W/W1/run_W1.log 2>&1 || { echo "[$(stamp)] W1 FAILED"; exit 1; }
ELAPSED=$(( $(date +%s) - T1 ))
echo "[$(stamp)] W1 4-seed block took $ELAPSED s"
if [ "$ELAPSED" -le 12600 ]; then
  echo "[$(stamp)] extension rule met (<= 3.5 h): running seeds 824-827"
  python -u scripts/run_wca_targeted_relax.py --stage W1 --seeds 824-827 > $W/W1/run_W1_ext.log 2>&1 || echo "[$(stamp)] W1 extension FAILED (continuing with the 4-seed block)"
fi
python -u scripts/analyze_wca_targeted_relax.py --stage W1 > $W/W1/analysis.log 2>&1 || { echo "[$(stamp)] W1 analysis FAILED"; exit 1; }
tail -8 $W/W1/analysis.log
COMMIT_MSG="WCA targeted-relax W1 closed: $(python -c "import json;d=json.load(open('$W/W1/rho_selection.json'));print('rho*', d['rho_star'], 'licensed', d['licensed'], 'h_read**', d['h_read_starstar'])"); freeze before W2"
commit $W/W1/rho_selection.json $W/W1/analysis.json $W/W1/provenance_*.json $W/W1/figures
if ! python -c "import json,sys;sys.exit(0 if json.load(open('$W/W1/rho_selection.json'))['licensed'] else 1)"; then
  echo "[$(stamp)] STOP = NO_COMPUTE_EFFICIENT_FR_RELAXATION. W2 not launched."; exit 0
fi

echo "[$(stamp)] === W2: confirmatory (seeds 900-915) ==="
mkdir -p $W/W2
python -u scripts/run_wca_targeted_relax.py --stage W2 > $W/W2/run_W2.log 2>&1 || echo "[$(stamp)] W2 runner exited non-zero (shards kept)"
python -u scripts/analyze_wca_targeted_relax.py --stage W2 > $W/W2/analysis.log 2>&1 || echo "[$(stamp)] W2 analysis refused or failed (incomplete block?)"
tail -12 $W/W2/analysis.log
if [ -f $W/W2/analysis.json ]; then
  COMMIT_MSG="WCA targeted-relax W2 closed: $(python -c "import json;d=json.load(open('$W/W2/analysis.json'));print(d['outcome'], 'R_C %.3f' % d['R_C'])")"
  commit $W/W2/analysis.json $W/W2/comparison.csv $W/W2/provenance_*.json $W/W2/figures
fi
echo "[$(stamp)] === overnight sequence finished ==="
