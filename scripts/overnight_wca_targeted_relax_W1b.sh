#!/usr/bin/env bash
# Amendment A2: W1b = the W1 ladder with the TI-scheme (projected) inner step, paired against the existing W1 abf/fr_uniform runs.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:scripts
W=results/targeted_relax_campaign/wca
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
echo "[$(stamp)] === W1b: projected-scheme ladder (seeds 820-823) ==="
python -u scripts/run_wca_targeted_relax.py --stage W1 --scheme projected > $W/W1/run_W1b.log 2>&1 || { echo "[$(stamp)] W1b FAILED"; exit 1; }
python -u scripts/analyze_wca_targeted_relax.py --stage W1 --scheme projected > $W/W1/analysis_projected.log 2>&1 || { echo "[$(stamp)] W1b analysis FAILED"; exit 1; }
tail -8 $W/W1/analysis_projected.log
git add $W/W1/rho_selection_projected.json $W/W1/analysis_projected.json $W/W1/provenance_*_projected.json $W/W1/figures_projected >/dev/null 2>&1
git commit -q -m "WCA targeted-relax W1b (amendment A2, projected scheme) closed: $(python -c "import json;d=json.load(open('$W/W1/rho_selection_projected.json'));print('rho*', d['rho_star'], 'licensed', d['licensed'], 'h_read**', d['h_read_starstar'])")

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && echo "[$(stamp)] committed: $(git log --oneline -1)"
echo "[$(stamp)] === W1b finished ==="
