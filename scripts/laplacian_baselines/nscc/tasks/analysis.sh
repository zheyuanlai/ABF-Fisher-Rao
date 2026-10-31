#!/bin/bash
# Analyses of the LRS Laplacian baseline on NSCC (CPU, a few minutes):
#  1. reproducibility of the ABF / FR re-runs against the archive's tracked analyses
#  2. three-arm paired analysis (analyze.py, eqb_metrics scorers) with the re-run ABF / FR arms
#  3. the archive's R2 sustained time-to-accuracy metric, Laplacian vs ABF (analyze_r2.sh, unchanged analyzer)
set -uo pipefail
cd "$LB_ROOT"
REF=results/laplacian_baselines/references
ITE300=$REF/ite300; ITE150=$REF/ite150
# the frozen LTA analyzer reads lta_T{T}_reference.npz from results/equal_budget_v2/references/: supply the rebuilt 80 / 350 K files there
for T in 80 350; do
  if [ -f "$REF/lta_T${T}_reference.npz" ] && [ ! -f "results/equal_budget_v2/references/lta_T${T}_reference.npz" ]; then
    cp -v "$REF/lta_T${T}_reference.npz" results/equal_budget_v2/references/
  fi
done
echo "== 1. reproducibility of the re-runs =="
python scripts/laplacian_baselines/reproducibility.py --ite300-refbase "$ITE300" --ite150-refbase "$ITE150"
echo "== 2. paired three-arm analysis =="
python scripts/laplacian_baselines/analyze.py --case all --arms-root results/laplacian_baselines/raw \
    --ite300-refbase "$ITE300" --ite150-refbase "$ITE150"
echo "== 3. archived R2 sustained time-to-accuracy metric, Laplacian vs ABF =="
LB_ABF_ROOT=results/laplacian_baselines/raw LB_ITE300_REFBASE=$ITE300 LB_ITE150_REFBASE=$ITE150 PY=python \
    bash scripts/laplacian_baselines/analyze_r2.sh
echo "== 4. summary tables =="
python scripts/laplacian_baselines/summarize.py
echo ANALYSIS_DONE
