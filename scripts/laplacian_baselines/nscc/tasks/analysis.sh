#!/bin/bash
# Analyses of the LRS Laplacian baseline on NSCC (CPU, a few minutes):
#  1. reproducibility of the ABF / FR re-runs against the archive's tracked analyses
#  2. three-arm paired analysis (analyze.py, eqb_metrics scorers) with the re-run ABF / FR arms
#  3. the archive's R2 sustained time-to-accuracy metric, Laplacian vs ABF (analyze_r2.sh, unchanged analyzer)
set -uo pipefail
cd "$LB_ROOT"
REF=results/laplacian_baselines/references
ITE300=$REF/ite300; ITE150=$REF/ite150
# convert any GPU sweep reference whose lta_T{T}_reference.npz is still missing (the GPU task's own conversion step
# failed in job 26074508: no numba in the torch env), then supply the rebuilt 80 / 350 K files to the frozen LTA analyzer,
# which reads lta_T{T}_reference.npz from results/equal_budget_v2/references/
for T in 150 80 350; do
  if [ -f "$REF/lta_sweep/reference_T$T.npz" ] && [ ! -f "$REF/lta_T${T}_reference.npz" ]; then
    cmp=""; [ "$T" = 150 ] && cmp="--compare results/equal_budget_v2/references/lta_T150_reference.npz"
    python scripts/laplacian_baselines/convert_lta_reference.py "$REF/lta_sweep/reference_T$T.npz" "$REF/lta_T${T}_reference.npz" $cmp
  fi
done
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
echo "== 3b. scale sensitivity on the pilot runs (secondary) =="
python scripts/laplacian_baselines/pilot_sensitivity.py --ite300-refbase "$ITE300" --ite150-refbase "$ITE150"
echo "== 4. summary tables =="
python scripts/laplacian_baselines/summarize.py
echo ANALYSIS_DONE
