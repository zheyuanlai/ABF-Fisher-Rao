#!/usr/bin/env bash
# M3: with the numerical floor measured and lowered (h 2e-3 -> 1e-3, b_mf 0.05
# -> 0.02, grid 129 -> 257), do the three pentane arms still separate?  The old
# comparison bottomed out at ~0.020 for every constrained arm; that number is now
# known to be estimator plus integrator, and the floor here should be ~0.005.
#
# Three arms only: warm stratified TI (the practical ceiling), persistent RC-WFR
# with the learned Metropolis lift, and cold stratified TI (no transport).
# Run concurrently -- the loop is launch-bound, so three processes on one GPU
# finish in well under three times one process.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
for arm in ti_warm ti_cold wfr_lmh; do
  python scripts/mol_run_many.py --spec results/mol/M3_${arm}_spec.json \
    > results/logs/mol_M3_${arm}.log 2>&1 &
done
wait
echo M3_DONE
