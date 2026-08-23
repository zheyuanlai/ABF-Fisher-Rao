#!/usr/bin/env bash
# Does the extrapolated crossover with the unbiased baselines actually happen?
# The confirmation's fitted rates put it at ~4e8 force evaluations against ABF;
# this runs 4.3e8 and measures it instead of extrapolating.  Plus the alanine
# transport-rate sweep.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/long_spec.json
python scripts/mol_run_many.py --spec results/mol/ala_kappa_spec.json
echo LONG_DONE
