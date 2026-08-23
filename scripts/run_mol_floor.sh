#!/usr/bin/env bash
# M1: separate the ~0.020 plateau into integrator bias, estimator smoothing bias
# and statistical error, on butane, using warm stratified constrained TI only.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_floor_study.py --system BUT --hs 2e-3,1e-3,5e-4,2.5e-4 \
  --bws 0.08,0.04,0.02 --ngrid 257 --N 1024 --rows 8 --time 400
echo FLOOR_DONE
