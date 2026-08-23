#!/usr/bin/env bash
# Is the ~0.008 that M3's arms share the time step, or not?
#
# The first pentane sweep ran at physical time 400, where the statistical floor
# is 0.016 -- it resolved the h=2e-3 bias and nothing below it.  This runs only
# the two h values that matter, at four times the physical time and four times
# the rows, putting the self-difference's noise floor near 0.002.  If
# ||F(1e-3) - F(5e-4)|| comes back under that, the residual is not the
# integrator, and the density-dependence of the kernel becomes the candidate.
# Three bandwidths ride along free, since they share the trajectory.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_floor_study.py --system PEN --warm \
  --hs 1e-3,5e-4 --bws 0.04,0.02,0.01 --ngrid 257 \
  --N 1024 --rows 32 --time 1600 --out results/mol/floor2
echo FLOOR_PEN2_DONE
