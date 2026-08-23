#!/usr/bin/env bash
# Stage B's only dial: z-resolution against replicas per window, at fixed M*R.
# Pentane recovers -0.484 after the snap with one replica per window; alanine
# reaches -0.220, and the visible reason is that one trajectory cannot explore a
# 60-coordinate fiber by time-averaging however long it runs.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/ala_MR_spec.json
echo MR_DONE
