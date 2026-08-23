#!/usr/bin/env bash
# The decisive next experiment: is RC-WFR's residual error carried in its DEPOSITS
# (removable by switching transport off and re-estimating) or in the sampler
# itself (not removable)?  Each run carries two accumulators, so one job reports
# both the keep-everything and the post-switch-only estimator.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/switch_spec.json
echo SWITCH_DONE
