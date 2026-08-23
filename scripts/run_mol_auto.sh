#!/usr/bin/env bash
# Automatic switch: thresholds calibrated on PENTANE only, then frozen and applied
# to alanine with no retuning.  It does not have to beat the best hindsight-chosen
# switch; it has to come close without knowing the answer in advance.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/auto_spec.json
echo AUTO_DONE
