#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/switch2_spec.json
echo SWITCH_DONE
