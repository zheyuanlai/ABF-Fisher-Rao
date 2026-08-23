#!/usr/bin/env bash
# Alanine, part B: freeze on the screening seeds, confirm on 16 fresh ones.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_freeze.py --system ALA --steps 400000 --seeds 16 --seed0 90000 \
  --out results/mol/frozen_ALA.json --spec results/mol/ala_confirm_spec.json \
  | tee results/mol/frozen_ALA_table.md
python scripts/mol_run_many.py --spec results/mol/ala_confirm_spec.json
echo PHASE3_DONE
