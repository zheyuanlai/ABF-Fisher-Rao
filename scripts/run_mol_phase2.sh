#!/usr/bin/env bash
# Everything after the first screen, chained: MH-arm screen -> freeze -> 32-seed
# confirmation -> transport-rate sweep -> ablations -> hexane.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/screen2_spec.json
python scripts/mol_freeze.py | tee results/mol/frozen_table.md
python scripts/mol_run_many.py --spec results/mol/confirm_spec.json
python scripts/mol_run_many.py --spec results/mol/kappa_spec.json
python scripts/mol_run_many.py --spec results/mol/ablation_spec.json
python scripts/mol_fiber_time.py --system PEN --B 32768 --steps 400000 --every 5000
python scripts/mol_fiber_time.py --system HEX --B 32768 --steps 400000 --every 5000
python scripts/mol_mode_diagnostic.py --system PEN
python scripts/mol_mode_diagnostic.py --system HEX
python scripts/mol_run_many.py --spec results/mol/hexane_spec.json
echo PHASE2_DONE
