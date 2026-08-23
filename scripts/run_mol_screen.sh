#!/usr/bin/env bash
# Screening stage in ONE process, then the hexane reference.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/screen_spec.json
python scripts/mol_reference.py --system HEX --B 131072 --steps 2000000 --nb 180 --blocks 8 --joint-nb 60
echo SCREEN_DONE
