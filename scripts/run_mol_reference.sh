#!/usr/bin/env bash
# Reference free energies: unbiased Brownian dynamics, ~2.2e11 force evaluations each.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_reference.py --system BUT --B 131072 --steps 2000000 --nb 180 --blocks 8
python scripts/mol_reference.py --system PEN --B 131072 --steps 2000000 --nb 180 --blocks 8
echo REFERENCE_DONE
