#!/usr/bin/env bash
# Does the switched arm actually cross BELOW the persistent one, or only look
# like it will?  Against production budget the switched arms converge at -0.28
# while persistent is parked at -0.044, and they are currently level.  This runs
# both to 8.6e8 force evaluations so the crossing is measured, not extrapolated.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_run_many.py --spec results/mol/xlong_spec.json
echo XLONG_DONE
