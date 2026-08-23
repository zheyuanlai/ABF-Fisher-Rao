#!/usr/bin/env bash
# S_k tau_k^2 validation across chain length: heptane offers three candidate
# hidden torsions at increasing distance from z, so the diagnostic can be checked
# against measured promotion gains over three contrasts instead of one.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_fiber_time.py --system HEP --B 32768 --steps 400000 --every 5000
python scripts/mol_mode_diagnostic.py --system HEP
python scripts/mol_run_many.py --spec results/mol/hep_spec.json
echo HEP_DONE
