#!/usr/bin/env bash
# Stage IV production, cell K128_T2ns, arm abf (prepare is idempotent; both arms share init.npz).
set -euo pipefail
CFG=configs/deca_fr_2026/production.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell K128_T2ns --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell K128_T2ns --method abf
echo PROD_K128_T2ns_abf_OK
