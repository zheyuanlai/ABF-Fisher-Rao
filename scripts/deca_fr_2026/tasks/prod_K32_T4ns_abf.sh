#!/usr/bin/env bash
# Stage IV production, cell K32_T4ns, arm abf (prepare is idempotent; both arms share init.npz).
set -euo pipefail
CFG=configs/deca_fr_2026/production.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell K32_T4ns --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell K32_T4ns --method abf
echo PROD_K32_T4ns_abf_OK
