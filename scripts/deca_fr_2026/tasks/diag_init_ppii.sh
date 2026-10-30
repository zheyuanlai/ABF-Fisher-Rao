#!/usr/bin/env bash
# Post-protocol diagnostic: ABF from a ppii initial population, K = 64 x 2 ns, 8 seeds (ABF only).
set -euo pipefail
CFG=configs/deca_fr_2026/diagnostics.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell init_ppii_K64 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell init_ppii_K64 --method abf
echo DIAG_INIT_ppii_OK
