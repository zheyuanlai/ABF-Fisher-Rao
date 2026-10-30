#!/usr/bin/env bash
# Post-protocol diagnostic: ABF on the extended region only ([2.6, 3.6] nm) from a restrained PPII
# start, K = 64 x 1 ns, 8 seeds (ABF only).  gdev-sized (~1 h).
set -euo pipefail
CFG=configs/deca_fr_2026/diagnostics.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell ext_ppii_K64 --prepare-only
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell ext_ppii_K64 --method abf
echo DIAG_EXT_PPII_OK
