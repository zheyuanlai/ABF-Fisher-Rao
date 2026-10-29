#!/usr/bin/env bash
# Stage I-B: timestep sensitivity, K=64, 200 ps, 8 seeds, dt 1 fs vs 0.5 fs.
set -euo pipefail
CFG=configs/deca_fr_2026/validation.yaml
for cell in dt1fs_K64 dt05fs_K64; do
  python scripts/deca_fr_2026/run_cell.py --config $CFG --cell $cell --prepare-only
  python scripts/deca_fr_2026/run_cell.py --config $CFG --cell $cell --method abf
done
echo VALIDATION_DT_OK
