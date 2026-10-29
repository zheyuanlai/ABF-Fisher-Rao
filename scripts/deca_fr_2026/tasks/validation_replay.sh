#!/usr/bin/env bash
# Stage I-A: the 2026-08 screen replayed with and without the half-kick defect (reference-free
# metrics: learned span, pinning above 2.80 nm).  ~2 x 30 min on one A100.
set -euo pipefail
CFG=configs/deca_fr_2026/validation.yaml
for cell in replay_legacy replay_fixed; do
  python scripts/deca_fr_2026/run_cell.py --config $CFG --cell $cell --prepare-only
  python scripts/deca_fr_2026/run_cell.py --config $CFG --cell $cell --method abf
done
echo VALIDATION_REPLAY_OK
