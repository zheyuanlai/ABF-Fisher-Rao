#!/usr/bin/env bash
# Stage I-B (direct): <f_loc | R> in 25 stiff windows x 48 replicas at dt = 1 fs and 0.5 fs, same seeds.
set -euo pipefail
python scripts/deca_fr_2026/dt_windows.py --dt 0.001  --out results/deca_fr_2026/validation/dt_windows_1fs
python scripts/deca_fr_2026/dt_windows.py --dt 0.0005 --out results/deca_fr_2026/validation/dt_windows_05fs
echo DT_WINDOWS_OK
