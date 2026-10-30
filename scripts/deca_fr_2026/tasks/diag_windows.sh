#!/usr/bin/env bash
# Post-protocol diagnostic: stiff windows on the extended region (10 centres 2.6-3.5 nm x 48
# replicas, k = 8000, 20 ps pull + 100 ps equil + 600 ps prod, 100 ps blocks) from the umbrella's
# diverse preset pool vs. from a PPII-only pool.  Same seeds.  ~40 min each on one A100.
set -euo pipefail
for init in pool ppii; do
  python scripts/deca_fr_2026/dt_windows.py --dt 0.001 --init $init --win-lo 2.6 --win-hi 3.5 --n-windows 10 \
      --n-rep 48 --prod-ps 600 --block-ps 100 --out results/deca_fr_2026/diagnostics/windows_$init
done
echo DIAG_WINDOWS_OK
