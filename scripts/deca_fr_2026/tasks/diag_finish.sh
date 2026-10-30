#!/usr/bin/env bash
# Analysis + figures of the diagnostic init cells (CPU job).
set -euo pipefail
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/diagnostics.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/diagnostics.yaml
echo DIAG_FINISH_OK
# also against the corrected reference (2026-10-31)
REF2=results/deca_fr_2026/reference_v2/reference.npz
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/diagnostics.yaml --reference $REF2 --tag v2
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/diagnostics.yaml --reference $REF2 --tag v2
python scripts/deca_fr_2026/diag_extended.py --reference $REF2 --out results/deca_fr_2026/diagnostics/analysis_v2
echo DIAG_FINISH_V2_OK
