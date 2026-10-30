#!/usr/bin/env bash
# Reference audit (2026-10-31): re-merge the three accepted builds WITHOUT the replicas that started
# from the left-handed-helix preset (pool index 5), which stay kinetically trapped above 2.9 nm with
# a conditional mean force ~10x the other five presets' (and ABF's).  Then re-score every stage
# against the corrected reference into analysis_v2/ and figures_v2/ (the pre-registered v1 files
# stay untouched).
set -euo pipefail
REF2=results/deca_fr_2026/reference_v2
python scripts/deca_fr_2026/merge_reference.py --dir results/deca_fr_2026/reference --out $REF2 --exclude-presets 5
for st in production fr_calibration estimator_pilot validation; do
  python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/$st.yaml --reference $REF2/reference.npz --tag v2
  python scripts/deca_fr_2026/plot.py    --config configs/deca_fr_2026/$st.yaml --reference $REF2/reference.npz --tag v2
done
python scripts/deca_fr_2026/diag_extended.py --reference $REF2/reference.npz --out results/deca_fr_2026/diagnostics/analysis_v2
echo REFERENCE_V2_OK
