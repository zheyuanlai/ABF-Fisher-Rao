#!/usr/bin/env bash
# Re-score the six LTA/ITE Laplacian positive cells against their archived ABF partners
# using the EXACT unchanged R2 analysis (geometric sustained time-to-accuracy speedup).
# The analyzer's internal label "fr" means "second arm" here: that arm is LAPLACIAN.
# Run only after all required simulations and references exist locally.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=${PY:-python}
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export ABFFR_DATA_ROOT=${ABFFR_DATA_ROOT:-"$ROOT"}
OUT=results/laplacian_baselines/analysis
mkdir -p "$OUT"
# NSCC (docs/LRS_LAPLACIAN_BASELINE.md "NSCC execution"): the archived raw files are absent, so the ABF arm comes from
# the re-runs (LB_ABF_ROOT) and the ITE references from the rebuilt bases; the 80 / 350 K LTA references must sit in
# results/equal_budget_v2/references/ (the analyzer's fixed path; copied there from results/laplacian_baselines/references/).
OLD=${LB_ABF_ROOT:-results/r2_campaign/raw}
NEW=results/laplacian_baselines/raw
ITE=${LB_ITE300_REFBASE:-results/overnight/ite/reference}
ITE150=${LB_ITE150_REFBASE:-results/overnight/ite150/reference}
COMMON="--allow-diff laplacian_scale"

# NOTE: --fr-glob is the original parser's positional name, not the method.
$PY scripts/adaptive_fr/analyze_online.py --block LRS_R2C1 --engine eri \
  --abf-dir "$OLD/R2C1_ite300_K1024_R2" --dir "$NEW/R2C1_ite300_K1024_R2" \
  --fr-glob 'laplacian_N1024_s*.npz' --K 1024 \
  --refbase "$ITE" --variant ite_itq3 --ladder kT --expect-seeds 1400000:16 \
  $COMMON --out "$OUT/R2C1_sgeo_lap_vs_abf.json"

$PY scripts/adaptive_fr/analyze_online.py --block LRS_R2C2 --engine eri \
  --abf-dir "$OLD/R2C2_ite150_K256_R2" --dir "$NEW/R2C2_ite150_K256_R2" \
  --fr-glob 'laplacian_N256_s*.npz' --K 256 \
  --refbase "$ITE150" --variant ite_itq3 --ladder abf --expect-seeds 1410000:16 \
  $COMMON --out "$OUT/R2C2_sgeo_lap_vs_abf.json"

for tup in 'R2C3 150 1420000 R2C3_lta150_K1024_R2' \
           'R2C4 300 1430000 R2C4_lta300_K1024_R2' \
           'R2H80 80 1450000 R2H80_lta80_K1024_R2' \
           'R2H350 350 1460000 R2H350_lta350_K1024_R2'; do
  read -r b temp first tag <<< "$tup"
  $PY scripts/adaptive_fr/analyze_online.py --block "LRS_$b" --engine lta \
    --abf-dir "$OLD/$tag" --dir "$NEW/$tag" \
    --fr-glob 'laplacian_N1024_s*.npz' --K 1024 --T-K "$temp" \
    --ladder abf --expect-seeds "${first}:16" \
    $COMMON --out "$OUT/${b}_sgeo_lap_vs_abf.json"
done
echo "All six Laplacian-vs-ABF original R2 sustained-speedup reports saved to $OUT"
