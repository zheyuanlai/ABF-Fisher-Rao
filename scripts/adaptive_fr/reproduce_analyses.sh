#!/usr/bin/env bash
# Re-run every analysis of the adaptive-FR campaign from the saved run files (no simulation; under a minute, one CPU).
# Overwrites results/adaptive_fr_campaign/{analysis,mechanism}/*.json and figures/adaptive_fr/*; run from any checkout
# of research/adaptive-fr-equal-budget after `bash scripts/convergence_fr/link_data.sh` (worktrees only).
#   bash scripts/adaptive_fr/reproduce_analyses.sh
# Not re-run here: the wall-clock benchmark (a measurement; scripts/adaptive_fr/wallclock_bench.py) and the blind
# calibrations (scripts/overnight/fr_calibrate.py results/adaptive_fr_campaign/Ecal_K{64,16} --K {64,16}).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=${PY:-/home/zheyuanlai/miniconda3/envs/abffr/bin/python}
export ABFFR_DATA_ROOT=${ABFFR_DATA_ROOT:-/home/zheyuanlai/ABF-Fisher-Rao}
export NUMBA_NUM_THREADS=1
AN=results/adaptive_fr_campaign/analysis
D=results/adaptive_fr_campaign
ITE=results/overnight/ite/reference
ITE150=results/overnight/ite150/reference
ERI=results/eri_campaign/reference

# Experiment A (fixed stop 10, paired with the historical EB ABF arm) + its four-configuration equal-budget analysis
$PY scripts/convergence_fr/analyze_block.py --block A --dir $D/A_ite300_K256_T200_stop10 \
    --abf-dir results/convergence_fr_campaign/EB_ite300_abf_K256_T200 --allow-diff fr_rate --K 256 --refbase $ITE \
    --variant ite_itq3 --ladder kT --expect-seeds 1140000:16 --out $AN/A.json
$PY scripts/adaptive_fr/equal_budget4.py

# Experiment C (frozen online rule R1), five cells
$PY scripts/adaptive_fr/analyze_online.py --block C1 --engine eri --dir $D/C1_ite300_K1024_R1 --K 1024 --refbase $ITE \
    --variant ite_itq3 --ladder kT --expect-seeds 1300000:16
$PY scripts/adaptive_fr/analyze_online.py --block C2 --engine eri --dir $D/C2_ite150_K256_R1 --K 256 --refbase $ITE150 \
    --variant ite_itq3 --ladder abf --expect-seeds 1310000:16
$PY scripts/adaptive_fr/analyze_online.py --block C3 --engine lta --dir $D/C3_lta150_K1024_R1 --K 1024 --T-K 150 \
    --ladder abf --expect-seeds 1320000:16
$PY scripts/adaptive_fr/analyze_online.py --block C4 --engine lta --dir $D/C4_lta300_K1024_R1 --K 1024 --T-K 300 \
    --ladder abf --expect-seeds 1330000:16
$PY scripts/adaptive_fr/analyze_online.py --block C5 --engine eri --dir $D/C5_eri600_K1024_R1 --K 1024 --refbase $ERI \
    --variant gard_tait --ladder kT --expect-seeds 1340000:16

# Experiment E (efficiency): R1 at K256 x T200 (paired with EB), frontier cells K64 and K16, frontier, wall-clock
$PY scripts/adaptive_fr/analyze_online.py --block EA_R1 --engine eri --dir $D/EA_ite300_K256_T200_R1 \
    --abf-dir results/convergence_fr_campaign/EB_ite300_abf_K256_T200 --allow-diff fr_rate --K 256 --refbase $ITE \
    --variant ite_itq3 --ladder kT --expect-seeds 1140000:16
$PY scripts/adaptive_fr/analyze_online.py --block EB64 --engine eri --dir $D/EB64_ite300_K64_T800_R1 --K 64 --refbase $ITE \
    --variant ite_itq3 --ladder kT --expect-seeds 1350000:16
$PY scripts/adaptive_fr/analyze_online.py --block EB16 --engine eri --dir $D/EB16_ite300_K16_T3200_R1 --K 16 --refbase $ITE \
    --variant ite_itq3 --ladder kT --expect-seeds 1370000:16
$PY scripts/adaptive_fr/equal_budget_frontier.py
$PY scripts/adaptive_fr/wallclock_tta.py

# Experiment D (exploratory mechanism)
$PY scripts/adaptive_fr/bias_variance.py
$PY scripts/adaptive_fr/clone_age.py

# Figures 1-8
$PY scripts/adaptive_fr/fig_campaign.py   # figures 1-8 (+ 3b frontier, 7b mean force and genealogy)
