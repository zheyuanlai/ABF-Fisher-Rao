#!/usr/bin/env bash
# Frozen analysis of the R2 campaign cells (docs/r2_campaign/PREREGISTRATION.md section 2): calls the UNCHANGED
# scripts/adaptive_fr/analyze_online.py with each cell's parameters (section 3), then the frozen verdict rules.
# No simulation; under a minute on one CPU.  Run from the worktree after the cells have finished.
#   bash scripts/r2_campaign/analyze_all.sh [block ...]      (default: every block whose 32 run files exist)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PY=${PY:-/home/zheyuanlai/miniconda3/envs/abffr/bin/python}
export ABFFR_DATA_ROOT=${ABFFR_DATA_ROOT:-/home/zheyuanlai/ABF-Fisher-Rao}
export NUMBA_NUM_THREADS=1
AN=results/r2_campaign/analysis
D=results/r2_campaign/raw
ITE=results/overnight/ite/reference
ITE150=results/overnight/ite150/reference
ERI=results/eri_campaign/reference
mkdir -p "$AN"

declare -A CMD
CMD[R2C1]="--block R2C1 --engine eri --dir $D/R2C1_ite300_K1024_R2 --K 1024 --refbase $ITE --variant ite_itq3 --ladder kT --expect-seeds 1400000:16"
CMD[R2C2]="--block R2C2 --engine eri --dir $D/R2C2_ite150_K256_R2 --K 256 --refbase $ITE150 --variant ite_itq3 --ladder abf --expect-seeds 1410000:16"
CMD[R2C5]="--block R2C5 --engine eri --dir $D/R2C5_eri600_K1024_R2 --K 1024 --refbase $ERI --variant gard_tait --ladder kT --expect-seeds 1440000:16"
CMD[R2N256]="--block R2N256 --engine eri --dir $D/R2N256_eri600_K256_R2 --K 256 --refbase $ERI --variant gard_tait --ladder kT --expect-seeds 1470000:16"
CMD[R2C3]="--block R2C3 --engine lta --dir $D/R2C3_lta150_K1024_R2 --K 1024 --T-K 150 --ladder abf --expect-seeds 1420000:16"
CMD[R2C4]="--block R2C4 --engine lta --dir $D/R2C4_lta300_K1024_R2 --K 1024 --T-K 300 --ladder abf --expect-seeds 1430000:16"
CMD[R2H80]="--block R2H80 --engine lta --dir $D/R2H80_lta80_K1024_R2 --K 1024 --T-K 80 --ladder abf --expect-seeds 1450000:16"
CMD[R2H350]="--block R2H350 --engine lta --dir $D/R2H350_lta350_K1024_R2 --K 1024 --T-K 350 --ladder abf --expect-seeds 1460000:16"
declare -A DIR
DIR[R2C1]=R2C1_ite300_K1024_R2; DIR[R2C2]=R2C2_ite150_K256_R2; DIR[R2C5]=R2C5_eri600_K1024_R2; DIR[R2N256]=R2N256_eri600_K256_R2
DIR[R2C3]=R2C3_lta150_K1024_R2; DIR[R2C4]=R2C4_lta300_K1024_R2; DIR[R2H80]=R2H80_lta80_K1024_R2; DIR[R2H350]=R2H350_lta350_K1024_R2

BLOCKS=("$@")
if [ ${#BLOCKS[@]} -eq 0 ]; then BLOCKS=(R2C3 R2C4 R2H80 R2H350 R2N256 R2C1 R2C2 R2C5); fi
for b in "${BLOCKS[@]}"; do
  n=$(ls "$D/${DIR[$b]}"/*_s*.npz 2>/dev/null | grep -v ckpt | wc -l || true)
  if [ "$n" -ne 32 ]; then echo "[$b] $n/32 run files: skipped"; continue; fi
  # shellcheck disable=SC2086
  $PY scripts/adaptive_fr/analyze_online.py ${CMD[$b]} --out "$AN/$b.json"
done
$PY scripts/r2_campaign/verdicts.py
