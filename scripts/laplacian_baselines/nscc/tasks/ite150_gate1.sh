#!/bin/bash
# ITE 150 K exact-MC reference: Gate 1 analysis with the parallel-bootstrap wrapper (scripts/laplacian_baselines/
# gate1_parallel.py).  The serial analysis of the 1024 windows exceeded a 2 h walltime twice (jobs 26074468, 26074906).
# Step 1 re-runs the 300 K Gate 1 analysis with the wrapper into a scratch base (MC windows symlinked) and requires it to be
# bitwise identical to the serial gate1.json / reference_profile.npz already on disk; step 2 runs tasks/ite_reference.sh,
# which skips the finished 300 K cell and the existing 150 K MC windows and now calls the wrapper.  Idempotent.
set -uo pipefail
cd "$LB_ROOT"
REF=results/laplacian_baselines/references
CHK=$REF/gate1check/ite300/ite_itq3
mkdir -p "$CHK"
ln -sfn "$PWD/$REF/ite300/ite_itq3/MC" "$CHK/MC"
echo "== 1. wrapper check on the 300 K cell (serial output: $REF/ite300/ite_itq3) =="
python scripts/laplacian_baselines/gate1_parallel.py --procs "$NPROCS" --variant ite_itq3 --base "$REF/gate1check/ite300" \
  --T 300 --tol 0.03 --tolup 0.045 --tags none --nboot 2000 | tail -n 3
python scripts/laplacian_baselines/gate1_parallel.py --check "$CHK" "$REF/ite300/ite_itq3" || { echo "wrapper check failed; not running the 150 K analysis"; exit 3; }
echo "== 2. 150 K cell =="
bash scripts/laplacian_baselines/nscc/tasks/ite_reference.sh
