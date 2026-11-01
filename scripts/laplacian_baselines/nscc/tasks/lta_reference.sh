#!/bin/bash
# LTA umbrella/WHAM references at 150 K (protocol validation against the exact-MC reference), 80 K and 350 K with the
# temperature-sweep rule (configs/uniform_campaign/lta_sweep_prereg.json: 40 windows, 256 replicas, kappa = kT/0.07^2,
# n_steps = round(150000 * 300/T), burn_in = round(30000 * 300/T), 100k unbiased check), one A100, then converted to
# the lta_T{T}_reference.npz layout (scripts/laplacian_baselines/convert_lta_reference.py).  Idempotent.
set -uo pipefail
cd "$LB_ROOT"
REF=results/laplacian_baselines/references/lta_sweep
mkdir -p "$REF"
rc_all=0
for T in 150 80 350; do
  if [ ! -f "$REF/reference_T$T.npz" ]; then
    kappa=$(python -c "print(0.008314462618*$T/0.07**2)")
    nsteps=$(python -c "print(round(150000*300/$T))")
    burn=$(python -c "print(round(30000*300/$T))")
    echo "== T=$T K: kappa $kappa, n_steps $nsteps, burn_in $burn =="
    python -u scripts/run_lta_reference.py --temperature "$T" --kappa "$kappa" --n-steps "$nsteps" --burn-in "$burn" \
        --n-rep 256 --unbiased-steps 100000 --out "$REF"
    rc=$?; [ $rc -ne 0 ] && { echo "run_lta_reference T=$T failed rc=$rc"; rc_all=1; continue; }
  fi
  cmp=""
  [ "$T" = 150 ] && cmp="--compare results/equal_budget_v2/references/lta_T150_reference.npz"
  # the converter imports lta_ladder_numba (numba): the deca26 torch env has no numba, so use the CPU env's python
  "${LB_CPU_PY:-/scratch/users/nus/e1297740/envs/ltafl/bin/python}" scripts/laplacian_baselines/convert_lta_reference.py \
      "$REF/reference_T$T.npz" "results/laplacian_baselines/references/lta_T${T}_reference.npz" $cmp || rc_all=1
done
echo LTA_REFERENCES_DONE rc=$rc_all
exit $rc_all
