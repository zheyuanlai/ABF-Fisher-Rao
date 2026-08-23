#!/usr/bin/env bash
# The complete-coordinate control: alanine with z = (phi, psi), so the hidden
# torsion is no longer hidden.  Whatever advantage survives cannot be
# "the method repairs an incomplete reaction coordinate".
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
[ -f results/mol/ref/ALA2D_tiref.npz ] || \
  python scripts/mol_ti_reference2d.py --N 4096 --rows 4 --steps 400000 --n-eq 60000
python scripts/mol_campaign2d.py --spec results/mol/ala2d_spec.json
echo ALA2D_DONE
