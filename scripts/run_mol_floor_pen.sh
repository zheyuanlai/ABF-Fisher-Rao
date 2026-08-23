#!/usr/bin/env bash
# The same floor decomposition as butane, on the system that has a SLOW TORSION
# IN THE FIBER.  Butane's fiber is bonds and angles; pentane's carries phi2 with
# tau_y = 1.3e5 steps, and M3's warm-TI ceiling parks at 0.0084 where butane's
# reached 0.0036, so the extra term is plausibly the projection acting on the
# fiber's slow direction rather than on its stiff ones.  Warm init draws phi2
# from the reference conditional, so this measures maintenance, not relaxation.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
python scripts/mol_floor_study.py --system PEN --warm \
  --hs 2e-3,1e-3,5e-4,2.5e-4 --bws 0.08,0.04,0.02 --ngrid 257 \
  --N 1024 --rows 8 --time 400
echo FLOOR_PEN_DONE
