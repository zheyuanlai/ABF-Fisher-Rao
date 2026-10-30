#!/usr/bin/env bash
# After the calibration arms: share the ABF arm with every rate cell (bit-identical by construction:
# same init.npz, same rng_seed, the abf method never reads fr_rate), then the frozen analysis + figures.
set -euo pipefail
OUT=results/deca_fr_2026/fr_calibration
python - <<'PY'
# The init files were built per cell on the GPU; the engine is not run-to-run bit-deterministic, so
# report how far the populations are apart (the npz zip timestamps differ, so cmp is useless).
import numpy as np
a = np.load("results/deca_fr_2026/fr_calibration/rate005/init.npz")["positions"]
for c in ("rate001", "rate002", "rate020", "rate050"):
    b = np.load(f"results/deca_fr_2026/fr_calibration/{c}/init.npz")["positions"]
    print(f"init {c} vs rate005: max |dx| = {np.abs(a - b).max():.3e} nm, identical = {np.array_equal(a, b)}")
PY
for c in rate001 rate002 rate020 rate050; do
  cp $OUT/rate005/abf.npz $OUT/$c/abf.npz; cp $OUT/rate005/abf.json $OUT/$c/abf.json
done
python - <<'PY'
import yaml
p = "configs/deca_fr_2026/fr_calibration.yaml"
cfg = yaml.safe_load(open(p))
for c in ("rate001", "rate002", "rate020", "rate050"):
    cfg["cells"][c]["methods"] = ["abf", "fr_uniform"]
txt = yaml.safe_dump(cfg, sort_keys=False); open(p, "w").write(txt)
PY
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/fr_calibration.yaml
echo CALIB_FINISH_OK
