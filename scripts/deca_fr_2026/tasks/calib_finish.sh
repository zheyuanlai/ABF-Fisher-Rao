#!/usr/bin/env bash
# After the three calibration tasks: share the ABF arm with the other rate cells (bit-identical by
# construction: same init.npz, same rng_seed, the abf method never reads fr_rate), then analyse.
set -euo pipefail
OUT=results/deca_fr_2026/fr_calibration
cmp $OUT/rate005/init.npz $OUT/rate020/init.npz && cmp $OUT/rate005/init.npz $OUT/rate050/init.npz
for c in rate020 rate050; do cp $OUT/rate005/abf.npz $OUT/$c/abf.npz; cp $OUT/rate005/abf.json $OUT/$c/abf.json; done
python - <<'PY'
import yaml
p = "configs/deca_fr_2026/fr_calibration.yaml"
cfg = yaml.safe_load(open(p))
for c in ("rate020", "rate050"):
    cfg["cells"][c]["methods"] = ["abf", "fr_uniform"]
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
PY
python scripts/deca_fr_2026/analyze.py --config configs/deca_fr_2026/fr_calibration.yaml
python scripts/deca_fr_2026/plot.py --config configs/deca_fr_2026/fr_calibration.yaml
echo CALIB_FINISH_OK
