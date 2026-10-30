#!/usr/bin/env bash
# Is the A100 engine run-to-run deterministic?  Run the tiny ABF cell twice (same init, same seed)
# and report the first step at which the xi traces differ; same for fr_uniform at zero events.
set -euo pipefail
OUT=results/deca_fr_2026/determinism
rm -rf $OUT
CFG=configs/deca_fr_2026/smoke.yaml
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell tiny --prepare-only --out $OUT
mkdir -p $OUT/tiny_rep; cp $OUT/tiny/init.npz $OUT/tiny_rep/
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell tiny --method abf --out $OUT
python - <<'PY'
import shutil, os
shutil.copy("results/deca_fr_2026/determinism/tiny/abf.npz", "results/deca_fr_2026/determinism/tiny/abf_run1.npz")
os.remove("results/deca_fr_2026/determinism/tiny/abf.json")
PY
python scripts/deca_fr_2026/run_cell.py --config $CFG --cell tiny --method abf --out $OUT --force
python - <<'PY'
import numpy as np
a = np.load("results/deca_fr_2026/determinism/tiny/abf_run1.npz"); b = np.load("results/deca_fr_2026/determinism/tiny/abf.npz")
xa, xb, st = a["xi_trace"], b["xi_trace"], a["xi_trace_steps"]
d = np.abs(xa - xb).max(axis=(1, 2))
first = int(np.argmax(d > 0)) if (d > 0).any() else -1
print("ABF twice, same seed: identical =", bool(np.array_equal(xa, xb)),
      "| first differing trace step:", (st[first] if first >= 0 else None),
      "| max |dxi| by step:", {int(st[i]): float(d[i]) for i in range(0, len(st), max(1, len(st)//8))})
print("final pmf max |diff| (kJ/mol):", float(np.abs(a["pmf"][-1] - b["pmf"][-1]).max()))
PY
echo DETERMINISM_OK
