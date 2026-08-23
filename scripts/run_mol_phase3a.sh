#!/usr/bin/env bash
# Alanine, part A: fiber timescale, conditional library, hyper-parameter screen.
# Independent of the pentane confirmation, so it runs alongside it -- these loops
# are launch-bound and a second process fills the gaps rather than competing.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=3
[ -f results/mol/ALA_fiber_time.npz ] || python scripts/mol_fiber_time.py --system ALA \
  --B 16384 --steps 400000 --every 5000 --z -0.5236 --y0 -0.9 --thr 60
[ -f results/mol/ref/ALA_conflib.npz ] || python scripts/mol_conf_library.py --system ALA \
  --B 16384 --steps 500000 --burn 60000 --every 100 --nz 64 --M 2048
python scripts/mol_mode_diagnostic.py --system ALA || true
python - <<'PY'
import json
base = dict(system="ALA", seeds=8, seed0=1000, N=256, steps=100_000, bw_mf=0.05,
            save_every=5000, z0=-0.5236, tag="screen")
K = "0.075,0.3,1.2"; TH = "0.3"
spec = [{**base, "arm": a, "kappa": K, "theta": TH}
        for a in ["wfr_rot", "wfr_shake", "wfr_ymap", "wfr_yref", "wfr_ymh"]]
spec += [{**base, "arm": "wfr_lmh", "kappa": K, "theta": TH,
          "decay": "0.997,0.999", "lift_bw_z": 0.25, "tag": "screen_bz0.25"}]
spec += [{**base, "arm": a, "n_windows": w, "tag": f"screen_w{w}"}
         for w in [64, 256] for a in ["ti_cold", "ti_warm"]]
spec += [{**base, "arm": "abf", "abf_nmin": float(n), "tag": f"screen_n{n}"}
         for n in [50, 200, 800]]
spec += [{**base, "arm": "wfr_qref", "kappa": K, "theta": TH}]
json.dump(spec, open("results/mol/ala_screen_spec.json", "w"), indent=1)
print(len(spec), "ALA screen runs")
PY
python scripts/mol_run_many.py --spec results/mol/ala_screen_spec.json
echo PHASE3A_DONE
