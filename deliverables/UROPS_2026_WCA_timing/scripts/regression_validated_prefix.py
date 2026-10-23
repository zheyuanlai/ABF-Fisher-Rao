#!/usr/bin/env python
"""Regression: the extended engine reproduces the VALIDATED Phase-V WCA confirmation BITWISE.

Re-runs seed 5100 of results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240/raw/ with the extended
src/wca_numba.py (stopping disabled) for the first 45 t.u. (360 000 steps: the ABF ramp, the estimator reset at
t = 20, the FR start at t = 40 and five t.u. of FR events) and compares every saved array at t = 0, 5, ..., 45
with the stored validated run:
  arm 'abf'        : job [hist_abf], default streams                  vs s5100_abf.npz    arm hist_abf
  arm 'fr_start40' : job [fr_start40], fr_stream_ids = [0] (child 1)  vs s5100_frsham.npz arm hist_fr_uniform
(the validated FR arm was the first FR arm of its job, i.e. FR stream 0; the sham does not perturb it).

    python regression_validated_prefix.py --arm abf|fr_start40 --out <json>
"""
import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402

wn = CW.engine()
VAL = os.path.join(CW.DATA_ROOT, "results", "numerical_validation", "wca_confirmatory", "dt0.000125_N256_T240", "raw")
KEYS = ("M_bias", "C_bias", "M_prod", "C_prod", "M_rep", "C_rep", "z_hist", "repl_cumulative", "frac_regions",
        "n_c2s", "n_s2c", "n_round_trips", "n_event_opportunities", "n_cap_binding", "ancestor_ess", "max_ancestor_frac",
        "n_unique_ancestor", "ancestor_ess_window")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("abf", "fr_start40"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end", type=float, default=45.0)
    a = ap.parse_args()
    dt = 1.25e-4
    cfg = CW.validated_cfg(fr_start_t=40.0)
    n = int(round(a.t_end / dt))
    save = np.arange(0, n + 1, int(round(5.0 / dt)))
    t0 = time.perf_counter()
    if a.arm == "abf":
        res = wn.run_ladder_point(5100, 256, n, [("hist_abf", False)], cfg=cfg, save_at=save, cap_min=0)
        ref = wn.load_result(os.path.join(VAL, "s5100_abf.npz"))
        i_ref = list(ref["arms"]).index("hist_abf")
    else:
        res = wn.run_ladder_point(5100, 256, n, [("fr_start40", True)], cfg=cfg, save_at=save, cap_min=0,
                                  fr_stream_ids=[0])
        ref = wn.load_result(os.path.join(VAL, "s5100_frsham.npz"))
        i_ref = list(ref["arms"]).index("hist_fr_uniform")
    wall = time.perf_counter() - t0
    k = save.size
    assert np.array_equal(np.asarray(ref["save_at"])[:k], save)
    assert ref["cfg"] == cfg, {x: (ref["cfg"].get(x), cfg.get(x)) for x in set(ref["cfg"]) | set(cfg) if ref["cfg"].get(x) != cfg.get(x)}
    rows = {}
    for key in KEYS:
        x = np.asarray(res[key])[0][:k]
        y = np.asarray(ref[key])[i_ref][:k]
        rows[key] = dict(bitwise=bool(np.array_equal(x, y, equal_nan=True)),
                         max_abs_diff=float(np.nanmax(np.abs(np.asarray(x, float) - np.asarray(y, float)))) if x.size else 0.0)
    out = dict(arm=a.arm, seed=5100, t_end=a.t_end, n_steps=n, saves_compared=save.tolist(), wall_s=wall,
               replacements_at_t_end=int(np.asarray(res["repl_cumulative"])[0, -1]),
               all_bitwise=all(r["bitwise"] for r in rows.values()), arrays=rows, engine=CW.engine_path(),
               engine_sha256=CW.sha256_file(CW.engine_path()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(a.arm, "all bitwise:", out["all_bitwise"], "replacements:", out["replacements_at_t_end"], f"wall {wall:.0f} s")


if __name__ == "__main__":
    main()
