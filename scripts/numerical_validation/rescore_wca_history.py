#!/usr/bin/env python
"""Rescore historical WCA ABF vs ABF+FR runs against the EXACT Gibbs reference (Metropolis MC).

    python scripts/numerical_validation/rescore_wca_history.py

The published scores (v2 TI reference; dt-consistent pooled-ABF references) are NOT modified; this writes
results/numerical_validation/wca/rescore_history.json next to them.  Sets:
  accepted_f64   results/wca_replica_ladder/validation_f64  (float64 numba replay of the accepted histogram
                 confirmation: N 1024, dt 0.002, seeds 3100-3115; equal to it within its own noise)
  ladder_dt0.002 results/wca_replica_ladder/production       (N 1024 ... 4)
  ladder_dt0.0005 results/wca_replica_ladder/production_dt0.0005
Read-out: the accepted scorer (wca_numba.score_ladder_result), with the reference replaced.
"""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "numerical_validation"))
import analyze_wca_validation as A  # noqa: E402
import wca_numba as wn  # noqa: E402

LAD = os.path.join(ROOT, "results", "wca_replica_ladder")
OUT = os.path.join(ROOT, "results", "numerical_validation", "wca")
ABF, FR = "hist_abf", "hist_fr_uniform"


def mc_reference():
    """The MC profiles in the scorer's format: free energy and mean force on linspace(-0.2, 1.2, 160)."""
    mc = A.load_group("mc", "mc")
    C = sum(r["C"] for r in mc)
    M = sum(r["M"] for r in mc)
    F = A.F_density(C)
    mf = M / np.maximum(C, 1e-300)
    grid = np.linspace(A.ZMIN, A.ZMAX, A.NB)
    return dict(label=f"exact Gibbs (Metropolis MC, {len(mc)} chains): F_density; mean force = MC bin means",
                grid=grid, free_energy=np.interp(grid, A.CEN, F), mean_force=np.interp(grid, A.CEN, mf),
                z_ti=grid), len(mc)


def boot_median(x, seed=20261009, n=10000):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = np.median(x[rng.integers(0, len(x), (n, len(x)))], 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def score_set(pattern, ref, arms=(ABF, FR)):
    files = sorted(glob.glob(os.path.join(LAD, pattern)))
    if not files:
        return None
    rows = []
    for f in files:
        res = wn.load_result(f)
        row = {}
        for arm in arms:
            if arm not in list(res["arms"]):
                continue
            sc = wn.score_ladder_result(res, arm, reference=ref)
            row[arm] = dict(I_F=float(sc["integrated_l2_f"]), Ibar_u=float(sc["mean_l2_f_u"]), e_F=float(sc["l2_f"]))
        rows.append(row)
    out = {}
    for arm in arms:
        if arm in rows[0]:
            out[arm] = {k: float(np.median([r[arm][k] for r in rows])) for k in ("I_F", "Ibar_u", "e_F")}
    if FR in rows[0] and ABF in rows[0]:
        for k in ("I_F", "e_F"):
            d = [100 * (r[FR][k] / r[ABF][k] - 1) for r in rows]
            out[f"FR_vs_ABF_{k}"] = dict(median=float(np.median(d)), ci95=boot_median(d), wins=int(np.sum(np.array(d) < 0)), n=len(d))
    return out


def main():
    ref_mc, nmc = mc_reference()
    ref_v2 = wn.load_reference()
    res = dict(reference_mc=ref_mc["label"], sets={})
    sets = [("accepted_f64 (N 1024, dt 0.002)", "validation_f64/raw/N1024_s*.npz")]
    sets += [(f"ladder dt 0.002 N {N}", f"production/raw/N{N}_s*.npz") for N in (1024, 256, 64, 16, 4)]
    sets += [(f"ladder dt 0.0005 N {N}", f"production_dt0.0005/raw/N{N}_s*.npz") for N in (1024, 256, 64, 16, 4)]
    for name, pat in sets:
        r_mc = score_set(pat, ref_mc)
        if r_mc is None:
            continue
        r_v2 = score_set(pat, ref_v2)
        res["sets"][name] = {"MC reference": r_mc, "v2 TI reference (published)": r_v2}
        for lab, r in (("v2 ", r_v2), ("MC ", r_mc)):
            if "FR_vs_ABF_I_F" in r:
                a, b = r["FR_vs_ABF_I_F"], r["FR_vs_ABF_e_F"]
                print(f"{name:34s} {lab}: ABF e_F {r[ABF]['e_F']:.4f} FR e_F {r[FR]['e_F']:.4f} | dI_F {a['median']:+.1f}% "
                      f"[{a['ci95'][0]:+.1f},{a['ci95'][1]:+.1f}] ({a['wins']}/{a['n']}) | de_F {b['median']:+.1f}% "
                      f"[{b['ci95'][0]:+.1f},{b['ci95'][1]:+.1f}] ({b['wins']}/{b['n']})", flush=True)
    json.dump(res, open(os.path.join(OUT, "rescore_history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
