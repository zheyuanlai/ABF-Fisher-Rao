"""Paired two-arm analysis (ABF = arm A, ABF + uniform FR = arm B), overnight PREREGISTRATION section 6.

    python scripts/overnight/analyze_two_arm.py results/overnight/<cell>/prod_<block> --K 1024 \
        --refbase results/overnight/<cell>/reference --variant ite_itq3 --T 300

Primary: per seed dI = Ibar_F(B) / Ibar_F(A) - 1; median, 95 % percentile bootstrap CI over seeds (10 000), wins.
Secondary: final e_F, final e_F', late-deposit e_F (deposits after fr_start), TV(p, uniform) time-average after
fr_start, true translocations, ancestor ESS / max lineage, replacements, force evaluations, wall-clock, pooled
long-time limits (D statistic FR vs ABF and each vs MC).
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt  # noqa: E402
from eri.model import KB  # noqa: E402


def boot_median(x, n=10000, seed=2026):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def per_run(r, Fr, Gr):
    c = r["cfg"]
    t = r["times"]
    e = mt.errors(r, Fr, Gr)
    k0 = int(c["fr_start_steps"])
    i0 = int(np.searchsorted(r["save_steps"], k0))
    Gl = mt.mean_force(r["sv_M"][-1] - r["sv_M"][i0], r["sv_C"][-1] - r["sv_C"][i0])
    late = float(mt.centred_rms(mt.histogram_pmf(Gl) - Fr))
    h = r["sv_h"][i0:]
    p = h / np.maximum(h.sum(1, keepdims=True), 1)
    pc = p.reshape(p.shape[0], 18, -1).sum(2)
    tv = 0.5 * np.abs(pc - 1 / 18).sum(1)
    return dict(Ibar_F=e["Ibar_F"], I_F=e["I_F"], eF_T=e["eF_T"], eG_T=e["eG_T"], late_eF=late,
                tv_mean=float(tv.mean()), transloc=int(r["sv_i"][-1, 1]), repl=int(r["sv_i"][-1, 2]),
                essw_min=float(np.nanmin(r["sv_u"][i0:, 2])), maxw_max=float(np.nanmax(r["sv_u"][i0:, 3])),
                nfe=int(r["n_force_evals"]), wall=float(r["wall_seconds_session"]), eF_curve=e["eF"], t=t,
                pre_fr=r["sv_C"][r["save_steps"] < k0].copy(),     # strictly before: the save AT fr_start follows the first FR opportunity
                C=r["sv_C"][-1], M=r["sv_M"][-1], seed=int(c["seed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--refbase", required=True)
    ap.add_argument("--variant", default="ite_itq3")
    ap.add_argument("--T", type=float, required=True)
    a = ap.parse_args()
    mt.set_reference_base(a.refbase)
    Fr = Gr = None
    arms = {}
    for method in ("abf", "fr"):
        for f in sorted(glob.glob(os.path.join(ROOT, a.dir, f"{method}_N{a.K}_s*.npz"))):
            if f.endswith((".ckpt.npz", ".tmp.npz")):
                continue
            s = int(re.search(r"_s(\d+)\.npz$", f).group(1))
            r = mt.load_run(f)
            Fr, Gr = mt.reference_for_grid(r["sv_C"].shape[1], a.variant)
            arms.setdefault(s, {})[method] = per_run(r, Fr, Gr)
    seeds = sorted(s for s in arms if len(arms[s]) == 2)
    assert len(seeds) == len(arms), "every seed must have both arms (no discarding)"
    for s in seeds:
        assert arms[s]["abf"]["nfe"] == arms[s]["fr"]["nfe"], "force evaluations must match"
    # pairing check (prereg section 6): the two arms are bitwise identical at every save before the first FR event
    paired = [bool(np.array_equal(arms[s]["abf"]["pre_fr"], arms[s]["fr"]["pre_fr"])) for s in seeds]
    out = {"n_seeds": len(seeds), "K": a.K, "seeds": seeds, "paired_bitwise_before_fr_start": paired}
    for key in ("Ibar_F", "eF_T", "eG_T", "late_eF", "tv_mean", "transloc"):
        if key == "transloc":      # a count: report the paired DIFFERENCE (fr - abf), not a ratio
            d = [arms[s]["fr"][key] - arms[s]["abf"][key] for s in seeds]
        else:
            d = [arms[s]["fr"][key] / arms[s]["abf"][key] - 1 for s in seeds]
        med, lo, hi = boot_median(d)
        out[key] = dict(median_rel=med, ci95=[lo, hi], wins_fr=int(sum(x < 0 for x in d)),
                        kind="difference" if key == "transloc" else "relative",
                        abf_median_abs=float(np.median([arms[s]["abf"][key] for s in seeds])),
                        fr_median_abs=float(np.median([arms[s]["fr"][key] for s in seeds])), per_seed=d)
    out["fr_genealogy"] = dict(min_essw=float(np.median([arms[s]["fr"]["essw_min"] for s in seeds])),
                               max_lineage=float(np.median([arms[s]["fr"]["maxw_max"] for s in seeds])),
                               replacements=float(np.median([arms[s]["fr"]["repl"] for s in seeds])))
    out["wall_seconds"] = dict(abf=float(np.median([arms[s]["abf"]["wall"] for s in seeds])),
                               fr=float(np.median([arms[s]["fr"]["wall"] for s in seeds])))
    # pooled long-time limits
    pooled = {}
    for m in ("abf", "fr"):
        C = sum(arms[s][m]["C"] for s in seeds)
        M = sum(arms[s][m]["M"] for s in seeds)
        pooled[m] = mt.histogram_pmf(mt.mean_force(M, C))       # same grid as Fr (last run loaded)
    out["pooled_rms_vs_ref"] = {m: float(mt.centred_rms(pooled[m] - Fr)) for m in pooled}
    out["pooled_rms_fr_vs_abf"] = float(mt.centred_rms(pooled["fr"] - pooled["abf"]))
    p = out["Ibar_F"]
    verdict = ("FR BETTER (useful)" if p["ci95"][1] < 0 and p["median_rel"] <= -0.10 else
               "FR BETTER (below the 10 % usefulness threshold)" if p["ci95"][1] < 0 else
               "FR WORSE" if p["ci95"][0] > 0 else "NO RESOLVED DIFFERENCE")
    out["primary_verdict"] = verdict
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_seed"})
                      for k, v in out.items()}, indent=1, default=float))
    json.dump(out, open(os.path.join(ROOT, a.dir, f"two_arm_K{a.K}.json"), "w"), indent=1, default=float)
    np.savez(os.path.join(ROOT, a.dir, f"two_arm_curves_K{a.K}.npz"), t=arms[seeds[0]]["abf"]["t"],
             eF_abf=np.array([arms[s]["abf"]["eF_curve"] for s in seeds]),
             eF_fr=np.array([arms[s]["fr"]["eF_curve"] for s in seeds]))


if __name__ == "__main__":
    main()
