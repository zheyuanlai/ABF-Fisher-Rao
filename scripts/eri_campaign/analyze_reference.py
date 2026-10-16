"""Gate 1 (MC reference) and Gate 2 (Langevin timestep ladder) analysis, frozen in PREREGISTRATION.md sections 3-4.

    python scripts/eri_campaign/analyze_reference.py [--variant gard_tait]

Writes results/eri_campaign/reference/<variant>/{gate1.json, gate2.json, reference_profile.npz} and figures under
figures/eri_campaign/reference/.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import refanalysis as ra  # noqa: E402
from eri.model import KB  # noqa: E402

TOL, TOL_UP = 0.05, 0.075        # ERI defaults; overridden by --tol / --tolup (ITE: 0.03 / 0.045)


def level(c):
    return "PASS" if (c["D"] <= TOL and c["D_upper95"] <= TOL_UP) else "FAIL"


def profiles(acc, kT, label):
    G = len(acc["groups"])
    dens = ra.Profile(f"{label} F_density", lambda idx: ra.F_density(acc, idx, kT)[0], G)
    mf = ra.Profile(f"{label} F_MF", lambda idx: ra.F_MF(acc, idx)[0], G)
    return dens, mf


def tau_int(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    v = x.var()
    if v == 0 or n < 10:
        return float("nan")
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (v * np.arange(n, 0, -1))
    t = 0.5
    for k in range(1, n // 2):
        t += ac[k]
        if k > 5 * t:
            break
    return float(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gard_tait")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--skip-gate1", action="store_true")
    ap.add_argument("--tags", default="", help="comma list of Gate 2 tags (default: all)")
    ap.add_argument("--base", default=os.path.join("results", "eri_campaign", "reference"))
    ap.add_argument("--T", type=float, default=600.0)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--tolup", type=float, default=0.075)
    a = ap.parse_args()
    global TOL, TOL_UP
    TOL, TOL_UP = a.tol, a.tolup
    kT = KB * a.T
    base = os.path.join(ROOT, a.base, a.variant)
    mc = ra.load(os.path.join(base, "MC"), groups=list(range(16)))
    G = len(mc["groups"])
    allg = np.arange(G)
    dens, mf = profiles(mc, kT, "MC")
    out = {"variant": a.variant, "kT": kT, "tolerance": [TOL, TOL_UP], "nboot": a.nboot}
    _, gam, cnt = ra.F_MF(mc, allg)
    obs = {k: ra.cond_mean(mc, allg, k) for k in ("Mux2", "Muz2", "Meta", "Mbond")}
    bond2 = ra.cond_mean(mc, allg, "Mbond2")
    bs = ra.barrier_stats(dens.F, ra.cond_mean(mc, allg, "MU"), kT)
    if a.skip_gate1:
        return gate2(a, base, dens, kT, cnt, obs, bond2, bs)
    # 1 overlap
    fine = mc["C_fine"].sum((0, 1))
    out["overlap_min_fine_count"] = float(fine.min())
    out["check1_overlap"] = "PASS" if fine.min() >= 200 else "FAIL"
    # 2 MC self-consistency
    c2 = ra.compare(dens, mf, nboot=a.nboot)
    out["check2_density_vs_MF"] = c2 | {"verdict": level(c2)}
    # 3 split half
    h1 = {k: v[:8] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == G else v for k, v in mc.items()}
    h2 = {k: v[8:] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == G else v for k, v in mc.items()}
    for h in (h1, h2):
        h["groups"] = np.arange(8)
    d1 = ra.Profile("MC groups 0-7", lambda idx: ra.F_density(h1, idx, kT)[0], 8)
    d2 = ra.Profile("MC groups 8-15", lambda idx: ra.F_density(h2, idx, kT)[0], 8)
    c3 = ra.compare(d1, d2, nboot=a.nboot)
    out["check3_split_half"] = c3 | {"verdict": "PASS" if c3["D"] <= TOL else "FAIL"}
    # 4 first vs second half of every window's samples
    first = ra.Profile("MC first half", lambda idx: ra.F_density(mc, idx, kT, key="C_fine_h1")[0], G)
    sec = dict(mc)
    sec["C_fine"] = mc["C_fine"] - mc["C_fine_h1"]
    second = ra.Profile("MC second half", lambda idx: ra.F_density(sec, idx, kT)[0], G)
    c4 = ra.compare(first, second, nboot=a.nboot)
    out["check4_equilibration"] = c4 | {"verdict": "PASS" if c4["D"] <= TOL else "FAIL"}
    # 5 periodic closure of the pooled mean force
    _, gam, cnt = ra.F_MF(mc, allg)
    loo = np.array([ra.F_MF(mc, np.delete(allg, g))[1].mean() for g in range(G)])
    se = math.sqrt((G - 1) / G * ((loo - loo.mean()) ** 2).sum())
    out["check5_closure"] = dict(mean_gamma=float(gam.mean()), se=se,
                                 verdict="PASS" if abs(gam.mean()) < 3 * se else "FAIL")
    # 6 precision
    out["check6_reference_noise_rms"] = float(np.sqrt(np.mean(dens.se ** 2)))
    out["gate1"] = "PASS" if all(out[k]["verdict"] == "PASS" if isinstance(out[k], dict) else out[k] == "PASS"
                                 for k in ("check1_overlap", "check2_density_vs_MF", "check3_split_half",
                                           "check4_equilibration", "check5_closure")) else "FAIL"
    # decomposition and conditional observables
    U = ra.cond_mean(mc, allg, "MU")
    F = dens.F
    Uc = U - U.mean()
    bs = ra.barrier_stats(F, U, kT)
    out["barrier"] = bs
    out["acceptance"] = {k: float(mc["cnt"][..., i].sum() / mc["cnt"][..., i + 1].sum())
                         for k, i in (("bead", 0), ("trans", 2), ("rot", 4))}
    obs = {k: ra.cond_mean(mc, allg, k) for k in ("Mux2", "Muz2", "Meta", "Mbond")}
    bond2 = ra.cond_mean(mc, allg, "Mbond2")
    np.savez(os.path.join(base, "reference_profile.npz"), F=F, F_se=dens.se, F_MF=mf.F, F_MF_se=mf.se, U=U,
             mTS=F - Uc - (F - Uc).mean(), counts=cnt, gamma=gam, ux2=obs["Mux2"], uz2=obs["Muz2"],
             eta=obs["Meta"], bond=obs["Mbond"], bond_var=bond2 - obs["Mbond"] ** 2, kT=kT,
             grid=(np.arange(ra.NB) + 0.5) * ra.DPHI - math.pi)
    bF_s = ra.bead_profile(mc, allg, kT, "Hs")
    bF_x = ra.bead_profile(mc, allg, kT, "Hx")
    out["bead_s_barrier_kT"] = float(bF_s[0])
    out["bead_x_barrier_kT"] = float(bF_x[0])
    print(json.dumps({k: out[k] for k in out if k not in ("barrier",)}, indent=1, default=float))
    print("barrier:", json.dumps(bs, default=float))
    json.dump(out, open(os.path.join(base, "gate1.json"), "w"), indent=1, default=float)
    if not a.tags:
        gate2(a, base, dens, kT, cnt, obs, bond2, bs)


def gate2(a, base, dens, kT, cnt, obs, bond2, bs):
    g2 = {}
    want = [t for t in a.tags.split(",") if t]
    for d in sorted(glob.glob(os.path.join(base, "*_*"))):
        tag = os.path.basename(d)
        if not os.path.isdir(d) or tag == "MC" or (want and tag not in want):
            continue
        try:
            L = ra.load(d, groups=list(range(16)))
        except FileNotFoundError as e:
            print(tag, "incomplete:", e)
            continue
        ldens, lmf = profiles(L, kT, tag)
        cd = ra.compare(ldens, dens, nboot=a.nboot)
        cm = ra.compare(lmf, dens, nboot=a.nboot)
        lobs = {k: ra.cond_mean(L, np.arange(16), k) for k in ("Mux2", "Muz2", "Meta", "Mbond")}
        lb2 = ra.cond_mean(L, np.arange(16), "Mbond2")
        w = cnt > 0
        rec = dict(D_density=cd, D_MF=cm,
                   eta_mean_abs_diff=float(np.mean(np.abs(lobs["Meta"] - obs["Meta"])[w])),
                   ux2_mean_abs_diff=float(np.mean(np.abs(lobs["Mux2"] - obs["Mux2"])[w])),
                   uz2_mean_abs_diff=float(np.mean(np.abs(lobs["Muz2"] - obs["Muz2"])[w])),
                   bond_var_rel=float(np.mean((lb2 - lobs["Mbond"] ** 2)[w]) / np.mean((bond2 - obs["Mbond"] ** 2)[w])
                                      - 1.0),
                   barrier=ra.barrier_stats(ldens.F, None, kT),
                   max_step_component=float(L["cnt"][..., 7].max()))
        rec["verdict"] = ("ADMISSIBLE" if (level(cd) == "PASS" and level(cm) == "PASS"
                                           and rec["eta_mean_abs_diff"] <= 0.02) else "NOT ADMISSIBLE")
        g2[tag] = rec
        print(f"{tag:10s} D_dens {cd['D']:.4f} [up {cd['D_upper95']:.4f}] D_MF {cm['D']:.4f} [up {cm['D_upper95']:.4f}]"
              f" |d eta| {rec['eta_mean_abs_diff']:.4f} bond var {100 * rec['bond_var_rel']:+.2f}% "
              f"barrier {rec['barrier']['F_barrier']:.3f} (MC {bs['F_barrier']:.3f}) -> {rec['verdict']}")
        json.dump(rec, open(os.path.join(base, f"gate2_{tag}.json"), "w"), indent=1, default=float)
    return g2


if __name__ == "__main__":
    main()
