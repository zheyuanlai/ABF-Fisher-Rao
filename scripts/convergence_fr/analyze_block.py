"""Frozen two-arm analysis (ABF = arm A, ABF + FR = arm B) for the convergence-FR campaign blocks
(docs/convergence_fr_campaign/PREREGISTRATION.md sections 2-5).  One block = one paired two-arm experiment.

    python scripts/convergence_fr/analyze_block.py --block E1 \
        --dir results/convergence_fr_campaign/E1_ite300_stop10 --K 1024 \
        --refbase results/overnight/ite/reference --variant ite_itq3 --ladder kT [--fr-glob 'fr_N1024_s*.npz']

--ladder kT  : eps = {0.080, 0.064, 0.048, 0.040, 0.032} x kBT (the atlas ladder, frozen before any data)
--ladder abf : eps = {3.0, 2.5, 2.0, 1.6, 1.3} x median ABF e_F(T) of THIS block (FR-blind; matrix cells)

PRIMARY (time to accuracy): per seed, the geometric mean over the ladder of the restricted SUSTAINED speedup
S_k = min(tau_ABF, T) / min(tau_FR, T); the median over seeds and its 95 % bootstrap CI (10 000 resamples).
Supported if the CI lies entirely above 1, contradicted if entirely below 1, otherwise unresolved.
COMPANION: post-ramp Ibar_F over [t_FR, T] (per-seed relative change).
Secondary (each with its own frozen rule in the prereg): early post-ramp Ibar_F over [t_FR, 2 t_stop] (H1),
e_F(T) (H2), late-window Ibar_F over [0.8 T, T], post-stop-only estimate error (H4; deposits after t_stop in both
arms), first-hit and windowed speedups per tolerance, e_F'(T), genealogy, translocations, establishment,
circulation c(t), pooled long-time limits.  Every seed is kept; the pairing check runs first.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]
LADDER_ABF = [3.0, 2.5, 2.0, 1.6, 1.3]


def circulation(r):
    """c(t) = (1/2pi) int Gamma_t dphi = mean over equal bins of the read-out Gamma; c_app(t) the same for the
    APPLIED bias force ramp * w(C) * Gamma (what the walkers feel)."""
    c = r["cfg"]
    G = mt.mean_force(r["sv_M"], r["sv_C"])
    k = r["save_steps"].astype(float)
    ramp = np.ones_like(k) if c["warmup_steps"] <= 0 else np.minimum(1.0, k / c["warmup_steps"])
    w = np.ones_like(G) if c["trust_count"] <= 0 else np.minimum(1.0, r["sv_C"] / c["trust_count"])
    return G.mean(1), (ramp[:, None] * w * G).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--refbase", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ladder", choices=("kT", "abf"), required=True)
    ap.add_argument("--fr-glob", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = S.load_eri_cell(a.dir, a.K, a.refbase, a.variant, fr_glob=a.fr_glob, keep_runs=True)
    t, T = d["t"], d["T"]
    A, B = d["arms"]["abf"], d["arms"]["fr"]
    ra, rb = A["runs"], B["runs"]
    seeds = d["seeds"]
    cfa, cfb = ra[0]["cfg"], rb[0]["cfg"]
    k0 = int(cfb["fr_start_steps"])
    kstop = int(cfb.get("fr_stop_steps", -1))
    out = dict(block=a.block, dir=a.dir, K=d["K"], T=T, T_K=d["T_K"], dt=d["dt"], bins=d["bins"], rate=d["rate"],
               t_fr=d["t_fr"], t_stop=d["t_stop"], n_seeds=len(seeds), seeds=seeds, ladder_rule=a.ladder)
    # ---- integrity: config equality apart from the method, pairing before the first FR opportunity, parity ----
    diff = {k for k in cfa if cfa[k] != cfb[k]}
    out["config_keys_differing_between_arms"] = sorted(diff)
    paired = []
    for x, y in zip(ra, rb):
        pre = x["save_steps"] < k0
        paired.append(bool(np.array_equal(x["sv_C"][pre], y["sv_C"][pre]) and np.array_equal(x["sv_M"][pre], y["sv_M"][pre])))
        assert int(x["n_force_evals"]) == int(y["n_force_evals"]), "force-evaluation parity"
    out["paired_bitwise_before_fr_start"] = paired
    out["force_evals_per_run"] = int(ra[0]["n_force_evals"])
    if kstop >= 0:
        post = rb[0]["save_steps"] >= kstop
        out["fr_events_after_stop"] = int(sum(int(y["sv_i"][-1, 2] - y["sv_i"][post, 2][0]) for y in rb))
    # ---- ladder ----
    if a.ladder == "kT":
        eps = [e * d["kT"] for e in LADDER_KT]
        out["ladder"] = dict(eps_kT=LADDER_KT, eps_kJmol=eps)
    else:
        base = float(np.median(A["eF"][-1]))
        eps = [m * base for m in LADDER_ABF]
        out["ladder"] = dict(multiples_of_median_abf_eF_T=LADDER_ABF, median_abf_eF_T=base, eps_kJmol=eps)
    # ---- PRIMARY ----
    logS = np.zeros(len(seeds))
    per_eps = {}
    for kind in ("sustained", "first", "windowed"):
        per_eps[kind] = []
        for e in eps:
            st = L.speedup_stats(L.tau(A["eF"], t, e, kind), L.tau(B["eF"], t, e, kind), T)
            st["eps_kJmol"] = e
            per_eps[kind].append(st)
            if kind == "sustained":
                logS += np.log(np.array(st["per_seed_S"])) / len(eps)
    med, lo, hi = L.boot_median(logS)
    bm = np.median(logS[np.random.default_rng(L.BOOT_SEED).integers(0, len(logS), size=(10000, len(logS)))], axis=1)
    ci975 = [float(np.exp(np.quantile(bm, 0.0125))), float(np.exp(np.quantile(bm, 0.9875)))]
    prim = dict(ci975=ci975, metric="per-seed geometric-mean restricted sustained speedup over the ladder",
                median=float(np.exp(med)), ci95=[float(np.exp(lo)), float(np.exp(hi))],
                fr_faster_seeds=int((logS > 0).sum()), abf_faster_seeds=int((logS < 0).sum()),
                per_seed=[float(np.exp(x)) for x in logS])
    prim["verdict"] = ("SUPPORTED (FR+ reaches the ladder sooner)" if prim["ci95"][0] > 1 else
                       "CONTRADICTED (ABF reaches the ladder sooner)" if prim["ci95"][1] < 1 else "UNRESOLVED")
    out["primary"] = prim
    out["time_to_accuracy"] = per_eps
    # ---- companion and secondary ----
    tr = d["t_fr"]
    sec = {}
    sec["Ibar_F_post_ramp"] = L.paired_rel(L.running_ibar(A["eF"], t, tr)[-1], L.running_ibar(B["eF"], t, tr)[-1])
    sec["Ibar_F_full"] = L.paired_rel(L.running_ibar(A["eF"], t)[-1], L.running_ibar(B["eF"], t)[-1])
    if d["t_stop"] is not None:
        t2 = 2 * d["t_stop"]
        i2 = int(np.searchsorted(t, t2 - 1e-9))
        sec["Ibar_F_early_post_ramp_to_2tstop"] = L.paired_rel(L.running_ibar(A["eF"], t, tr)[i2],
                                                               L.running_ibar(B["eF"], t, tr)[i2])
        sec["Ibar_F_early_post_ramp_to_2tstop"]["window"] = [tr, float(t[i2])]
    sec["Ibar_F_late_window"] = L.paired_rel(L.running_ibar(A["eF"], t, 0.8 * T)[-1], L.running_ibar(B["eF"], t, 0.8 * T)[-1])
    sec["Ibar_F_late_window"]["window"] = [0.8 * T, T]
    sec["eF_T"] = L.paired_rel(A["eF"][-1], B["eF"][-1])
    sec["eG_T"] = L.paired_rel(A["eG"][-1], B["eG"][-1])
    for frac in (0.25, 0.5, 0.75):
        i = int(np.searchsorted(t, frac * T - 1e-9))
        sec[f"eF_at_{frac:g}T"] = L.paired_rel(A["eF"][i], B["eF"][i])
    # post-stop (H4) and post-FR-start (continuous: late-deposit) read-only estimates, both arms on the same window
    Fr, Gr = d["F_ref"], d["G_ref"]
    kref = kstop if kstop >= 0 else k0
    ea, eb, la, lb = [], [], [], []
    for x, y in zip(ra, rb):
        i0 = int(np.searchsorted(x["save_steps"], kref))
        assert x["save_steps"][i0] == kref
        for r_, lst, lim in ((x, ea, la), (y, eb, lb)):
            Mp = r_["sv_M"][-1] - r_["sv_M"][i0]
            Cp = r_["sv_C"][-1] - r_["sv_C"][i0]
            Fp = mt.histogram_pmf(mt.mean_force(Mp, Cp))
            lst.append(float(mt.centred_rms(Fp - Fr)))
            lim.append((Mp, Cp))
    sec["post_window_estimate_eF"] = L.paired_rel(ea, eb)
    sec["post_window_estimate_eF"]["window"] = [kref * d["dt"], T]
    pooled = {}
    for name, lim in (("abf", la), ("fr", lb)):
        Mp = sum(m for m, _ in lim)
        Cp = sum(c for _, c in lim)
        pooled[name] = mt.histogram_pmf(mt.mean_force(Mp, Cp))
    full = {}
    for name, rr in (("abf", ra), ("fr", rb)):
        full[name] = mt.histogram_pmf(mt.mean_force(sum(r_["sv_M"][-1] for r_ in rr), sum(r_["sv_C"][-1] for r_ in rr)))
    sec["pooled_post_window_rms_vs_ref"] = {k: float(mt.centred_rms(v - Fr)) for k, v in pooled.items()}
    sec["pooled_post_window_rms_fr_vs_abf"] = float(mt.centred_rms(pooled["fr"] - pooled["abf"]))
    sec["pooled_full_rms_vs_ref"] = {k: float(mt.centred_rms(v - Fr)) for k, v in full.items()}
    sec["pooled_full_rms_fr_vs_abf"] = float(mt.centred_rms(full["fr"] - full["abf"]))
    out["secondary"] = sec
    # ---- mechanism / bookkeeping ----
    out["crossing_eF"] = L.crossing(A["eF"], B["eF"], t, t_min=tr)
    esa = L.establishment_time(A["win"], t, d["win_share"])
    esb = L.establishment_time(B["win"], t, d["win_share"])
    out["establishment"] = dict(abf_median=float(np.median(esa)), fr_median=float(np.median(esb)),
                                paired_difference=L.boot_median(esb - esa) if np.isfinite(esa).all() and np.isfinite(esb).all() else None,
                                abf=esa.tolist(), fr=esb.tolist())
    i0 = t >= tr
    out["genealogy_fr"] = dict(min_essw_median=float(np.median([np.nanmin(y["sv_u"][i0, 2]) for y in rb])),
                               max_lineage_median=float(np.median([np.nanmax(y["sv_u"][i0, 3]) for y in rb])),
                               replacements_median=float(np.median([y["sv_i"][-1, 2] for y in rb])),
                               opportunities_with_event_median=float(np.median([y["sv_i"][-1, 8] for y in rb])))
    out["translocations"] = L.paired_rel([max(1, x["sv_i"][-1, 1]) for x in ra], [max(1, y["sv_i"][-1, 1]) for y in rb])
    ca = np.array([circulation(x)[0] for x in ra]).T
    cb = np.array([circulation(y)[0] for y in rb]).T
    caa = np.array([circulation(x)[1] for x in ra]).T
    cba = np.array([circulation(y)[1] for y in rb]).T
    late = t >= 0.5 * T
    out["circulation"] = dict(definition="c(t) = mean over bins of Gamma_t (read-out); c_app = same for the applied force",
                              late_half_mean_abf=float(np.median(ca[late].mean(0))),
                              late_half_mean_fr=float(np.median(cb[late].mean(0))),
                              late_half_abs_abf=float(np.median(np.abs(ca[late]).mean(0))),
                              late_half_abs_fr=float(np.median(np.abs(cb[late]).mean(0))),
                              paired_late_mean_difference=L.boot_median(cb[late].mean(0) - ca[late].mean(0)),
                              applied_late_half_mean_abf=float(np.median(caa[late].mean(0))),
                              applied_late_half_mean_fr=float(np.median(cba[late].mean(0))))
    out["wall_seconds_median"] = dict(abf=float(np.median([x["wall_seconds_session"] for x in ra])),
                                      fr=float(np.median([y["wall_seconds_session"] for y in rb])))
    p = out["primary"]
    print(f"[{a.block}] n={len(seeds)} paired {sum(paired)}/{len(paired)}  PRIMARY S_geo {p['median']:.3f} "
          f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] FR faster {p['fr_faster_seeds']}/{len(seeds)} -> {p['verdict']}")
    for k in ("Ibar_F_post_ramp", "Ibar_F_full", "Ibar_F_early_post_ramp_to_2tstop", "Ibar_F_late_window", "eF_T",
              "eG_T", "post_window_estimate_eF"):
        if k in sec:
            x = sec[k]
            print(f"   {k:34s} {x['median_rel']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] FR wins {x['wins_fr']}/{x['n']}")
    for st in per_eps["sustained"]:
        print(f"   sustained eps {st['eps_kJmol']:.3f}: S {st['S_median']:.2f} [{st['S_ci95'][0]:.2f}, {st['S_ci95'][1]:.2f}] "
              f"reach A{st['reached_abf']} B{st['reached_fr']}  FR faster {st['fr_faster']}  ABF faster {st['abf_faster']}")
    outp = a.out or os.path.join(S.ROOT, "results", "convergence_fr_campaign", "analysis", f"{a.block}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1, default=float)
    np.savez(outp.replace(".json", "_curves.npz"), t=t, eF_abf=A["eF"], eF_fr=B["eF"], eG_abf=A["eG"], eG_fr=B["eG"],
             win_abf=A["win"], win_fr=B["win"], c_abf=ca, c_fr=cb, capp_abf=caa, capp_fr=cba,
             F_post_pooled_abf=pooled["abf"], F_post_pooled_fr=pooled["fr"], F_full_pooled_abf=full["abf"],
             F_full_pooled_fr=full["fr"], F_ref=Fr)
    print("wrote", outp)


if __name__ == "__main__":
    main()
