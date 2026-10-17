"""Frozen paired two-arm analysis for the adaptive-FR campaign (ABF vs ABF + FR, any FR schedule), for both the
ITE/ERI engine (src/eri/engine.py) and the LTA numba engine (src/lta_ladder_numba.py).  Definitions are those of
scripts/convergence_fr/atlas_lib.py (PREREGISTRATION.md of the previous campaign, section 2); on a fixed-stop block
this script reproduces analyze_block.py (checked on E1).

    python scripts/adaptive_fr/analyze_online.py --block C1 --engine eri --dir <fr+abf dir> [--abf-dir <dir>] \
        --K 1024 --refbase results/overnight/ite/reference --variant ite_itq3 --ladder kT --expect-seeds 1300000:16
    python scripts/adaptive_fr/analyze_online.py --block C3 --engine lta --dir <dir> --K 1024 --T-K 150 --ladder abf ...

Per-seed stop: the FR arm's effective stop step (fixed fr_stop_steps, or the online controller's decision; a run that
never stopped uses its FR start for the post-stop window and counts as 'never stopped').
Early window (H1) = [t_FR, 0.4 T].  PRIMARY = per-seed geometric mean over the ladder of the restricted SUSTAINED
speedup; median, 95 % CI and the Bonferroni 99 % CI (5 cells of Experiment C).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
import atlas_lib as L  # noqa: E402
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

LADDER_KT = [0.080, 0.064, 0.048, 0.040, 0.032]
LADDER_ABF = [3.0, 2.5, 2.0, 1.6, 1.3]


# ----------------------------------------------------------------------------------------------- loaders
def eri_run(path, Fr, Gr):
    r = mt.load_run(path)
    c = r["cfg"]
    e = mt.errors(r, Fr, Gr)
    stop = int(r["fr_stop_effective_steps"]) if "fr_stop_effective_steps" in r else int(c.get("fr_stop_steps", -1))
    return dict(cfg=c, seed=int(c["seed"]), t=r["times"], steps=r["save_steps"], eF=e["eF"], eG=e["eG"],
                C=r["sv_C"], M=r["sv_M"], win=r["sv_i"][:, 3] / float(c["N"]), repl=r["sv_i"][:, 2],
                nopp=r["sv_i"][:, 7], essw=r["sv_u"][:, 2], maxw=r["sv_u"][:, 3], transloc=int(r["sv_i"][-1, 1]),
                nfe=int(r["n_force_evals"]), wall=float(r["wall_seconds_session"]), stop=stop,
                fr_start=int(c["fr_start_steps"]), dt=float(c["dt"]), K=int(c["N"]), T_K=float(c["T_K"]),
                win_share=2.0 * float(c["window_phi"]) / (2 * np.pi), occ=r["sv_h"],
                ctl=dict(t1=float(r["fr_controller_t1"]), t_stop=float(r["fr_controller_t_stop"]),
                         reason=str(r["fr_controller_reason"])) if "fr_controller_t1" in r else None,
                Fr=Fr, Gr=Gr, read=lambda M_, C_: mt.histogram_pmf(mt.mean_force(M_, C_)))


def lta_gamma(Mp, Cp, Ma, Ca):
    """equal-budget LTA convention: production accumulators once they hold any count, else all-steps."""
    use_p = Cp.sum(-1) > 0
    M_ = np.where(use_p[..., None], Mp, Ma)
    C_ = np.where(use_p[..., None], Cp, Ca)
    return np.where(C_ > 0, M_ / np.maximum(C_, 1.0), 0.0)


def lta_run(path, Fr, Gr):
    import lta_ladder_numba as LE
    r = LE.load_result(path)
    c = json.loads(str(r["cfg_json"]))
    G = lta_gamma(r["M_prod"], r["C_prod"], r["M_all"], r["C_all"])
    F = mt.histogram_pmf(G)
    eF = mt.centred_rms(F - Fr[None, :])
    eG = np.sqrt(((G - Gr[None, :]) ** 2).mean(-1))
    stop = int(r["fr_stop_effective_steps"]) if "fr_stop_effective_steps" in r else int(c.get("fr_stop_steps", -1))
    ns = len(r["save_step"])
    return dict(cfg=c, seed=int(c["seed"]), t=np.asarray(r["save_t"], float), steps=np.asarray(r["save_step"]),
                eF=eF, eG=eG, C=r["C_all"], M=r["M_all"], Cp=r["C_prod"], Mp=r["M_prod"],
                win=np.asarray(r["region_frac"], float)[:, 2], repl=np.asarray(r.get("cum_deaths", np.zeros(ns))),
                nopp=np.asarray(r.get("cum_opps", np.zeros(ns))), essw=np.asarray(r.get("gen_ess_win", np.ones(ns))),
                maxw=np.asarray(r.get("gen_max_frac_win", np.zeros(ns))), transloc=int(np.asarray(r["events_translocations"]).ravel()[-1]),
                nfe=int(r["n_force_evals"]), wall=float(r["wall_s"]), stop=stop, fr_start=int(c["fr_start_steps"]),
                dt=float(c["h"]), K=int(c["N"]), T_K=float(c["T_K"]), win_share=2 * float(c["window_half"]) / 11.919,
                occ=np.asarray(r["hist_inst"], float),
                ctl=dict(t1=float(r["fr_controller_t1"]), t_stop=float(r["fr_controller_t_stop"]),
                         reason=str(r["fr_controller_reason"])) if "fr_controller_t1" in r else None,
                Fr=Fr, Gr=Gr, read=None)


def collect(a):
    if a.engine == "eri":
        mt.set_reference_base(S.dpath(a.refbase))
        refs = {}
    else:
        z = np.load(S.dpath(f"results/equal_budget_v2/references/lta_T{int(a.T_K)}_reference.npz"), allow_pickle=False)
        Fr_l, Gr_l = np.asarray(z["F_ref"], float), np.asarray(z["gamma_ref"], float)
        Fr_l = Fr_l - Fr_l.mean()
    arms = {}
    for arm, d, pat in (("abf", a.abf_dir or a.dir, f"abf_N{a.K}_s*.npz"), ("fr", a.dir, a.fr_glob or f"fr_N{a.K}_s*.npz")):
        for f in S.glob_both(d, pat):
            if a.engine == "eri":
                z = np.load(f, allow_pickle=False)
                ng = z["sv_C"].shape[1]
                if ng not in refs:
                    refs[ng] = mt.reference_for_grid(ng, a.variant)
                run = eri_run(f, *refs[ng])
            else:
                run = lta_run(f, Fr_l, Gr_l)
            arms.setdefault(run["seed"], {})[arm] = run
    seeds = sorted(s for s in arms if len(arms[s]) == 2)
    if len(seeds) != len(arms):
        raise RuntimeError(f"unpaired seeds {sorted(set(arms) - set(seeds))}")
    return seeds, arms


# ----------------------------------------------------------------------------------------------- analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--engine", choices=("eri", "lta"), required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--abf-dir", default=None)
    ap.add_argument("--fr-glob", default=None)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--refbase", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--T-K", type=float, default=None)
    ap.add_argument("--ladder", choices=("kT", "abf"), required=True)
    ap.add_argument("--expect-seeds", default=None)
    ap.add_argument("--allow-diff", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds, arms = collect(a)
    ra = [arms[s]["abf"] for s in seeds]
    rb = [arms[s]["fr"] for s in seeds]
    t = ra[0]["t"]
    T = float(t[-1])
    for x, y in zip(ra, rb):
        assert np.allclose(x["t"], t) and np.allclose(y["t"], t), "save grids differ"
    out = dict(block=a.block, engine=a.engine, K=a.K, T=T, T_K=ra[0]["T_K"], n_seeds=len(seeds), cfg_seeds=seeds,
               ladder_rule=a.ladder)
    # ---- integrity ----
    ca, cb = ra[0]["cfg"], rb[0]["cfg"]
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    allowed = {"fr_stop_steps", "fr_controller"} | set(a.allow_diff)
    if "diag_age_steps" in diff:         # read-only diagnostic key added after the EB runs: allowed only when OFF
        assert all(int(r_["cfg"].get("diag_age_steps") or 0) == 0 for r_ in ra + rb), "age diagnostic active"
        allowed = allowed | {"diag_age_steps"}
    assert diff <= allowed, f"arms differ in {diff}"
    strip = lambda c: {k: v for k, v in c.items() if k != "seed"}
    for rr in (ra, rb):
        assert all(strip(r_["cfg"]) == strip(rr[0]["cfg"]) for r_ in rr), "configs differ within an arm"
    if a.expect_seeds:
        s0, n = (int(v) for v in a.expect_seeds.split(":"))
        assert seeds == list(range(s0, s0 + n)), f"seed block {seeds} != {s0}+{n}"
    k0 = rb[0]["fr_start"]
    paired = []
    for x, y in zip(ra, rb):
        pre = x["steps"] < k0
        paired.append(bool(np.array_equal(x["C"][pre], y["C"][pre]) and np.array_equal(x["M"][pre], y["M"][pre])))
        assert x["nfe"] == y["nfe"], "force-evaluation parity"
        assert int(np.asarray(x["repl"])[-1]) == 0, "ABF arm has FR events"
    assert all(paired), "arms not bitwise paired before the first FR opportunity"
    out.update(paired_bitwise_before_fr_start=paired, force_evals_per_run=ra[0]["nfe"], allowed_config_diff=sorted(diff))
    # ---- FR schedule actually realised ----
    dt = rb[0]["dt"]
    stops = np.array([y["stop"] for y in rb])
    never = stops < 0
    after = []
    for y in rb:
        if y["stop"] >= 0:
            i = int(np.searchsorted(y["steps"], y["stop"]))
            after.append(int(np.asarray(y["repl"])[-1] - np.asarray(y["repl"])[min(i, len(y["steps"]) - 1)]))
    assert sum(after) == 0, "FR events after the stop"
    sched = dict(stop_t=[float(s * dt) if s >= 0 else None for s in stops], never_stopped=int(never.sum()),
                 events_after_stop_total=int(sum(after)),
                 replacements=[int(np.asarray(y["repl"])[-1]) for y in rb])
    if rb[0]["ctl"] is not None:
        sched.update(t1=[y["ctl"]["t1"] for y in rb], t_stop=[y["ctl"]["t_stop"] for y in rb],
                     reason=[y["ctl"]["reason"] for y in rb])
    st = np.array([s * dt for s in stops if s >= 0])
    if st.size:
        sched.update(stop_t_median=float(np.median(st)), stop_t_iqr=[float(np.quantile(st, 0.25)), float(np.quantile(st, 0.75))],
                     stop_t_min=float(st.min()), stop_t_max=float(st.max()))
    out["fr_schedule"] = sched
    # ---- ladder + PRIMARY ----
    EA = np.array([x["eF"] for x in ra]).T
    EB = np.array([y["eF"] for y in rb]).T
    GA = np.array([x["eG"] for x in ra]).T
    GB = np.array([y["eG"] for y in rb]).T
    kT = 0.0083144626 * ra[0]["T_K"]
    if a.ladder == "kT":
        eps = [e * kT for e in LADDER_KT]
        out["ladder"] = dict(eps_kT=LADDER_KT, eps_kJmol=eps)
    else:
        base = float(np.median(EA[-1]))
        eps = [m * base for m in LADDER_ABF]
        out["ladder"] = dict(multiples_of_median_abf_eF_T=LADDER_ABF, median_abf_eF_T=base, eps_kJmol=eps)
    logS = np.zeros(len(seeds))
    per_eps = {}
    for kind in ("sustained", "first", "windowed"):
        per_eps[kind] = []
        for e in eps:
            ta, tb = L.tau(EA, t, e, kind), L.tau(EB, t, e, kind)
            stt = L.speedup_stats(ta, tb, T)
            stt["eps_kJmol"] = e
            stt["restricted_mean_tau"] = dict(abf=float(np.mean(np.minimum(ta, T))), fr=float(np.mean(np.minimum(tb, T))))
            per_eps[kind].append(stt)
            if kind == "sustained":
                logS += np.log(np.array(stt["per_seed_S"])) / len(eps)
    med, lo, hi = L.boot_median(logS)
    bm = np.median(logS[np.random.default_rng(L.BOOT_SEED).integers(0, len(logS), size=(10000, len(logS)))], axis=1)
    ci99 = [float(np.exp(np.quantile(bm, 0.005))), float(np.exp(np.quantile(bm, 0.995)))]
    prim = dict(median=float(np.exp(med)), ci95=[float(np.exp(lo)), float(np.exp(hi))], ci99_bonferroni5=ci99,
                fr_faster_seeds=int((logS > 0).sum()), abf_faster_seeds=int((logS < 0).sum()),
                per_seed=[float(np.exp(x)) for x in logS])
    prim["verdict"] = "SUPPORTED" if prim["ci95"][0] > 1 else "CONTRADICTED" if prim["ci95"][1] < 1 else "UNRESOLVED"
    prim["verdict_bonferroni"] = "SUPPORTED" if ci99[0] > 1 else "CONTRADICTED" if ci99[1] < 1 else "UNRESOLVED"
    out["primary"] = prim
    out["time_to_accuracy"] = per_eps
    # ---- secondary ----
    tr = k0 * dt
    sec = {}
    sec["Ibar_F_post_ramp"] = L.paired_rel(L.running_ibar(EA, t, tr)[-1], L.running_ibar(EB, t, tr)[-1])
    sec["Ibar_F_full"] = L.paired_rel(L.running_ibar(EA, t)[-1], L.running_ibar(EB, t)[-1])
    i_e = int(np.searchsorted(t, 0.4 * T - 1e-9))
    sec["Ibar_F_early"] = L.paired_rel(L.running_ibar(EA, t, tr)[i_e], L.running_ibar(EB, t, tr)[i_e])
    sec["Ibar_F_early"]["window"] = [tr, float(t[i_e])]
    sec["Ibar_F_late_window"] = L.paired_rel(L.running_ibar(EA, t, 0.8 * T)[-1], L.running_ibar(EB, t, 0.8 * T)[-1])
    sec["eF_T"] = L.paired_rel(EA[-1], EB[-1])
    sec["eG_T"] = L.paired_rel(GA[-1], GB[-1])
    for frac in (0.25, 0.5, 0.75):
        i = int(np.searchsorted(t, frac * T - 1e-9))
        sec[f"eF_at_{frac:g}T"] = L.paired_rel(EA[i], EB[i])
    # post-stop-only estimate: deposits after the FR arm's stop (or FR start if it never stopped), both arms
    ea, eb = [], []
    for x, y in zip(ra, rb):
        kref = y["stop"] if y["stop"] >= 0 else k0
        i0 = int(np.searchsorted(x["steps"], kref))
        for r_, lst in ((x, ea), (y, eb)):
            if a.engine == "eri":
                Fp = mt.histogram_pmf(mt.mean_force(r_["M"][-1] - r_["M"][i0], r_["C"][-1] - r_["C"][i0]))
            else:
                Fp = mt.histogram_pmf(mt.mean_force(r_["Mp"][-1] - r_["Mp"][i0], r_["Cp"][-1] - r_["Cp"][i0]))
            lst.append(float(mt.centred_rms(Fp - r_["Fr"])))
    sec["post_stop_estimate_eF"] = L.paired_rel(ea, eb)
    out["secondary"] = sec
    # ---- mechanism / bookkeeping ----
    out["crossing_eF"] = L.crossing(EA, EB, t, t_min=tr)
    WA = np.array([x["win"] for x in ra]).T
    WB = np.array([y["win"] for y in rb]).T
    esa, esb = L.establishment_time(WA, t, ra[0]["win_share"]), L.establishment_time(WB, t, ra[0]["win_share"])
    out["establishment"] = dict(abf_median=float(np.median(esa)), fr_median=float(np.median(esb)),
                                paired_difference=L.boot_median(esb - esa) if np.isfinite(esa).all() and np.isfinite(esb).all() else None)
    fr_on = [(y["steps"] > k0) & ((y["steps"] <= y["stop"]) if y["stop"] >= 0 else True) for y in rb]
    out["genealogy_fr"] = dict(min_essw_median=float(np.median([np.nanmin(np.asarray(y["essw"])[m]) for y, m in zip(rb, fr_on)])),
                               max_lineage_median=float(np.median([np.nanmax(np.asarray(y["maxw"])[m]) for y, m in zip(rb, fr_on)])),
                               replacements_median=float(np.median(sched["replacements"])))

    def tv(occ):
        p = occ / np.maximum(occ.sum(-1, keepdims=True), 1)
        pc = p.reshape(p.shape[0], 18, -1).sum(-1)
        return 0.5 * np.abs(pc - 1 / 18).sum(-1)
    out["marginal_tv_time_avg_after_fr_start"] = L.paired_rel([float(tv(x["occ"])[x["steps"] >= k0].mean()) for x in ra],
                                                              [float(tv(y["occ"])[y["steps"] >= k0].mean()) for y in rb])
    out["translocations"] = dict(abf_median=float(np.median([x["transloc"] for x in ra])),
                                 fr_median=float(np.median([y["transloc"] for y in rb])))
    out["wall_seconds_median"] = dict(abf=float(np.median([x["wall"] for x in ra])), fr=float(np.median([y["wall"] for y in rb])))
    p = out["primary"]
    print(f"[{a.block}] n={len(seeds)} paired {sum(paired)}/{len(paired)}  PRIMARY S_geo {p['median']:.3f} [{p['ci95'][0]:.3f}, "
          f"{p['ci95'][1]:.3f}] (99 %: [{ci99[0]:.3f}, {ci99[1]:.3f}]) FR faster {p['fr_faster_seeds']}/{len(seeds)} -> {p['verdict']}")
    if st.size:
        print(f"   FR stop: median {sched['stop_t_median']:.2f} t.u. IQR {sched['stop_t_iqr']} range [{sched['stop_t_min']}, {sched['stop_t_max']}]"
              f" never stopped {sched['never_stopped']}" + (f"; reasons {sorted(set(sched.get('reason', [])))}" if 'reason' in sched else ""))
    for k in ("Ibar_F_post_ramp", "Ibar_F_full", "Ibar_F_early", "Ibar_F_late_window", "eF_T", "eG_T", "post_stop_estimate_eF"):
        x = sec[k]
        print(f"   {k:24s} {x['median_rel']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] FR wins {x['wins_fr']}/{x['n']}")
    for s_ in per_eps["sustained"]:
        print(f"   sustained eps {s_['eps_kJmol']:.3f}: S {s_['S_median']:.2f} [{s_['S_ci95'][0]:.2f}, {s_['S_ci95'][1]:.2f}] "
              f"reach A{s_['reached_abf']} B{s_['reached_fr']}  FR faster {s_['fr_faster']}  ABF faster {s_['abf_faster']}")
    outp = a.out or os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "analysis", f"{a.block}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1, default=float)
    np.savez(outp.replace(".json", "_curves.npz"), t=t, eF_abf=EA, eF_fr=EB, eG_abf=GA, eG_fr=GB, win_abf=WA, win_fr=WB)
    print("wrote", outp)


if __name__ == "__main__":
    main()
