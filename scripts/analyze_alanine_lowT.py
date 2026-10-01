#!/usr/bin/env python
"""Analysis for the low-temperature alanine study (docs/ALANINE_LOWT.md).  Safe on partial data.

    python scripts/analyze_alanine_lowT.py --root results/alanine_T150 --ref results/alanine_T150/reference/reference.npz

Predictor (read on the ABF arm): t_est = first save at which >= 95 % of the 8 kT mask cells carry a
raw count >= c_min AND C7ax has been visited in every seed; t_F = first save at which the ABF error is
within 2x its final value; ESTABLISHMENT-LIMITED iff t_est >= 30 ps and t_est > t_F.  FR arms vs the
matched ABF arm with the frozen statistics/verdicts; support-target vs torus-target paired at equal rate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "..", "src"))

from analyze_fr_start_timing import (FLOORS, _colors, _style, flat_stats, integrate,  # noqa: E402
                                     paired_stats, speedups, verdict, write_csv)
from analyze_alanine_histogram import READOUTS, arm_info, at_times, med_int          # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks, smooth_reference             # noqa: E402

W0, W1 = (1.0, 100.0), (5.0, 100.0)
T_EST_MIN, EST_FRAC = 30.0, 0.95


def load_runs(root):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "final_pmf" not in d.files:
            continue
        meta = json.loads(str(d["meta"]))
        keys = [k for k in d.files if k not in ("meta", "pmf", "marg_hist", "final_f1s", "final_f2s", "acc_f1s", "acc_f2s")]
        data = {k: np.asarray(d[k]) for k in keys}
        data["pmf"] = np.asarray(d["pmf"])
        runs[meta["stage"]] = dict(data=data, meta=meta, path=f, t=np.asarray(d["times"], float))
    return runs


def references(ref):
    refd = np.load(ref, allow_pickle=True)
    F = refd["F"]; rmeta = json.loads(str(refd["meta"]))
    kT, n = float(rmeta["kT_kJ"]), int(rmeta["n_grid"])
    pack = build_masks(F, kT); F_sm = smooth_reference(F, 0.08, n)
    boot = os.path.join(os.path.dirname(ref), "bootstrap.npz")
    floor = float("nan")
    if os.path.exists(boot):
        se = np.load(boot, allow_pickle=True)["F_se"]; w = pack["weights"]["equilibrium"]
        ok = (w > 0) & np.isfinite(se); floor = float(np.sqrt((se[ok] ** 2 * w[ok]).sum() / w[ok].sum()))
    acc = os.path.join(os.path.dirname(ref), "acceptance.json")
    acceptance = json.load(open(acc)) if os.path.exists(acc) else None
    return dict(F=F, kT=kT, n=n, pack=pack, F_sm=F_sm, floor=floor, temperature=rmeta.get("temperature"),
                acceptance=acceptance)


def error_series(runs, refs):
    import torch
    from alkanes import density2d as d2
    n = refs["n"]; pack = refs["pack"]; w = pack["weights"]["equilibrium"]; wu = pack["weights"]["uniform8"]
    g1, g2, _, _ = d2.torus_grid(n, n, dtype=torch.float64)
    K1, K2 = d2.kernels(g1, g2, 0.08, 0.08); K1, K2 = K1 / K1.sum(1, keepdim=True), K2 / K2.sum(1, keepdim=True)
    for run in runs.values():
        pmf = run["data"]["pmf"]; T, R = pmf.shape[:2]
        e = {k: np.zeros((T, R)) for k in READOUTS}
        for ti in range(T):
            psm = d2.smooth2(torch.as_tensor(pmf[ti]), K1, K2).numpy()
            for r in range(R):
                e["raw"][ti, r] = aligned_l2(pmf[ti, r], pack["F"], w)
                e["km"][ti, r] = aligned_l2(pmf[ti, r], refs["F_sm"], w)
                e["raw_u8"][ti, r] = aligned_l2(pmf[ti, r], pack["F"], wu)
                e["sm"][ti, r] = aligned_l2(psm[r], refs["F_sm"], w)
        run["e"] = e; del run["data"]["pmf"]


def predictor(run, refs, t):
    """t_est, t_F and the regime call from the ABF arm (per seed, medians reported)."""
    d = run["data"]; info = arm_info(run); mask = refs["pack"]["mask8"]
    R = d["first_hit"].shape[0]
    t_est = np.full(R, np.nan); t_cov = np.full(R, np.nan)
    if "acc_csum" in d:
        cs = d["acc_csum"]                                   # (T, R, n, n)
        for r in range(R):
            frac = (cs[:, r][:, mask] >= info["c_min"]).mean(1)
            hit = d["first_hit"][r, 2]
            ok = (frac >= EST_FRAC) & (t >= (hit * 1e-3 if hit >= 0 else np.inf))
            t_est[r] = t[int(np.argmax(ok))] if ok.any() else np.inf
            cov = (cs[:, r][:, mask] >= 1).mean(1)
            okc = cov >= EST_FRAC
            t_cov[r] = t[int(np.argmax(okc))] if okc.any() else np.inf
    e = run["e"]["raw"]; t_F = np.full(R, np.nan)
    for r in range(R):
        ok = e[:, r] <= 2.0 * e[-1, r]; t_F[r] = t[int(np.argmax(ok))] if ok.any() else np.inf
    fh = d["first_hit"][:, 2].astype(float) * 1e-3
    med = lambda x: float(np.median(x))
    if med(t_est) < T_EST_MIN:
        reg = "FAST"
    elif med(t_F) >= med(t_est):
        reg = "ESTABLISHMENT-LIMITED (error still falling when counts establish: favourable)"
    else:
        reg = "COUNT-STARVED, ERROR-CONVERGED (t_F < t_est: the ZIF-8 pattern, NOT allocation-limited)"
    return dict(t_est_med=med(t_est), t_est_max=float(np.max(t_est)), t_cov_med=med(t_cov), t_F_med=med(t_F),
                c7ax_first_hit_med=med(np.where(fh < 0, np.inf, fh)), c7ax_censored=int((fh < 0).sum()),
                regime=reg, rule=f"FAST if t_est < {T_EST_MIN:g} ps; favourable iff t_F >= t_est (amendment A1, docs/ALANINE_NLADDER.md)")


def matched_abf(runs, info, n_replicas=None, gamma=None):
    for n, r in runs.items():
        i = arm_info(r)
        if (i["method"] == "abf" and i["estimator"] == info["estimator"] and i["c_min"] == info["c_min"]
                and i["warmup_ps"] == info["warmup_ps"]
                and (n_replicas is None or int(r["meta"]["n_replicas"]) == int(n_replicas))
                and float(r["meta"].get("gamma", 1.0)) == float(gamma if gamma is not None else r["meta"].get("gamma", 1.0))):
            return n
    return None


def fr_row(name, run, abf, t):
    info = arm_info(run); e, ea = run["e"]["raw"], abf["e"]["raw"]; T = float(t[-1])
    own = (max(info["fr_start_ps"], 1.0), T)
    rec = dict(arm=name, **info, matched_abf=abf["name"], own_window=f"[{own[0]:g};{own[1]:g}]")
    prim = paired_stats(integrate(t, ea, *own), integrate(t, e, *own)); rec.update(flat_stats("dIF_own", prim))
    for ro in READOUTS:
        rec.update(flat_stats(f"dIF_W1_{ro}", paired_stats(integrate(t, abf["e"][ro], *W1), integrate(t, run["e"][ro], *W1))))
        rec.update(flat_stats(f"final_{ro}", paired_stats(abf["e"][ro][-1], run["e"][ro][-1])))
    fin = paired_stats(ea[-1], e[-1]); rec.update(flat_stats("final", fin))
    sp, e0 = speedups(t, ea, e, T, own[0])
    for k, v in sp.items():
        rec[f"S_{k}"] = v["speedup"]; rec[f"cens_{k}"] = f"{v['censored_abf']}/{v['censored_arm']}"
    ratio = np.median(e / ea, 1); rec["ratio_min"], rec["ratio_min_t"], rec["ratio_final"] = float(ratio.min()), float(t[int(np.argmin(ratio))]), float(ratio[-1])
    d = run["data"]; N = int(run["meta"]["n_replicas"]); sel = t >= own[0] - 1e-9
    n_opp = max(int((int(run["meta"]["n_steps"]) - int(round(info["fr_start_ps"] * 1000))) // info["fr_every"]) + 1, 1)
    ev = d["total_events"].astype(float)
    rec["events_per_opp_med"] = float(np.median(ev)) / n_opp; rec["event_frac_per_opp_max"] = float(ev.max()) / N / n_opp
    rec["ess_age_min"] = float(np.nanmin(d["ess_age"][sel])); rec["wmax_max"] = float(np.nanmax(d["wmax"][sel]))
    floors_ok = rec["ess_age_min"] >= FLOORS["ess"] and rec["wmax_max"] <= FLOORS["wmax"] and rec["event_frac_per_opp_max"] < FLOORS["event_frac"]
    rec["floors_ok"] = bool(floors_ok)
    for k in ("kl_uniform", "kl_support"):
        if k in d:
            rec[f"{k}_final_arm"], rec[f"{k}_final_abf"] = float(np.median(d[k][-1])), float(np.median(abf["data"][k][-1]))
            rec[f"{k}_min_arm"] = float(np.median(d[k], 1).min())
    rec["c7ax_final_arm"], rec["c7ax_final_abf"] = float(np.median(d["basin_frac"][-1, :, 2])), float(np.median(abf["data"]["basin_frac"][-1, :, 2]))
    fh = d["first_hit"][:, 2].astype(float) * 1e-3; rec["c7ax_first_hit_med"] = float(np.median(np.where(fh < 0, np.inf, fh)))
    rec["verdict"] = verdict(prim, fin, floors_ok)
    return rec


def pct(r, k):
    return f"{100 * r[k + '_median']:+6.2f} % [{100 * r[k + '_lo']:+6.2f}, {100 * r[k + '_hi']:+6.2f}] {int(round(r[k + '_win'] * r[k + '_n']))}/{r[k + '_n']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "analysis"); os.makedirs(out, exist_ok=True)
    runs = load_runs(a.root)
    if not runs:
        print("no runs"); return
    refs = references(a.ref); t = runs[next(iter(runs))]["t"]
    runs = {k: v for k, v in runs.items() if np.allclose(v["t"], t)}
    error_series(runs, refs)
    L = [f"# Low-T alanine scoreboard ({a.root}; reference T = {refs['temperature']} K, kT = {refs['kT']:.3f} kJ/mol; raw read-out floor {refs['floor']:.3f} kJ/mol)", "",
         f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Reference acceptance: {json.dumps(refs['acceptance'].get('verdict', refs['acceptance'].get('accepted', 'see acceptance.json'))) if refs['acceptance'] else 'acceptance.json absent'}", ""]
    pred = {}
    abf_names = [n for n, r in runs.items() if arm_info(r)["method"] == "abf"]
    for n in abf_names:
        pred[n] = predictor(runs[n], refs, t)
        r = runs[n]; e = r["e"]["raw"]
        L += [f"## Predictor on `{n}` (N = {r['meta']['n_replicas']}, gamma = {r['meta'].get('gamma', 1.0)} /ps): **{pred[n]['regime']}**", "",
              f"- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median {pred[n]['t_est_med']:.0f} ps (max {pred[n]['t_est_max']:.0f}); coverage-only t_cov {pred[n]['t_cov_med']:.0f} ps; t_F (error within 2x final): {pred[n]['t_F_med']:.0f} ps; rule: {pred[n]['rule']}",
              f"- C7ax first hit median {pred[n]['c7ax_first_hit_med']:.2f} ps, censored {pred[n]['c7ax_censored']}/16; ABF raw error @1/5/20/50/100 ps: " + " / ".join(f"{v:.3f}" for v in at_times(t, e, (1, 5, 20, 50, 100)).values()) + f"; I_F W1 {med_int(t, e, W1):.2f}; KL(p||U) @1/100: " + "/".join(f"{v:.2f}" for v in at_times(t, r['data']['kl_uniform'], (1, 100)).values()) + (("; KL(p||support) @1/100: " + "/".join(f"{v:.2f}" for v in at_times(t, r['data']['kl_support'], (1, 100)).values())) if "kl_support" in r["data"] else ""), ""]
    rows = []
    for n, r in sorted(runs.items()):
        i = arm_info(r)
        if i["method"] == "abf":
            continue
        m = matched_abf(runs, i, r["meta"]["n_replicas"], r["meta"].get("gamma", 1.0))
        if m is None:
            print(f"{n}: no matched ABF arm yet"); continue
        rows.append(fr_row(n, r, dict(runs[m], name=m), t))
    if rows:
        L += ["## FR vs matched ABF (own window, raw read-out; frozen rules)", "",
              "| arm | method | rate | dI_F own | dI_F W1 | final | S(e0/2) | S(e0/4) | S(e0/8) | ratio min (t) | ratio final | events/opp | ESS min | KL_U final arm/abf | KL_support final arm/abf | C7ax final arm/abf | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            S = ["-" if r[f"S_{k}"] is None else f"{r[f'S_{k}']:.2f}" for k in ("e0/2", "e0/4", "e0/8")]
            ks = f"{r['kl_support_final_arm']:.3f}/{r['kl_support_final_abf']:.3f}" if "kl_support_final_arm" in r else "-"
            L.append(f"| {r['arm']} | {r['method']} | {r['fr_rate']:g} | {pct(r, 'dIF_own')} | {pct(r, 'dIF_W1_raw')} | {pct(r, 'final')} | {S[0]} | {S[1]} | {S[2]} | {r['ratio_min']:.3f} ({r['ratio_min_t']:g}) | {r['ratio_final']:.3f} | {r['events_per_opp_med']:.1f} | {r['ess_age_min']:.2f} | {r['kl_uniform_final_arm']:.3f}/{r['kl_uniform_final_abf']:.3f} | {ks} | {r['c7ax_final_arm']:.3f}/{r['c7ax_final_abf']:.3f} | {r['verdict']} |")
        # support vs torus at equal rate
        pairs = []
        for r in rows:
            if r["method"] == "fr_support":
                u = [x for x in rows if x["method"] == "fr_uniform" and x["fr_rate"] == r["fr_rate"] and x["fr_start_ps"] == r["fr_start_ps"]]
                if u:
                    ru, rs = runs[u[0]["arm"]], runs[r["arm"]]; own = (max(r["fr_start_ps"], 1.0), float(t[-1]))
                    s = paired_stats(integrate(t, ru["e"]["raw"], *own), integrate(t, rs["e"]["raw"], *own)); f = paired_stats(ru["e"]["raw"][-1], rs["e"]["raw"][-1])
                    pairs.append(f"- rate {r['fr_rate']:g}: support vs torus target, own window {100*s['median']:+.2f} % [{100*s['lo']:+.2f}, {100*s['hi']:+.2f}] {int(round(s['win_rate']*s['n']))}/{s['n']}; final {100*f['median']:+.2f} % [{100*f['lo']:+.2f}, {100*f['hi']:+.2f}]")
        if pairs:
            L += ["", "## Support target vs torus target (paired, equal rate)", ""] + pairs
    open(os.path.join(out, "scoreboard.md"), "w").write("\n".join(L) + "\n")
    write_csv(os.path.join(out, "fr.csv"), rows)
    json.dump(dict(root=a.root, reference=a.ref, floor=refs["floor"], predictor=pred, fr={r["arm"]: r for r in rows},
                   arms={n: arm_info(r) for n, r in runs.items()}), open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    if not a.no_figs:
        plt = _style(); fd = os.path.join(out, "figures"); os.makedirs(fd, exist_ok=True)
        names = sorted(runs); cols = dict(zip(names, _colors(len(names))))
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained")
        ax = axes[0, 0]
        for n in names:
            ax.plot(t, np.median(runs[n]["e"]["raw"], 1), color=cols[n], lw=1.8 if arm_info(runs[n])["method"] == "abf" else 1.1, label=n)
        ax.axhline(refs["floor"], color="k", ls=":", lw=0.7); ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel("aligned L2 FES error, raw (kJ/mol)")
        ax.set_title(f"T = {refs['temperature']} K: ABF vs ABF + FR (dotted: reference floor)", fontsize=9); ax.legend(fontsize=6, frameon=False)
        ax = axes[0, 1]
        for r in rows:
            m = r["matched_abf"]; ax.plot(t, np.median(runs[r["arm"]]["e"]["raw"] / runs[m]["e"]["raw"], 1), color=cols[r["arm"]], lw=1.1, label=r["arm"])
        ax.axhline(1, color="k", lw=0.7, ls=":"); ax.axhline(0.9, color="gray", lw=0.6, ls="--"); ax.set_xlabel("t (ps)"); ax.set_ylabel("e_FR / e_ABF (median per-seed ratio)"); ax.legend(fontsize=6, frameon=False)
        ax = axes[1, 0]
        for n in names:
            d = runs[n]["data"]
            ax.plot(t, np.median(d["kl_uniform"], 1), color=cols[n], lw=1.0, label=n + " (torus)")
            if "kl_support" in d:
                ax.plot(t, np.median(d["kl_support"], 1), color=cols[n], lw=1.0, ls="--")
        ax.set_xlabel("t (ps)"); ax.set_ylabel("KL(p_t || target): solid torus-uniform, dashed visited-support"); ax.legend(fontsize=6, frameon=False)
        ax = axes[1, 1]
        for n in names:
            d = runs[n]["data"]
            if "acc_csum" in d:
                cov = (d["acc_csum"][:, :, refs["pack"]["mask8"]] >= arm_info(runs[n])["c_min"]).mean(2)
                ax.plot(t, np.median(cov, 1), color=cols[n], lw=1.0, label=n)
        ax.axhline(EST_FRAC, color="k", ls=":", lw=0.7); ax.set_xlabel("t (ps)"); ax.set_ylabel("fraction of 8 kT mask cells with count >= c_min"); ax.legend(fontsize=6, frameon=False)
        ax.set_title("marginal establishment", fontsize=9)
        fig.savefig(os.path.join(fd, "lowT_curves.png"), dpi=160); fig.savefig(os.path.join(fd, "lowT_curves.pdf")); plt.close(fig)
    print(open(os.path.join(out, "scoreboard.md")).read())


if __name__ == "__main__":
    main()
