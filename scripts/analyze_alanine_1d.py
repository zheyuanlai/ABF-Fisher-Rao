#!/usr/bin/env python
"""Analysis for the 1-D (incomplete-CV) alanine study (docs/ALANINE_1D_CV.md).  Safe on partial data.

    python scripts/analyze_alanine_1d.py [--no-figs]

Per CV (phi, psi): every arm's 1-D FES error against the FULL marginal reference (and, for psi, the
visited-side phi < 0 conditional marginal), the hidden-coordinate conditional distance D_cond, C7ax
discovery and occupancy, FR vs matched ABF with the frozen paired statistics and verdict rules, the
kernel check arm vs the histogram arm, and the 2-D histogram arm marginalised as context.
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
from alanine.basins import from_reference                                           # noqa: E402
from alanine.core1d_ala import marginal_reference_1d                                # noqa: E402
from alanine.metrics_ala import aligned_l2, build_masks                             # noqa: E402

ROOT = os.path.join(SCRIPTS, "..")
REF = os.path.join(ROOT, "results/alanine/reference/reference.npz")
RUN_ROOT = os.path.join(ROOT, "results/alanine_1d")
CTX_2D = os.path.join(ROOT, "results/alanine_histogram/hA_c800_w5/raw")
OUT = os.path.join(RUN_ROOT, "analysis")
W0, W1 = (1.0, 100.0), (5.0, 100.0)
MIN_WALKERS = 200            # samples per pooled CV bin in a time window
POOL = 4                     # 97 cells -> 24 pooled bins of ~15 deg (last cell folded into the last bin)
WINDOWS = ((1.0, 5.0), (5.0, 20.0), (20.0, 50.0), (50.0, 100.0))
AXIS = {"phi": 0, "psi": 1}


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUN_ROOT, "*", "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "final_pmf" not in d.files:
            continue
        meta = json.loads(str(d["meta"]))
        keys = [k for k in d.files if k not in ("meta", "acc_fs", "acc_cs", "marg_hist", "counts")]
        runs[meta["stage"]] = dict(data={k: np.asarray(d[k]) for k in keys}, meta=meta, path=f,
                                   t=np.asarray(d["times"], float))
    return runs


def info(run):
    m = run["meta"]
    return dict(cv=m["cv"], method=m["method"], estimator=m.get("abf_estimator", "histogram"),
                c_min=float(m.get("abf_min_count", 800.0)), warmup_ps=float(m.get("abf_warmup_steps", 5000)) * 1e-3,
                fr_start_ps=float(m.get("fr_start_steps", 0)) * 1e-3, fr_rate=float(m.get("fr_rate", 0.02)),
                fr_every=int(m.get("fr_every", 500)))


def references():
    refd = np.load(REF, allow_pickle=True)
    F2 = refd["F"]
    rmeta = json.load(open(os.path.join(os.path.dirname(REF), "meta.json")))
    kT = float(rmeta["kT_kJ"]); n = int(rmeta["n_grid"])
    grid = -np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n
    phi_neg = (grid < 0)[:, None] & np.ones((1, n), bool)
    out = dict(kT=kT, n=n, grid=grid, F2=F2)
    for cv, ax in AXIS.items():
        F1 = marginal_reference_1d(F2, kT, ax)
        out[cv] = dict(full=F1, pack=build_masks(F1, kT))
        Fv = marginal_reference_1d(F2, kT, ax, mask=phi_neg)
        out[cv]["visited"] = Fv
        out[cv]["pack_visited"] = build_masks(Fv, kT)
    finite = np.isfinite(F2)
    P2 = np.where(finite, np.exp(-(np.where(finite, F2, 0.0) - np.nanmin(F2[finite])) / kT), 0.0)
    out["P2"] = P2 / P2.sum()
    bm, _ = from_reference(REF)
    out["basins"] = bm
    out["c7ax_ref_pop"] = float(bm.population(F2, kT)[bm.names[2]])
    return out


def error_series(runs, refs):
    for run in runs.values():
        cv = info(run)["cv"]
        pmf = run["data"]["pmf"]                       # (T, R, n)
        T, R = pmf.shape[:2]
        pk, pv = refs[cv]["pack"], refs[cv]["pack_visited"]
        e_full = np.zeros((T, R)); e_vis = np.zeros((T, R)); e_u8 = np.zeros((T, R))
        for ti in range(T):
            for r in range(R):
                e_full[ti, r] = aligned_l2(pmf[ti, r], pk["F"], pk["weights"]["equilibrium"])
                e_u8[ti, r] = aligned_l2(pmf[ti, r], pk["F"], pk["weights"]["uniform8"])
                e_vis[ti, r] = aligned_l2(pmf[ti, r], pv["F"], pv["weights"]["equilibrium"])
        run["e"] = dict(full=e_full, visited=e_vis, full_u8=e_u8)
        run["dcond"] = cond_distance(run, refs)


def _pool(a, axis):
    """Sum blocks of POOL cells along ``axis`` (the 97th cell joins the last block)."""
    n = a.shape[axis]; nb = n // POOL
    idx = np.minimum(np.arange(n) // POOL, nb - 1)
    out = np.zeros(a.shape[:axis] + (nb,) + a.shape[axis + 1:])
    for b in range(nb):
        out[(slice(None),) * axis + (b,)] = a.take(np.nonzero(idx == b)[0], axis=axis).sum(axis)
    return out


def cond_distance(run, refs):
    """Hidden-coordinate conditional distance, per time WINDOW and seed: joint walker histograms are
    summed over the saves in the window and pooled to ~15-deg bins on both axes; for every pooled CV
    bin with >= MIN_WALKERS samples, TV(p_walkers(hidden | xi), p_ref(hidden | xi)); equal-weight
    mean over qualifying bins (never weighted by the walker marginal).  Alongside: the multinomial
    sampling floor E[TV] ~ 0.5 sum_i sqrt(2 q_i (1 - q_i) / (pi n)) at the same counts, and the
    hidden-side mass (fraction of walkers with phi > 0, meaningful for the psi CV) vs the reference."""
    cv = info(run)["cv"]
    t = run["t"]
    jh = run["data"]["joint_hist"].astype(float)        # (T, R, n_phi, n_psi)
    if cv == "psi":
        jh = np.swapaxes(jh, 2, 3)                      # -> (T, R, n_cv, n_hidden)
    P2 = refs["P2"] if cv == "phi" else refs["P2"].T
    P2p = _pool(_pool(P2, 0), 1)
    pref = P2p / np.maximum(P2p.sum(1, keepdims=True), 1e-300)
    n_phi_pos = (refs["grid"] > 0)
    ref_mass_pos = float(refs["P2"][n_phi_pos].sum())
    # the proper target under a psi bias: the psi-UNIFORM average of the reference conditional
    # p_ref(phi > 0 | psi) (a flat psi marginal weights every psi equally), per pooled psi bin and overall
    P2T = refs["P2"].T                                                       # (n_psi, n_phi)
    cond_pos = P2T[:, n_phi_pos].sum(1) / np.maximum(P2T.sum(1), 1e-300)     # per psi cell
    cond_pos_pooled = _pool(P2T, 0)[:, n_phi_pos].sum(1) / np.maximum(_pool(P2T, 0).sum(1), 1e-300)
    ref_mass_pos_uniform = float(cond_pos[np.isfinite(cond_pos)].mean())
    T, R = jh.shape[:2]
    out = {}
    for (a, b) in WINDOWS:
        sel = (t >= a - 1e-9) & (t <= b + 1e-9)
        J = jh[sel].sum(0)                              # (R, n_cv, n_hidden)
        Jp = _pool(_pool(J, 1), 2)
        d = np.full(R, np.nan); fl = np.full(R, np.nan); nb = np.zeros(R); mpos = np.full(R, np.nan)
        pos_by_cv = np.full((R, Jp.shape[1]), np.nan)
        for r in range(R):
            cnt = Jp[r].sum(1); ok = cnt >= MIN_WALKERS
            if ok.any():
                pw = Jp[r][ok] / cnt[ok, None]
                d[r] = (0.5 * np.abs(pw - pref[ok]).sum(1)).mean()
                fl[r] = (0.5 * np.sqrt(2.0 * pref[ok] * (1 - pref[ok]) / (np.pi * cnt[ok, None])).sum(1)).mean()
                nb[r] = ok.sum()
            if cv == "psi":
                tot = J[r].sum()
                mpos[r] = J[r][:, n_phi_pos].sum() / tot if tot > 0 else np.nan
                rowtot = _pool(J[r], 0).sum(1)
                pos_by_cv[r] = np.where(rowtot >= MIN_WALKERS, _pool(J[r], 0)[:, n_phi_pos].sum(1) / np.maximum(rowtot, 1), np.nan)
        out[f"[{a:g},{b:g}]"] = dict(dcond=d, floor=fl, nbins=nb, mass_phi_pos=mpos, pos_by_cv=pos_by_cv)
    run["ref_mass_phi_pos"] = ref_mass_pos
    run["ref_mass_phi_pos_uniform"] = ref_mass_pos_uniform
    run["ref_pos_by_cv"] = cond_pos_pooled
    return out


def context_2d(refs, t_ref):
    """The 2-D histogram ABF arm (c800/w5) marginalised over the hidden angle, scored on the 1-D
    references (a 2-D method's implied 1-D result).  Untrusted cells early carry B ~ 0, so this is
    context, honest only once every cell is trusted (~5 ps)."""
    files = glob.glob(os.path.join(CTX_2D, "*.npz"))
    if not files:
        return None
    d = np.load(files[0], allow_pickle=True)
    pmf2 = d["pmf"]; t = np.asarray(d["times"], float)
    if not np.allclose(t, t_ref):
        return None
    kT = refs["kT"]
    out = {}
    for cv, ax in AXIS.items():
        pk = refs[cv]["pack"]
        T, R = pmf2.shape[:2]
        e = np.zeros((T, R))
        for ti in range(T):
            for r in range(R):
                B = pmf2[ti, r] - pmf2[ti, r].min()
                F1 = -kT * np.log(np.exp(-B / kT).sum(axis=1 - ax))
                e[ti, r] = aligned_l2(F1, pk["F"], pk["weights"]["equilibrium"])
        out[cv] = e
    return out


def at(t, series, times):
    return {f"@{tt:g}": float(np.nanmedian(series[int(np.argmin(np.abs(t - tt)))])) for tt in times}


def arm_rows(runs, refs, ctx, t):
    rows = []
    for name, run in sorted(runs.items()):
        i = info(run)
        d = run["data"]
        rec = dict(arm=name, **i, n_seeds=int(run["e"]["full"].shape[1]), ms_per_step=run["meta"].get("ms_per_step"))
        for ro in ("full", "visited", "full_u8"):
            e = run["e"][ro]
            rec[f"IF_W0_{ro}"] = float(np.median(integrate(t, e, *W0)))
            rec[f"IF_W1_{ro}"] = float(np.median(integrate(t, e, *W1)))
            rec[f"final_{ro}"] = float(np.median(e[-1]))
        rec["final_full_over_visited"] = rec["final_full"] / max(rec["final_visited"], 1e-12)
        fh = d["first_hit"][:, 2].astype(float) * 1e-3
        rec["c7ax_first_hit_med_ps"] = float(np.median(np.where(fh < 0, np.nan, fh))) if np.isfinite(np.where(fh < 0, np.nan, fh)).any() else float("nan")
        rec["c7ax_censored"] = int((fh < 0).sum())
        rec.update({f"c7ax{k}": v for k, v in at(t, d["basin_frac"][:, :, 2], (5, 20, 100)).items()})
        for wn, v in run["dcond"].items():
            rec[f"dcond{wn}"] = float(np.nanmedian(v["dcond"])); rec[f"dcond_floor{wn}"] = float(np.nanmedian(v["floor"]))
            rec[f"dcond_nbins{wn}"] = float(np.nanmedian(v["nbins"])); rec[f"mass_phi_pos{wn}"] = float(np.nanmedian(v["mass_phi_pos"]))
        rec["ref_mass_phi_pos"] = run["ref_mass_phi_pos"]
        rec["ref_mass_phi_pos_uniform"] = run["ref_mass_phi_pos_uniform"]
        if i["cv"] == "phi":                          # the phi<0-side reference is meaningful for psi only
            rec["final_visited"] = float("nan"); rec["final_full_over_visited"] = float("nan")
        rec.update({f"kl{k}": v for k, v in at(t, d["kl_uniform"], (1, 5, 20, 100)).items()})
        rec.update({f"T{k}": v for k, v in at(t, np.asarray(d["temperature"], float)[:, None], (1, 5, 100)).items()})
        rec["trust@1"], rec["trust@100"] = float(d["trust_frac"][int(np.argmin(np.abs(t - 1)))]), float(d["trust_frac"][-1])
        if ctx is not None:
            rec["ctx2d_IF_W1_full"] = float(np.median(integrate(t, ctx[i["cv"]], *W1)))
            rec["ctx2d_final_full"] = float(np.median(ctx[i["cv"]][-1]))
        rows.append(rec)
    return rows


def matched_abf(runs, i):
    for n, r in runs.items():
        j = info(r)
        if j["method"] == "abf" and j["cv"] == i["cv"] and j["estimator"] == i["estimator"] and j["c_min"] == i["c_min"] and j["warmup_ps"] == i["warmup_ps"]:
            return n
    return None


def fr_rows(runs, t):
    rows = []
    for name, run in sorted(runs.items()):
        i = info(run)
        if i["method"] == "abf":
            continue
        m = matched_abf(runs, i)
        if m is None:
            print(f"[fr] {name}: no matched ABF arm yet"); continue
        abf = runs[m]
        T = float(t[-1]); own = (max(i["fr_start_ps"], 1.0), T)
        e, ea = run["e"]["full"], abf["e"]["full"]
        rec = dict(arm=name, **i, matched_abf=m, own_window=f"[{own[0]:g},{own[1]:g}]")
        prim = paired_stats(integrate(t, ea, *own), integrate(t, e, *own)); rec.update(flat_stats("dIF_own", prim))
        rec.update(flat_stats("dIF_W1", paired_stats(integrate(t, ea, *W1), integrate(t, e, *W1))))
        rec.update(flat_stats("dIF_own_visited", paired_stats(integrate(t, abf["e"]["visited"], *own), integrate(t, run["e"]["visited"], *own))))
        fin = paired_stats(ea[-1], e[-1]); rec.update(flat_stats("final", fin))
        rec.update(flat_stats("final_visited", paired_stats(abf["e"]["visited"][-1], run["e"]["visited"][-1])))
        if i["cv"] == "phi":                          # the phi<0-side reference is defined for psi only
            for k in list(rec):
                if "visited" in k:
                    rec[k] = float("nan")
        last = list(run["dcond"])[-1]
        rec.update(flat_stats("dcond_final", paired_stats(abf["dcond"][last]["dcond"], run["dcond"][last]["dcond"])))
        sp, e0 = speedups(t, ea, e, T, own[0])
        for k, v in sp.items():
            rec[f"S_{k}"] = v["speedup"]
        d = run["data"]; N = int(run["meta"]["n_replicas"])
        n_opp = max(int((int(run["meta"]["n_steps"]) - int(round(i["fr_start_ps"] * 1000))) // i["fr_every"]) + 1, 1)
        ev = d["total_events"].astype(float); sel = t >= own[0] - 1e-9
        rec["events_per_opp_med"] = float(np.median(ev)) / n_opp
        rec["event_frac_per_opp_max"] = float(ev.max()) / N / n_opp
        rec["ess_age_min"] = float(np.nanmin(d["ess_age"][sel])); rec["wmax_max"] = float(np.nanmax(d["wmax"][sel]))
        floors_ok = rec["ess_age_min"] >= FLOORS["ess"] and rec["wmax_max"] <= FLOORS["wmax"] and rec["event_frac_per_opp_max"] < FLOORS["event_frac"]
        rec["floors_ok"] = bool(floors_ok)
        rec["c7ax_censored_arm"], rec["c7ax_censored_abf"] = int((d["first_hit"][:, 2] < 0).sum()), int((abf["data"]["first_hit"][:, 2] < 0).sum())
        rec["c7ax_final_arm"], rec["c7ax_final_abf"] = float(np.median(d["basin_frac"][-1, :, 2])), float(np.median(abf["data"]["basin_frac"][-1, :, 2]))
        rec["kl_final_arm"], rec["kl_final_abf"] = float(np.median(d["kl_uniform"][-1])), float(np.median(abf["data"]["kl_uniform"][-1]))
        ratio = np.median(e / ea, 1); rec["ratio_min"], rec["ratio_min_t"] = float(ratio.min()), float(t[int(np.argmin(ratio))])
        rec["verdict"] = verdict(prim, fin, floors_ok)
        rows.append(rec)
    return rows


def estimator_rows(runs, t):
    rows = []
    for cv in AXIS:
        h = [n for n, r in runs.items() if info(r)["cv"] == cv and info(r)["method"] == "abf" and info(r)["estimator"] == "histogram"]
        k = [n for n, r in runs.items() if info(r)["cv"] == cv and info(r)["method"] == "abf" and info(r)["estimator"] == "kernel"]
        if h and k:
            a, b = runs[h[0]], runs[k[0]]
            rec = dict(cv=cv, histogram=h[0], kernel=k[0])
            rec.update(flat_stats("kernel_vs_hist_IF_W1_full", paired_stats(integrate(t, a["e"]["full"], *W1), integrate(t, b["e"]["full"], *W1))))
            rec.update(flat_stats("kernel_vs_hist_final_full", paired_stats(a["e"]["full"][-1], b["e"]["full"][-1])))
            last = list(a["dcond"])[-1]
            rec.update(flat_stats("kernel_vs_hist_dcond_final", paired_stats(a["dcond"][last]["dcond"], b["dcond"][last]["dcond"])))
            rows.append(rec)
    return rows


def pct(r, k):
    return f"{100 * r[k + '_median']:+6.2f} % [{100 * r[k + '_lo']:+6.2f}, {100 * r[k + '_hi']:+6.2f}] {int(round(r[k + '_win'] * r[k + '_n']))}/{r[k + '_n']}"


def scoreboard(rows_a, rows_f, rows_e, refs, out):
    L = ["# Alanine 1-D (incomplete CV) study: scoreboard", "",
         f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Errors: equilibrium-weighted aligned L2 of the 1-D PMF vs the FULL marginal reference (8 kT mask); 'visited' = vs the phi < 0 conditional marginal. D_cond = equal-weight mean TV of p(hidden | xi) over CV bins with >= {MIN_WALKERS} walkers. Reference C7ax population {100 * refs['c7ax_ref_pop']:.2f} %.", "",
         "## Arms", "",
         "| arm | CV | method | estimator | FR start | rate | I_F W1 full | final full | final visited (psi) | full/visited | C7ax first hit (ps) | censored | C7ax @20/100 | D_cond by window (floor) | mass phi>0 by window (target = psi-uniform average of the reference conditional) | KL @1/100 | T @1 | ctx 2-D final | ms/step |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    wins = [f"[{a:g},{b:g}]" for a, b in WINDOWS]
    for r in rows_a:
        dc = " ".join(f"{r[f'dcond{w}']:.3f}({r[f'dcond_floor{w}']:.3f})" for w in wins)
        mp = " ".join(f"{r[f'mass_phi_pos{w}']:.3f}" for w in wins) + f" (target {r['ref_mass_phi_pos_uniform']:.3f}; equilibrium {r['ref_mass_phi_pos']:.3f})" if r["cv"] == "psi" else "-"
        L.append(f"| {r['arm']} | {r['cv']} | {r['method']} | {r['estimator']} | {r['fr_start_ps']:g} | {r['fr_rate']:g} | {r['IF_W1_full']:.3f} | {r['final_full']:.4f} | {r['final_visited']:.4f} | {r['final_full_over_visited']:.2f} | {r['c7ax_first_hit_med_ps']:.2f} | {r['c7ax_censored']}/{r['n_seeds']} | {r['c7ax@20']:.4f}/{r['c7ax@100']:.4f} | {dc} | {mp} | {r['kl@1']:.2f}/{r['kl@100']:.2f} | {r['T@1']:.1f} | {r.get('ctx2d_final_full', float('nan')):.4f} | {r['ms_per_step']:.2f} |")
    if rows_f:
        L += ["", "## FR vs matched ABF (own window, full reference)", "",
              "| arm | CV | rate | dI_F own (full) | dI_F W1 | final (full) | dI_F own (visited) | D_cond final | C7ax censored arm/abf | C7ax final arm/abf | events/opp | ESS min | ratio min (t) | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows_f:
            vis = pct(r, 'dIF_own_visited') if np.isfinite(r['dIF_own_visited_median']) else "-"
            L.append(f"| {r['arm']} | {r['cv']} | {r['fr_rate']:g} | {pct(r, 'dIF_own')} | {pct(r, 'dIF_W1')} | {pct(r, 'final')} | {vis} | {pct(r, 'dcond_final')} | {r['c7ax_censored_arm']}/{r['c7ax_censored_abf']} | {r['c7ax_final_arm']:.4f}/{r['c7ax_final_abf']:.4f} | {r['events_per_opp_med']:.1f} | {r['ess_age_min']:.2f} | {r['ratio_min']:.3f} ({r['ratio_min_t']:g}) | {r['verdict']} |")
    if rows_e:
        L += ["", "## Estimator check: kernel ABF vs histogram ABF (paired)", ""]
        for r in rows_e:
            L.append(f"- {r['cv']}: I_F W1 {pct(r, 'kernel_vs_hist_IF_W1_full')}; final {pct(r, 'kernel_vs_hist_final_full')}; D_cond final {pct(r, 'kernel_vs_hist_dcond_final')}")
    open(os.path.join(out, "scoreboard.md"), "w").write("\n".join(L) + "\n")


def figures(runs, refs, ctx, t, out):
    plt = _style()
    fd = os.path.join(out, "figures"); os.makedirs(fd, exist_ok=True)
    for cv in AXIS:
        names = sorted(n for n, r in runs.items() if info(r)["cv"] == cv)
        if not names:
            continue
        cols = dict(zip(names, _colors(len(names))))
        ncol = 3 if cv == "psi" else 2
        fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 6.8), layout="constrained")
        ax = axes[0, 0]
        for n in names:
            ax.plot(t, np.median(runs[n]["e"]["full"], 1), color=cols[n], lw=1.2, label=n)
            if cv == "psi":
                ax.plot(t, np.median(runs[n]["e"]["visited"], 1), color=cols[n], lw=0.8, ls="--")
        if ctx is not None:
            ax.plot(t, np.median(ctx[cv], 1), "k:", lw=1.2, label="2-D histogram ABF, marginalised (context)")
        ax.set_yscale("log"); ax.set_xlabel("t (ps)"); ax.set_ylabel(f"aligned L2 error of F({cv}) (kJ/mol)")
        ax.set_title(f"{cv} alone: vs full marginal (solid){', vs phi<0 conditional (dashed)' if cv == 'psi' else ''}", fontsize=8); ax.legend(fontsize=6, frameon=False)
        ax = axes[0, 1]
        xs = [0.5 * (a + b) for a, b in WINDOWS]
        for n in names:
            dc = runs[n]["dcond"]
            ax.plot(xs, [np.nanmedian(dc[w]["dcond"]) for w in dc], "o-", color=cols[n], lw=1.2, ms=4, label=n)
            ax.plot(xs, [np.nanmedian(dc[w]["floor"]) for w in dc], ":", color=cols[n], lw=0.8)
        ax.set_xlabel("window centre (ps)"); ax.set_ylabel("D_cond = TV of p(hidden | xi) vs reference")
        ax.text(0.98, 0.02, "dotted: multinomial sampling floor", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.set_ylim(0, None); ax.legend(fontsize=6, frameon=False); ax.set_title("hidden coordinate: conditional distance by window", fontsize=9)
        ax = axes[1, 0]
        for n in names:
            ax.plot(t, np.median(runs[n]["data"]["basin_frac"][:, :, 2], 1), color=cols[n], lw=1.2, label=n)
        ax.axhline(refs["c7ax_ref_pop"], color="k", ls=":", lw=0.8); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.0)
        ax.set_xlabel("t (ps)"); ax.set_ylabel("C7ax occupancy (dotted: reference)"); ax.set_title("rare-basin discovery through the hidden angle", fontsize=9)
        ax = axes[1, 1]
        g = np.degrees(refs["grid"])
        Ff = refs[cv]["full"]; Fv = refs[cv]["visited"]
        ax.plot(g, Ff - np.nanmin(Ff[np.isfinite(Ff)]), "k-", lw=1.6, label="reference (full marginal)")
        if cv == "psi":
            ax.plot(g, Fv - np.nanmin(Fv[np.isfinite(Fv)]), "k--", lw=1.0, label="reference, phi<0 side")
        for n in names:
            P = np.median(runs[n]["data"]["final_pmf"], 0)
            pk = refs[cv]["pack"]; w = pk["weights"]["equilibrium"]; ok = w > 0
            c = ((P - pk["F"])[ok] * w[ok]).sum() / w[ok].sum()
            ax.plot(g, P - c, color=cols[n], lw=1.0, label=n)
        ax.set_xlabel(f"{cv} (deg)"); ax.set_ylabel("F (kJ/mol)"); ax.set_ylim(-2, 45); ax.legend(fontsize=6, frameon=False)
        ax.set_title("final PMF (median over seeds, aligned on the equilibrium weight)", fontsize=9)
        if cv == "psi":
            nb = len(runs[names[0]]["ref_pos_by_cv"])
            gp = -180 + (np.arange(nb) + 0.5) * 360.0 / nb
            ax = axes[0, 2]
            ax.plot(gp, runs[names[0]]["ref_pos_by_cv"], "k-", lw=1.6, label="reference p(phi>0 | psi)")
            for n in names:
                last = list(runs[n]["dcond"])[-1]
                ax.plot(gp, np.nanmedian(runs[n]["dcond"][last]["pos_by_cv"], 0), "o-", color=cols[n], ms=3, lw=1.0, label=f"{n} [50,100] ps")
            ax.set_xlabel("psi (deg)"); ax.set_ylabel("fraction of walkers at phi > 0"); ax.set_yscale("log"); ax.legend(fontsize=6, frameon=False)
            ax.set_title("hidden phi: mass on the C7ax side, per psi bin", fontsize=9)
            ax = axes[1, 2]
            xs = [0.5 * (a + b) for a, b in WINDOWS]
            for n in names:
                dc = runs[n]["dcond"]
                ax.plot(xs, [np.nanmedian(dc[w]["mass_phi_pos"]) for w in dc], "o-", color=cols[n], ms=4, lw=1.2, label=n)
            ax.axhline(runs[names[0]]["ref_mass_phi_pos_uniform"], color="k", ls="-", lw=1.0, label="target (psi-uniform average)")
            ax.axhline(runs[names[0]]["ref_mass_phi_pos"], color="k", ls=":", lw=0.8, label="equilibrium mass")
            ax.set_xlabel("window centre (ps)"); ax.set_ylabel("walker mass at phi > 0"); ax.legend(fontsize=6, frameon=False)
            ax.set_title("hidden phi: total mass on the C7ax side", fontsize=9)
        fig.savefig(os.path.join(fd, f"{cv}_panels.png"), dpi=160); fig.savefig(os.path.join(fd, f"{cv}_panels.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-figs", action="store_true"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    runs = load_runs()
    if not runs:
        print("no runs"); return
    refs = references()
    t = runs[next(iter(runs))]["t"]
    error_series(runs, refs)
    ctx = context_2d(refs, t)
    rows_a = arm_rows(runs, refs, ctx, t); rows_f = fr_rows(runs, t); rows_e = estimator_rows(runs, t)
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, "arms.csv"), rows_a); write_csv(os.path.join(a.out, "fr.csv"), rows_f)
    json.dump(dict(reference=REF, kT=refs["kT"], c7ax_ref_pop=refs["c7ax_ref_pop"], windows=dict(W0=W0, W1=W1), min_walkers=MIN_WALKERS,
                   arms={r["arm"]: r for r in rows_a}, fr={r["arm"]: r for r in rows_f}, estimator=rows_e),
              open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)
    scoreboard(rows_a, rows_f, rows_e, refs, a.out)
    if not a.no_figs:
        figures(runs, refs, ctx, t, a.out)
    print(open(os.path.join(a.out, "scoreboard.md")).read())


if __name__ == "__main__":
    main()
