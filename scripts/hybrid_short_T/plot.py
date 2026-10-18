#!/usr/bin/env python
"""Figures A-F of the hybrid short-T wall-clock study (prereg section 7), PNG + PDF in figures/hybrid_short_T/.

    python scripts/hybrid_short_T/plot.py [--only A B ...]

Inputs: results/hybrid_short_T/bench/summary.json, routing/routing.json, analysis/summary.json and
analysis/curves_<kind>_<system>.npz.  Style: scripts/equal_budget/short_T_ladder.py (same colours and rcParams);
every figure passes scripts/equal_budget/fig_legibility.check_figure_safe or the failure is printed.

A  CPU throughput vs N: S1 and the best walker-parallel P_t for ABF and FR (median, min-max over the benchmark
   repeats), the frozen route per N (filled marker), the chosen thread count; H200 NOT MEASURED (user instruction
   2026-10-18) -- the 2026-10-10 Experiment III torch-eager H200 points at N = 512 are drawn as historical markers
B  all N-T settings, e_F vs MEASURED wall seconds: per N the median over seeds of e_F at each save against the median
   wall stamp of that save, IQR band, ABF and ABF + FR, backend, threshold; N = 1 is ABF only
C  the same vs physical time
D  best-N frontier: per method the minimum over N of the median e_F at wall budget w (dashed where the best
   configuration's runs have completed and their final estimate is carried forward)
E  time to the mid threshold per N (median W_mid, bootstrap 95 % CI of the median; censored medians drawn at the
   top edge with an upward triangle), the best N of each method starred
F  synthesis: best hybrid ABF vs best hybrid ABF + FR (exploration and, if run, confirmation) and the fixed largest N
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402
import benchmark as BM  # noqa: E402

sys.path.insert(0, os.path.join(H.ROOT, "scripts", "equal_budget"))
import fig_legibility as LG  # noqa: E402
import short_T_ladder as ST  # noqa: E402

C_ABF, C_FR = ST.C_ABF, ST.C_FR
INK, INK2, MUTED, GRID = ST.INK, ST.INK2, ST.MUTED, ST.GRID
E_UNIT = {"gateway": "reduced units", "lta300": "kJ/mol", "lta150": "kJ/mol"}
AN = os.path.join(H.OUT_ROOT, "analysis")
FIG = H.FIG_ROOT
# Experiment III (docs/mechanism/EXP3_PARALLEL_EFFICIENCY.md; results/mechanism/parallel_benchmark/analysis/
# summary.json cost_projection): torch eager on one H200 at N = 512, us per step (gateway: bench_step_costs only)
HIST_H200_US_PER_STEP = {"gateway": dict(abf=590.0, fr=None), "lta300": dict(abf=683.98, fr=852.96),
                         "lta150": dict(abf=681.08, fr=857.22)}


def save(fig, name, checks):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p + ".png", dpi=170)
    fig.savefig(p + ".pdf")
    c = LG.check_figure_safe(fig)
    checks[name] = c["status"]
    if c["status"] != "pass":
        print(LG.one_line(name, c))
    plt.close(fig)
    return p + ".png"


def load():
    bench = C.read_json(os.path.join(H.OUT_ROOT, "bench", "summary.json"))
    routing = C.read_json(os.path.join(H.OUT_ROOT, "routing", "routing.json"))
    summ = C.read_json(os.path.join(AN, "summary.json")) if os.path.exists(os.path.join(AN, "summary.json")) else None
    return bench, routing, summ


def curves(kind, system):
    p = os.path.join(AN, f"curves_{kind}_{system}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fnum(x, p=3):
    if x is None:
        return "n/a"
    x = float(x)
    if math.isinf(x):
        return "cens."
    if math.isnan(x):
        return "n/a"
    return f"{x:.{p}g}"


# ================================================================================================ A
def fig_A(bench, routing, checks):
    summ = bench["configs"]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.9), layout="constrained")
    for ax, s in zip(axs, H.SYSTEMS):
        spec = H.system_spec(s)
        Ns = sorted(spec["N_ladder"])
        for m, col, dx in (("abf", C_ABF, 0.97), ("fr", C_FR, 1.03)):
            xs, s1, s1lo, s1hi, bp, bplo, bphi, bt = [], [], [], [], [], [], [], []
            for N in Ns:
                if m == "fr" and N == 1:
                    continue
                k1 = f"{s}|{H.config_name(m, N, 'serial', 1)}"
                if k1 not in summ or not summ[k1]["n"]:
                    continue
                a = summ[k1]
                xs.append(N)
                s1.append(1 / a["c_ss_median"])
                s1lo.append(1 / a["c_ss_max"])
                s1hi.append(1 / a["c_ss_min"])
                par = [BM.entry(summ, s, m, N, "parallel", t) for t in (1, 2, 4, 8, 16) if t <= N]   # A2.5 windows
                par = [p for p in par if p and p["n"]]
                b = min(par, key=lambda p: p["c_ss_median"])
                bp.append(1 / b["c_ss_median"])
                bplo.append(1 / b["c_ss_max"])
                bphi.append(1 / b["c_ss_min"])
                bt.append(b["threads"])
            x = np.array(xs, float) * dx
            ax.errorbar(x, s1, yerr=[np.array(s1) - s1lo, np.array(s1hi) - s1], color=col, lw=1.3, ls=(0, (4, 2)),
                        marker="o", ms=4.5, mfc="white", capsize=0, elinewidth=0.9, zorder=3)
            ax.errorbar(x, bp, yerr=[np.array(bp) - bplo, np.array(bphi) - bp], color=col, lw=1.7, marker="s",
                        ms=4.5, capsize=0, elinewidth=0.9, zorder=3)
            if m == "abf":
                for xi, yi, t in zip(x, bp, bt):
                    ax.annotate(f"t{t}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center",
                                fontsize=6.8, color=INK2)
            hist = HIST_H200_US_PER_STEP[s][m]
            if hist is None:            # gateway FR: Exp III never measured it (its 590 us is the ABF step cost)
                continue
            ax.plot([512 * (0.97 if m == "abf" else 1.03)], [512 / (hist * 1e-6)], marker="D", ms=6, mfc="none",
                    mec=col, mew=1.4, ls="none", zorder=4)
        for N in Ns:
            r = routing["routes"][s][str(N)]["common"]
            ax.axvspan(N / 1.41, N * 1.41, color=GRID if r["impl"] == "serial" else "#d9e8d0", alpha=0.45, lw=0,
                       zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("replicas N")
        ax.set_ylabel("walker-steps per second (steady state)")
        ax.set_title(f"{spec['label']}", loc="left")
        ax.text(0.02, 0.97, "H200: NOT MEASURED in this study\n(all four H200s busy with other users' jobs;\n"
                "user instruction: CPU only). Open diamonds:\nhistorical Exp III torch-eager H200, N 512\n(gateway: step-cost estimate, ABF only)",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.2, color=INK2,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
    handles = [Line2D([], [], color=C_ABF, lw=1.3, ls=(0, (4, 2)), marker="o", mfc="white", label="ABF, 1 core (S1)"),
               Line2D([], [], color=C_FR, lw=1.3, ls=(0, (4, 2)), marker="o", mfc="white",
                      label="ABF + FR, 1 core (S1)"),
               Line2D([], [], color=C_ABF, lw=1.7, marker="s", label="ABF, best walker-parallel P_t (t <= 16)"),
               Line2D([], [], color=C_FR, lw=1.7, marker="s", label="ABF + FR, best walker-parallel P_t"),
               Line2D([], [], color=INK2, marker="D", mfc="none", ls="none", label="H200 torch eager (historical)"),
               matplotlib.patches.Patch(color="#d9e8d0", alpha=0.6, label="frozen route: parallel"),
               matplotlib.patches.Patch(color=GRID, alpha=0.6, label="frozen route: S1")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Figure A. CPU throughput vs N (median over 3 benchmark repeats, bars min-max); route = ABF timing, "
                 ">= 5 % faster and every repeat faster", fontsize=10.5, x=0.01, ha="left")
    return save(fig, "A_throughput_crossover", checks)


# ================================================================================================ B / C
def _median_curve(e, wall):
    return np.median(wall, 0), np.median(e, 0), np.percentile(e, 25, 0), np.percentile(e, 75, 0)


def fig_BC(system, summ, routing, checks, xkind="wall"):
    cv = curves("prod", system)
    if cv is None:
        return None
    res = summ["results"][system]
    spec = H.system_spec(system)
    rows = {r["N"]: r for r in res["per_N"]}
    Ns = spec["N_ladder"]
    nc = 4
    nr = int(math.ceil(len(Ns) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.15 * nc, 3.15 * nr + 1.0), sharey=True, squeeze=False,
                            layout="constrained")
    eps = spec["thresholds"]["mid"]
    for i, N in enumerate(Ns):
        ax = axs.flat[i]
        row = rows.get(N, {})
        xmax = 0.0
        for m, col in (("abf", C_ABF), ("fr", C_FR)):
            tag = f"{m}{N}"
            if f"{tag}_e" not in cv:
                continue
            e, wall = cv[f"{tag}_e"], cv[f"{tag}_wall"]
            x = np.median(wall, 0) if xkind == "wall" else cv[f"{tag}_t"]
            med, lo, hi = np.median(e, 0), np.percentile(e, 25, 0), np.percentile(e, 75, 0)
            ax.fill_between(x, lo, hi, color=col, alpha=0.17, lw=0, zorder=2)
            ax.plot(x, med, color=col, lw=1.7, zorder=3)
            xmax = max(xmax, float(x[-1]))
            if xkind == "wall":
                W = row.get(f"W_mid_{m}", {}).get("median", math.inf)
                if W is not None and math.isfinite(float(W)):
                    ax.axvline(float(W), color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(eps, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.set_yscale("log")
        if xkind == "wall":                      # log wall axis: crossings at ~1-10 s, runs up to ~250 s
            ax.set_xscale("log")
            ax.set_xlim(0.03, xmax * 1.3 if xmax > 0 else 1)
        else:
            ax.set_xlim(0, xmax * 1.02 if xmax > 0 else 1)
        be = row.get("backend", "?")
        ax.set_title(f"N = {N},  T_N = {row.get('T_N', H.T_N(spec, N)):.4g},  {be}", loc="left", fontsize=9.6)
        wa, wf = row.get("W_mid_abf", {}), row.get("W_mid_fr", {})

        def wtxt(w):
            v = w.get("median", math.nan)
            v = math.inf if v in ("inf", None) and w.get("n_censored", 0) * 2 > w.get("n", 0) else v
            return ("median censored" if (isinstance(v, (int, float)) and math.isinf(v)) or v == "inf"
                    else f"{fnum(v)} s")
        txt = f"W_mid ABF {wtxt(wa)} [{wa.get('n_censored', 0)}/{wa.get('n', 0)} cens.]"
        if N > 1 and wf:
            pw = row.get("pair_W_mid", {})
            r = pw.get("ratio")
            txt += f"\nW_mid FR {wtxt(wf)} [{wf.get('n_censored', 0)}/{wf.get('n', 0)} cens.]"
            if r is None or (isinstance(r, float) and math.isnan(r)):
                txt += "\nFR/ABF undefined (both medians censored)"
            elif float(r) == 0.0:
                txt += "\nFR/ABF 0 (ABF median censored)"
            else:
                txt += (f"\nFR/ABF {fnum(r, 2)} [{fnum(pw.get('lo', math.nan), 2)}, "
                        f"{fnum(pw.get('hi', math.nan), 2)}]")
        else:
            txt += "\nABF only (no FR at N = 1)"
        if xkind == "wall":
            ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=INK2,
                    bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=5)
        if i % nc == 0:
            ax.set_ylabel(f"e_F ({E_UNIT[system]})")
        ax.set_xlabel("measured wall-clock (s since the timed start)" if xkind == "wall" else
                      "physical time t (t.u.)")
    for ax in axs.flat[len(Ns):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=C_ABF, lw=1.7, label="ABF (median over seeds, IQR band)"),
               Line2D([], [], color=C_FR, lw=1.7, label="ABF + FR (median, IQR)"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label=f"mid threshold e_F = {eps:g}")]
    if xkind == "wall":
        handles.append(Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="median W_mid (persistent)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    what = ("MEASURED wall-clock (x = median wall stamp of each save over seeds; no projection)" if xkind == "wall"
            else "physical time (algorithmic view, hardware-independent)")
    fig.suptitle(f"Figure {'B' if xkind == 'wall' else 'C'}. {spec['label']}: every N-T setting at the shortened "
                 f"budget B = {spec['B']:.4g} walker-steps, e_F vs {what}", fontsize=10.2, x=0.01, ha="left")
    return save(fig, f"{'B' if xkind == 'wall' else 'C'}_{system}_all_NT_{'wallclock' if xkind == 'wall' else 'physical_time'}",
                checks)


# ================================================================================================ D
def fig_D(summ, checks):
    fig = plt.figure(figsize=(15.5, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.0])
    for c, s in enumerate(H.SYSTEMS):
        res = summ["results"].get(s)
        if not res:
            continue
        ax = fig.add_subplot(gs[0, c])
        axn = fig.add_subplot(gs[1, c], sharex=ax)
        fr = res["frontier"]
        w = np.array(fr["w"])
        for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
            if m not in fr:
                continue
            b = np.array(fr[m]["best"])
            done = np.array(fr[m]["completed_frac"]) >= 0.5
            ax.plot(np.where(~done, w, np.nan), np.where(~done, b, np.nan), color=col, lw=1.8, label=lab)
            ax.plot(np.where(done, w, np.nan), np.where(done, b, np.nan), color=col, lw=1.3, ls=(0, (4, 2)))
            Nb = np.array(fr[m]["best_N"], float)
            axn.step(w, np.log2(Nb) + (0.12 if m == "fr" else -0.12), where="post", color=col, lw=1.4)
        ax.axhline(res["thresholds"]["mid"], color=MUTED, lw=0.9, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(f"best median e_F ({E_UNIT[s]})")
        ax.set_title(res["label"], loc="left")
        ax.tick_params(labelbottom=False)
        axn.set_yticks(range(0, 12, 2))
        axn.set_yticklabels([str(2 ** k) for k in range(0, 12, 2)], fontsize=7.5)
        axn.set_ylabel("best N")
        axn.set_xlabel("wall-clock budget w (s)")
    handles = [Line2D([], [], color=C_ABF, lw=1.8, label="ABF: best N at each w (still sampling)"),
               Line2D([], [], color=C_FR, lw=1.8, label="ABF + FR: best N at each w"),
               Line2D([], [], color=INK2, lw=1.3, ls=(0, (4, 2)),
                      label="dashed: best configuration completed; final estimate carried forward"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="mid threshold")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Figure D. Descriptive best-of-N wall-clock frontier (measured runs; selection over N on the same "
                 "seeds: not a confirmatory claim). Lower strips: the minimising N", fontsize=10.5, x=0.01, ha="left")
    return save(fig, "D_wallclock_frontier", checks)


# ================================================================================================ E
def _boot_med(x, rng, n=4000):
    x = np.asarray(x, float)
    idx = rng.integers(0, x.size, (n, x.size))
    S = np.sort(x[idx], 1)
    k = x.size
    p = 0.5 * (k - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    a, b = S[:, lo], S[:, hi]
    with np.errstate(invalid="ignore"):
        v = np.where(np.isinf(a) | np.isinf(b), np.inf, a + (b - a) * (p - lo))
    return np.percentile(v[np.isfinite(v)], [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan), \
        float(np.mean(np.isinf(v)))


def fig_E(summ, checks, kind="prod"):
    rng = np.random.default_rng(20261018)
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.7), layout="constrained")
    for ax, s in zip(axs, H.SYSTEMS):
        cv = curves(kind, s)
        res = summ["results"].get(s)
        if cv is None or not res:
            ax.set_visible(False)
            continue
        spec = H.system_spec(s)
        top = 0.0
        pts = {}
        for m, col, dx in (("abf", C_ABF, 0.96), ("fr", C_FR, 1.04)):
            for N in spec["N_ladder"]:
                k = f"{m}{N}_W_mid"
                if k not in cv:
                    continue
                W = cv[k]
                med = _inf_median(W)
                if math.isfinite(med):
                    (lo_, hi_), _ = _boot_med(W, rng)
                    top = max(top, med, float(hi_) if np.isfinite(hi_) else med)
                pts[(m, N)] = (W, col, dx)
        top *= 2.0
        for (m, N), (W, col, dx) in pts.items():
            med = _inf_median(W)
            (lo, hi), fcens = _boot_med(W, rng)
            x = N * dx
            if math.isfinite(med):
                ax.errorbar([x], [med], yerr=[[med - lo], [max(hi - med, 0)]] if np.isfinite(hi) else None,
                            color=col, marker="o", ms=5, capsize=0, elinewidth=1.0)
                if not np.isfinite(hi) or fcens > 0.025:
                    ax.plot([x, x], [med, top], color=col, lw=0.8, ls=(0, (1, 1.5)))
            else:
                ax.plot([x], [top], color=col, marker="^", ms=7, mfc="white")
            nc = int(np.sum(np.isinf(W)))
            if nc:
                ax.annotate(f"{nc}", (x, top if not math.isfinite(med) else med), textcoords="offset points",
                            xytext=(-7 if m == "abf" else 7, -11 if not math.isfinite(med) else 5),
                            ha="right" if m == "abf" else "left", fontsize=6.3, color=col)
        for m, col, mk in (("abf", C_ABF, "*"), ("fr", C_FR, "*")):
            b = res["best"]["mid"].get(m)
            if b and b.get("best_N") is not None and math.isfinite(b["best_value"]):
                ax.plot([b["best_N"] * (0.96 if m == "abf" else 1.04)], [b["best_value"]], marker=mk, ms=15,
                        mfc="none", mec=col, mew=1.5, ls="none", zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(top=top * 1.35)
        ax.set_xticks(sorted(spec["N_ladder"]))
        ax.set_xticklabels([str(n) for n in sorted(spec["N_ladder"])], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlabel("replicas N (route per N in Figure A)")
        ax.set_ylabel("W_mid (measured s)")
        R = res["best"]["mid"].get("R_wall", {})
        ax.set_title(f"{res['label']}\nexploratory best FR / best ABF = {fnum(R.get('point', math.nan))} "
                     f"[{fnum((R.get('ci95') or [math.nan])[0])}, {fnum((R.get('ci95') or [0, math.nan])[1])}]",
                     loc="left", fontsize=9.5)
    handles = [Line2D([], [], color=C_ABF, marker="o", ls="none", label="ABF median W_mid (95 % CI of the median)"),
               Line2D([], [], color=C_FR, marker="o", ls="none", label="ABF + FR"),
               Line2D([], [], color=INK2, marker="^", mfc="white", ls="none",
                      label="median censored (> half the runs never sustain the threshold)"),
               Line2D([], [], color=INK2, marker="*", mfc="none", ms=12, ls="none", label="best N of each method"),
               Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="CI reaches censoring"),
               Line2D([], [], color="none", label="small numbers: censored runs")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Figure E. Measured wall-clock time to the mid threshold at every N (exploration seeds)",
                 fontsize=10.5, x=0.01, ha="left")
    return save(fig, "E_time_to_accuracy", checks)


def _inf_median(W):
    import eqb_metrics as M
    return M.quantile_inf(np.asarray(W, float), 50)


# ================================================================================================ F
def fig_F(summ, checks):
    rng = np.random.default_rng(7)
    fig, axs = plt.subplots(1, 3, figsize=(18.5, 5.8), layout="constrained")
    for ax, s in zip(axs, H.SYSTEMS):
        res = summ["results"].get(s)
        cv = curves("prod", s)
        if not res or cv is None:
            ax.set_visible(False)
            continue
        spec = H.system_spec(s)
        rows = {r["N"]: r for r in res["per_N"]}
        b = res["best"]["mid"]
        items = []                                   # (label, W array, colour, marker, group)
        for m in ("abf", "fr"):
            if b.get(m) and b[m].get("best_N") is not None:
                N = int(b[m]["best_N"])
                items.append((f"{'ABF' if m == 'abf' else 'FR'} {N}\n{rows[N]['backend']}\nexplor.",
                              cv[f"{m}{N}_W_mid"], C_ABF if m == "abf" else C_FR, "o", "exploration"))
        cf = curves("confirm", s)
        conf = (summ.get("confirm") or {}).get(s)
        if cf is not None and conf:
            for m in ("abf", "fr"):
                arms = sorted((k for k in conf["arms"] if k.startswith(m) and k[len(m):].isdigit()),
                              key=lambda k: int(k[len(m):]))
                for k in arms:
                    if f"{k}_W_mid" in cf:
                        N = int(k[len(m):])
                        items.append((f"{'ABF' if m == 'abf' else 'FR'} {N}\n{conf['arms'][k]['backend']}\nconfirm",
                                      cf[f"{k}_W_mid"], C_ABF if m == "abf" else C_FR, "s", "confirmation"))
        ys = []
        for i, (lab, W, col, mk, grp) in enumerate(items):
            med = _inf_median(W)
            (lo, hi), fc = _boot_med(W, rng)
            ys.append(med)
            if math.isfinite(med):
                ax.errorbar([i], [med], yerr=[[med - lo], [max(hi - med, 0)]] if np.isfinite(hi) else None,
                            color=col, marker=mk, ms=7, capsize=3, elinewidth=1.2, mfc=col if grp == "confirmation"
                            else "white", mew=1.4)
            ncen = int(np.sum(np.isinf(W)))
            ax.annotate(f"{ncen}/{W.size} cens." if ncen else f"n={W.size}", (i, med if math.isfinite(med) else 1),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=6.8, color=INK2)
        n_ex = sum(1 for it in items if it[4] == "exploration")
        if n_ex and n_ex < len(items):
            ax.axvline(n_ex - 0.5, color=MUTED, lw=0.9, ls=(0, (3, 2)))
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([it[0] for it in items], fontsize=7.2)
        ax.set_xlabel("exploration best | confirmation arms")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_ylabel("W_mid (s): median, 95 % CI")
        R = b.get("R_wall", {})
        t = (f"{res['label']}\nexploratory R_wall = {fnum(R.get('point', math.nan))} "
             f"[{fnum((R.get('ci95') or [math.nan])[0])}, {fnum((R.get('ci95') or [0, math.nan])[1])}]")
        if conf and conf.get("R_conf"):
            u, e = conf["R_conf"], conf["R_conf_e2e"]
            t += (f"\nconfirm. R: W {fnum(u['ratio'])} [{fnum(u['lo'])}, {fnum(u['hi'])}], "
                  f"W_e2e {fnum(e['ratio'])} [{fnum(e['lo'])}, {fnum(e['hi'])}]: {conf['decision']}")
        N0 = spec["N0"]
        r0 = rows[N0]
        pe = r0.get("pair_final_e_F", {})
        t += (f"\nN {N0}: censored ABF {r0['W_mid_abf']['n_censored']}/{r0['W_mid_abf']['n']}, "
              f"FR {r0['W_mid_fr']['n_censored']}/{r0['W_mid_fr']['n']}; e_F(T_N) FR/ABF {fnum(pe.get('ratio'), 2)} "
              f"[{fnum(pe.get('lo'), 2)}, {fnum(pe.get('hi'), 2)}]")
        ax.set_title(t, loc="left", fontsize=8.6)
    handles = [Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="exploration seeds (selected best)"),
               Line2D([], [], color=INK2, marker="s", ls="none", label="confirmation: 32 fresh seeds per arm"),
               Line2D([], [], color=C_ABF, lw=1.5, label="ABF"), Line2D([], [], color=C_FR, lw=1.5, label="ABF + FR")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Figure F. Synthesis: best hybrid ABF + FR vs best hybrid ABF, measured wall-clock to the mid "
                 "threshold (R < 1 favours FR; 'unresolved' = CI includes 1, never 'equivalent')", fontsize=10.5,
                 x=0.01, ha="left")
    return save(fig, "F_synthesis", checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    ST.setup_style()
    bench, routing, summ = load()
    checks = {}
    want = set(a.only or "ABCDEF")
    if "A" in want:
        print(fig_A(bench, routing, checks))
    if summ:
        for s in H.SYSTEMS:
            if s not in summ["results"]:
                continue
            if "B" in want:
                print(fig_BC(s, summ, routing, checks, "wall"))
            if "C" in want:
                print(fig_BC(s, summ, routing, checks, "time"))
        if "D" in want:
            print(fig_D(summ, checks))
        if "E" in want:
            print(fig_E(summ, checks))
        if "F" in want:
            print(fig_F(summ, checks))
    C.write_json(os.path.join(FIG, "legibility.json"), checks)
    print(checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
