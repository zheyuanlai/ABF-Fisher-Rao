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
        ax.set_xlim(0, xmax * 1.02 if xmax > 0 else 1)
        be = row.get("backend", "?")
        ax.set_title(f"N = {N},  T_N = {row.get('T_N', H.T_N(spec, N)):.4g},  {be}", loc="left", fontsize=9.6)
        wa, wf = row.get("W_mid_abf", {}), row.get("W_mid_fr", {})
        txt = f"W_mid ABF {fnum(wa.get('median', math.nan))} s [{wa.get('n_censored', 0)}/{wa.get('n', 0)} cens.]"
        if N > 1 and wf:
            pw = row.get("pair_W_mid", {})
            txt += (f"\nW_mid FR {fnum(wf.get('median', math.nan))} s [{wf.get('n_censored', 0)}/{wf.get('n', 0)}]"
                    f"\nFR/ABF {fnum(pw.get('ratio', math.nan), 2)} [{fnum(pw.get('lo', math.nan), 2)}, "
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
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
    for ax, s in zip(axs, H.SYSTEMS):
        res = summ["results"].get(s)
        if not res:
            ax.set_visible(False)
            continue
        fr = res["frontier"]
        w = np.array(fr["w"])
        for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
            if m not in fr:
                continue
            b = np.array(fr[m]["best"])
            done = np.array(fr[m]["completed_frac"]) >= 0.5
            ax.plot(np.where(~done, w, np.nan), np.where(~done, b, np.nan), color=col, lw=1.8, label=lab)
            ax.plot(np.where(done, w, np.nan), np.where(done, b, np.nan), color=col, lw=1.3, ls=(0, (4, 2)))
            Nb = np.array(fr[m]["best_N"])
            ch = np.flatnonzero(np.diff(Nb) != 0) + 1
            last = -np.inf
            for k in [0] + ch.tolist():          # label the best N where it changes, >= 0.4 decades apart
                if np.isfinite(b[k]) and math.log10(w[k]) - last >= 0.4:
                    ax.annotate(f"N{Nb[k]}", (w[k], b[k]), textcoords="offset points",
                                xytext=(2, 5 if m == "abf" else -11), fontsize=6.6, color=col)
                    last = math.log10(w[k])
        ax.axhline(res["thresholds"]["mid"], color=MUTED, lw=0.9, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock budget w (s)")
        ax.set_ylabel(f"best median e_F over N ({E_UNIT[s]})")
        ax.set_title(res["label"], loc="left")
    handles = [Line2D([], [], color=C_ABF, lw=1.8, label="ABF: best N at each w (still sampling)"),
               Line2D([], [], color=C_FR, lw=1.8, label="ABF + FR: best N at each w"),
               Line2D([], [], color=INK2, lw=1.3, ls=(0, (4, 2)),
                      label="dashed: best configuration completed; final estimate carried forward"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="mid threshold")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Figure D. Descriptive best-of-N wall-clock frontier (measured runs; selection over N on the same "
                 "seeds: not a confirmatory claim)", fontsize=10.5, x=0.01, ha="left")
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
                fin = W[np.isfinite(W)]
                top = max(top, float(fin.max()) if fin.size else 0.0, float(np.max(cv[f"{m}{N}_completion"])))
                pts[(m, N)] = (W, col, dx)
        top *= 1.6
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
        ax.set_ylim(top=top * 1.25)
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
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), layout="constrained")
    for ax, s in zip(axs, H.SYSTEMS):
        res = summ["results"].get(s)
        cv = curves("prod", s)
        if not res or cv is None:
            ax.set_visible(False)
            continue
        spec = H.system_spec(s)
        items = []
        b = res["best"]["mid"]
        rows = {r["N"]: r for r in res["per_N"]}
        for m in ("abf", "fr"):
            if b.get(m) and b[m].get("best_N") is not None:
                N = int(b[m]["best_N"])
                items.append((f"best {m.upper() if m == 'abf' else 'FR'}\nN {N} {rows[N]['backend']}\nexploration",
                              cv[f"{m}{N}_W_mid"], C_ABF if m == "abf" else C_FR))
        cf = curves("confirm", s)
        conf = (summ.get("confirm") or {}).get(s)
        if cf is not None and conf:
            for m in ("abf", "fr"):
                key = conf.get(f"best_{m}")
                if key and f"{key}_W_mid" in cf:
                    N = int(key[len(m):])
                    items.append((f"best {m.upper() if m == 'abf' else 'FR'}\nN {N}\nconfirmation", cf[f"{key}_W_mid"],
                                  C_ABF if m == "abf" else C_FR))
        N0 = spec["N0"]
        for m in ("abf", "fr"):
            items.append((f"{'ABF' if m == 'abf' else 'FR'} N {N0}\n(fixed largest N)\n{rows[N0]['backend']}",
                          cv[f"{m}{N0}_W_mid"], C_ABF if m == "abf" else C_FR))
        ys = []
        for i, (lab, W, col) in enumerate(items):
            med = _inf_median(W)
            (lo, hi), fc = _boot_med(W, rng)
            ys.append(med)
            if math.isfinite(med):
                ax.errorbar([i], [med], yerr=[[med - lo], [max(hi - med, 0)]] if np.isfinite(hi) else None,
                            color=col, marker="o", ms=7, capsize=3, elinewidth=1.2)
            ncen = int(np.sum(np.isinf(W)))
            ax.annotate(f"{ncen}/{W.size} cens." if ncen else f"n={W.size}", (i, med if math.isfinite(med) else 1),
                        textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7, color=INK2)
        fin = [y for y in ys if math.isfinite(y)]
        for i, y in enumerate(ys):
            if not math.isfinite(y):
                ax.plot([i], [max(fin) * 2 if fin else 1], marker="^", ms=9, color=items[i][2], mfc="white")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([it[0] for it in items], fontsize=7.4)
        ax.set_xlabel("configuration (method, N, backend, seed block)")
        ax.set_yscale("log")
        ax.set_ylabel("W_mid (s), median and 95 % CI")
        R = b.get("R_wall", {})
        t = (f"{res['label']}\nexploratory R_wall = {fnum(R.get('point', math.nan))} "
             f"[{fnum((R.get('ci95') or [math.nan])[0])}, {fnum((R.get('ci95') or [0, math.nan])[1])}]")
        if conf and conf.get("R_wall_conf"):
            u = conf["R_wall_conf"]
            t += f"\nconfirmatory R_wall = {fnum(u['ratio'])} [{fnum(u['lo'])}, {fnum(u['hi'])}]: {u['decision']}"
        fx = rows[N0].get("pair_W_mid", {})
        t += f"\nfixed N {N0}: FR/ABF {fnum(fx.get('ratio', math.nan), 2)} [{fnum(fx.get('lo', math.nan), 2)}, " \
             f"{fnum(fx.get('hi', math.nan), 2)}]"
        ax.set_title(t, loc="left", fontsize=8.8)
    fig.suptitle("Figure F. Synthesis: best hybrid ABF + FR vs best hybrid ABF (measured wall-clock to the mid "
                 "threshold), and the fixed largest N", fontsize=10.5, x=0.01, ha="left")
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
