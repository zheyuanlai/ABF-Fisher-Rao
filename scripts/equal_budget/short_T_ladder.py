#!/usr/bin/env python
"""Shortened-T equal-budget ladders: ABF vs ABF+FR at every N-T setting.

Rule (2026-10-15).  At the largest-N rung (gateway N = 2048, LTA N = 1024) take T* = the first time on the uniform
budget grid after which the ABF+FR median e_F stays at or below the system's frozen 'mid' accuracy threshold, while
the ABF median is still above it.  The ladder is re-anchored at (N_max, T*): new budget B' = N_max T* / h, so every
rung keeps N T = const and gets T_N = (T* / T0) T_N^orig.  The same budget fraction u* = T* / T0 applies to every N.

No new dynamics.  Every schedule in both engines is in physical time (gateway ramp_t, LTA warmup / burn-in / FR start
in steps), never a fraction of the run, so a run stopped at T_N is the stored run truncated at T_N.  T_N falls on the
shared uniform u grid (k / 200), which is a save point of every run.  Metrics come from the per-run cache
results/equal_budget_v2/<sys>/analysis/run_metrics (eqb_metrics/2): e_F(t) against the frozen references;
Ibar_F over [0, T_N] = mean of e_F over the uniform-grid saves with u <= u*, the original definition truncated.

Outputs: figures/equal_budget_v2/short_T/<sys>/ and results/equal_budget_v2/short_T/{summary.json,tables.md}.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fig_legibility as LG  # noqa: E402

C_ABF, C_FR = "#2a78d6", "#eb6834"          # same slots as plot_config.py
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

SYSTEMS = {
    "gateway": dict(label="Entropic gateway", cfg="gateway_production.json", unit="t.u."),
    "lta_T300": dict(label="Ethane / LTA, 300 K", cfg="lta_300K_production.json", unit="t.u."),
    "lta_T150": dict(label="Ethane / LTA, 150 K", cfg="lta_150K_production.json", unit="t.u."),
}
E_UNIT = {"gateway": "reduced", "lta_T300": "kJ/mol", "lta_T150": "kJ/mol"}
N_BOOT = 10000
RNG = np.random.default_rng(20261015)


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 0.85, "legend.facecolor": "white", "legend.edgecolor": "none",
        "pdf.fonttype": 42, "mathtext.default": "regular", "lines.solid_capstyle": "round",
    })


# ------------------------------------------------------------------------------------------------ data
def load_runs(sysn, N, method, run_size):
    """{seed: dict(t, u, iu, e)} from the metric cache; refuses a cache entry whose source file changed size."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, f"results/equal_budget_v2/{sysn}/analysis/run_metrics/N{N}/s*_{method}.npz"))):
        seed = int(re.search(r"/s(\d+)_", f).group(1))
        js = json.load(open(f[:-4] + ".json"))
        src = os.path.join(ROOT, f"results/equal_budget_v2/{sysn}/N{N}/s{seed}_{method}.npz")
        if os.path.getsize(src) != js["file"]["src_size"]:
            raise RuntimeError(f"stale cache: {f}")
        d = np.load(f)
        out[seed] = dict(t=d["save_t"], u=d["save_u"], iu=d["uniform_index"], e=d["e_F"])
        run_size[0] += 1
    return out


def find_ustar(A, F, eps):
    """First uniform-grid u after which the FR median e_F stays <= eps; also the ABF one and the ABF median at u*."""
    r0 = next(iter(A.values()))
    iu, u, t = r0["iu"], r0["u"], r0["t"]
    ma = np.median([r["e"] for r in A.values()], 0)[iu]
    mf = np.median([r["e"] for r in F.values()], 0)[iu]

    def settle(m):
        ok = m <= eps
        for j in range(len(ok)):
            if ok[j:].all():
                return j
        return None

    jf, ja = settle(mf), settle(ma)
    if jf is None or (ja is not None and ja <= jf):
        raise RuntimeError("rule not satisfiable: FR never settles below eps before ABF")
    return dict(u_star=float(u[iu[jf]]), T_star=float(t[iu[jf]]), eps=eps, abf_median_at_T_star=float(ma[jf]),
                fr_median_at_T_star=float(mf[jf]),
                abf_settle_u=None if ja is None else float(u[iu[ja]]),
                abf_settle_t=None if ja is None else float(t[iu[ja]]))


def at_ustar(r, u_star):
    """e_F at T_N and Ibar_F over [0, T_N] for one run (u* is a uniform-grid point of every run)."""
    uu = r["u"][r["iu"]]
    k = np.flatnonzero(np.isclose(uu, u_star, rtol=0, atol=1e-12))
    if len(k) != 1:
        raise RuntimeError(f"u* = {u_star} is not on the uniform grid")
    k = int(k[0])
    return float(r["e"][r["iu"][k]]), float(np.mean(r["e"][r["iu"][: k + 1]])), int(r["iu"][k])


def boot_median_ratio_paired(a, f):
    lr = np.log(np.asarray(f) / np.asarray(a))
    idx = RNG.integers(0, len(lr), (N_BOOT, len(lr)))
    b = np.exp(np.median(lr[idx], 1))
    return float(np.exp(np.median(lr))), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_median_ratio_unpaired(a, f):
    a, f = np.asarray(a), np.asarray(f)
    ba = np.median(a[RNG.integers(0, len(a), (N_BOOT, len(a)))], 1)
    bf = np.median(f[RNG.integers(0, len(f), (N_BOOT, len(f)))], 1)
    b = bf / ba
    return float(np.median(f) / np.median(a)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------------------------------------ analysis
def analyse(sysn):
    cfg = json.load(open(os.path.join(ROOT, "configs/equal_budget_v2", SYSTEMS[sysn]["cfg"])))
    Ns = list(cfg["N_ladder"])
    N0, T0 = cfg["anchor"]["N0"], cfg["anchor"]["T0"]
    eps = cfg["thresholds"]["e_F"][1]
    n = [0]
    runs = {N: {m: load_runs(sysn, N, m, n) for m in ("abf", "fr")} for N in Ns}
    star = find_ustar(runs[N0]["abf"], runs[N0]["fr"], eps)
    u_star = star["u_star"]
    h = cfg["engine_cfg"]["h"]
    rows = []
    for N in Ns:
        A, F = runs[N]["abf"], runs[N]["fr"]
        T_orig = cfg["B"] * h / N
        row = dict(N=N, T_orig=T_orig, T_N=u_star * T_orig, n_abf=len(A), n_fr=len(F))
        fa = {s: at_ustar(r, u_star) for s, r in A.items()}
        ff = {s: at_ustar(r, u_star) for s, r in F.items()}
        for key, j in (("final", 0), ("Ibar", 1)):
            va = np.array([fa[s][j] for s in sorted(fa)])
            row[f"{key}_abf_median"] = float(np.median(va))
            row[f"{key}_abf_q25"], row[f"{key}_abf_q75"] = map(float, np.percentile(va, [25, 75]))
            row[f"{key}_abf_values"] = va.tolist()
            if F:
                seeds = sorted(set(fa) & set(ff))
                pa = np.array([fa[s][j] for s in seeds])
                pf = np.array([ff[s][j] for s in seeds])
                vf = np.array([ff[s][j] for s in sorted(ff)])
                row[f"{key}_fr_median"] = float(np.median(vf))
                row[f"{key}_fr_q25"], row[f"{key}_fr_q75"] = map(float, np.percentile(vf, [25, 75]))
                row[f"{key}_fr_values"] = vf.tolist()
                r, lo, hi = boot_median_ratio_paired(pa, pf)
                row[f"{key}_ratio"], row[f"{key}_ratio_lo"], row[f"{key}_ratio_hi"] = r, lo, hi
                row[f"{key}_fr_wins"] = int(np.sum(pf < pa))
                row[f"{key}_n_pairs"] = len(seeds)
        rows.append(row)
    # best allocation at the new budget (median over seeds; independent runs across N -> unpaired bootstrap)
    best = {}
    for key in ("final", "Ibar"):
        ba = min(rows, key=lambda r: r[f"{key}_abf_median"])
        bf = min((r for r in rows if r["n_fr"]), key=lambda r: r[f"{key}_fr_median"])
        r, lo, hi = boot_median_ratio_unpaired(ba[f"{key}_abf_values"], bf[f"{key}_fr_values"])
        best[key] = dict(best_abf_N=ba["N"], best_abf_median=ba[f"{key}_abf_median"], best_fr_N=bf["N"],
                         best_fr_median=bf[f"{key}_fr_median"], ratio_best_fr_over_best_abf=r, ci95=[lo, hi])
    return dict(system=sysn, label=SYSTEMS[sysn]["label"], N0=N0, T0=T0, h=h, B_orig=cfg["B"],
                B_new=u_star * cfg["B"], threshold_name="mid", **star, rows=rows, best=best, n_runs=n[0]), runs


# ------------------------------------------------------------------------------------------------ plots
def band(ax, t, Y, color, label):
    med, lo, hi = np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.17, lw=0, zorder=2)
    ax.plot(t, med, color=color, lw=1.8, zorder=3, label=label)


def panel(ax, res, row, runs, show_ylabel=True, show_xlabel=True):
    u_star = res["u_star"]
    A, F = runs[row["N"]]["abf"], runs[row["N"]]["fr"]
    r0 = next(iter(A.values()))
    k = at_ustar(r0, u_star)[2]
    t = r0["t"][: k + 1]
    band(ax, t, np.array([r["e"][: k + 1] for r in A.values()]), C_ABF, "ABF")
    if F:
        band(ax, t, np.array([r["e"][: k + 1] for r in F.values()]), C_FR, "ABF + FR")
    ax.axhline(res["eps"], color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0, row["T_N"])
    ax.set_title(f"N = {row['N']},  T = {row['T_N']:.4g}", loc="left")
    if F:
        txt = (f"final FR/ABF {row['final_ratio']:.2f} [{row['final_ratio_lo']:.2f}, {row['final_ratio_hi']:.2f}]"
               f"  {row['final_fr_wins']}/{row['final_n_pairs']}\n"
               f"Ibar FR/ABF {row['Ibar_ratio']:.2f} [{row['Ibar_ratio_lo']:.2f}, {row['Ibar_ratio_hi']:.2f}]"
               f"  {row['Ibar_fr_wins']}/{row['Ibar_n_pairs']}")
    else:
        txt = "ABF only (no FR arm at N = 1)"
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=INK2,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85), zorder=5)
    if show_xlabel:
        ax.set_xlabel(f"physical time t ({SYSTEMS[res['system']]['unit']})")
    if show_ylabel:
        ax.set_ylabel(f"e_F ({E_UNIT[res['system']]})")


def legend_handles(res):
    return [Line2D([], [], color=C_ABF, lw=1.8, label="ABF (median, IQR band)"),
            Line2D([], [], color=C_FR, lw=1.8, label="ABF + FR (median, IQR band)"),
            Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label=f"mid threshold e_F = {res['eps']:g}")]


def subtitle(res):
    return (f"{res['label']}: re-anchored at N = {res['N0']}, T* = {res['T_star']:.4g} "
            f"(ABF+FR median settles below {res['eps']:g}; ABF median {res['abf_median_at_T_star']:.3g}).  "
            f"Every rung N T = {res['N0']} x {res['T_star']:.4g}  (u* = {res['u_star']:.3f} of the original budget)")


def grid_figure(res, runs, outdir):
    rows = res["rows"]
    nc = 4
    nr = int(np.ceil(len(rows) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.1 * nc, 3.05 * nr + 0.9), sharey=True, squeeze=False,
                            layout="constrained")
    for i, row in enumerate(rows):
        ax = axs.flat[i]
        panel(ax, res, row, runs, show_ylabel=(i % nc == 0))
    for ax in axs.flat[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(subtitle(res), fontsize=10, color=INK, x=0.01, ha="left")
    fig.legend(handles=legend_handles(res), loc="outside lower center", ncol=3, frameon=False)
    path = os.path.join(outdir, f"{res['system']}_all_NT_settings")
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    chk = LG.check_figure_safe(fig)
    plt.close(fig)
    return path + ".png", chk


def single_figures(res, runs, outdir):
    out = {}
    for row in res["rows"]:
        fig, ax = plt.subplots(figsize=(6.2, 4.6), layout="constrained")
        panel(ax, res, row, runs)
        fig.legend(handles=legend_handles(res), loc="outside lower center", ncol=2, frameon=False, fontsize=8)
        path = os.path.join(outdir, f"N{row['N']:04d}")
        fig.savefig(path + ".png", dpi=160)
        out[row["N"]] = LG.check_figure_safe(fig)
        plt.close(fig)
    return out


def summary_figure(res, outdir):
    rows = res["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
    for ax, key, ttl in ((axs[0], "final", "e_F at T_N"), (axs[1], "Ibar", "Ibar_F over [0, T_N]")):
        for m, c, lab, dx in (("abf", C_ABF, "ABF", 0.97), ("fr", C_FR, "ABF + FR", 1.03)):
            rr = [r for r in rows if f"{key}_{m}_median" in r]
            N = np.array([r["N"] for r in rr], float) * dx
            med = np.array([r[f"{key}_{m}_median"] for r in rr])
            lo = med - np.array([r[f"{key}_{m}_q25"] for r in rr])
            hi = np.array([r[f"{key}_{m}_q75"] for r in rr]) - med
            ax.errorbar(N, med, yerr=[lo, hi], color=c, lw=1.6, marker="o", ms=5, capsize=0, elinewidth=1.0,
                        label=lab, zorder=3)
        b = res["best"][key]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("replicas N   (each rung runs T_N = N0 T* / N)")
        ax.set_ylabel(f"{ttl} ({E_UNIT[res['system']]})")
        ax.set_title(f"{ttl}\nbest FR (N = {b['best_fr_N']}) / best ABF (N = {b['best_abf_N']}) = "
                     f"{b['ratio_best_fr_over_best_abf']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]",
                     loc="left", fontsize=9.5)
    fig.legend(handles=[Line2D([], [], color=C_ABF, lw=1.6, marker="o", ms=5, label="ABF"),
                        Line2D([], [], color=C_FR, lw=1.6, marker="o", ms=5, label="ABF + FR")],
               loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"{res['label']}: equal-budget ladder at the shortened budget (median, IQR bars)",
                 fontsize=10.5, x=0.01, ha="left")
    path = os.path.join(outdir, f"{res['system']}_summary_vs_N")
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    chk = LG.check_figure_safe(fig)
    plt.close(fig)
    return path + ".png", chk


def budget_sweep(res, runs, n_boot=2000):
    """Best-over-N ABF+FR vs best-over-N ABF at EVERY shortened budget u on the uniform grid (the same re-anchoring
    rule with T* replaced by u T0).  The bootstrap CI holds the selected N fixed (selection is not resampled)."""
    Ns = [r["N"] for r in res["rows"]]
    r0 = next(iter(runs[Ns[0]]["abf"].values()))
    iu = r0["iu"]
    u = r0["u"][iu]
    med, boot = {}, {}
    for N in Ns:
        for m in ("abf", "fr"):
            if not runs[N][m]:
                continue
            for r in runs[N][m].values():
                if not np.allclose(r["u"][r["iu"]], u, rtol=0, atol=1e-12):
                    raise RuntimeError(f"N = {N}: uniform grid differs")
            E = np.array([r["e"][r["iu"]] for r in runs[N][m].values()])   # each run's OWN uniform indices
            for key, Y in (("final", E), ("Ibar", np.cumsum(E, 1) / np.arange(1, E.shape[1] + 1))):
                med[N, m, key] = np.median(Y, 0)
                idx = RNG.integers(0, len(Y), (n_boot, len(Y)))
                boot[N, m, key] = np.median(Y[idx], 1)                      # (n_boot, n_u)
    out = dict(u=u.tolist(), T_at_N0=(u * res["T0"]).tolist())
    for key in ("final", "Ibar"):
        nA = [N for N in Ns if (N, "abf", key) in med]
        nF = [N for N in Ns if (N, "fr", key) in med]
        MA = np.array([med[N, "abf", key] for N in nA])
        MF = np.array([med[N, "fr", key] for N in nF])
        ja, jf = MA.argmin(0), MF.argmin(0)
        cols = np.arange(len(u))
        ratio = MF[jf, cols] / MA[ja, cols]
        bA = np.array([boot[N, "abf", key] for N in nA])[ja, :, cols]   # (n_u, n_boot)
        bF = np.array([boot[N, "fr", key] for N in nF])[jf, :, cols]
        rb = bF / bA
        out[key] = dict(ratio=ratio.tolist(), lo=np.percentile(rb, 2.5, 1).tolist(),
                        hi=np.percentile(rb, 97.5, 1).tolist(), best_abf_N=[nA[j] for j in ja],
                        best_fr_N=[nF[j] for j in jf],
                        fixed_N0_ratio=(med[res["N0"], "fr", key] / med[res["N0"], "abf", key]).tolist())
    return out


def sweep_figure(sweeps, outdir):
    fig, axs = plt.subplots(1, len(sweeps), figsize=(5.0 * len(sweeps), 4.4), layout="constrained", squeeze=False)
    for ax, (sysn, (res, sw)) in zip(axs[0], sweeps.items()):
        T = np.array(sw["T_at_N0"])
        f, ib = sw["final"], sw["Ibar"]
        ax.fill_between(T, f["lo"], f["hi"], color=INK, alpha=0.10, lw=0)
        ax.plot(T, f["ratio"], color=INK, lw=1.7)
        ax.plot(T, ib["ratio"], color=INK2, lw=1.5, ls=(0, (5, 2)))
        ax.plot(T, f["fixed_N0_ratio"], color=C_FR, lw=1.3, ls=(0, (1, 1.5)))
        ax.axhline(1.0, color=MUTED, lw=0.9)
        ax.axvline(res["T_star"], color=MUTED, lw=0.9, ls=(0, (3, 2)))
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0.22, 2.3)
        ax.set_xlim(0, res["T0"])
        ax.set_xlabel(f"T at N = {res['N0']}  (rung N runs T {res['N0']} / N)")
        ax.set_ylabel("error ratio, ABF+FR / ABF")
        ax.set_title(f"{res['label']}  (T* = {res['T_star']:.3g})", loc="left")
    handles = [Line2D([], [], color=INK, lw=1.7, label="best-N ABF+FR / best-N ABF, e_F(T_N) (95 % band, N fixed)"),
               Line2D([], [], color=INK2, lw=1.5, ls=(0, (5, 2)), label="best-N ABF+FR / best-N ABF, Ibar_F"),
               Line2D([], [], color=C_FR, lw=1.3, ls=(0, (1, 1.5)), label="same N (largest), e_F(T_N)"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label="chosen T*")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Every shortened budget: does the best ABF+FR allocation beat the best ABF allocation?  "
                 "(below 1 = ABF+FR better)", fontsize=10.5, x=0.01, ha="left")
    path = os.path.join(outdir, "budget_sweep_best_allocation")
    fig.savefig(path + ".png", dpi=160)
    fig.savefig(path + ".pdf")
    chk = LG.check_figure_safe(fig)
    plt.close(fig)
    return path + ".png", chk


def table_md(res):
    L = [f"### {res['label']}", "",
         f"T* = {res['T_star']:.4g} at N = {res['N0']} (original T0 = {res['T0']:g}; u* = {res['u_star']:.3f}); "
         f"ABF+FR median e_F settles at <= {res['eps']:g} there, ABF median is {res['abf_median_at_T_star']:.3g} "
         f"(ABF settles at t = {res['abf_settle_t'] if res['abf_settle_t'] is not None else 'never (within T0)'}). "
         f"New budget B' = {res['B_new']:.4g} walker-steps (was {res['B_orig']:.4g}).", "",
         "| N | T_N | e_F(T_N) ABF | e_F(T_N) ABF+FR | ratio [95 % CI] | FR wins | Ibar ABF | Ibar ABF+FR | ratio [95 % CI] | FR wins |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r["n_fr"]:
            L.append(f"| {r['N']} | {r['T_N']:.4g} | {r['final_abf_median']:.3g} | {r['final_fr_median']:.3g} | "
                     f"{r['final_ratio']:.2f} [{r['final_ratio_lo']:.2f}, {r['final_ratio_hi']:.2f}] | "
                     f"{r['final_fr_wins']}/{r['final_n_pairs']} | {r['Ibar_abf_median']:.3g} | {r['Ibar_fr_median']:.3g} | "
                     f"{r['Ibar_ratio']:.2f} [{r['Ibar_ratio_lo']:.2f}, {r['Ibar_ratio_hi']:.2f}] | "
                     f"{r['Ibar_fr_wins']}/{r['Ibar_n_pairs']} |")
        else:
            L.append(f"| {r['N']} | {r['T_N']:.4g} | {r['final_abf_median']:.3g} | -- | -- | -- | "
                     f"{r['Ibar_abf_median']:.3g} | -- | -- | -- |")
    for key, name in (("final", "e_F(T_N)"), ("Ibar", "Ibar_F")):
        b = res["best"][key]
        L.append("")
        L.append(f"Best allocation on {name}: ABF N = {b['best_abf_N']} ({b['best_abf_median']:.3g}) vs ABF+FR "
                 f"N = {b['best_fr_N']} ({b['best_fr_median']:.3g}); best FR / best ABF = "
                 f"{b['ratio_best_fr_over_best_abf']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}] (unpaired bootstrap).")
    return "\n".join(L) + "\n"


def main():
    setup_style()
    res_dir = os.path.join(ROOT, "results/equal_budget_v2/short_T")
    os.makedirs(res_dir, exist_ok=True)
    summary, md, checks = {}, ["# Shortened-T equal-budget ladders: ABF vs ABF+FR", "",
                               "Generated by `scripts/equal_budget/short_T_ladder.py` (see its docstring for the rule). "
                               "Ratios are FR / ABF medians; < 1 means ABF+FR has the lower error. "
                               "Per-N ratios are paired by seed (bootstrap over seeds, 95 %).", ""], {}
    sweeps = {}
    for sysn in SYSTEMS:
        res, runs = analyse(sysn)
        sw = budget_sweep(res, runs)
        sweeps[sysn] = (res, sw)
        res["budget_sweep"] = sw
        fig_dir = os.path.join(ROOT, "figures/equal_budget_v2/short_T", sysn)
        os.makedirs(fig_dir, exist_ok=True)
        p1, c1 = grid_figure(res, runs, fig_dir)
        p2, c2 = summary_figure(res, fig_dir)
        cs = single_figures(res, runs, fig_dir)
        checks[sysn] = dict(grid=c1["status"], summary=c2["status"],
                            singles={str(k): v["status"] for k, v in cs.items()})
        for name, c in (("grid", c1), ("summary", c2), *((f"N{k}", v) for k, v in cs.items())):
            if c["status"] != "pass":
                print(LG.one_line(f"{sysn}/{name}", c))
        slim = {k: v for k, v in res.items() if k != "rows"}
        slim["rows"] = [{k: v for k, v in r.items() if not k.endswith("_values")} for r in res["rows"]]
        summary[sysn] = slim
        md.append(table_md(res))
        print(f"{sysn}: T* = {res['T_star']:.4g} (u* {res['u_star']:.3f}), {res['n_runs']} runs -> {p1}, {p2}")
    p3, c3 = sweep_figure(sweeps, os.path.join(ROOT, "figures/equal_budget_v2/short_T"))
    checks["budget_sweep"] = c3["status"]
    if c3["status"] != "pass":
        print(LG.one_line("budget_sweep", c3))
    md.append("### Every shortened budget (sweep)\n")
    for sysn, (res, sw) in sweeps.items():
        for key in ("final", "Ibar"):
            s = sw[key]
            sig = [T for T, hi in zip(sw["T_at_N0"], s["hi"]) if hi < 1]
            md.append(f"* {res['label']}, {key}: best-N ABF+FR / best-N ABF ranges {min(s['ratio']):.2f}-"
                      f"{max(s['ratio']):.2f}; upper 95 % bound < 1 at {len(sig)} of {len(s['ratio'])} budgets"
                      + (f" (T at N0 in [{min(sig):.3g}, {max(sig):.3g}])" if sig else "") + ".")
    md.append("")
    print(f"sweep -> {p3}")
    summary["legibility"] = checks
    json.dump(summary, open(os.path.join(res_dir, "summary.json"), "w"), indent=1)
    open(os.path.join(res_dir, "tables.md"), "w").write("\n".join(md))
    print(open(os.path.join(res_dir, "tables.md")).read())


if __name__ == "__main__":
    main()
