"""Figures of the NSCC A100 study (figures/nscc_a100/).

    python scripts/nscc_a100/plot.py            # every figure whose inputs exist

A  throughput crossover: steady-state ns per walker-step vs N, S1 / best P_t / G1 (A100), ABF and FR, per system
B  A100 per-step time split by phase (device clock of block 0): move + force, deposit, grid barrier, FR
C  median W_mid vs N, ABF and FR, on the routed backend (exploration), GPU-routed rungs marked
D  confirmation: R_W and R_e2e with 95 % CIs per system
E  e_F against measured wall-clock, selected arms (median and 25-75 % band over the confirmation seeds)
F  e_F against measured wall-clock for EVERY (N, T_N) rung, ABF vs ABF + FR on the rung's routed backend (median and
   25-75 % band over the 24 production seeds), one page per system, timed and process-start (e2e) clocks:
   figures/nscc_a100/all_N_T_wallclock/ (PNG + PDF per system, one multipage PDF, the plotted quantiles as CSV).
   Each panel marks and prints each method's median W_mid (W_mid_e2e on the e2e clock) over the same seeds; the
   quartiles and censored counts go to all_N_T_W_mid_<clock>.csv

E and F are descriptive.  A run's error is its value at the latest save at or before t; the curves span only the
window where EVERY seed has a save (from the latest first save to the earliest last save), so nothing is extrapolated
before a run's first save or carried past its completion.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402
import eqb_metrics as M  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(NC.ROOT, "figures", "nscc_a100")
LABEL = {"gateway": "Entropic gateway", "lta300": "Ethane / LTA 300 K", "lta150": "Ethane / LTA 150 K"}
COL = {"S1": "#7f7f7f", "P": "#1f77b4", "G1": "#d62728", "abf": "#1f77b4", "fr": "#d62728"}


def jload(p):
    with open(p) as f:
        return json.load(f)


def band(runs, n=200):
    """(t, q25, q50, q75) of e_F over runs [(wall, e_F), ...] on a log grid covered by every run (module docstring);
    None if the common window is empty."""
    t0 = max(float(w[w > 0][0]) for w, *_ in runs)
    t1 = min(float(w[-1]) for w, *_ in runs)
    if not t1 > t0:
        return None
    t = np.logspace(np.log10(t0), np.log10(t1), n)
    E = np.array([e[np.searchsorted(w, t, side="right") - 1] for w, e, *_ in runs])
    q25, q50, q75 = np.percentile(E, [25, 50, 75], axis=0)
    return t, q25, q50, q75


def load_curves(kind, system, config, clock="timed"):
    """[(wall, e_F, W_mid)] of the complete records of results/nscc_a100/<kind>/<system>/<config>/; clock 'e2e' adds
    the run's process-start overhead (and takes W_mid_e2e).  W_mid is inf for a run that never persistently reached
    the mid threshold."""
    runs = []
    for q in sorted(glob.glob(os.path.join(NC.OUT_ROOT, kind, system, config, "s*.json"))):
        r = jload(q)
        if r.get("status") != "complete":
            continue
        w = np.asarray(r["wall_s_at_save"], float)
        if clock == "e2e":
            w = w + NC.C.num(r["start_overhead_s"])
        W = NC.C.num(r["endpoints"]["W_mid_e2e" if clock == "e2e" else "W_mid"])
        runs.append((w, np.asarray(r["e_F"], float), W))
    return runs


BACKEND = {"S1": "CPU, 1 core", "P1": "CPU, 1 core", "G1": "A100"}


def fig_all_N(clock):
    """Figure F (module docstring) for one clock: 'timed' (engine's timed start) or 'e2e' (process start)."""
    from matplotlib.backends.backend_pdf import PdfPages
    out = os.path.join(FIG, "all_N_T_wallclock")
    os.makedirs(out, exist_ok=True)
    xlab = "wall-clock since the engine's timed start (s)" if clock == "timed" else "wall-clock since process start (s)"
    rows, wrows = [], []
    with PdfPages(os.path.join(out, f"nscc_a100_all_N_T_eF_vs_wallclock_{clock}.pdf")) as pdf:
        for system in NC.SYSTEMS:
            spec = NC.H.system_spec(system)
            Ns = sorted(spec["N_ladder"])
            fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharey=True)
            for ax in axes.flat[len(Ns):]:
                ax.set_visible(False)
            for ax, N in zip(axes.flat, Ns):
                backend, wtxt = None, []
                for m in NC.H.methods_for(N):
                    cfg = glob.glob(os.path.join(NC.OUT_ROOT, "prod", system, f"{m}{N}_*"))
                    if len(cfg) != 1:
                        continue
                    config = os.path.basename(cfg[0])
                    backend = config.split("_", 1)[1]
                    runs = load_curves("prod", system, config, clock)
                    W = np.array([r[2] for r in runs])
                    wq = [M.quantile_inf(W, q) for q in (25, 50, 75)]  # inf-aware, as analyze.py
                    n_cens = int(np.sum(np.isinf(W)))
                    wrows.append((system, N, spec["h"] * NC.H.n_steps_for(spec, N), m, backend, clock, len(runs),
                                  n_cens, *map(float, wq)))
                    lab = "ABF" if m == "abf" else "ABF+FR"
                    if math.isfinite(wq[1]):
                        ax.axvline(wq[1], color=COL[m], ls=":", lw=1.2)
                        ax.plot(wq[1], spec["thresholds"]["mid"], "o", color=COL[m], ms=5, zorder=5)
                        wtxt.append(f"{lab}: {wq[1]:.3g} s" + (f" ({n_cens}/{len(runs)} cens.)" if n_cens else ""))
                    else:
                        wtxt.append(f"{lab}: not reached ({n_cens}/{len(runs)} cens.)")
                    b = band(runs) if runs else None
                    if b is None:
                        continue
                    t, q25, q50, q75 = b
                    ax.plot(t, q50, color=COL[m], lw=1.4, label=lab)
                    ax.fill_between(t, q25, q75, color=COL[m], alpha=0.2, lw=0)
                    rows += [(system, N, spec["h"] * NC.H.n_steps_for(spec, N), m, backend, clock, len(runs),
                              *v) for v in zip(t, q25, q50, q75)]
                ax.axhline(spec["thresholds"]["mid"], color="k", ls="--", lw=0.8)
                ax.text(0.98, 0.97, "median W_mid\n" + "\n".join(wtxt), transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
                ax.set_xscale("log")
                ax.set_yscale("log")
                # plain-number labels: the sci-notation minor labels overlap on panels spanning under a decade
                ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False,
                                                                            minor_thresholds=(1, 0.4)))
                ax.grid(alpha=0.3, which="both")
                bl = BACKEND.get(backend, f"CPU, {backend[1:]} cores" if backend else "?")
                ax.set_title(f"N = {N}, T_N = {spec['h'] * NC.H.n_steps_for(spec, N):g}  ({bl})", fontsize=9)
            for ax in axes[:, 0]:
                ax.set_ylabel("e_F")
            for ax in axes[-1, :]:
                ax.set_xlabel(xlab, fontsize=8)
            for ax in axes.flat:
                if ax.get_visible():
                    ax.tick_params(labelbottom=True)
            axes.flat[1].legend(fontsize=8, loc="lower left")  # N = 1 has no FR arm
            fig.suptitle(f"{LABEL[system]}: e_F vs measured wall-clock at every (N, T_N) of the ladder, routed backend "
                         f"(median, 25-75 % over 24 production seeds; dashed: mid threshold;\n"
                         f"dot / dotted line: median W_mid, the wall time to persistent mid-threshold accuracy, "
                         f"which can exceed the median curve's first crossing)", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(out, f"{system}_all_N_T_eF_vs_wallclock_{clock}.{ext}"), dpi=130)
            pdf.savefig(fig)
            plt.close(fig)
    with open(os.path.join(out, f"all_N_T_eF_vs_wallclock_{clock}.csv"), "w") as f:
        f.write("system,N,T_N,method,backend,clock,n_seeds,t_s,eF_q25,eF_median,eF_q75\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    with open(os.path.join(out, f"all_N_T_W_mid_{clock}.csv"), "w") as f:
        f.write("system,N,T_N,method,backend,clock,n_seeds,n_censored,W_mid_q25,W_mid_median,W_mid_q75\n")
        for r in wrows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")


def fig_all_N_timed():
    fig_all_N("timed")


def fig_all_N_e2e():
    fig_all_N("e2e")


def fig_throughput():
    p = os.path.join(NC.OUT_ROOT, "bench", "summary.json")
    if not os.path.exists(p):
        return
    S = jload(p)["summary"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=False)
    for c, system in enumerate(NC.SYSTEMS):
        for r, m in enumerate(("abf", "fr")):
            ax = axes[r, c]
            ent = S.get(system, {}).get(m, {})
            Ns = sorted(int(k) for k in ent)
            if not Ns:
                ax.set_visible(False)
                continue
            for be, lab, col in (("S1", "1 core (S1)", COL["S1"]), ("P", "best P_t (<= 16 cores)", COL["P"]),
                                 ("G1", "A100 (G1)", COL["G1"])):
                xs, ys, lo, hi = [], [], [], []
                for N in Ns:
                    e = ent[str(N)]
                    if be == "P":
                        par = {k: v for k, v in e.items() if k.startswith("P")}
                        if not par:
                            continue
                        v = min(par.values(), key=lambda q: q["median"])
                    else:
                        if be not in e:
                            continue
                        v = e[be]
                    xs.append(N)
                    ys.append(v["median"])
                    lo.append(v["median"] - v["min"])
                    hi.append(v["max"] - v["median"])
                ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=4, lw=1.4, capsize=2, color=col, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{LABEL[system]}: {'ABF' if m == 'abf' else 'ABF + FR'}", fontsize=10)
            ax.set_xlabel("replicas N")
            if c == 0:
                ax.set_ylabel("ns per walker-step (steady state)")
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Throughput on NSCC ASPIRE2A: AMD EPYC 7713 cores vs one A100 (median, min-max of 3 repeats)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "A_throughput_crossover.png"), dpi=150)
    plt.close(fig)


def fig_gpu_split():
    recs = []
    for p in glob.glob(os.path.join(NC.OUT_ROOT, "bench", "*", "*_G1", "s*_r0.json")):
        r = jload(p)
        if r.get("status") == "complete":
            recs.append(r)
    if not recs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, system in enumerate(NC.SYSTEMS):
        ax = axes[c]
        rs = sorted([r for r in recs if r["system"] == system and r["method"] == "fr"], key=lambda r: r["N"])
        if not rs:
            ax.set_visible(False)
            continue
        Ns = [r["N"] for r in rs]
        parts = {"move + force": [], "deposit": [], "grid barrier": [], "FR": []}
        for r in rs:
            t = r["timers"]
            steps = r["n_steps"]
            parts["move + force"].append(t["t_force_move_s"] / steps * 1e6)
            parts["deposit"].append(t["t_abf_s"] / steps * 1e6)
            parts["grid barrier"].append(t["t_sync_s"] / steps * 1e6)
            parts["FR"].append(t["t_fr_s"] / steps * 1e6)
        x = np.arange(len(Ns))
        bot = np.zeros(len(Ns))
        for (k, v), col in zip(parts.items(), ("#4c72b0", "#55a868", "#c44e52", "#8172b2")):
            v = np.nan_to_num(np.array(v))
            ax.bar(x, v, bottom=bot, color=col, label=k)
            bot += v
        ax.set_xticks(x, [str(n) for n in Ns], fontsize=8)
        ax.set_xlabel("replicas N")
        ax.set_ylabel("us per step (A100, ABF + FR)")
        ax.set_title(LABEL[system], fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "B_gpu_phase_split.png"), dpi=150)
    plt.close(fig)


def fig_wmid():
    p = os.path.join(NC.OUT_ROOT, "analysis", "prod_summary.json")
    if not os.path.exists(p):
        return
    S = jload(p)["systems"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, system in enumerate(NC.SYSTEMS):
        ax = axes[c]
        per = S.get(system, {}).get("per", {})
        for m in ("abf", "fr"):
            rows = sorted([v for v in per.values() if v["method"] == m], key=lambda v: v["N"])
            if not rows:
                continue
            xs = [v["N"] for v in rows]
            ys = [v["median_W_mid"] if math.isfinite(v["median_W_mid"]) else np.nan for v in rows]
            ax.plot(xs, ys, marker="o", color=COL[m], label="ABF" if m == "abf" else "ABF + FR")
            for v in rows:
                if v["backend"] == "G1" and math.isfinite(v["median_W_mid"]):
                    ax.plot(v["N"], v["median_W_mid"], marker="s", ms=10, mfc="none", mec="k", lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("replicas N")
        ax.set_ylabel("median W_mid (s)")
        ax.set_title(LABEL[system] + "  (square = A100)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "C_wmid_vs_N.png"), dpi=150)
    plt.close(fig)


def fig_confirm():
    p = os.path.join(NC.OUT_ROOT, "analysis", "confirm_summary.json")
    if not os.path.exists(p):
        return
    S = jload(p)["systems"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = []
    for i, system in enumerate([s for s in NC.SYSTEMS if s in S]):
        for k, (key, lab) in enumerate((("W_mid", "W"), ("W_mid_e2e", "W_e2e"))):
            R = S[system][key]["R"]
            y = i * 3 + k
            ys.append((y, f"{LABEL[system]}  {lab}"))
            lo, hi = R["ci95"]
            ax.errorbar(R["point"], y, xerr=[[R["point"] - lo], [hi - R["point"]]], fmt="o",
                        color="k" if k == 0 else "#888", capsize=3)
    ax.axvline(1.0, color="r", lw=1)
    ax.set_yticks([y for y, _ in ys], [t for _, t in ys], fontsize=8)
    ax.set_xlabel("median FR / median ABF (best deployments), 95 % CI; < 1 favours FR")
    ax.set_xscale("log")
    ticks = [0.5, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "D_confirmation_ratios.png"), dpi=150)
    plt.close(fig)


def fig_ef_wall():
    p = os.path.join(NC.OUT_ROOT, "selection", "selection.json")
    if not os.path.exists(p):
        return
    sel = jload(p)["systems"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, system in enumerate(NC.SYSTEMS):
        ax = axes[c]
        spec = NC.H.system_spec(system)
        for m in ("abf", "fr"):
            arm = sel.get(system, {}).get(m)
            if not arm:
                continue
            runs = load_curves("confirm", system, f"{m}{arm['N']}_{arm['backend']}")
            b = band(runs) if runs else None
            if b is None:
                continue
            grid, q25, q50, q75 = b
            ax.plot(grid, q50, color=COL[m], label=f"{'ABF' if m == 'abf' else 'ABF + FR'} N={arm['N']} "
                                                    f"({arm['backend']})")
            ax.fill_between(grid, q25, q75, color=COL[m], alpha=0.2)
        ax.axhline(spec["thresholds"]["mid"], color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock since timed start (s)")
        ax.set_ylabel("e_F")
        ax.set_title(LABEL[system], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E_eF_vs_wall_selected.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_throughput, fig_gpu_split, fig_wmid, fig_confirm, fig_ef_wall, fig_all_N_timed, fig_all_N_e2e):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
