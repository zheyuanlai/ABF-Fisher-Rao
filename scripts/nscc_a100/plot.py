"""Figures of the NSCC A100 study (figures/nscc_a100/).

    python scripts/nscc_a100/plot.py            # every figure whose inputs exist

A  throughput crossover: steady-state ns per walker-step vs N, S1 / best P_t / G1 (A100), ABF and FR, per system
B  A100 per-step time split by phase (device clock of block 0): move + force, deposit, grid barrier, FR
C  median W_mid vs N, ABF and FR, on the routed backend (exploration), GPU-routed rungs marked
D  confirmation: R_W and R_e2e with 95 % CIs per system
E  e_F against measured wall-clock, selected arms (median and 25-75 % band over the confirmation seeds)
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(NC.ROOT, "figures", "nscc_a100")
LABEL = {"gateway": "Entropic gateway", "lta300": "Ethane / LTA 300 K", "lta150": "Ethane / LTA 150 K"}
COL = {"S1": "#7f7f7f", "P": "#1f77b4", "G1": "#d62728", "abf": "#1f77b4", "fr": "#d62728"}


def jload(p):
    with open(p) as f:
        return json.load(f)


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
            recs = [jload(q) for q in glob.glob(os.path.join(NC.OUT_ROOT, "confirm", system,
                                                             f"{m}{arm['N']}_{arm['backend']}", "s*.json"))]
            recs = [r for r in recs if r.get("status") == "complete"]
            if not recs:
                continue
            grid = np.logspace(-2, np.log10(max(r["wall_total_s"] for r in recs)), 200)
            E = []
            for r in recs:
                w = np.asarray(r["wall_s_at_save"])
                e = np.asarray(r["e_F"])
                idx = np.searchsorted(w, grid, side="right") - 1
                E.append(np.where(idx >= 0, e[np.clip(idx, 0, None)], np.nan))
            E = np.array(E)
            q25, q50, q75 = np.nanpercentile(E, [25, 50, 75], axis=0)
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
    for f in (fig_throughput, fig_gpu_split, fig_wmid, fig_confirm, fig_ef_wall):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
