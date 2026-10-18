#!/usr/bin/env python
"""Figures 2-6 of docs/PENTANE_R15_DT_FR.md from the committed raw runs (Figure 1: analyze_stage_a.py).

    python scripts/pentane_r15_dt_fr/make_figures.py
-> results/pentane_r15_dt_fr/figures/fig{2..6}_*.{png,pdf}
Uses confirmatory runs when present (16 seeds), else pilot runs (4 seeds); every panel title says which.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r15metrics as M  # noqa: E402

FIG = os.path.join(M.RES, "figures")
COL = {5e-4: "#c0392b", 2.5e-4: "#e67e22", 1.25e-4: "#2471a3", 6.25e-5: "#17a589"}
ARMC = {"A": "#34495e", "F0.02": "#c0392b", "Fsec": "#8e44ad"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=150)
    plt.close(fig)


def band(ax, t, Y, color, label, ls="-"):
    Y = np.asarray(Y)
    ax.plot(t, np.median(Y, 0), color=color, lw=1.8, ls=ls, label=label)
    ax.fill_between(t, np.quantile(Y, 0.25, 0), np.quantile(Y, 0.75, 0), color=color, alpha=0.18, lw=0)


def collect(stage, ref):
    out = {}
    for r in M.load_runs(stage):
        m = r["meta"]
        key = (m["arm"], float(m["fr_rate"]) if m["arm"] == "fr" else 0.0, float(m["dt"]), float(m["T"]), m.get("deposit_interval"))
        out.setdefault(key, []).extend(M.per_seed(r, ref))
    return out


def fig2(ref, pilot, confirm):
    """ABF timestep convergence at equal physical time (+ equal force evaluations, deposit stride)."""
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    src = "pilot, 4 seeds"
    for (arm, rate, dt, T, dep), rows in sorted(pilot.items(), key=lambda kv: -kv[0][2]):
        if arm != "abf" or dep is not None:
            continue
        rows40 = [dict(r) for r in rows]
        t = rows[0]["t"]; sel = t <= 40 + 1e-9
        if T > 40 and any(k[0] == "abf" and k[2] == dt and k[3] == 40 and k[4] is None for k in pilot):
            pass
        band(ax[0], t[sel], [r["eF"][sel] for r in rows40], COL.get(dt, "k"), f"dt {dt:g} (T {T:g})")
        band(ax[1], t[sel], [r["eFp"][sel] for r in rows40], COL.get(dt, "k"), f"dt {dt:g}")
        fe = (np.round(t / dt) + 1) * 1024
        band(ax[2], fe, [r["eF"] for r in rows40], COL.get(dt, "k"), f"dt {dt:g}, to T {T:g}")
    for (arm, rate, dt, T, dep), rows in pilot.items():
        if arm == "abf" and dep is not None:
            t = rows[0]["t"]
            band(ax[3], t, [r["eF"] for r in rows], COL.get(dt, "k"), f"dt {dt:g}, deposit every {dep:g}", ls="--")
            base = [v for k, v in pilot.items() if k[0] == "abf" and k[2] == dt and k[4] is None]
            if base:
                bt = base[0][0]["t"]; s = bt <= 40 + 1e-9
                band(ax[3], bt[s], [r["eF"][s] for r in base[0]], COL.get(dt, "k"), f"dt {dt:g}, deposit every step")
    if confirm:
        for (arm, rate, dt, T, dep), rows in confirm.items():
            if arm == "abf":
                band(ax[0], rows[0]["t"], [r["eF"] for r in rows], COL.get(dt, "k"), f"dt {dt:g} confirmatory (16)", ls=":")
    ax[0].set_title(f"(a) ABF e_F(t), equal physical time ({src})", fontsize=10); ax[0].set_xlabel("physical time t")
    ax[1].set_title("(b) ABF e_F'(t), equal physical time", fontsize=10); ax[1].set_xlabel("physical time t")
    ax[2].set_title("(c) ABF e_F vs force evaluations (per seed)", fontsize=10); ax[2].set_xlabel("force evaluations (1024 walkers x steps)"); ax[2].set_xscale("log")
    ax[3].set_title("(d) deposition frequency at fixed dt", fontsize=10); ax[3].set_xlabel("physical time t")
    ax[0].set_ylabel("windowed RMS error of F (v2 reference)"); ax[1].set_ylabel("windowed RMS error of F'")
    for a in ax:
        a.legend(fontsize=7); a.set_yscale("log"); a.grid(alpha=0.3)
    fig.suptitle("Figure 2. ABF timestep convergence on pentane R15 (beta 2); median and IQR over seeds", fontsize=12)
    fig.tight_layout(); save(fig, "fig2_abf_timestep_convergence")


def arms_at(data, dt, sec):
    A = next((v for k, v in data.items() if k[0] == "abf" and k[2] == dt and k[4] is None and k[3] == 40.0), None)
    F = next((v for k, v in data.items() if k[0] == "fr" and k[1] == 0.02 and k[2] == dt), None)
    S = next((v for k, v in data.items() if sec and k[0] == "fr" and k[1] == sec and k[2] == dt), None)
    return A, F, S


def fig3(ref, data, src, dts, sec):
    fig, ax = plt.subplots(1, len(dts) + 1, figsize=(5.2 * (len(dts) + 1), 4.6))
    pos = 0
    for i, dt in enumerate(dts):
        A, F, S = arms_at(data, dt, sec)
        if not A:
            continue
        t = A[0]["t"]
        band(ax[i], t, [r["eF"] for r in A], ARMC["A"], "ABF")
        if F:
            band(ax[i], t, [r["eF"] for r in F], ARMC["F0.02"], "ABF + mFR (0.02)")
        if S:
            band(ax[i], t, [r["eF"] for r in S], ARMC["Fsec"], f"ABF + mFR ({sec:g})")
        ax[i].axvline(6, color="0.5", ls=":", lw=1); ax[i].set_yscale("log"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
        ax[i].set_title(f"(abc'[i]) dt {dt:g}: e_F(t) ({src})".replace("(abc'[i])", f"({'abc'[i]})"), fontsize=10); ax[i].set_xlabel("physical time t (= force evaluations / 1024 / dt)")
        for j, (X, lab) in enumerate(((F, "0.02"), (S, f"{sec:g}" if sec else None))):
            if X and lab:
                c = M.paired(X, A, "I_F")
                x = pos + 0.25 * j
                ax[-1].scatter(np.full(len(c["values"]), x) + np.linspace(-0.05, 0.05, len(c["values"])), 100 * np.array(c["values"]), s=12,
                               color=ARMC["F0.02" if j == 0 else "Fsec"])
                ax[-1].errorbar(x + 0.1, 100 * c["median"], yerr=[[100 * (c["median"] - c["ci95"][0])], [100 * (c["ci95"][1] - c["median"])]], fmt="ks", capsize=3)
                ax[-1].text(x, ax[-1].get_ylim()[1] if False else 100 * c["ci95"][1] + 0.5, f"dt {dt:g}\nrate {lab}", fontsize=7, ha="center")
        pos += 1
    ax[0].set_ylabel("windowed RMS error of F")
    ax[-1].axhline(0, color="k", lw=0.8); ax[-1].axhspan(-10, 10, color="0.9", zorder=0)
    ax[-1].set_ylabel("Delta I_F = I_F(mFR)/I_F(ABF) - 1  (%)"); ax[-1].set_xticks([]); ax[-1].set_title("(d) paired effect per seed, median [95% CI]; grey = +-10%", fontsize=10)
    fig.suptitle("Figure 3. ABF vs ABF + uniform mFR, free-energy error through time (median, IQR)", fontsize=12)
    fig.tight_layout(); save(fig, "fig3_abf_vs_mfr")


def fig4(ref, data, src, dt, sec):
    A, F, S = arms_at(data, dt, sec)
    if not A:
        return
    win = ref["win"]; g = ref["grid"]
    fig, ax = plt.subplots(1, 5, figsize=(25, 4.8))
    U = 1.0 / (win.sum() * ref["dz"])
    # (e) deposit-weighted torsional-family composition per conditional R bin vs the exact conditional
    edges = ref["cond_edges"]; kb = [k for k in range(len(edges) - 1) if edges[k + 1] > 2.0 and edges[k] < 3.55]
    famcol = ["#2e86c1", "#28b463", "#f1c40f", "#cb4335"]
    sets = [("ref", M.fam4_from9(ref["cond_basin"] / np.clip(ref["cond_basin"].sum(-1, keepdims=True), 1e-300, None)))]
    for X, lab in ((A, "ABF"), (F, "mFR 0.02"), (S, f"mFR {sec}")):
        if X:
            cbs = np.sum([r["cond_basin_end"] for r in X], 0)
            sets.append((lab, M.fam4_from9(cbs / np.clip(cbs.sum(-1, keepdims=True), 1, None))))
    w = 0.8 / len(sets)
    for j, (lab, comp) in enumerate(sets):
        x = np.arange(len(kb)) + (j - (len(sets) - 1) / 2) * w
        bottom = np.zeros(len(kb))
        for f in range(4):
            ax[4].bar(x, comp[kb, f], w * 0.95, bottom=bottom, color=famcol[f], edgecolor="k" if lab == "ref" else "none", lw=0.6,
                      label=(M.FAM[f] if j == 0 else None), hatch=(None if lab == "ref" else ["", "//", "xx"][min(j - 1, 2)]))
            bottom += comp[kb, f]
    ax[4].set_xticks(np.arange(len(kb))); ax[4].set_xticklabels([f"{0.5 * (edges[k] + edges[k + 1]):.2f}" for k in kb], fontsize=7, rotation=45)
    ax[4].set_xlabel("R15 bin centre"); ax[4].set_ylabel("family share of deposits")
    ax[4].set_title("(e) torsional family per R bin: exact (outlined) | " + " | ".join(l for l, _ in sets[1:]) + " (hatched)", fontsize=9)
    ax[4].legend(fontsize=7, loc="lower right")
    for X, lab, c in ((A, "ABF", ARMC["A"]), (F, "mFR 0.02", ARMC["F0.02"]), (S, f"mFR {sec}", ARMC["Fsec"])):
        if not X:
            continue
        P = np.array([r["p_hat_end"] for r in X]); P = P / (P[:, win].sum(1, keepdims=True) * ref["dz"])
        ax[0].plot(g[win], np.median(P[:, win], 0), color=c, lw=1.6, label=lab)
        t = X[0]["t"]
        fc = np.array([r["famc"] for r in X]); fa = np.array([r["fama"] for r in X])
        for k, ls in zip(range(4), ["-", "--", ":", "-."]):
            ax[1].plot(t, np.median(fc[:, :, k], 0), color=c, ls=ls, lw=1.4, label=(f"{lab}: {M.FAM[k]}" if True else None))
            ax[2].plot(t, np.median(fa[:, :, k] / np.clip(fc[:, :, k], 1e-9, None), 0), color=c, ls=ls, lw=1.4)
        band(ax[3], t, [r["lowsup"] for r in X], c, lab)
        if lab != "ABF":
            bh = np.array([r["birth_hist"] for r in X]).sum(0); dh = np.array([r["death_hist"] for r in X]).sum(0)
            ax0b = ax[0].twinx() if not hasattr(fig4, "_tw") else fig4._tw
            fig4._tw = ax0b
            ax0b.step(g, bh / max(bh.sum(), 1), color=c, lw=0.8, ls="--", where="mid", label=f"{lab} births")
            ax0b.step(g, -dh / max(dh.sum(), 1), color=c, lw=0.8, ls=":", where="mid", label=f"{lab} deaths (neg.)")
            ax0b.set_ylabel("birth / death share per bin"); ax0b.legend(fontsize=7, loc="lower right")
    if hasattr(fig4, "_tw"):
        del fig4._tw
    ax[0].axhline(U, color="k", ls="--", lw=1, label="uniform target on window")
    ax[0].set_title(f"(a) final walker R marginal (window), dt {dt:g} ({src})", fontsize=10); ax[0].set_xlabel("R15"); ax[0].legend(fontsize=7, loc="upper left")
    ax[1].set_title("(b) walker share per torsional family", fontsize=10); ax[1].set_xlabel("t"); ax[1].legend(fontsize=6, ncol=2); ax[1].set_yscale("log")
    ax[2].set_title("(c) distinct lineages / walkers per family\n(1 = all independent; clones lower it)", fontsize=10); ax[2].set_xlabel("t")
    ax[3].set_title("(d) under-supported fraction of the window", fontsize=10); ax[3].set_xlabel("t"); ax[3].legend(fontsize=8)
    fig.suptitle("Figure 4. Regional exploration: genuine discovery vs descendants (medians over seeds)", fontsize=12)
    fig.tight_layout(); save(fig, "fig4_regional_exploration")


def fig5(ref, data, src, calib, dts, sec):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sets = []
    for dt in dts:
        A, F, S = arms_at(data, dt, sec)
        if F:
            sets.append((F, f"0.02, dt {dt:g} ({src})", COL.get(dt, "k"), "-"))
        if S:
            sets.append((S, f"{sec:g}, dt {dt:g} ({src})", COL.get(dt, "k"), "--"))
    for (arm, rate, dt, T, dep), rows in sorted(calib.items()):
        sets.append((rows, f"{rate:g}, dt {dt:g} (calibration)", {0.02: "#7f8c8d", 0.05: "#16a085", 0.1: "#8e44ad"}.get(rate, "k"), ":"))
    for rows, lab, c, ls in sets:
        t = rows[0]["t"]
        ax[0].plot(t, np.median([r["repl"] for r in rows], 0), color=c, ls=ls, lw=1.6, label=lab)
        ax[1].plot(t, np.median([r["ess"] for r in rows], 0), color=c, ls=ls, lw=1.6, label=lab)
        ax[2].plot(t, np.median([r["maxf"] for r in rows], 0), color=c, ls=ls, lw=1.6, label=lab)
    ax[0].set_title("(a) cumulative replacements / N", fontsize=10); ax[1].set_title("(b) ancestor ESS / N", fontsize=10)
    ax[2].set_title("(c) largest ancestor population fraction", fontsize=10)
    ax[1].axhline(0.5, color="k", ls="--", lw=1); ax[2].axhline(0.05, color="k", ls="--", lw=1)
    for a in ax:
        a.set_xlabel("physical time t"); a.axvline(6, color="0.5", ls=":", lw=1); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("Figure 5. FR activity and genealogical diversity (dashed lines: frozen safety thresholds)", fontsize=12)
    fig.tight_layout(); save(fig, "fig5_fr_activity_genealogy")


def fig6(ref, data, src, dts, sec):
    win = ref["win"]; g = ref["grid"]
    fig, ax = plt.subplots(2, len(dts), figsize=(6.2 * len(dts), 8.2), squeeze=False)
    for i, dt in enumerate(dts):
        A, F, S = arms_at(data, dt, sec)
        if not A:
            continue
        ax[0, i].plot(g[win], ref["F"][win] - ref["F"][win].mean(), "k", lw=2.2, label="reference v2 (exact IS)")
        ax[1, i].plot(g[win], ref["Fp"][win], "k", lw=2.2, label="reference v2")
        for X, lab, c in ((A, "ABF", ARMC["A"]), (F, "mFR 0.02", ARMC["F0.02"]), (S, f"mFR {sec}", ARMC["Fsec"])):
            if not X:
                continue
            P = np.array([r["pmf_end"] for r in X]); P = P - (P[:, win] - ref["F"][win]).mean(1, keepdims=True) - ref["F"][win].mean()
            ax[0, i].plot(g[win], np.median(P[:, win], 0), color=c, lw=1.4, label=lab)
            ax[0, i].fill_between(g[win], np.quantile(P[:, win], 0.1, 0), np.quantile(P[:, win], 0.9, 0), color=c, alpha=0.15, lw=0)
            Q = np.array([r["mf_end"] for r in X])
            ax[1, i].plot(g[win], np.median(Q[:, win], 0), color=c, lw=1.2, label=lab)
        ax[0, i].set_title(f"(a{i + 1}) F(R15) at t = 40, dt {dt:g} ({src}); band 10-90%", fontsize=10)
        ax[1, i].set_title(f"(b{i + 1}) F'(R15) at t = 40, dt {dt:g}", fontsize=10)
        for a in ax[:, i]:
            a.set_xlabel("R15"); a.grid(alpha=0.3); a.legend(fontsize=8)
            a.axvspan(ref["window"][0], 2.4, color="#f9e79f", alpha=0.25, lw=0)
    ax[0, 0].set_ylabel("free energy (window-centred)"); ax[1, 0].set_ylabel("mean force")
    fig.suptitle("Figure 6. Final free-energy and mean-force profiles vs the corrected reference (yellow: compact region R < 2.4)", fontsize=12)
    fig.tight_layout(); save(fig, "fig6_profiles")


def main():
    ref = M.load_ref()
    pilot = collect("pilot", ref)
    confirm = collect("confirm", ref) if os.path.isdir(os.path.join(M.RES, "confirm", "raw")) else {}
    calib = collect("calib", ref) if os.path.isdir(os.path.join(M.RES, "calib", "raw")) else {}
    sa = os.path.join(M.RES, "stage_a", "summary.json"); sel = json.load(open(sa))["selected_dt"] if os.path.exists(sa) else None
    cs = os.path.join(M.RES, "calib", "selection.json"); sec = json.load(open(cs))["secondary_rate"] if os.path.exists(cs) else None
    data, src = (confirm, "confirmatory, 16 seeds") if confirm else (pilot, "pilot, 4 seeds")
    dts = [d for d in ([sel] if sel else []) + [5e-4] if d is not None]
    dts = list(dict.fromkeys(dts))
    fig2(ref, pilot, confirm)
    fig3(ref, data, src, dts, sec)
    fig4(ref, data, src, dts[0], sec)
    fig5(ref, data, src, calib, dts, sec)
    fig6(ref, data, src, dts, sec)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
