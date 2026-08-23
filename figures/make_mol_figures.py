"""Every molecular-campaign figure, redrawn from the stored result archives."""
from __future__ import annotations

import glob, json, math, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
CAM = os.path.join(ROOT, "results", "mol", "campaign")
REF = os.path.join(ROOT, "results", "mol", "ref")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "lines.linewidth": 1.3,
})
C = {"wfr_rot": "#888888", "wfr_shake": "#c0392b", "wfr_ymap": "#2980b9",
     "wfr_yref": "#16a085", "wfr_lmap": "#8e44ad", "wfr_lref": "#d35400",
     "wfr_ymh": "#111111", "wfr_lmh": "#e8b31f", "wfr_qref": "#95a5a6",
     "ti_cold": "#34495e", "ti_warm": "#7f8c8d", "abf": "#e67e22"}
LBL = {"wfr_rot": "RC-WFR naive (rotation)", "wfr_shake": "RC-WFR min-norm SHAKE",
       "wfr_ymap": "RC-WFR + oracle y-map", "wfr_yref": "RC-WFR + oracle y-refresh",
       "wfr_lmap": "RC-WFR + learned y-map", "wfr_lref": "RC-WFR + learned y-refresh",
       "wfr_ymh": "RC-WFR + Metropolis y-move (oracle proposal)",
       "wfr_lmh": "RC-WFR + Metropolis y-move (learned)",
       "wfr_qref": "RC-WFR + full conditional refresh",
       "ti_cold": "stratified TI (cold)", "ti_warm": "stratified TI (warm, oracle init)",
       "abf": "ABF"}


def load(tag, arm, system="PEN"):
    p = os.path.join(CAM, f"{system}_{arm}_{tag}.npz")
    return np.load(p) if os.path.exists(p) else None


_SYS = ["PEN"]


def med(a, axis=-1):
    return np.median(a, axis=axis)


def fig_profiles(out):
    """F(phi) reference profiles and the pentane conditional."""
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.2),
                           gridspec_kw=dict(wspace=0.42))
    for i, (s, t) in enumerate([("BUT", "butane"), ("PEN", "pentane")]):
        d = np.load(os.path.join(REF, f"{s}_ref.npz"))
        c, b = d["centers"], float(d["beta"])
        p = d["H1"][:, 0].sum(0); F = -np.log(p / p.sum()) / b; F -= F.min()
        ax[i].plot(np.rad2deg(c), F, color="#2c3e50")
        ax[i].set_xlabel(r"$\phi_1$ (deg)" if i else r"$\phi$ (deg)")
        ax[i].set_ylabel(r"$F$ (kcal/mol)")
        ax[i].set_title(f"{t}: reference $F$", loc="left")
        ax[i].set_xticks([-180, -90, 0, 90, 180])
    d = np.load(os.path.join(REF, "PEN_ref.npz"))
    c = d["centers"]; H2 = d["H2"].sum(0); b = float(d["beta"])
    P = H2 / np.maximum(H2.sum(1, keepdims=True), 1)
    W = -np.log(np.maximum(P, 1e-12)).T / b
    W = W - W.min()
    im = ax[2].imshow(W, origin="lower", aspect="auto",
                      extent=[-180, 180, -180, 180], cmap="viridis", vmax=4.0)
    ax[2].set_xlabel(r"$\phi_1$ (deg)"); ax[2].set_ylabel(r"$\phi_2$ (deg)")
    ax[2].set_title(r"pentane: $-\beta^{-1}\log p(\phi_2|\phi_1)$", loc="left")
    ax[2].plot([115, 115], [-180, 180], color="w", lw=.6, ls=":")
    ax[2].annotate("g$-$ suppressed 28x", xy=(115, -115), xytext=(-165, -155),
                   color="w", fontsize=6,
                   arrowprops=dict(arrowstyle="->", color="w", lw=.6))
    ax[2].set_xticks([-180, 0, 180]); ax[2].set_yticks([-180, 0, 180])
    plt.colorbar(im, ax=ax[2], label="kcal/mol")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_curves(tag, arms, out, floor=None, title="", unit="kcal/mol", system="PEN"):
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.5),
                           gridspec_kw=dict(wspace=0.32))
    for a in arms:
        d = load(tag, a, system)
        if d is None:
            continue
        n_cfg = int(d["n_cfg"]) if "n_cfg" in d else 1
        ns = d["e_F"].shape[1] // n_cfg
        e = d["e_F"].reshape(d["e_F"].shape[0], n_cfg, ns)[:, 0]
        dc = d["dcond"].reshape(d["dcond"].shape[0], n_cfg, ns)[:, 0]
        fe = d["fe"]
        ax[0].loglog(fe, med(e), color=C.get(a), label=LBL.get(a, a))
        ax[0].fill_between(fe, np.quantile(e, .25, -1), np.quantile(e, .75, -1),
                           color=C.get(a), alpha=.15, lw=0)
        m = dc[:, 0] > 0
        if dc.max() > 0:
            ax[1].loglog(fe, np.maximum(med(dc), 1e-4), color=C.get(a))
    if floor:
        ax[0].axhline(floor, color="k", ls=":", lw=1)
        ax[0].text(ax[0].get_xlim()[0] * 1.2, floor * 1.08, "estimator floor", fontsize=6)
    ax[0].set_xlabel("force evaluations"); ax[0].set_ylabel(rf"$e_F$ ({unit})")
    ax[1].set_xlabel("force evaluations"); ax[1].set_ylabel(r"$D^{y}_{\rm cond}$ (nats)")
    ax[0].set_title(title or "free-energy error", loc="left")
    ax[1].set_title("conditional error of the hidden mode", loc="left")
    ax[1].legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=6,
                 loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_kappa(out, arms=("wfr_shake", "wfr_rot", "wfr_yref", "wfr_ymap",
                         "wfr_ymh", "wfr_lmh")):
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5),
                           gridspec_kw=dict(wspace=0.30))
    for a in arms:
        d = load("kappa", a)
        if d is None:
            continue
        g = d["cfg_grid"]; n_cfg = int(d["n_cfg"]); ns = int(d["n_seed"])
        k = g[:, 0]
        e = d["e_F_final"].reshape(n_cfg, ns)
        dc = d["dcond"][-1].reshape(n_cfg, ns)
        ax[0].loglog(k, med(e), "o-", color=C.get(a), ms=3, label=LBL.get(a, a))
        ax[0].fill_between(k, np.quantile(e, .25, -1), np.quantile(e, .75, -1),
                           color=C.get(a), alpha=.15, lw=0)
        ax[1].loglog(k, np.maximum(med(dc), 1e-4), "o-", color=C.get(a), ms=3)
    for x in ax:
        x.set_xlabel(r"$\kappa_W$  (rad$^2$/time)")
    ax[0].set_ylabel(r"$e_F$ (kcal/mol)"); ax[1].set_ylabel(r"$D^{y}_{\rm cond}$")
    ax[0].set_title("faster transport: what it costs", loc="left")
    ax[1].set_title("...and what it buys the conditional", loc="left")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_kappa_screen(out, arms=("wfr_shake", "wfr_rot", "wfr_ymap", "wfr_yref"),
                     theta=0.3, floor=0.0127):
    """The transport-rate stress test, read out of the screening grid."""
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for a in arms:
        d = load("screen", a)
        if d is None:
            continue
        g = d["cfg_grid"]; n_cfg = int(d["n_cfg"]); ns = int(d["n_seed"])
        m = np.abs(g[:, 1] - theta) < 1e-9
        k = g[m, 0]
        e = d["e_F_final"].reshape(n_cfg, ns)[m]
        dc = d["dcond"][-1].reshape(n_cfg, ns)[m]
        o = np.argsort(k)
        ax[0].loglog(k[o], np.median(e, 1)[o], "o-", color=C.get(a), ms=3.5,
                     label=LBL.get(a, a))
        ax[0].fill_between(k[o], np.quantile(e, .25, 1)[o], np.quantile(e, .75, 1)[o],
                           color=C.get(a), alpha=.15, lw=0)
        ax[1].loglog(k[o], np.median(dc, 1)[o], "o-", color=C.get(a), ms=3.5)
    ax[0].axhline(floor, color="k", ls=":", lw=1)
    ax[0].text(ax[0].get_xlim()[0] * 1.05, floor * 1.1, "estimator floor", fontsize=6)
    for x in ax:
        x.set_xlabel(r"transport rate  $\kappa_W$")
    ax[0].set_ylabel(r"$e_F$ (kcal/mol)")
    ax[1].set_ylabel(r"$D^{\psi_2}_{\rm cond}$ (nats)")
    ax[0].set_title("faster transport: what it costs", loc="left")
    ax[1].set_title("...and what it buys the conditional", loc="left")
    ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_summary(tag, out, floor=0.0127, budget=1.0, unit="kcal/mol"):
    """One bar per arm at a fixed budget, sorted, with the floor marked."""
    arms = [a for a in LBL if load(tag, a) is not None]
    vals, los, his, names, cols = [], [], [], [], []
    for a in arms:
        d = load(tag, a)
        n_cfg = int(d["n_cfg"]) if "n_cfg" in d else 1
        ns = d["e_F"].shape[1] // n_cfg
        e = d["e_F"].reshape(d["e_F"].shape[0], n_cfg, ns)[:, 0]
        i = int(np.argmin(np.abs(d["fe"] - budget * d["fe"][-1])))
        vals.append(np.median(e[i])); los.append(np.quantile(e[i], .25))
        his.append(np.quantile(e[i], .75)); names.append(LBL[a]); cols.append(C.get(a))
    o = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 0.26 * len(o) + 1.0))
    y = np.arange(len(o))
    ax.barh(y, np.array(vals)[o], color=[cols[i] for i in o], height=.7)
    ax.errorbar(np.array(vals)[o], y,
                xerr=[np.array(vals)[o] - np.array(los)[o],
                      np.array(his)[o] - np.array(vals)[o]],
                fmt="none", ecolor="k", elinewidth=.7, capsize=1.6)
    ax.axvline(floor, color="k", ls=":", lw=1)
    ax.text(floor * 1.05, len(o) - 0.4, "estimator floor", fontsize=6, rotation=90,
            va="top")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in o], fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlabel(rf"$e_F$ ({unit}), matched force evaluations")
    ax.set_title("every arm at the same budget", loc="left")
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_final_profiles(tag, arms, out):
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4))
    ref = None
    for a in arms:
        d = load(tag, a)
        if d is None:
            continue
        ref = d["F_ref"][0]
        n_cfg = int(d["n_cfg"]) if "n_cfg" in d else 1
        ns = d["F"].shape[1] // n_cfg
        F = d["F"][-1].reshape(n_cfg, ns, -1)[0]
        x = np.linspace(-180, 180, F.shape[-1])
        m = np.median(F, 0); m = m - m.mean()
        ax.plot(x, m - (ref - ref.mean()), color=C.get(a), label=LBL.get(a, a))
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel(r"$\phi_1$ (deg)"); ax.set_ylabel(r"$\hat F - F_{\rm ref}$ (kcal/mol)")
    ax.set_title("where the lift error lives", loc="left")
    ax.set_xticks([-180, -90, 0, 90, 180]); ax.legend(frameon=False, fontsize=6)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


def fig_hexane(out):
    """Which fiber mode has to be promoted -- prediction against measurement."""
    import json
    rows = [("HEX_wfr_rot_hex", "none", "#888888"),
            ("HEX_wfr_ymh_hex_p1", r"$\phi_2$", "#16a085"),
            ("HEX_wfr_ymh_hex_p2", r"$\phi_3$", "#c0392b"),
            ("HEX_wfr_ymh_hex_p12", "both", "#2980b9")]
    v, lo, hi, lab, col = [], [], [], [], []
    for f, name, c in rows:
        p_ = os.path.join(CAM, f + ".npz")
        if not os.path.exists(p_):
            return
        d = np.load(p_)["e_F_final"]
        v.append(np.median(d)); lo.append(np.quantile(d, .25))
        hi.append(np.quantile(d, .75)); lab.append(name); col.append(c)
    dg = json.load(open(os.path.join(ROOT, "results/mol/HEX_mode_diagnostic.json")))
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3),
                           gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.5]))
    ax[0].bar([0, 1], dg["damage"], color=["#16a085", "#c0392b"], width=.6)
    ax[0].set_yscale("log"); ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([r"$\phi_2$", r"$\phi_3$"])
    ax[0].set_ylabel(r"predicted damage  $S_k\,\tau_k^2$")
    ax[0].set_title("predicted, before any run", loc="left")
    x = np.arange(len(v))
    ax[1].bar(x, v, color=col, width=.65)
    ax[1].errorbar(x, v, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)],
                   fmt="none", ecolor="k", elinewidth=.7, capsize=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylabel(r"$e_F$ (kcal/mol)")
    ax[1].set_xlabel("fiber mode promoted")
    ax[1].set_title("measured", loc="left")
    ax[0].set_xlabel(r"$\phi_2$ adjacent   |   $\phi_3$ distal, 1.6x SLOWER",
                     fontsize=6.5)
    ax[1].annotate("promoting the SLOWER mode\nbuys nothing", xy=(2, hi[2]),
                   xytext=(1.55, hi[2] * 1.28), fontsize=6, ha="center",
                   arrowprops=dict(arrowstyle="->", lw=.6))
    ax[1].set_ylim(0, hi[2] * 1.55)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE), exist_ok=True)
    fig_profiles(os.path.join(HERE, "figMOL1_systems"))
    arms = ["ti_cold", "ti_warm", "abf", "wfr_shake", "wfr_rot", "wfr_ymap",
            "wfr_yref", "wfr_lmap", "wfr_lref", "wfr_ymh", "wfr_lmh"]
    if load("confirm", "wfr_rot") is not None:
        fig_curves("confirm", arms, os.path.join(HERE, "figMOL2_curves"), floor=0.0127)
        fig_final_profiles("confirm", arms, os.path.join(HERE, "figMOL4_profiles"))
    if load("screen", "wfr_rot") is not None:
        fig_kappa_screen(os.path.join(HERE, "figMOL3_kappa_screen"))
    if load("confirm", "wfr_rot") is not None:
        fig_summary("confirm", os.path.join(HERE, "figMOL5_summary"))
    if load("kappa", "wfr_rot") is not None:
        fig_kappa(os.path.join(HERE, "figMOL3_kappa"))
    if os.path.exists(os.path.join(CAM, "HEX_wfr_ymh_hex_p12.npz")):
        fig_hexane(os.path.join(HERE, "figMOL6_hexane"))
    if load("confirm", "wfr_rot", "ALA") is not None:
        fig_curves("confirm", arms, os.path.join(HERE, "figMOL7_alanine"),
                   floor=0.156, unit="kJ/mol", system="ALA")
    print("figures written")
