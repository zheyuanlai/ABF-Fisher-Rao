#!/usr/bin/env python
"""Analyse the gateway replica ladder (docs/GATEWAY_REPLICA_LADDER.md, configs/gateway_replica_ladder/design.json).

  python scripts/analyze_gateway_replica_ladder.py --validate     # gate (a)+(b), before reading the ladder
  python scripts/analyze_gateway_replica_ladder.py                # ladder tables + figures

Every e_F is the accepted engine's own read-out: eb.HistogramABFEstimator.pmf_profile on the saved bin
accumulators (M_j, C_j), centred on the eval window, RMS at the grid nodes vs the analytic F.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import eb_abffr_core as eb  # noqa: E402

DESIGN = os.path.join(ROOT, "configs", "gateway_replica_ladder", "design.json")
OUT = os.path.join(ROOT, "results", "gateway_replica_ladder")
ACCEPTED = os.path.join(ROOT, "results", "histogram_abf", "gateway", "confirmation", "raw", "hist_nbins45.npz")
CPU, DT = torch.device("cpu"), torch.float64

# colours: categorical slots 1-2 of the reference palette (validated pair), neutral ink for the serial control
C_ABF, C_FR, C_SER, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#8a8984"


def em_mean_force(x, cell):
    """Mean force the Euler-Maruyama sampler actually converges to at fixed x: the y-update
    y <- (1 - omega^2 dt) y + sqrt(2 dt / beta) z has stationary variance 1 / (beta omega^2 (1 - omega^2 dt / 2)),
    so <omega omega' y^2 | x> = omega' / (beta omega (1 - omega^2 dt / 2)) instead of omega' / (beta omega).
    POST-HOC (found 2026-10-04 in this study): the analytic reference omits it; at dt 4e-4, omega_in 32 the
    factor is 1/(1 - 0.2048) = 1.26 at the gate and the resulting F floor is 0.00138 RMS -- every arm's
    seed-pooled endpoint converges to that floor under the analytic reference, and to ~0 under this one."""
    b, H, oo, oi, s, dt = (cell[k] for k in ("beta", "H", "omega_out", "omega_in", "s", "dt"))
    e = np.exp(-x * x / (2 * s * s))
    om = oo + (oi - oo) * e
    dom = -(oi - oo) * (x / (s * s)) * e
    return 4 * H * x * (x * x - 1) + dom / (om * b) / (1.0 - om * om * dt / 2.0)


class Scorer:
    def __init__(self, cell, reference="analytic"):
        self.x_grid, self.dx, self.eval_mask, self.idx0 = eb.build_grid(CPU, DT)
        t = lambda v: torch.tensor([[float(v)]], dtype=DT)
        self.p = (t(cell["beta"]), t(cell["H"]), t(cell["omega_out"]), t(cell["omega_in"]), t(cell["s"]))
        self.F_ref, self.Fp_ref = eb.reference_profiles(self.x_grid, self.eval_mask, *self.p)
        self.reference, self.cell = reference, cell
        if reference == "em":
            xf = np.linspace(eb.XMIN, eb.XMAX, 180 * 400 + 1)
            fp = em_mean_force(xf, cell)
            F = np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(xf))])
            Fg = np.interp(self.x_grid.numpy(), xf, F)
            Fg = Fg - Fg[self.eval_mask.numpy()].mean()
            self.F_ref = torch.tensor(Fg[None], dtype=DT)
        self._est = {}

    def est(self, nb, R):
        key = (nb, R)
        if key not in self._est:
            e = eb.HistogramABFEstimator(R, nb, 1.0, self.x_grid, CPU, DT)
            sub = eb.reference_mean_force(e.sub_x.unsqueeze(0), *self.p)
            if self.reference == "em":
                sub = torch.tensor(em_mean_force(e.sub_x.numpy(), self.cell)[None], dtype=DT)
            mask = (e.sub_x >= eb.EVAL_LO) & (e.sub_x <= eb.EVAL_HI)
            self._est[key] = (e, sub, mask)
        return self._est[key]

    def score(self, M, C):
        """M, C: (n_saves, nb) -> e_F, e_F' (function space) per save."""
        R, nb = M.shape
        e, sub, mask = self.est(nb, R)
        e.M = torch.as_tensor(M, dtype=DT); e.C = torch.as_tensor(C, dtype=DT)
        Fp_bins = e.bin_mean_force()
        B = e.pmf_profile(self.idx0, Fp_bins)
        Bc = B - B[:, self.eval_mask].mean(dim=1, keepdim=True)
        eF = eb.l2_error(Bc, self.F_ref.expand(R, -1), self.eval_mask).numpy()
        eFp = e.fp_error(sub, mask, Fp_bins).numpy()
        return eF, eFp


def boot_median(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# --------------------------------------------------------------------------------------------- validation
def validate(sc, design):
    dt = design["cell"]["dt"]
    acc = np.load(ACCEPTED, allow_pickle=True)
    n = int(acc["n_rows"])
    rows = [dict(seed=int(acc[f"r{i}/seed"]), init=str(acc[f"r{i}/init"]), method=str(acc[f"r{i}/method"]), i=i) for i in range(n)]
    # (a) the scorer reproduces the accepted own read-out from the accepted accumulators
    worst = 0.0
    for r in rows:
        eF, _ = sc.score(acc[f"r{r['i']}/Mh_t"], acc[f"r{r['i']}/Ch_t"])
        worst = max(worst, float(np.max(np.abs(eF - acc[f"r{r['i']}/l2_f_t"]))))
    gate_a = worst < 1e-10
    print(f"(a) scorer vs accepted l2_f_t: worst |diff| = {worst:.2e} -> {'PASS' if gate_a else 'FAIL'}")
    # (b) numba N=2048 vs accepted GPU rows, left init
    gpu = {}
    for r in rows:
        if r["init"] != "left":
            continue
        t = acc[f"r{r['i']}/t"]
        gpu.setdefault(r["method"], {})[r["seed"]] = dict(I=trapz(acc[f"r{r['i']}/l2_f_t"], t), fin=float(acc[f"r{r['i']}/final_l2_f"]),
                                                          ev=float(acc[f"r{r['i']}/n_die"]) + float(acc[f"r{r['i']}/n_clone"]))
    cpu = {"abf": {}, "fr_uniform": {}}
    for f in sorted(glob.glob(os.path.join(OUT, "validation", "raw", "N2048_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        t = (z["save_at"] - 1) * dt
        for a, name in enumerate(meta["arms"]):
            eF, _ = sc.score(z["M"][a], z["C"][a])
            key = "abf" if name.startswith("abf") else "fr_uniform"
            cpu[key][meta["seed"]] = dict(I=trapz(eF, t), fin=float(eF[-1]), ev=float(z["die"][a, -1] + z["clone"][a, -1]))
    res = {}
    for src, D in (("gpu", gpu), ("cpu", cpu)):
        seeds = sorted(set(D["abf"]) & set(D["fr_uniform"]))
        dI = [100 * (D["fr_uniform"][s]["I"] / D["abf"][s]["I"] - 1) for s in seeds]
        dF = [100 * (D["fr_uniform"][s]["fin"] / D["abf"][s]["fin"] - 1) for s in seeds]
        res[src] = dict(n=len(seeds), abf_I=float(np.median([D["abf"][s]["I"] for s in seeds])),
                        abf_fin=float(np.median([D["abf"][s]["fin"] for s in seeds])),
                        fr_I=float(np.median([D["fr_uniform"][s]["I"] for s in seeds])),
                        fr_fin=float(np.median([D["fr_uniform"][s]["fin"] for s in seeds])),
                        dI=boot_median(dI), dF=boot_median(dF), wins_I=int(np.sum(np.array(dI) < 0)),
                        events=float(np.median([D["fr_uniform"][s]["ev"] for s in seeds])))
        r = res[src]
        print(f"  {src}: n={r['n']}  ABF I_F {r['abf_I']:.4f} e_F(T) {r['abf_fin']:.5f} | FR I_F {r['fr_I']:.4f} e_F(T) {r['fr_fin']:.5f} | "
              f"dI_F {r['dI'][0]:+.1f}% [{r['dI'][1]:+.1f},{r['dI'][2]:+.1f}] ({r['wins_I']}/{r['n']})  d e_F(T) {r['dF'][0]:+.1f}%  events {r['events']:.0f}")
    g, c = res["gpu"], res["cpu"]
    checks = dict(abf_I=abs(c["abf_I"] / g["abf_I"] - 1) <= 0.10, abf_fin=abs(c["abf_fin"] / g["abf_fin"] - 1) <= 0.10,
                  effect_I=abs(c["dI"][0] - g["dI"][0]) <= 8.0, events=abs(c["events"] / g["events"] - 1) <= 0.10)
    gate_b = all(checks.values())
    print(f"(b) {checks} -> {'PASS' if gate_b else 'FAIL'}")
    out = dict(gate_a=dict(worst_abs_diff=worst, passed=gate_a), gate_b=dict(checks=checks, passed=bool(gate_b), gpu=g, cpu=c))
    os.makedirs(os.path.join(OUT, "validation"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "validation", "validation.json"), "w"), indent=2, default=float)
    return gate_a and gate_b


# --------------------------------------------------------------------------------------------- ladder
def load_ladder(sc, design):
    recs = []
    for f in sorted(glob.glob(os.path.join(OUT, "production", "raw", "N*_s*.npz"))):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        u = z["u"]
        lin = np.isclose(u * 200, np.round(u * 200)) & (u > 0)
        for a, name in enumerate(meta["arms"]):
            nb = meta["arm_nbins"][a]
            eF, eFp = sc.score(z["M"][a, :, :nb], z["C"][a, :, :nb])
            recs.append(dict(N=meta["N"], seed=meta["seed"], arm=name, method=("fr" if meta["arm_use_fr"][a] else "abf"), nb=nb,
                             u=u, eF=eF, eFp=eFp, Ibar=float(np.mean(eF[lin])), fin=float(eF[-1]), finp=float(eFp[-1]),
                             events=float(z["die"][a, -1] + z["clone"][a, -1]), deaths=float(z["die"][a, -1]),
                             min_ess=float(z["ess"][a].min()), final_ess=float(z["ess"][a, -1]),
                             T=meta["T"], cap=meta["cap"], n_fr=meta["n_fr_opportunities"], wall=meta["wall_seconds"]))
    return recs


def u_eps(u, e, eps):
    ok = e <= eps
    if not ok[-1]:
        return np.inf
    bad = np.where(~ok)[0]
    return float(u[0]) if len(bad) == 0 else float(u[bad[-1] + 1])


def tables(recs, design):
    Ns = sorted({r["N"] for r in recs}, reverse=True)
    widths = sorted({r["nb"] for r in recs})
    idx = {(r["N"], r["seed"], r["method"], r["nb"]): r for r in recs}
    out = {"per_width": {}}
    for nb in widths:
        eps = float(np.median([r["fin"] for r in recs if r["N"] == max(Ns) and r["method"] == "abf" and r["nb"] == nb]))
        rows = []
        for N in Ns:
            row = dict(N=N, T=float(N and design["budget"]["B_walker_steps"] // N * design["cell"]["dt"]))
            for m in ("abf", "fr"):
                rr = [r for r in recs if r["N"] == N and r["method"] == m and r["nb"] == nb]
                if not rr:
                    continue
                row[m] = dict(n=len(rr), Ibar=[float(np.median([r["Ibar"] for r in rr])), float(np.percentile([r["Ibar"] for r in rr], 25)), float(np.percentile([r["Ibar"] for r in rr], 75))],
                              fin=[float(np.median([r["fin"] for r in rr])), float(np.percentile([r["fin"] for r in rr], 25)), float(np.percentile([r["fin"] for r in rr], 75))],
                              finp=float(np.median([r["finp"] for r in rr])),
                              u_eps=float(np.median([u_eps(r["u"], r["eF"], eps) for r in rr])),
                              frac_reach_eps=float(np.mean([np.isfinite(u_eps(r["u"], r["eF"], eps)) for r in rr])),
                              events=float(np.median([r["events"] for r in rr])), deaths=float(np.median([r["deaths"] for r in rr])),
                              min_ess=float(np.median([r["min_ess"] for r in rr])), cap=rr[0]["cap"], n_fr=rr[0]["n_fr"])
            seeds = sorted({s for (n_, s, m, b) in idx if n_ == N and b == nb and m == "fr"})
            if seeds:
                dI = [100 * (idx[(N, s, "fr", nb)]["Ibar"] / idx[(N, s, "abf", nb)]["Ibar"] - 1) for s in seeds]
                dF = [100 * (idx[(N, s, "fr", nb)]["fin"] / idx[(N, s, "abf", nb)]["fin"] - 1) for s in seeds]
                dFp = [100 * (idx[(N, s, "fr", nb)]["finp"] / idx[(N, s, "abf", nb)]["finp"] - 1) for s in seeds]
                row["effect"] = dict(n=len(seeds), dIbar=boot_median(dI), dfin=boot_median(dF), dfinp=boot_median(dFp),
                                     wins_Ibar=int(np.sum(np.array(dI) < 0)), wins_fin=int(np.sum(np.array(dF) < 0)))
            rows.append(row)
        out["per_width"][str(nb)] = dict(eps=eps, rows=rows)
    return out


def scoreboard(tab, design):
    lines = ["# Gateway replica ladder at equal force-evaluation budget -- scoreboard", "",
             f"Budget B = {design['budget']['B_walker_steps']:,} walker-steps per arm; n_steps = B/N; T = n_steps x dt. "
             "Ibar_F = (1/B) int_0^B e_F(b) db (mean over 200 budget fractions); effects are per-seed paired (FR-ABF)/ABF, median [bootstrap 95 % CI], wins = seeds with FR < ABF.", ""]
    for nb, blk in tab["per_width"].items():
        lines += [f"## {nb} bins (Delta {3.6 / int(nb):.3f}){' -- PRIMARY' if int(nb) == design['primary_width'] else ''}; eps = {blk['eps']:.5f} (median ABF e_F(B) at N=2048)", "",
                  "| N | T | ABF Ibar_F | FR Ibar_F | dIbar_F (FR vs ABF) | wins | ABF e_F(B) | FR e_F(B) | d e_F(B) | wins | ABF u_eps | FR u_eps | FR events | FR min ESS/N | cap |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in blk["rows"]:
            a, f, e = r.get("abf"), r.get("fr"), r.get("effect")
            fmt = lambda v: "inf" if not np.isfinite(v) else f"{v:.3f}"
            if f and e:
                lines.append(f"| {r['N']} | {r['T']:g} | {a['Ibar'][0]:.5f} | {f['Ibar'][0]:.5f} | {e['dIbar'][0]:+.1f} % [{e['dIbar'][1]:+.1f}, {e['dIbar'][2]:+.1f}] | {e['wins_Ibar']}/{e['n']} | "
                             f"{a['fin'][0]:.5f} | {f['fin'][0]:.5f} | {e['dfin'][0]:+.1f} % [{e['dfin'][1]:+.1f}, {e['dfin'][2]:+.1f}] | {e['wins_fin']}/{e['n']} | "
                             f"{fmt(a['u_eps'])} | {fmt(f['u_eps'])} | {f['events']:.0f} | {f['min_ess']:.2f} | {f['cap']} |")
            else:
                lines.append(f"| {r['N']} | {r['T']:g} | {a['Ibar'][0]:.5f} | -- | -- | -- | {a['fin'][0]:.5f} | -- | -- | -- | {fmt(a['u_eps'])} | -- | -- | -- | -- |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- figures
def figures(recs, tab, design, figdir, sfx=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C_MUTED,
                         "axes.labelcolor": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.titlesize": 9.5,
                         "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.8, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6})
    os.makedirs(figdir, exist_ok=True)
    nbP = int(design["primary_width"])
    Ns = sorted({r["N"] for r in recs}, reverse=True)
    reflab = "  [e_F vs dt-consistent reference, POST-HOC]" if tab.get("reference") == "em" else "  [e_F vs analytic F, preregistered]"

    def curves(N, m, nb):
        rr = [r for r in recs if r["N"] == N and r["method"] == m and r["nb"] == nb]
        if not rr:
            return None
        E = np.stack([r["eF"] for r in rr])
        return rr[0]["u"], np.median(E, 0), np.percentile(E, 25, 0), np.percentile(E, 75, 0)

    ser = curves(1, "abf", nbP)

    for nb in sorted({r["nb"] for r in recs}):
        tag = f"h{nb}"
        # ---- small multiples: e_F vs normalised budget, one panel per N --------------------------------
        fig, axs = plt.subplots(3, 4, figsize=(11, 7.6), sharex=True, sharey=True)
        for ax, N in zip(axs.flat, Ns):
            s1 = curves(1, "abf", nb)
            ax.plot(s1[0], s1[1], color=C_SER, lw=1.2, ls=(0, (4, 2)), label="ABF, N = 1 (serial)")
            for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
                c = curves(N, m, nb)
                if c is None:
                    continue
                ax.fill_between(c[0], c[2], c[3], color=col, alpha=0.15, lw=0)
                ax.plot(c[0], c[1], color=col, label=lab)
            T = design["budget"]["B_walker_steps"] // N * design["cell"]["dt"]
            ax.set_title(f"N = {N}   (T = {T:g})")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="major")
        for ax in axs[-1]:
            ax.set_xlabel("budget used  b / B")
        for ax in axs[:, 0]:
            ax.set_ylabel("e_F  (L2 error of F)")
        axs.flat[0].legend(loc="lower left", fontsize=8)
        fig.suptitle(f"Entropic gateway, equal force-evaluation budget B = N x n_steps = 2048 x 1e5;  median over 32 seeds, IQR band  ({nb} bins){reflab}",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(figdir, f"fig_ladder_curves_{tag}{sfx}.{ext}"), dpi=170)
        plt.close(fig)

    # ---- headline: (a) curves for 4 rungs on one axis, (b) FR effect vs N, (c) absolute error vs N ----
    rows = tab["per_width"][str(nbP)]["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ax = axs[0]
    show = [2048, 128, 8]
    styles = {2048: "-", 128: (0, (5, 1.5)), 8: (0, (1.5, 1.2))}
    for N in show:
        for m, col in (("abf", C_ABF), ("fr", C_FR)):
            c = curves(N, m, nbP)
            ax.plot(c[0], c[1], color=col, ls=styles[N], lw=1.7)
    ax.plot(ser[0], ser[1], color=C_SER, lw=2.2, ls="-", alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True)
    ax.set_xlabel("budget used  b / B   (b = N x steps)"); ax.set_ylabel("e_F  (median of 32 seeds)")
    ax.set_title("(a) error vs normalised budget", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_ABF, lw=2, label="ABF"), Line2D([], [], color=C_FR, lw=2, label="ABF + FR"),
         Line2D([], [], color=C_SER, lw=2.2, label="ABF, N = 1 (serial)")]
    h += [Line2D([], [], color=C_MUTED, ls=styles[N], lw=1.5, label=f"N = {N}") for N in show]
    lo_a = min(np.nanmin(curves(N, m, nbP)[1]) for N in show for m in ("abf", "fr")); lo_a = min(lo_a, np.nanmin(ser[1]))
    ax.set_ylim(lo_a / 3.0, 0.5); ax.legend(handles=h, loc="lower left", fontsize=7.5, ncol=2)

    ax = axs[1]
    rr = [r for r in rows if "effect" in r]
    xN = np.array([r["N"] for r in rr], float)
    for key, col, mk, lab, off in (("dIbar", C_INK, "o", "integrated error  (1/B) \u222b e_F db", 0.95),
                                   ("dfin", C_MUTED, "D", "final error  e_F(B)", 1.05)):
        v = np.array([r["effect"][key] for r in rr])
        ax.errorbar(xN * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=4.5, color=col,
                    mfc=col, mec="white", mew=0.8, lw=1, capsize=2, label=lab)
        ax.plot(xN * off, v[:, 0], color=col, lw=1, alpha=0.6)
    ax.axhline(0, color=C_MUTED, lw=0.8)
    ax.text(0.02, 0.04, "FR helps", transform=ax.transAxes, color=C_MUTED, fontsize=8, va="bottom")
    ax.text(0.02, 0.96, "FR hurts", transform=ax.transAxes, color=C_MUTED, fontsize=8, va="top")
    ax.set_xscale("log", base=2); ax.grid(True, axis="y")
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512", "1k", "2k"])
    ax.set_xlabel("replicas N   (n_steps = B / N)"); ax.set_ylabel("ABF+FR vs ABF, paired median change (%)")
    ax.set_title("(b) Fisher-Rao benefit vs number of replicas", loc="left")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.08))

    ax = axs[2]
    for m, col, lab in (("abf", C_ABF, "ABF"), ("fr", C_FR, "ABF + FR")):
        sel = [r for r in rows if m in r]
        x = np.array([r["N"] for r in sel], float)
        for key, ls, mk, kind in (("Ibar", "-", "o", "integrated"), ("fin", (0, (4, 2)), "^", "final")):
            v = np.array([r[m][key] for r in sel])
            ax.plot(x, v[:, 0], color=col, ls=ls, marker=mk, ms=4, label=f"{lab}, {kind}")
            ax.fill_between(x, v[:, 1], v[:, 2], color=col, alpha=0.12, lw=0)
    s1 = [r for r in rows if r["N"] == 1][0]["abf"]
    ax.plot([1], [s1["fin"][0]], marker="*", ms=11, color=C_SER, ls="none", zorder=5)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(True, which="both")
    allv = np.concatenate([np.array([r[m][k] for r in rows if m in r]).ravel() for m in ("abf", "fr") for k in ("Ibar", "fin")])
    lo_, hi_ = allv.min() / 1.4, allv.max() * 1.3
    yt = [v for v in (0.0002, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02) if lo_ <= v <= hi_]
    ax.set_ylim(lo_, hi_); ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.minorticks_off()
    ax.annotate(f"serial ABF (N = 1)\nfinal {s1['fin'][0]:.4f}", (1, s1["fin"][0]), xytext=(1.25, lo_ * 1.08),
                fontsize=7.5, color=C_INK, arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.7))
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k"])
    ax.set_xlabel("replicas N   (n_steps = B / N)"); ax.set_ylabel("e_F at equal budget  (median, IQR)")
    ax.set_title("(c) which (N, T) split is best at fixed budget", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Entropic gateway: replica-time trade-off at a fixed number of force evaluations B = 2.048e8 per arm ({nbP} bins, 32 seeds per N){reflab}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"fig_ladder_headline{sfx}.{ext}"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--reference", default="analytic", choices=["analytic", "em"],
                    help="analytic = preregistered; em = POST-HOC dt-consistent reference (see em_mean_force)")
    a = ap.parse_args()
    design = json.load(open(DESIGN))
    sc = Scorer(design["cell"], "analytic" if a.validate else a.reference)
    sfx = "" if a.reference == "analytic" else "_emref"
    if a.validate:
        ok = validate(sc, design)
        print("VALIDATION", "PASS" if ok else "FAIL")
        return
    recs = load_ladder(sc, design)
    print(f"{len(recs)} arm-runs loaded")
    tab = tables(recs, design)
    tab["reference"] = a.reference
    json.dump(tab, open(os.path.join(OUT, f"summary{sfx}.json"), "w"), indent=2, default=float)
    md = scoreboard(tab, design)
    if a.reference == "em":
        md = md.replace("-- scoreboard", "-- scoreboard, POST-HOC dt-consistent (Euler-Maruyama) reference", 1)
    open(os.path.join(OUT, f"scoreboard{sfx}.md"), "w").write(md)
    print(md)
    # compact per-run table for later re-analysis
    np.savez_compressed(os.path.join(OUT, f"per_run_curves{sfx}.npz"),
                        **{f"{r['arm']}_N{r['N']}_s{r['seed']}": np.stack([r["u"], r["eF"], r["eFp"]]) for r in recs})
    figures(recs, tab, design, os.path.join(OUT, "figures"), sfx)


if __name__ == "__main__":
    main()
