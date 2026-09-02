#!/usr/bin/env python
"""Analyzer for the WCA FR + targeted solvent relaxation campaign: stages W0, W1, W2.

Prereg: configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json.  Written before the
data.  W0: tau_f(z), v_constr(z), validity gates -> tau_map.json.  W1: read-out intersection
first, then Delta I_F(F_rho, F), exact compute accounting, rho* rule -> rho_selection.json.
W2: primary F_R vs F, controls, compute endpoint R_C, frozen outcome labels.

    python scripts/analyze_wca_targeted_relax.py --stage W0|W1|W2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_uniform_wca import tau as tau_to, PERSIST, FRACTIONS, ESS_FLOOR, WMAX_CAP   # noqa: E402
from analyze_wca_bandwidth_audit import readouts, Z_LO, Z_HI, LEGACY_KEY               # noqa: E402
import wca_abffr_core as core                                                          # noqa: E402

PREREG = os.path.join(ROOT, "configs/targeted_relax_campaign/wca_fr_targeted_relax_prereg.json")
CAMPAIGN = os.path.join(ROOT, "results/targeted_relax_campaign/wca")
N_BOOT, BOOT_SEED = 10000, 20260906
PLATEAU_TOL = 0.02
TAU_FLOOR, TAU_CAP = 0.02, 20.0
LADDER_RHO = (0.25, 0.5, 1.0)
MARGIN, COMPUTE_GATE = -10.0, 0.80


# ----------------------------------------------------------------------------- helpers
def boot_median(d, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    med = np.median(np.asarray(d)[idx], axis=1)
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def paired(arm, ref, k):
    d = 100.0 * (np.asarray(arm) - np.asarray(ref)) / np.asarray(ref)
    return dict(median=float(np.median(d)), ci95=boot_median(d, BOOT_SEED + k), wins=int((d < 0).sum()), n=int(len(d)),
                per_seed=[float(v) for v in d])


def fmt(c):
    return f"{c['median']:+7.2f}% [{c['ci95'][0]:+7.2f},{c['ci95'][1]:+7.2f}] {c['wins']:2d}/{c['n']}"


def load_runs(raw_dir, stage):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"{stage}__*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        r = {k: d[k] for k in d.files}
        runs.setdefault(str(r["name"]), {})[int(r["seed"])] = r
    return runs


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_axis(run):
    """Cumulative force evaluations (replica-steps) at every save: outer N x step + inner steps so far."""
    steps = np.asarray(run["profile_steps"], float); N = float(run["n_replicas"])
    inner = np.cumsum(np.asarray(run["relax_steps"], float)) if "relax_steps" in run else np.zeros_like(steps)
    return N * steps + inner


def first_reach(axis, curve, eps, persist=PERSIST):
    """First axis value at which curve <= eps and stays so for persist x (axis span)."""
    return tau_to(axis, curve, eps, persist)


# ----------------------------------------------------------------------------- W0
def stage_w0(a):
    d0 = os.path.join(CAMPAIGN, "W0")
    sel = json.load(open(os.path.join(d0, "selection.json")))
    cz = np.load(os.path.join(d0, "constrained.npz"), allow_pickle=True)
    zk = np.asarray(cz["z_sites"], float); kind = np.array([str(k) for k in cz["kind"]]); dt = float(cz["dt"]); rec = int(cz["record_every"])
    grid = np.asarray(sel["grid"], float); vhat = np.asarray(sel["vhat_median_final"], float)
    n = len(zk)
    print(f"W0: {n} constrained sites ({int((kind == 'site').sum())} quantile sites, {int((kind == 'control').sum())} controls); n_prod {int(cz['n_prod'])} steps"
          + ("  [extended]" if sel.get("extended") else ""))
    rows, curves = [], {}
    for j in range(n):
        f = np.asarray(cz[f"f_{j}"], np.float64)              # (B, T)
        B, T = f.shape
        tau_full, rho = core.autocorrelation_time(f, dt * rec, max_lag=min(T // 2, 4000))
        halves = [core.autocorrelation_time(f[:B // 2], dt * rec, max_lag=min(T // 2, 4000))[0],
                  core.autocorrelation_time(f[B // 2:], dt * rec, max_lag=min(T // 2, 4000))[0],
                  core.autocorrelation_time(f[:, :T // 2], dt * rec, max_lag=min(T // 4, 4000))[0],
                  core.autocorrelation_time(f[:, T // 2:], dt * rec, max_lag=min(T // 4, 4000))[0]]
        halves = [h for h in halves if np.isfinite(h) and h > 0]
        stable = bool(len(halves) >= 2 and max(halves) / min(halves) <= 2.0)
        resolved = bool(np.isfinite(tau_full) and tau_full <= (T * dt * rec) / 5.0)
        v_c = float(f.var(ddof=1))
        v_h = float(np.interp(zk[j], grid, vhat))
        rows.append(dict(z=float(zk[j]), kind=kind[j], tau_f=float(tau_full), tau_halves=[float(h) for h in halves], stable=stable,
                         resolved=resolved, v_constr=v_c, v_hat=v_h, mean_f=float(f.mean()), n_rep=int(B), n_samples=int(T)))
        curves[j] = rho[:min(len(rho), 2000)]
        print(f"  z={zk[j]:+.3f} {kind[j]:>7}: tau_f {tau_full:8.4f} (halves {', '.join(f'{h:.3f}' for h in halves)}) stable={stable} resolved={resolved}"
              f"  v_constr {v_c:9.4f}  v_hat {v_h:9.4f}")
    sites = [r for r in rows if r["kind"] == "site"]; ctrl = [r for r in rows if r["kind"] == "control"]
    rho_s = spearman([r["v_hat"] for r in rows], [r["v_constr"] for r in rows])
    top = sorted(sites, key=lambda r: -r["v_hat"])[: max(1, len(sites) // 2)]
    sep = (np.median([r["v_constr"] for r in top]) > max(r["v_constr"] for r in ctrl)) if ctrl else True
    # sites carrying >= 50% of the sensitivity mass: by cumulative v_hat mass among sites (equal-mass quantiles -> the top half)
    mass_sites = sorted(sites, key=lambda r: -r["v_hat"])[: max(1, len(sites) // 2)]
    stab_frac = float(np.mean([r["stable"] and r["resolved"] for r in mass_sites]))
    passed_v = bool(rho_s >= 0.6 and sep)
    passed_tau = bool(stab_frac > 0.5)
    print(f"\n  Spearman(v_hat, v_constr) = {rho_s:.3f} (>= 0.6 required); top-half sites above both controls: {sep}")
    print(f"  tau stable & resolved on the sensitivity-carrying sites: {stab_frac:.2f} (> 0.5 required)")
    outcome = "W0_PASS" if (passed_v and passed_tau) else ("SENSITIVITY_INVALID" if not passed_v else "TAU_MAP_UNRESOLVED")
    print(f"  W0 outcome: {outcome}")
    # tau map on the sim grid: linear between sites (sorted by z), nearest-endpoint outside, floor/cap
    good = sorted([r for r in rows if np.isfinite(r["tau_f"]) and r["tau_f"] > 0], key=lambda r: r["z"])
    zs = np.array([r["z"] for r in good]); ts = np.array([r["tau_f"] for r in good])
    tau_grid = np.clip(np.interp(grid, zs, ts), TAU_FLOOR, TAU_CAP)
    h = hashlib.sha256(tau_grid.astype(np.float64).tobytes()).hexdigest()
    json.dump(dict(passed=bool(outcome == "W0_PASS"), outcome=outcome, tau_grid=tau_grid.tolist(), grid=grid.tolist(), sha256=h,
                   floor=TAU_FLOOR, cap=TAU_CAP, sites=rows, spearman=rho_s, controls_separated=bool(sep), tau_stability_fraction=stab_frac,
                   rule="linear in z between sites, nearest-endpoint outside, floor 0.02, cap 20"),
              open(os.path.join(d0, "tau_map.json"), "w"), indent=2, default=float)
    json.dump(dict(outcome=outcome, spearman=rho_s, sep=bool(sep), stability_fraction=stab_frac, sites=rows, tau_map_sha256=h),
              open(os.path.join(d0, "analysis.json"), "w"), indent=2, default=float)
    if not a.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(d0, "figures"); os.makedirs(fd, exist_ok=True)
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
            axes[0].plot(grid, vhat, color="#1b9e77", lw=1.5, label=r"online $\hat v(z)$ (median of 8 ABF runs)")
            axes[0].scatter([r["z"] for r in rows], [r["v_constr"] for r in rows], c=["#d95f02" if r["kind"] == "site" else "#7570b3" for r in rows], s=30, zorder=3, label="constrained Var(f | z)")
            axes[0].set_xlabel("z"); axes[0].set_ylabel("conditional variance of the local force"); axes[0].legend(fontsize=7, frameon=False); axes[0].grid(alpha=0.25)
            for j in range(n):
                lag = np.arange(len(curves[j])) * dt * rec
                axes[1].plot(lag, curves[j], lw=1, color=("#7570b3" if kind[j] == "control" else "#d95f02"), alpha=0.7)
            axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xlabel("lag (time)"); axes[1].set_ylabel(r"$C_f(t;z)/C_f(0;z)$"); axes[1].set_xlim(0, min(5 * max(1e-3, max(r["tau_f"] for r in good)), lag[-1])); axes[1].grid(alpha=0.25)
            axes[2].plot(grid, tau_grid, "k-", lw=1.4, label=r"frozen $\tau_f(z)$ map"); axes[2].scatter(zs, ts, color="#d95f02", s=25, zorder=3)
            axes[2].set_yscale("log"); axes[2].set_xlabel("z"); axes[2].set_ylabel(r"$\tau_f$ (time)"); axes[2].legend(fontsize=7, frameon=False); axes[2].grid(alpha=0.25, which="both")
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fd, f"fig_W1W2_sensitivity_and_tau.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  (figures skipped: {e})")
    print(f"  wrote W0/tau_map.json (sha256 {h[:12]}), analysis.json")


# ----------------------------------------------------------------------------- shared W1/W2 machinery
def prepare(runs, arms):
    seeds = sorted(set.intersection(*[set(runs[m]) for m in arms]))
    assert seeds, "no complete seed block"
    any_run = runs[arms[0]][seeds[0]]
    grid = np.asarray(any_run["grid"], float); ref_F = np.asarray(any_run["reference_free_energy"], float)
    t = np.asarray(any_run["profile_times"], float); mask = (grid >= Z_LO) & (grid <= Z_HI); sigma = float(any_run.get("abf_smooth_sigma", 0.5))
    ro = {m: {s: readouts(runs[m][s], grid, mask, ref_F, sigma) for s in seeds} for m in arms}
    dev = max(abs(float(ro[m][s][LEGACY_KEY][-1]) - float(runs[m][s]["l2_f"])) for m in arms for s in seeds)
    assert dev < 1e-5, dev
    return seeds, grid, t, mask, ro


def plateau_intersection(ro, arms, seeds, ladder):
    """Largest ladder level inside EVERY arm's 2% plateau (relative to that arm's own ladder minimum of median e_F(T))."""
    per_arm = {}
    for m in arms:
        e = {lab: float(np.median([ro[m][s][lab][-1] for s in seeds])) for lab in ladder}
        emin = min(e.values()); per_arm[m] = {lab: dict(eF_T=e[lab], on_plateau=bool(e[lab] <= (1 + PLATEAU_TOL) * emin)) for lab in ladder}
    for lab in ladder:                                   # ladder is ordered from the widest kernel to raw
        if all(per_arm[m][lab]["on_plateau"] for m in arms):
            return lab, per_arm
    return "raw", per_arm


def curves_and_compute(runs, ro, arms, seeds, lab, t):
    I = {m: np.array([np.trapezoid(ro[m][s][lab], t) for s in seeds]) for m in arms}
    fin = {m: np.array([ro[m][s][lab][-1] for s in seeds]) for m in arms}
    med = {m: np.median([ro[m][s][lab] for s in seeds], axis=0) for m in arms}
    caxis = {m: np.median([compute_axis(runs[m][s]) for s in seeds], axis=0) for m in arms}
    return I, fin, med, caxis


def compute_to(med, caxis, m, eps):
    return first_reach(caxis[m], med[m], eps)


# ----------------------------------------------------------------------------- W1
def stage_w1(a):
    d1 = os.path.join(CAMPAIGN, "W1")
    runs = load_runs(os.path.join(d1, "raw"), "W1")
    arms = ["abf", "fr_uniform"] + [f"{p}_targ{r:g}" for r in LADDER_RHO for p in ("abf", "fr")]
    missing = [m for m in arms if m not in runs]
    assert not missing, f"missing arms {missing}"
    seeds, grid, t, mask, ro = prepare(runs, arms)
    ladder = list(ro[arms[0]][seeds[0]].keys())
    print(f"W1: {len(seeds)} seeds {seeds}; arms {arms}; ladder {ladder}")
    h2, per_arm = plateau_intersection(ro, arms, seeds, ladder)
    print(f"read-out intersection (2% plateau per arm): h_read** = {h2}")
    for m in arms:
        print(f"    {m:>14}: " + "  ".join(f"{lab} {per_arm[m][lab]['eF_T']:.5f}{'*' if per_arm[m][lab]['on_plateau'] else ' '}" for lab in ladder))
    I, fin, med, caxis = curves_and_compute(runs, ro, arms, seeds, h2, t)
    eps_F = float(med["fr_uniform"][-1]); eps_A = float(med["abf"][-1])
    print(f"\ncontrasts at {h2} (paired by seed; {len(seeds)} seeds -> descriptive CIs); compute = outer N x step + inner replica-steps (exact)")
    res, k = {}, 0
    C_F = compute_to(med, caxis, "fr_uniform", eps_F); C_A = compute_to(med, caxis, "abf", eps_A)
    for rho in LADDER_RHO:
        F, A = f"fr_targ{rho:g}", f"abf_targ{rho:g}"
        cF = paired(I[F], I[fr := "fr_uniform"], k); cFf = paired(fin[F], fin[fr], k + 1); cA = paired(I[A], I["abf"], k + 2); cAf = paired(fin[A], fin["abf"], k + 3); k += 4
        CF = compute_to(med, caxis, F, eps_F); ratio = CF / C_F if np.isfinite(C_F) and C_F > 0 else float("inf")
        cost = float(np.median([float(runs[F][s]["relax_cost_ratio"]) for s in seeds]))
        wall_in = float(np.median([float(runs[F][s]["relax_inner_wall_seconds"]) for s in seeds])); wall = float(np.median([float(runs[F][s]["wall_seconds"]) for s in seeds]))
        act = float(np.median([float(np.mean(runs[F][s]["relax_active_frac"][8:])) for s in seeds]))
        passes = bool(cF["median"] <= MARGIN and ratio <= COMPUTE_GATE)
        res[f"{rho:g}"] = dict(rho=rho, F_vs_F0=dict(d_int=cF, d_fin=cFf), A_vs_A0=dict(d_int=cA, d_fin=cAf), C_F_rho=CF, C_F=C_F, compute_ratio=ratio,
                              actual_cost_ratio=cost, active_frac=act, wall_seconds=wall, inner_wall_seconds=wall_in, passes=passes)
        print(f"  rho={rho:g}: F_rho vs F  dI_F {fmt(cF)}  final {cFf['median']:+.1f}%;  A_rho vs A dI_F {fmt(cA)}  final {cAf['median']:+.1f}%;  "
              f"C(eps_F) ratio {ratio:.3f};  actual cost {cost:.3f}x, active {act:.3f}, inner wall {wall_in:.0f}/{wall:.0f}s -> {'PASS' if passes else 'fail'}")
    cF0 = paired(I["fr_uniform"], I["abf"], 99)
    print(f"  positive control F vs A: dI_F {fmt(cF0)}")
    passing = [r for r in LADDER_RHO if res[f"{r:g}"]["passes"]]
    rho_star = min(passing) if passing else None
    licensed = rho_star is not None
    # mechanism: occupancy, v_hat, budget fraction on the grid (median over seeds; the top rho as the illustration)
    mech = {}
    for rho in LADDER_RHO:
        F = f"fr_targ{rho:g}"
        hist = np.median([np.asarray(runs[F][s]["relax_budget_hist"], float) for s in seeds], axis=0)
        occ = np.median([np.asarray(runs[F][s]["final_eff_counts"], float) for s in seeds], axis=0)
        vh = np.median([np.asarray(runs[F][s]["final_vhat"], float) for s in seeds], axis=0)
        mech[f"{rho:g}"] = dict(budget_frac=(hist / max(hist.sum(), 1e-300)).tolist(), occupancy_frac=(occ / max(occ.sum(), 1e-300)).tolist(), vhat=vh.tolist())
        # how concentrated: fraction of z-window carrying 80% of the budget vs 80% of the occupancy
        def width80(w):
            w = np.asarray(w) / max(np.sum(w), 1e-300); o = np.argsort(w)[::-1]; c = np.cumsum(w[o]); return float((c < 0.8).sum() + 1) / len(w)
        mech[f"{rho:g}"]["width80_budget"] = width80(hist); mech[f"{rho:g}"]["width80_occupancy"] = width80(occ)
        print(f"  rho={rho:g}: grid fraction carrying 80% of the relaxation budget {mech[f'{rho:g}']['width80_budget']:.3f} vs 80% of the occupancy {mech[f'{rho:g}']['width80_occupancy']:.3f}")
    print(f"\n  rho* = {rho_star} ({'licensed' if licensed else 'STOP = NO_COMPUTE_EFFICIENT_FR_RELAXATION'})")
    sel = dict(prereg=os.path.relpath(PREREG, ROOT), seeds=seeds, h_read_starstar=h2, readout_per_arm=per_arm, eps_F=eps_F, eps_A=eps_A,
               per_rho=res, rho_star=rho_star, licensed=bool(licensed), positive_control=cF0,
               stop=(None if licensed else "NO_COMPUTE_EFFICIENT_FR_RELAXATION"),
               rule="smallest rho <= 1 with median dI_F(F_rho, F) <= -10% and C_{F_rho}(eps_F)/C_F(eps_F) <= 0.8")
    json.dump(sel, open(os.path.join(d1, "rho_selection.json"), "w"), indent=2, default=float)
    json.dump(dict(sel, mechanism=mech, grid=grid.tolist(), tau_grid=np.asarray(runs[f"fr_targ{LADDER_RHO[0]:g}"][seeds[0]]["tau_grid"], float).tolist()),
              open(os.path.join(d1, "analysis.json"), "w"), indent=2, default=float)
    if not a.no_figures:
        w1_figs(d1, grid, t, med, caxis, arms, mech, runs, seeds, h2)
    print(f"  wrote W1/rho_selection.json, analysis.json")


def w1_figs(d1, grid, t, med, caxis, arms, mech, runs, seeds, h2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figures skipped: {e})"); return
    fd = os.path.join(d1, "figures"); os.makedirs(fd, exist_ok=True)
    col = {"abf": "#4d4d4d", "fr_uniform": "#d95f02"}
    ls = {"0.25": ":", "0.5": "-.", "1": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    key = f"{LADDER_RHO[-1]:g}"
    axes[0].plot(grid, mech[key]["occupancy_frac"], color="#4d4d4d", lw=1.3, label="occupancy (kernel counts)")
    vh = np.asarray(mech[key]["vhat"]); axes[0].plot(grid, vh / max(vh.sum(), 1e-300), color="#1b9e77", lw=1.3, label=r"$\hat v(z)$ (normalised)")
    axes[0].plot(grid, mech[key]["budget_frac"], color="#7570b3", lw=1.6, label=f"relaxation budget (rho {key})")
    axes[0].set_xlabel("z"); axes[0].set_ylabel("fraction per grid bin"); axes[0].legend(fontsize=7, frameon=False); axes[0].grid(alpha=0.25)
    for m in arms:
        base = "abf" if m.startswith("abf") else "fr_uniform"; l = ls.get(m.split("targ")[-1], "-") if "targ" in m else "-"
        axes[1].plot(t, med[m], color=col[base], ls=l, lw=1.2, label=m)
        axes[2].plot(caxis[m], med[m], color=col[base], ls=l, lw=1.2, label=m)
    axes[1].set_yscale("log"); axes[1].set_xlabel("outer time"); axes[1].set_ylabel(f"median $e_F$ at {h2}"); axes[1].legend(fontsize=6, frameon=False, ncol=2); axes[1].grid(alpha=0.25, which="both")
    axes[2].set_xscale("log"); axes[2].set_yscale("log"); axes[2].set_xlabel("total force evaluations (replica-steps)"); axes[2].grid(alpha=0.25, which="both")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, f"fig_W3W4W5_mechanism_time_compute.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- W2
def stage_w2(a):
    d2 = os.path.join(CAMPAIGN, "W2")
    sel = json.load(open(os.path.join(CAMPAIGN, "W1", "rho_selection.json")))
    rho = float(sel["rho_star"]); h2 = sel["h_read_starstar"]
    A, F, AR, FR, FRAND = "abf", "fr_uniform", f"abf_targ{rho:g}", f"fr_targ{rho:g}", f"fr_rand{rho:g}"
    runs = load_runs(os.path.join(d2, "raw"), "W2")
    arms = [A, F, AR, FR, FRAND]
    assert all(m in runs for m in arms), f"missing arms: {[m for m in arms if m not in runs]}"
    pre = json.load(open(PREREG)); want = pre["seeds"]["W2"][1] - pre["seeds"]["W2"][0] + 1
    seeds, grid, t, mask, ro = prepare(runs, arms)
    if len(seeds) < want:
        print(f"*** INCOMPLETE BLOCK: {len(seeds)} of {want} seeds -- the prereg forbids analysing it; stopping ***")
        return
    print(f"W2: {len(seeds)} seeds; rho* {rho:g}; h_read** {h2} (frozen from W1)")
    I, fin, med, caxis = curves_and_compute(runs, ro, arms, seeds, h2, t)
    NAMED = [("PRIMARY", FR, F), ("S_FR_vs_A", FR, A), ("S_AR_vs_A", AR, A), ("S_FR_vs_Frand", FR, FRAND), ("S_F_vs_A", F, A), ("S_Frand_vs_F", FRAND, F)]
    res, k = {}, 0
    print(f"\n{'tag':>14} {'contrast':>34} {'d I_F':>8} {'CI95':>20} {'wins':>6} {'d e_F(T)':>9} {'CI95':>20}")
    for tag, arm, ref in NAMED:
        ci = paired(I[arm], I[ref], k); cf = paired(fin[arm], fin[ref], k + 1); k += 2
        res[tag] = dict(arm=arm, ref=ref, d_int=ci, d_fin=cf)
        print(f"{tag:>14} {arm + ' vs ' + ref:>34} {ci['median']:+8.2f} [{ci['ci95'][0]:+8.2f},{ci['ci95'][1]:+8.2f}] {ci['wins']:3d}/{ci['n']} {cf['median']:+9.2f} [{cf['ci95'][0]:+8.2f},{cf['ci95'][1]:+8.2f}]")
    # sensitivity read-outs
    other = {}
    for lab in ro[A][seeds[0]]:
        if lab == h2:
            continue
        I2 = {m: np.array([np.trapezoid(ro[m][s][lab], t) for s in seeds]) for m in (FR, F)}
        other[lab] = paired(I2[FR], I2[F], 500 + hash(lab) % 100)
    print("primary at other read-outs: " + "; ".join(f"{lab} {fmt(c)}" for lab, c in other.items()))
    # genealogy on every FR-type run
    N = float(runs[F][seeds[0]]["n_replicas"])
    health = {}
    for m in (F, FR, FRAND):
        ess = [float(runs[m][s]["min_ancestor_ess_window"]) / N for s in seeds]; wm = [float(runs[m][s]["max_ancestor_frac_over_time"]) for s in seeds]
        health[m] = dict(min_ess_frac=min(ess), max_wmax=max(wm), ok=bool(min(ess) >= ESS_FLOOR and max(wm) <= WMAX_CAP))
    health_ok = all(v["ok"] for v in health.values())
    nan_ok = not any(bool(runs[m][s]["had_nan"]) for m in arms for s in seeds)
    print("genealogy: " + "; ".join(f"{m} ESS/N {v['min_ess_frac']:.3f} wmax {v['max_wmax']:.3f} ok={v['ok']}" for m, v in health.items()) + f"; NaN-free {nan_ok}")
    # compute endpoint
    e0 = float(med[A][0]); eps = {f"e0/{int(1 / f)}": e0 * f for f in FRACTIONS}; eps["abf_final"] = float(med[A][-1]); eps["fr_final"] = float(med[F][-1])
    cta = {}
    for nm, ep in eps.items():
        cta[nm] = {m: compute_to(med, caxis, m, ep) for m in arms}
        print(f"  compute to {nm:>9} (eps {ep:.5f}): " + "  ".join(f"{m} {cta[nm][m]:.3g}" for m in arms))
    R_C = cta["fr_final"][FR] / cta["fr_final"][F] if np.isfinite(cta["fr_final"][F]) and cta["fr_final"][F] > 0 else float("inf")
    cost = {m: float(np.median([float(runs[m][s].get("relax_cost_ratio", 0.0)) for s in seeds])) for m in arms}
    print(f"  R_C = C_FR(eps_F)/C_F(eps_F) = {R_C:.3f} (<= 0.80 for the compute verdict); actual cost ratios {cost}")
    # frozen outcome
    P = res["PRIMARY"]; resolved = P["d_int"]["median"] <= MARGIN and P["d_int"]["ci95"][1] < 0
    reversal = P["d_fin"]["ci95"][0] > 0
    spec = res["S_FR_vs_Frand"]; specific = spec["d_int"]["median"] <= MARGIN and spec["d_int"]["ci95"][1] < 0
    replicated = res["S_F_vs_A"]["d_int"]["ci95"][1] < 0
    if not (health_ok and nan_ok):
        outcome = "UNSAFE"
    elif not replicated:
        outcome = "FAILED_REPLICATION_OF_POSITIVE_CONTROL"
    elif not resolved or reversal:
        outcome = "NULL"
    elif not specific:
        outcome = "TARGETING_NOT_SPECIFIC"
    elif R_C <= COMPUTE_GATE:
        outcome = "MOLECULAR_TRANSFER_POSITIVE"
    else:
        outcome = "RELAXATION_HELPS_BUT_NOT_COMPUTE_EFFICIENT"
    print(f"\n  PRIMARY F_R vs F: {fmt(P['d_int'])} final {P['d_fin']['median']:+.2f}% -> resolved={resolved}, late reversal={reversal}")
    print(f"  targeting specificity F_R vs F_rand: {fmt(spec['d_int'])} -> {specific};  positive control replicated: {replicated}")
    print(f"  OUTCOME: {outcome}")
    summary = dict(prereg=os.path.relpath(PREREG, ROOT), n_seeds=len(seeds), seeds=seeds, rho_star=rho, h_read_starstar=h2, contrasts=res,
                   primary_other_readouts=other, health=health, nan_free=nan_ok, compute_to_accuracy=cta, R_C=R_C, actual_cost_ratio=cost,
                   decisions=dict(resolved=bool(resolved), reversal=bool(reversal), specific=bool(specific), replicated=bool(replicated)), outcome=outcome)
    json.dump(summary, open(os.path.join(d2, "analysis.json"), "w"), indent=2, default=float)
    with open(os.path.join(d2, "comparison.csv"), "w") as fh:
        fh.write("seed," + ",".join(f"{tag}_dI,{tag}_dF" for tag, _, _ in NAMED) + "\n")
        for i, s in enumerate(seeds):
            fh.write(f"{s}," + ",".join(f"{res[tag]['d_int']['per_seed'][i]:.3f},{res[tag]['d_fin']['per_seed'][i]:.3f}" for tag, _, _ in NAMED) + "\n")
    if not a.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(d2, "figures"); os.makedirs(fd, exist_ok=True)
            col = {A: "#4d4d4d", F: "#d95f02", AR: "#4d4d4d", FR: "#d95f02", FRAND: "#7570b3"}; ls = {A: "-", F: "-", AR: "--", FR: "--", FRAND: ":"}
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for m in arms:
                axes[0].plot(t, med[m], color=col[m], ls=ls[m], lw=1.4, label=m); axes[1].plot(caxis[m], med[m], color=col[m], ls=ls[m], lw=1.4, label=m)
            axes[0].set_yscale("log"); axes[0].set_xlabel("outer time"); axes[0].set_ylabel(f"median $e_F$ at {h2}"); axes[0].legend(fontsize=7, frameon=False)
            axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("total force evaluations (replica-steps)")
            for ax in axes:
                ax.grid(alpha=0.25, which="both")
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(fd, f"fig_W4W5W6_confirmatory.{ext}"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            print(f"  (figures skipped: {e})")
    print(f"  wrote W2/analysis.json, comparison.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["W0", "W1", "W2"], required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    {"W0": stage_w0, "W1": stage_w1, "W2": stage_w2}[a.stage](a)


if __name__ == "__main__":
    main()
