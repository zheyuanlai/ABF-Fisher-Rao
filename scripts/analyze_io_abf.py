"""Endpoint, tables and figures for the IO-ABF transfer campaign.

Frozen protocol: ``docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md`` sections 9-13.

Reads the thresholds this system froze during calibration and refuses to run if
they are missing.  Nothing here recomputes ``eps1`` or ``eps2``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "results", "io_abf_overnight")

BOOT_N = 10_000
BOOT_SEED = 20260827
ARMS = ("A0", "A6b", "A6c")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_phase(system, phase):
    out = {}
    for path in sorted(glob.glob(os.path.join(OUT_ROOT, system, phase, "*.npz"))):
        with np.load(path, allow_pickle=True) as d:
            rec = {k: d[k] for k in d.files}
        arm = str(rec.get("io_arm", "A0"))
        out.setdefault(arm, {})[int(rec["seed"])] = rec
    return out


def thresholds(system):
    path = os.path.join(OUT_ROOT, system, "calibration", "thresholds.json")
    if not os.path.exists(path):
        raise SystemExit(f"[{system}] thresholds not frozen; run calibration first")
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #
def hitting_time(t, e, eps, n_consecutive=3):
    """First time three consecutive evaluation frames are at or below ``eps``.

    A single frame below threshold is a fluctuation of the estimate, not an
    arrival: the ABF error curve is noisy at the frame scale and a one-frame rule
    would score the noise.  Returns NaN when the run never gets there, so
    censoring is visible instead of being absorbed into a number.
    """
    e = np.asarray(e, dtype=float)
    below = e <= float(eps)
    run = 0
    for k, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= n_consecutive:
            return float(t[k - n_consecutive + 1])
    return float("nan")


def restricted(tau, T):
    """``min(tau, T)`` with a censored run charged the full horizon."""
    return np.where(np.isfinite(tau), np.minimum(tau, T), T)


def speedup(tau_ref, tau_arm, T):
    return float(np.mean(restricted(tau_ref, T)) / np.mean(restricted(tau_arm, T)))


def paired_bootstrap(tau_ref, tau_arm, T, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    m = tau_ref.size
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = speedup(tau_ref[idx], tau_arm[idx], T)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def arm_table(system, phase):
    thr = thresholds(system)
    data = load_phase(system, phase)
    if "A0" not in data:
        raise SystemExit(f"[{system}/{phase}] no A0 rows")
    seeds = sorted(set.intersection(*[set(v) for v in data.values()]))
    T = float(thr["T_total"])
    t = data["A0"][seeds[0]]["t"]

    rows, curves = [], {}
    for arm in ARMS:
        if arm not in data:
            continue
        e = np.stack([data[arm][s]["l2_f_t"] for s in seeds])
        ef = np.stack([data[arm][s]["l2_f_full_t"] for s in seeds])
        curves[arm] = dict(t=t, e=e, e_full=ef, seeds=seeds)
        rows.append(dict(
            arm=arm,
            tau1=np.array([hitting_time(t, e[i], thr["eps1"]) for i in range(len(seeds))]),
            tau2=np.array([hitting_time(t, e[i], thr["eps2"]) for i in range(len(seeds))]),
            final=e[:, -1], final_full=ef[:, -1]))
    return thr, seeds, T, rows, curves, data


def endpoint(system, phase):
    thr, seeds, T, rows, curves, data = arm_table(system, phase)
    by = {r["arm"]: r for r in rows}
    ref = by["A0"]
    out = dict(system=system, phase=phase, n_seeds=len(seeds), T=T,
               eps1=thr["eps1"], eps2=thr["eps2"],
               gamma_unresolved=bool(thr["gamma_unresolved"]),
               R_gamma=thr["R_gamma"]["ratio"], role=thr["role"], arms={})
    for arm, r in by.items():
        rec = dict(arm=arm)
        for lab, key in (("eps1", "tau1"), ("eps2", "tau2")):
            S = speedup(ref[key], r[key], T)
            lo, hi = paired_bootstrap(ref[key], r[key], T)
            rec[f"S_{lab}"] = S
            rec[f"S_{lab}_ci"] = [lo, hi]
            rec[f"hit_{lab}"] = float(np.mean(np.isfinite(r[key])))
            rec[f"median_tau_{lab}"] = float(np.nanmedian(r[key])) \
                if np.isfinite(r[key]).any() else float("nan")
        rec["final_median"] = float(np.median(r["final"]))
        rec["final_full_median"] = float(np.median(r["final_full"]))
        rec["final_ratio_to_A0"] = float(np.median(r["final"]) / np.median(ref["final"]))
        rec["final_full_ratio_to_A0"] = float(
            np.median(r["final_full"]) / np.median(ref["final_full"]))
        rec["n_improved_eps2"] = int(np.sum(
            restricted(r["tau2"], T) < restricted(ref["tau2"], T)))
        out["arms"][arm] = rec

    # A6c diagnostics: is Fisher-Rao load-bearing, and what did it cost?
    if "A6c" in data:
        lam = np.concatenate([data["A6c"][s]["io_lam_t"] for s in seeds])
        ess = np.concatenate([data["A6c"][s]["io_ess_predicted_t"] for s in seeds])
        tv = np.concatenate([data["A6c"][s]["io_tv_to_unconstrained_t"] for s in seeds])
        essu = np.concatenate([data["A6c"][s]["io_mass_ess_unconstrained_t"]
                               for s in seeds])
        out["a6c"] = dict(
            p_lambda_positive=float(np.mean(lam > 0)),
            mass_ess_median=float(np.median(ess)),
            mass_ess_min=float(np.min(ess)),
            mass_ess_unconstrained_median=float(np.median(essu)),
            tv_to_a6b_median=float(np.median(tv)))
        if "A6b" in out["arms"]:
            sb = out["arms"]["A6b"]["S_eps2"] - 1.0
            sc = out["arms"]["A6c"]["S_eps2"] - 1.0
            out["a6c"]["R_retain"] = float(sc / sb) if abs(sb) > 1e-9 else float("nan")

    # the preregistered verdict, computed from the frozen rule and nothing else
    if "A6b" in out["arms"]:
        a = out["arms"]["A6b"]
        checks = dict(
            speedup_at_least_1_15=bool(a["S_eps2"] >= 1.15),
            ci_lower_above_1=bool(a["S_eps2_ci"][0] > 1.0),
            censoring_not_worse=bool(
                a["hit_eps2"] >= out["arms"]["A0"]["hit_eps2"] - 0.05),
            final_within_10pct=bool(a["final_ratio_to_A0"] <= 1.10),
            final_full_within_10pct=bool(a["final_full_ratio_to_A0"] <= 1.10))
        out["verdict_checks"] = checks
        out["verdict"] = "POSITIVE" if all(checks.values()) else "NOT POSITIVE"
    return out


# --------------------------------------------------------------------------- #
# the difficulty decomposition
# --------------------------------------------------------------------------- #
def decomposition(system, phase):
    data = load_phase(system, phase)
    arm = "A0" if "A0" in data else sorted(data)[0]
    seeds = sorted(data[arm])
    a_cell = data[arm][seeds[0]]["io_a_cell"]
    scored = a_cell > 0

    def stat(name):
        v = np.stack([data[arm][s][f"io_{name}"] for s in seeds])[:, scored]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 4:
            return dict(q10=float("nan"), q90=float("nan"), ratio=float("nan"))
        q10, q90 = float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))
        return dict(q10=q10, q90=q90, ratio=q90 / max(q10, 1e-300))

    tau = np.stack([data[arm][s]["io_tau"] for s in seeds])[:, scored]
    out = dict(system=system, arm=arm, n_scored_cells=int(scored.sum()),
               sigma2=stat("sigma2"), tau=stat("tau"), gamma=stat("gamma"),
               valid_tau_fraction=float(np.mean(np.isfinite(tau) & (tau > 0))))

    # Spearman(Gamma early, Gamma late) -- does the difficulty map hold still?
    g_t = np.stack([data[arm][s]["io_gamma_t"] for s in seeds])   # (S, n_opp, J)
    if g_t.ndim == 3 and g_t.shape[1] >= 4:
        n = g_t.shape[1]
        early = np.log(np.maximum(g_t[:, :n // 3].mean(axis=1)[:, scored], 1e-300))
        late = np.log(np.maximum(g_t[:, -n // 3:].mean(axis=1)[:, scored], 1e-300))
        from scipy.stats import spearmanr
        rs = [spearmanr(early[i], late[i]).statistic for i in range(early.shape[0])]
        out["spearman_gamma_early_late"] = float(np.median(rs))
        # which factor carries the spread?
        s_r, t_r = out["sigma2"]["ratio"], out["tau"]["ratio"]
        out["dominant_source"] = (
            "sigma2" if s_r > 3 * t_r else "tau" if t_r > 3 * s_r else "both")
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(system, phase, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    thr, seeds, T, rows, curves, data = arm_table(system, phase)
    by = {r["arm"]: r for r in rows}
    colors = {"A0": "#444444", "A6b": "#c0392b", "A6c": "#2471a3"}
    rng = np.random.default_rng(BOOT_SEED)

    def band(ax, t, E, c, label):
        med = np.median(E, axis=0)
        bs = np.array([np.median(E[rng.integers(0, E.shape[0], E.shape[0])], axis=0)
                       for _ in range(400)])
        ax.plot(t, med, color=c, lw=1.8, label=label)
        ax.fill_between(t, np.quantile(bs, 0.025, axis=0),
                        np.quantile(bs, 0.975, axis=0), color=c, alpha=0.18, lw=0)

    # Fig 1 -- convergence, primary and full domain
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "e", "primary (evaluation mask)"),
                           (axes[1], "e_full", "full domain")):
        for arm in ARMS:
            if arm in curves:
                band(ax, curves[arm]["t"], curves[arm][key], colors[arm], arm)
        if key == "e":
            ax.axhline(thr["eps1"], ls=":", c="grey", lw=1)
            ax.axhline(thr["eps2"], ls="--", c="grey", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"$e_A(t)$")
        ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{system} / {phase} -- free-energy error "
                 f"(median, bootstrap band, n={len(seeds)})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_convergence.png"), dpi=150)
    plt.close(fig)

    # Fig 2 -- paired time-to-accuracy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, lab in ((axes[0], "tau1", r"$\tau_{\epsilon_1}$"),
                         (axes[1], "tau2", r"$\tau_{\epsilon_2}$")):
        ref = restricted(by["A0"][key], T)
        for arm in ARMS:
            if arm == "A0" or arm not in by:
                continue
            y = restricted(by[arm][key], T)
            ax.scatter(ref, y, s=26, alpha=0.75, color=colors[arm], label=arm,
                       edgecolor="none")
        lim = [0, T * 1.03]
        ax.plot(lim, lim, c="k", lw=0.8, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"A0 {lab}"); ax.set_ylabel(f"arm {lab}")
        ax.set_title(f"{lab}  (below the diagonal = faster)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{system} / {phase} -- paired restricted time-to-accuracy")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_time_to_accuracy.png"), dpi=150)
    plt.close(fig)

    # cell centres for the mechanism panels
    ref_rec = data["A0"][seeds[0]]
    edges = ref_rec["io_cell_edges"]
    zc = 0.5 * (edges[1:] + edges[:-1])

    def cellmean(arm, name):
        return np.median(np.stack([data[arm][s][name] for s in seeds]), axis=0)

    # Fig 3 -- difficulty decomposition
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name, lab in ((axes[0], "io_sigma2", r"$\sigma^2(z)$"),
                          (axes[1], "io_tau", r"$\tau(z)$"),
                          (axes[2], "io_gamma", r"$\Gamma(z)=\sigma^2\tau$")):
        v = cellmean("A0", name)
        ax.plot(zc, v, "o-", ms=3, color="#444444")
        ax.set_yscale("log"); ax.set_xlabel("z"); ax.set_title(lab)
    fig.suptitle(f"{system} -- local information cost, measured on A0")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_difficulty.png"), dpi=150)
    plt.close(fig)

    # Fig 4 -- the information target
    a = ref_rec["io_a_cell"]
    g = cellmean("A0", "io_gamma")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(zc, a, "o-", ms=3, color="#7d3c98"); axes[0].set_title(r"$a(z)$ (leverage)")
    axes[1].plot(zc, a * g, "o-", ms=3, color="#b7950b"); axes[1].set_title(r"$a(z)\Gamma(z)$")
    axes[1].set_yscale("log")
    if "A6b" in data:
        rs = np.median(np.stack([data["A6b"][s]["io_r_star_t"][-1] for s in seeds]), axis=0)
        axes[2].plot(zc, rs, "o-", ms=3, color=colors["A6b"], label="A6b")
    if "A6c" in data:
        rs = np.median(np.stack([data["A6c"][s]["io_r_star_t"][-1] for s in seeds]), axis=0)
        axes[2].plot(zc, rs, "s-", ms=3, color=colors["A6c"], label="A6c")
    axes[2].axhline(1.0 / zc.size, ls="--", c="grey", lw=1, label="uniform")
    axes[2].set_title(r"$r^\star(z)$"); axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("z")
    fig.suptitle(f"{system} -- from difficulty to allocation")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_target.png"), dpi=150)
    plt.close(fig)

    # Fig 5 -- target vs realised occupancy
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, arm in zip(axes, ("A6b", "A6c")):
        if arm not in data:
            ax.axis("off"); continue
        rs = np.median(np.stack([data[arm][s]["io_r_star_t"][-1] for s in seeds]), axis=0)
        oc = np.median(np.stack([data[arm][s]["io_occupancy_t"][-1] for s in seeds]), axis=0)
        ax.plot(zc, rs, "o-", ms=3, color=colors[arm], label=r"target $r^\star$")
        ax.plot(zc, oc, "s--", ms=3, color="#16a085", label="realised occupancy")
        oc0 = np.median(np.stack([data["A0"][s]["io_occupancy_t"][-1] for s in seeds]), axis=0)
        ax.plot(zc, oc0, ":", color="#888888", label="A0 occupancy")
        ax.set_title(f"{arm}: TV(target, realised) = "
                     f"{0.5 * np.abs(rs - oc).sum():.3f}")
        ax.set_xlabel("z"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{system} -- does the bias actually deliver the allocation?")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig5_occupancy.png"), dpi=150)
    plt.close(fig)

    # Fig 6 -- where Fisher-Rao bites
    if "A6c" in data:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        q = np.median(np.stack([data["A6c"][s]["io_q_t"][-1] for s in seeds]), axis=0)
        rb = np.median(np.stack([data["A6c"][s]["io_r_star_unconstrained_t"][-1]
                                 for s in seeds]), axis=0)
        rc = np.median(np.stack([data["A6c"][s]["io_r_star_t"][-1] for s in seeds]), axis=0)
        lam = np.median(np.concatenate([data["A6c"][s]["io_lam_t"] for s in seeds]))
        ess = np.median(np.concatenate([data["A6c"][s]["io_ess_predicted_t"]
                                        for s in seeds]))
        ax.plot(zc, q, "^-", ms=3, color="#117a65", label=r"$q(z)$ (FR mass)")
        ax.plot(zc, rb, "o-", ms=3, color=colors["A6b"], label=r"$r^\star$ unconstrained")
        ax.plot(zc, rc, "s-", ms=3, color=colors["A6c"], label=r"$r^\star$ ESS-constrained")
        ax.set_yscale("log"); ax.set_xlabel("z")
        ax.set_title(f"{system}: $\\lambda$={lam:.3g}, ESS$_M$/K={ess:.3f}, "
                     f"TV={0.5 * np.abs(rc - rb).sum():.3f}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig6_fr_separation.png"), dpi=150)
        plt.close(fig)
    print(f"  figures -> {os.path.relpath(outdir, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--phase", default="confirmatory")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    ep = endpoint(a.system, a.phase)
    dec = decomposition(a.system, a.phase)
    outdir = os.path.join(OUT_ROOT, a.system, "analysis")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{a.phase}_endpoint.json"), "w") as fh:
        json.dump(dict(endpoint=ep, decomposition=dec), fh, indent=2, default=str)
    print(json.dumps(dict(endpoint=ep, decomposition=dec), indent=2, default=str))
    if not a.no_figures:
        figures(a.system, a.phase, os.path.join(outdir, f"figures_{a.phase}"))


if __name__ == "__main__":
    main()
