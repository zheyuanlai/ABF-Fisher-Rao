#!/usr/bin/env python
"""Analyzer for the WCA capped-OT confirmatory campaign M3 (docs/WCA_OT_CONFIRMATORY_M3.md).

    --stage calibration   J_KL = int KL(p_hat_t || U) dt per arm from the stored walker marginal
                          ONLY; alpha* = argmin |log(J_OT / J_F)| -> calibration/alpha_star.json.
                          Reads no error field (blind).
    --stage core          A/F/T at h**: compute-normalised I_F^(C), e_F(C*), H-B1/H-B2, read-out
                          sensitivity, C(eps_A) -> core/analysis.json + core/go_nogo.json.
    --stage repair        + R/F+R/T+R on the common budget C* -> repair/analysis.json.
Figures under <stage>/figures/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
ROOT = os.path.join(SCRIPTS, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyze_wca_bandwidth_audit import Z_HI, Z_LO, readouts   # noqa: E402

CAMPAIGN = os.path.join(ROOT, "results/ot_repair_campaign/wca/M3")
H_STAR = "0.00625"
SENS = ("raw", "0.00625", "0.0125")
N_BOOT, BOOT_SEED = 10000, 20260905
MARGIN = 10.0
FINAL_MARGIN = 5.0
M1_SLOPE = 500.0
MIN_COUNT = 200


# ----------------------------------------------------------------------------- io / stats
def load(stage):
    runs = {}
    for f in sorted(glob.glob(os.path.join(CAMPAIGN, stage, "raw", f"M3_{stage}__*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        runs.setdefault(str(d["name"]), {})[int(d["seed"])] = {k: d[k] for k in d.files}
    return runs


def ci_of_median(d, level, k):
    rng = np.random.default_rng(BOOT_SEED + k)
    d = np.asarray(d, float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    med = np.median(d[idx], axis=1)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return [float(np.percentile(med, lo)), float(np.percentile(med, hi))]


def paired(arm, ref, k):
    d = 100.0 * (np.asarray(arm) - np.asarray(ref)) / np.asarray(ref)
    return dict(median=float(np.median(d)), ci95=ci_of_median(d, 95, k), ci90=ci_of_median(d, 90, k + 500),
                wins=int((d < 0).sum()), n=int(len(d)), per_seed=[float(v) for v in d])


def fmt(c):
    return f"{c['median']:+7.2f}% [{c['ci95'][0]:+7.2f},{c['ci95'][1]:+7.2f}] (90%: [{c['ci90'][0]:+6.2f},{c['ci90'][1]:+6.2f}]) {c['wins']:2d}/{c['n']}"


def superior(c):
    return bool(c["median"] <= -MARGIN and c["ci95"][1] < 0)


def equivalent(c):
    return bool(c["ci90"][0] >= -MARGIN and c["ci90"][1] <= MARGIN)


def compute_axis(run):
    steps = np.asarray(run["profile_steps"], float); N = float(run["n_replicas"])
    inner = np.cumsum(np.asarray(run["relax_steps"], float)) if "relax_steps" in run else np.zeros_like(steps)
    return N * steps + inner


def integral_on_budget(caxis, curve, C_star):
    """(1/C*) int_0^{C*} e(C) dC, the curve interpolated on the compute axis and truncated at C*."""
    caxis = np.asarray(caxis, float); curve = np.asarray(curve, float)
    if caxis[-1] < C_star - 1e-6:
        raise ValueError("curve does not reach the common budget")
    grid = np.linspace(0.0, C_star, 2001)
    e = np.interp(grid, caxis, curve)
    return float(np.trapezoid(e, grid) / C_star), float(np.interp(C_star, caxis, curve))


def first_reach_persist(axis, curve, eps, n_persist=2):
    """First axis value from which the curve stays <= eps for n_persist saves (the final save
    counts on its own, so an arm whose last point defines eps reaches it there)."""
    below = np.asarray(curve) <= eps
    for i in range(len(axis)):
        if below[i:min(i + n_persist, len(axis))].all():
            return float(axis[i])
    return float("inf")


def kl_to_uniform(p_hat_t, grid, z_lo, z_hi):
    """KL(p_hat || U) per save from the stored walker marginal (a density on the grid), with
    trapezoid weights so that the uniform density gives exactly 0 (rectangle sums on a
    160-point grid over-count the range by one bin and make a flat marginal read -0.6 %)."""
    p = np.clip(np.asarray(p_hat_t, float), 0.0, None)
    g = np.asarray(grid, float)
    w = np.gradient(g); w[0] *= 0.5; w[-1] *= 0.5                    # trapezoid weights, sum = z_hi - z_lo
    p = p / np.maximum((p * w).sum(1, keepdims=True), 1e-300)
    U = 1.0 / (z_hi - z_lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = (np.where(p > 0, p * np.log(p / U), 0.0) * w).sum(1)
    return kl


# ----------------------------------------------------------------------------- calibration (blind)
def stage_calibration(a):
    runs = load("calibration")
    assert "fr_uniform" in runs and "abf" in runs, sorted(runs)
    any_run = next(iter(runs["abf"].values()))
    grid = np.asarray(any_run["grid"], float); t = np.asarray(any_run["profile_times"], float)
    z_lo, z_hi = float(grid[0]), float(grid[-1])
    t_start = 20000 * 0.002
    sel = t >= t_start - 1e-9
    J = {}
    for arm in sorted(runs):
        vals = {}
        for s, r in runs[arm].items():
            kl = kl_to_uniform(r["p_hat_t"], grid, z_lo, z_hi)
            vals[s] = float(np.trapezoid(kl[sel], t[sel]))
        J[arm] = dict(per_seed=vals, median=float(np.median(list(vals.values()))))
    JF = J["fr_uniform"]["median"]
    ots = sorted([m for m in runs if m.startswith("ot_a")], key=lambda m: float(m[4:]))
    rows = []
    for m in ots:
        ratio = J[m]["median"] / JF
        capped = float(np.median([float(r["ot_capped_frac"]) for r in runs[m].values()]))
        absdz = float(np.median([float(r["ot_absdz_mean"]) for r in runs[m].values()]))
        nan = any(bool(r["had_nan"]) for r in runs[m].values())
        rows.append(dict(arm=m, alpha=float(m[4:]), J_KL=J[m]["median"], ratio=ratio, log_ratio=float(np.log(ratio)) if ratio > 0 else float("nan"),
                         capped_frac=capped, absdz_mean=absdz, had_nan=nan, n_seeds=len(runs[m])))
    ok = [r for r in rows if r["capped_frac"] < 0.05 and not r["had_nan"] and r["ratio"] > 0]
    pool = ok if ok else rows
    best = min(pool, key=lambda r: abs(r["log_ratio"]) if np.isfinite(r["log_ratio"]) else float("inf"))
    in_band = 0.9 <= best["ratio"] <= 1.1
    out = dict(alpha_star=best["alpha"], arm=best["arm"], ratio=best["ratio"], in_band=bool(in_band), capped_frac=best["capped_frac"],
               J_KL_F=JF, J_KL_A=J["abf"]["median"], ladder=rows, t_start=t_start, rule="argmin |log(J_OT/J_F)|, capped<5%, no NaN",
               blind=True, note="only p_hat_t / ot_* fields were read; no free-energy error was consulted")
    os.makedirs(os.path.join(CAMPAIGN, "calibration", "figures"), exist_ok=True)
    json.dump(out, open(os.path.join(CAMPAIGN, "calibration", "alpha_star.json"), "w"), indent=2)
    print(f"M3-A calibration ({len(runs['abf'])} seeds): J_KL(A) = {J['abf']['median']:.4f}, J_KL(F) = {JF:.4f}")
    for r in rows:
        print(f"  alpha {r['alpha']:<5g}: J_KL {r['J_KL']:.4f}  ratio {r['ratio']:.3f}  capped {r['capped_frac']:.3f}  |dz| mean {r['absdz_mean']:.4f}  nan={r['had_nan']}")
    print(f"  -> alpha* = {best['alpha']:g} (ratio {best['ratio']:.3f}, {'in' if in_band else 'OUTSIDE'} [0.9, 1.1]; closest in log-ratio)")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
    ax.plot([r["alpha"] for r in rows], [r["J_KL"] for r in rows], "o-", color="tab:red", label="capped OT")
    ax.axhline(JF, color="tab:blue", ls="--", label="uniform FR"); ax.axhline(J["abf"]["median"], color="k", ls=":", label="ABF")
    ax.axvline(best["alpha"], color="gray", lw=0.7)
    ax.set_xscale("log"); ax.set_xlabel("alpha"); ax.set_ylabel("J_KL = int KL(p_t || U) dt (t >= 40)")
    ax.set_title("M3-1: blind marginal-action calibration", fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(CAMPAIGN, "calibration", "figures", "m3_1_calibration.png"), dpi=160)
    return 0


# ----------------------------------------------------------------------------- core / repair
LABEL = {"abf": "A", "fr_uniform": "F", "abf_rej": "R", "fr_rej": "F+R"}


def label_of(name):
    if name in LABEL:
        return LABEL[name]
    return "T+R" if name.endswith("_rej") else "T"


def prepare(runs):
    any_run = next(iter(next(iter(runs.values())).values()))
    grid = np.asarray(any_run["grid"], float); ref_F = np.asarray(any_run["reference_free_energy"], float)
    Fp_ref = np.asarray(any_run["reference_mean_force"], float)
    t = np.asarray(any_run["profile_times"], float); mask = (grid >= Z_LO) & (grid <= Z_HI); sigma = float(any_run.get("abf_smooth_sigma", 0.5))
    ro = {m: {s: readouts(r, grid, mask, ref_F, sigma) for s, r in runs[m].items()} for m in runs}
    return grid, ref_F, Fp_ref, t, mask, ro


def contrasts_on_budget(runs, ro, arms, C_star, seeds, lab):
    I, fin = {}, {}
    for m in arms:
        I[m], fin[m] = {}, {}
        for s in seeds:
            cax = compute_axis(runs[m][s])
            I[m][s], fin[m][s] = integral_on_budget(cax, ro[m][s][lab], C_star)
    return I, fin


def stage_core(a, with_repair=False):
    runs = load("core")
    if with_repair:
        runs.update(load("repair"))
    runs = {label_of(m): v for m, v in runs.items()}
    seeds = sorted(set.intersection(*[set(v) for v in runs.values()]))
    assert seeds, "no complete seed block"
    grid, ref_F, Fp_ref, t, mask, ro = prepare(runs)
    N = int(next(iter(runs["A"].values()))["n_replicas"]); n_steps = int(next(iter(runs["A"].values()))["n_steps"])
    C_star = float(N * n_steps)
    arms = [m for m in ("A", "F", "T", "R", "F+R", "T+R") if m in runs]
    stage = "repair" if with_repair else "core"
    out_dir = os.path.join(CAMPAIGN, stage); os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    print(f"M3-{'C' if with_repair else 'B'}: arms {arms}, {len(seeds)} seeds {seeds[0]}-{seeds[-1]}, C* = {C_star:.4g}, h** = {H_STAR}")
    res = dict(seeds=seeds, C_star=C_star, arms={}, contrasts={}, sensitivity={}, compute_to_epsA={})
    k = 0
    contrast_list = [("T", "A", "H-B1"), ("T", "F", "H-B2"), ("F", "A", "F vs A")]
    if with_repair:
        contrast_list += [("T+R", "R", "H-C1"), ("T+R", "F+R", "H-C2"), ("T+R", "T", "repair at equal compute"), ("R", "A", "R vs A"), ("F+R", "F", "F+R vs F"), ("T+R", "A", "T+R vs A")]
    for lab in SENS:
        I, fin = contrasts_on_budget(runs, ro, arms, C_star, seeds, lab)
        res["sensitivity"][lab] = {}
        for arm, ref, tag in contrast_list:
            cI = paired([I[arm][s] for s in seeds], [I[ref][s] for s in seeds], k); cF = paired([fin[arm][s] for s in seeds], [fin[ref][s] for s in seeds], k + 1); k += 2
            res["sensitivity"][lab][f"{arm} vs {ref}"] = dict(dI=cI, dfin=cF)
            if lab == H_STAR:
                res["contrasts"][f"{arm} vs {ref}"] = dict(tag=tag, dI_C=cI, d_final=cF, superior=superior(cI), equivalent=equivalent(cI),
                                                            final_noninferior=bool(cF["median"] <= FINAL_MARGIN))
        if lab == H_STAR:
            for m in arms:
                res["arms"][m] = dict(I_C_median=float(np.median([I[m][s] for s in seeds])), final_median=float(np.median([fin[m][s] for s in seeds])),
                                      cost_ratio=float(np.median([compute_axis(runs[m][s])[-1] for s in seeds]) / C_star))
    print(f"\nprimary at h** (compute-normalised I_F^(C) on the common budget C*; paired; 95% and 90% CIs; wins/{len(seeds)})")
    for m in arms:
        print(f"  {m:>4}: I_F^(C) {res['arms'][m]['I_C_median']:.5f}  e_F(C*) {res['arms'][m]['final_median']:.5f}  total compute {res['arms'][m]['cost_ratio']:.2f}x")
    for key, c in res["contrasts"].items():
        print(f"  {key:>12} [{c['tag']}]: dI_F^(C) {fmt(c['dI_C'])}   final {fmt(c['d_final'])}   superior={c['superior']} equivalent={c['equivalent']}")
    print("read-out sensitivity (median dI_F^(C) at raw / 0.00625 / 0.0125):")
    for key in res["contrasts"]:
        meds = [res["sensitivity"][lab][key]["dI"]["median"] for lab in SENS]
        stable = len({np.sign(v) for v in meds}) == 1
        res["contrasts"][key]["readout_sensitive"] = not stable
        print(f"  {key:>12}: " + "  ".join(f"{lab} {v:+.2f}%" for lab, v in zip(SENS, meds)) + ("" if stable else "   <- SIGN CHANGES: read-out-sensitive"))
    # time-to-accuracy on the compute axis
    med = {m: np.median([ro[m][s][H_STAR] for s in seeds], axis=0) for m in arms}
    caxis = {m: np.median([compute_axis(runs[m][s]) for s in seeds], axis=0) for m in arms}
    eps_A = float(med["A"][-1]); C_A = first_reach_persist(caxis["A"], med["A"], eps_A)
    for m in arms:
        C = first_reach_persist(caxis[m], med[m], eps_A)
        res["compute_to_epsA"][m] = dict(C=C, ratio=(C / C_A if np.isfinite(C_A) and C_A > 0 else None))
    print("compute to eps_A (median ABF final error; persistence 2 saves): " + "  ".join(f"{m} {res['compute_to_epsA'][m]['ratio']:.3f}x" if res['compute_to_epsA'][m]['ratio'] else f"{m} n/a" for m in arms))
    # mechanism
    mech = {}
    for m in arms:
        r0 = next(iter(runs[m].values()))
        if "ot_absdz_t" in r0:
            mech[m] = dict(absdz_t=np.median([r["ot_absdz_t"] for r in runs[m].values()], 0).tolist(),
                           capped_frac=float(np.median([float(r["ot_capped_frac"]) for r in runs[m].values()])),
                           absdz_mean=float(np.median([float(r["ot_absdz_mean"]) for r in runs[m].values()])))
        z_lo, z_hi = float(grid[0]), float(grid[-1])
        mech.setdefault(m, {})["kl_uniform_t"] = np.median([kl_to_uniform(r["p_hat_t"], grid, z_lo, z_hi) for r in runs[m].values()], 0).tolist()
    res["mechanism"] = mech
    # |b_post| vs |dz| for the OT arms from the (z, |dz|) table
    inj = {}
    for m in arms:
        r0 = next(iter(runs[m].values()))
        if "ot_C2_post" not in r0 or float(r0["ot_alpha"]) == 0.0:
            continue
        C2 = sum(np.asarray(r["ot_C2_post"], float) for r in runs[m].values()); S2 = sum(np.asarray(r["ot_Sf2_post"], float) for r in runs[m].values())
        edges = np.asarray(r0["ot_absdz_edges"], float)
        rows = []
        for j in range(C2.shape[1]):
            c = C2[:, j]; ok = mask & (c >= MIN_COUNT)
            if ok.sum() < 3:
                continue
            bias = S2[ok, j] / c[ok] - Fp_ref[ok]; w = c[ok] / c[ok].sum()
            mid = 0.5 * (edges[j] + min(edges[j + 1], 0.0176))
            rows.append(dict(absdz_mid=float(mid), rms_bias=float(np.sqrt(np.sum(w * bias ** 2))), signed=float(np.sum(w * bias)), n_samples=float(c[ok].sum()), n_bins=int(ok.sum())))
        inj[m] = rows
    res["injection_vs_dz"] = inj
    # go / no-go (core only)
    if not with_repair:
        c1 = res["contrasts"]["T vs A"]
        go = bool(c1["superior"] and c1["final_noninferior"]) or bool(c1["d_final"]["ci95"][1] < 0 and c1["dI_C"]["ci95"][1] <= MARGIN)
        json.dump(dict(go=go, H_B1=c1["superior"] and c1["final_noninferior"], T_vs_A=c1, H_B2_equivalent=res["contrasts"]["T vs F"]["equivalent"]),
                  open(os.path.join(out_dir, "go_nogo.json"), "w"), indent=2)
        print(f"\nGO/NO-GO for M3-C: {'GO' if go else 'NO-GO'}  (H-B1 {'holds' if c1['superior'] and c1['final_noninferior'] else 'fails'}; H-B2 equivalence {'holds' if res['contrasts']['T vs F']['equivalent'] else 'fails'})")
    json.dump(res, open(os.path.join(out_dir, "analysis.json"), "w"), indent=1, default=float)
    figures(runs, ro, arms, seeds, t, grid, Fp_ref, mask, res, os.path.join(out_dir, "figures"), with_repair)
    print(f"wrote {os.path.join(out_dir, 'analysis.json')}")
    return 0


def figures(runs, ro, arms, seeds, t, grid, Fp_ref, mask, res, fig_dir, with_repair):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    cols = {"A": "k", "F": "tab:blue", "T": "tab:red", "R": "tab:gray", "F+R": "tab:cyan", "T+R": "tab:orange"}
    med = {m: np.median([ro[m][s][H_STAR] for s in seeds], axis=0) for m in arms}
    caxis = {m: np.median([compute_axis(runs[m][s]) for s in seeds], axis=0) for m in arms}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
    for m in arms:
        axes[0].plot(t, med[m], color=cols[m], lw=1.3, label=m)
        axes[1].plot(caxis[m] / 1e6, med[m], color=cols[m], lw=1.3, label=m)
    axes[1].axvline(res["C_star"] / 1e6, color="gray", lw=0.7, ls=":")
    for ax, xl, ttl in ((axes[0], "t (time units)", "M3-2: equal physical time"), (axes[1], "force evaluations (millions, inner steps charged)", "M3-3: equal compute")):
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(f"e_F at h = {H_STAR}"); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(fig_dir, "m3_2_3_convergence.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained")
    for m in arms:
        axes[0].plot(t, res["mechanism"][m]["kl_uniform_t"], color=cols[m], lw=1.2, label=m)
        if "absdz_t" in res["mechanism"][m]:
            axes[1].plot(t, res["mechanism"][m]["absdz_t"], color=cols[m], lw=1.2, label=m)
    axes[0].set_xlabel("t"); axes[0].set_ylabel("KL(p_t || U)"); axes[0].set_title("M3-4a: marginal", fontsize=9); axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("E|dz| per event"); axes[1].set_title("M3-4b: OT displacement self-limits", fontsize=9); axes[1].legend(fontsize=7, frameon=False)
    ax = axes[2]
    for m, rows in res["injection_vs_dz"].items():
        ax.plot([r["absdz_mid"] for r in rows], [r["rms_bias"] for r in rows], "o-", ms=3, color=cols[m], label=f"{m} (live sampler)")
    xs = np.logspace(-4, np.log10(0.0176), 20); ax.plot(xs, M1_SLOPE * xs, "k--", lw=0.8, label="M1: 500 |dz|")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("|dz| of the event"); ax.set_ylabel("RMS deposit bias after the event"); ax.set_title("M3-5: injection vs displacement", fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(fig_dir, "m3_4_5_mechanism.png"), dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
    for m in arms:
        mf = np.median([np.asarray(runs[m][s][f"readout_mean_force_t__h{H_STAR}"], float)[-1] for s in seeds], axis=0)
        ax.plot(grid, mf - Fp_ref, color=cols[m], lw=1.1, label=m)
    ax.axhline(0, color="k", lw=0.6); ax.set_xlim(Z_LO, Z_HI); ax.set_xlabel("z"); ax.set_ylabel("F'_hat(T) - F'_ref"); ax.set_title("M3-6: final signed mean-force error", fontsize=9); ax.legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(fig_dir, "m3_6_signed_error.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("calibration", "core", "repair"))
    a = ap.parse_args()
    if a.stage == "calibration":
        return stage_calibration(a)
    return stage_core(a, with_repair=(a.stage == "repair"))


if __name__ == "__main__":
    raise SystemExit(main())
