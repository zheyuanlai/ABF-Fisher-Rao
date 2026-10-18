#!/usr/bin/env python
"""Stage A analysis: integrator validation against exact Gibbs (frozen gates: configs/pentane_r15_dt_fr/prereg_stage_a.json).

    python scripts/pentane_r15_dt_fr/analyze_stage_a.py
-> results/pentane_r15_dt_fr/stage_a/{summary.json, SUMMARY.md}, figures/fig1_timestep_validity.{png,pdf}
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STAGE = os.path.join(ROOT, "results", "pentane_r15_dt_fr", "stage_a")
FIG = os.path.join(ROOT, "results", "pentane_r15_dt_fr", "figures")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
PREREG = json.load(open(os.path.join(ROOT, "configs", "pentane_r15_dt_fr", "prereg_stage_a.json")))
BETA = 2.0
TAU = PREREG["gate"]["tolerance_tau_F"]
N_BOOT = 2000
FAM = ["TT", "TG", "GG same-sign", "GG opposite-sign"]


def integrate(fp, dz):
    """Trapezoid integral of a bin-centred derivative over a contiguous bin run (F(first) = 0)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * dz)])


def centre(x):
    return x - x.mean(-1, keepdims=True)


def ref_noise_var(se, dz, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((n, se.size)) * se
    F = np.stack([centre(integrate(d, dz)) for d in draws])
    return F.var(0)


def debiased_D(dev, var_ref, rng=None):
    """dev (G, B): per-group centred profile minus reference. Returns debiased RMS D."""
    G = dev.shape[0]
    dbar = dev.mean(0); s2 = dev.var(0, ddof=1) if G > 1 else np.zeros(dev.shape[1])
    D2 = np.mean(dbar ** 2) - np.mean(s2) / G - np.mean(var_ref)
    return math.sqrt(max(D2, 0.0)), math.sqrt(np.mean(dbar ** 2))


def boot_upper(dev, var_ref, seed=1):
    rng = np.random.default_rng(seed); G = dev.shape[0]
    vals = [debiased_D(dev[rng.integers(0, G, G)], var_ref)[0] for _ in range(N_BOOT)]
    return float(np.quantile(vals, 0.95))


def load_cell(ens, dt, clip):
    fs = sorted(glob.glob(os.path.join(STAGE, "raw", f"{ens}__dt{dt:g}__clip{clip:g}__g*.npz")))
    return [np.load(f, allow_pickle=True) for f in fs]


def main():
    ref = np.load(REF_V2)
    grid = ref["grid"]; dz = float(ref["dz"]); win = ref["window_mask"]; Fref = ref["F"]; Fpref = ref["Fprime"]; se = ref["Fprime_se"]
    widx = np.nonzero(win)[0]
    Fref_w = centre(Fref[win]); var_ref_w = ref_noise_var(se[win], dz)
    # A1 sub-window: largest contiguous run of window bins with F_ref - min <= 2.5
    low = win & ((Fref - Fref[win].min()) <= 2.5)
    runs, cur = [], None
    for i, o in enumerate(list(low) + [False]):
        if o and cur is None:
            cur = i
        if not o and cur is not None:
            runs.append((cur, i)); cur = None
    a, b = max(runs, key=lambda t: t[1] - t[0]); sub = np.zeros_like(win); sub[a:b] = True
    Fref_s = centre(Fref[sub]); var_ref_s = ref_noise_var(se[sub], dz)
    ex = np.load(os.path.join(STAGE, "exact_accumulators.npz"))
    dts = sorted({float(os.path.basename(f).split("__")[1][2:]) for f in glob.glob(os.path.join(STAGE, "raw", "*.npz"))}, reverse=True)
    summary = dict(prereg=PREREG["gate"], window=[float(ref["window_lo"]), float(ref["window_hi"])], a1_subwindow=[float(grid[a] - dz / 2), float(grid[b - 1] + dz / 2)],
                   cells={}, gate={})
    exact = {}
    for tag in ("gibbs", "flat"):
        w = float(ex[f"{tag}_wsum"]); m1 = ex[f"{tag}_bond_m1"] / w; m2 = ex[f"{tag}_bond_m2"] / w
        exact[tag] = dict(bond_mean=m1, bond_var=m2 - m1 ** 2, bond_hist=ex[f"{tag}_bond"] / ex[f"{tag}_bond"].sum(1, keepdims=True),
                          bend_hist=ex[f"{tag}_bend"] / ex[f"{tag}_bend"].sum(1, keepdims=True), fam=ex[f"{tag}_fam"] / ex[f"{tag}_fam"].sum(),
                          b9=ex[f"{tag}_b9"] / ex[f"{tag}_b9"].sum())
    profiles = {}
    for dt in dts:
        for ens in ("flat", "gibbs"):
            for clip in (200.0, 0.0):
                runs_ = load_cell(ens, dt, clip)
                if not runs_:
                    continue
                key = f"{ens}|{dt:g}|{clip:g}"
                c = np.stack([r["c"] for r in runs_]); fsum = np.stack([r["fs"] for r in runs_])
                G = c.shape[0]
                cell = dict(n_groups=G, n_dep=int(sum(int(r["n_dep"]) for r in runs_)),
                            clip_frac=float(sum(int(r["n_clip"]) for r in runs_) / sum(int(r["n_atom_evals"]) for r in runs_)),
                            max_raw_force=float(max(float(r["max_raw_force"]) for r in runs_)),
                            mean_clip_excess=float(np.mean([float(r["mean_clip_excess"]) for r in runs_])),
                            max_disp=float(max(float(r["max_disp"]) for r in runs_)), n_nonfinite=int(sum(int(r["n_nonfinite"]) for r in runs_)),
                            n_fsample_clip=int(sum(int(r["n_fsample_clip"]) for r in runs_)),
                            wall_s=float(np.mean([float(r["wall_s"]) for r in runs_])))
                # bonds / bends vs exact
                bm = np.mean([r["bond_mean"] for r in runs_], 0); bv = np.mean([r["bond_var"] for r in runs_], 0)
                ex_t = exact[ens]
                cell["bond_mean_minus_exact"] = (bm - ex_t["bond_mean"]).tolist()
                cell["bond_var_ratio"] = (bv / ex_t["bond_var"]).tolist()
                bh = np.sum([r["bond"] for r in runs_], 0); bh = bh / bh.sum(1, keepdims=True)
                eh = np.sum([r["bend"] for r in runs_], 0); eh = eh / eh.sum(1, keepdims=True)
                cell["bond_tv"] = (0.5 * np.abs(bh - ex_t["bond_hist"]).sum(1)).tolist()
                cell["bend_tv"] = (0.5 * np.abs(eh - ex_t["bend_hist"]).sum(1)).tolist()
                fam = np.stack([r["fam_series"] for r in runs_]).astype(float)       # (G, S, 4)
                famf = fam / fam.sum(-1, keepdims=True)
                cell["fam_t"] = (np.stack([r["snap_steps"] for r in runs_])[0] * dt).tolist()
                cell["fam_mean"] = famf.mean(0).tolist(); cell["fam_se"] = (famf.std(0, ddof=1) / math.sqrt(G)).tolist()
                cell["fam_exact"] = ex_t["fam"].tolist()
                cell["fam_drift_end"] = (famf[:, -1] - famf[:, 0]).mean(0).tolist()
                cell["fam_drift_end_se"] = ((famf[:, -1] - famf[:, 0]).std(0, ddof=1) / math.sqrt(G)).tolist()
                mask = win if ens == "flat" else sub
                Fr = Fref_w if ens == "flat" else Fref_s; vr = var_ref_w if ens == "flat" else var_ref_s
                ok = (c[:, mask] > 0).all()
                cell["min_count_per_bin_group"] = int(c[:, mask].min())
                if ok:
                    fp = fsum[:, mask] / c[:, mask]
                    F_mf = np.stack([centre(integrate(fp[g], dz)) for g in range(G)])
                    if ens == "flat":
                        F_dn = centre(-np.log(c[:, mask]) / BETA + Fref[mask][None, :])
                    else:
                        F_dn = centre(-np.log(c[:, mask]) / BETA)
                    D_mf, raw_mf = debiased_D(F_mf - Fr, vr); D_dn, raw_dn = debiased_D(F_dn - Fr, vr)
                    D_md, _ = debiased_D(F_mf - F_dn, np.zeros_like(vr))
                    fpb = fp.mean(0) - Fpref[mask]; fpv = fp.var(0, ddof=1) / G
                    cell.update(D_mf=D_mf, D_mf_raw=raw_mf, U95_mf=boot_upper(F_mf - Fr, vr), D_dens=D_dn, D_dens_raw=raw_dn,
                                U95_dens=boot_upper(F_dn - Fr, vr), D_mf_vs_dens=D_md,
                                Fp_bias_rms=float(math.sqrt(max(np.mean(fpb ** 2) - np.mean(fpv) - np.mean(se[mask] ** 2), 0.0))),
                                Fp_bias_rms_raw=float(math.sqrt(np.mean(fpb ** 2))))
                    profiles[key] = dict(F_mf=F_mf.mean(0) - Fr, F_dn=F_dn.mean(0) - Fr, Fp=fp.mean(0) - Fpref[mask], F_mf_g=F_mf, R=grid[mask])
                summary["cells"][key] = cell
        # clip materiality (paired runs, flat ensemble)
        kc, k0 = f"flat|{dt:g}|200", f"flat|{dt:g}|0"
        if kc in profiles and k0 in profiles:
            dif = profiles[kc]["F_mf_g"] - profiles[k0]["F_mf_g"]
            Dc, rawc = debiased_D(dif, np.zeros(dif.shape[1]))
            summary["cells"][kc]["D_clip_vs_noclip"] = Dc; summary["cells"][kc]["D_clip_vs_noclip_raw"] = rawc
    # gates
    for dt in dts:
        c = summary["cells"].get(f"flat|{dt:g}|200")
        if c is None or "D_mf" not in c:
            continue
        clip_ok = (c["clip_frac"] <= 1e-5) and (c.get("D_clip_vs_noclip", 0.0) <= 0.05)
        finite = c["n_nonfinite"] == 0
        if not finite or c["D_mf"] > TAU or c["D_dens"] > TAU:
            v = "FAIL"
        elif c["U95_mf"] <= TAU and c["U95_dens"] <= TAU and clip_ok:
            v = "ADMISSIBLE"
        elif not clip_ok:
            v = "FAIL (clip material)"
        else:
            v = "MARGINAL"
        summary["gate"][f"{dt:g}"] = dict(verdict=v, D_mf=c["D_mf"], U95_mf=c["U95_mf"], D_dens=c["D_dens"], U95_dens=c["U95_dens"],
                                         clip_frac=c["clip_frac"], D_clip_vs_noclip=c.get("D_clip_vs_noclip"), nonfinite=c["n_nonfinite"])
    adm = [float(d) for d, g in summary["gate"].items() if g["verdict"] == "ADMISSIBLE"]
    summary["selected_dt"] = max(adm) if adm else None
    summary["selection_rule"] = PREREG["gate"]["selection_rule"]
    os.makedirs(STAGE, exist_ok=True)
    json.dump(summary, open(os.path.join(STAGE, "summary.json"), "w"), indent=1, default=float)
    # group-mean deviation profiles (for the ABF deposit-bias decomposition in analyze_runs.py)
    np.savez(os.path.join(STAGE, "profiles.npz"), **{f"{k}|{q}": v[q] for k, v in profiles.items() for q in ("R", "F_mf", "F_dn", "Fp")})
    write_md(summary, dts)
    plot(summary, profiles, dts, ref, exact)
    print(json.dumps(summary["gate"], indent=1, default=float)); print("selected dt:", summary["selected_dt"])


def write_md(s, dts):
    L = ["# Stage A: integrator validation against exact Gibbs (pentane R15, beta 2)", "",
         f"Frozen gate: `configs/pentane_r15_dt_fr/prereg_stage_a.json` (tolerance {TAU} on the debiased windowed RMS D; ADMISSIBLE needs both U95 <= tolerance, clip immaterial, no non-finite walker).",
         f"Window {s['window'][0]:.3f}-{s['window'][1]:.3f}; A1 (unbiased) sub-window {s['a1_subwindow'][0]:.3f}-{s['a1_subwindow'][1]:.3f}.", "",
         "| ensemble | dt | clip | D_MF [U95] | D_density [U95] | D(MF vs density) | F' bias RMS | clip frac | D(clip vs no clip) | max step disp | non-finite | bond var ratio (4 bonds) | bend TV |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for dt in dts:
        for ens in ("flat", "gibbs"):
            for clip in (200.0, 0.0):
                c = s["cells"].get(f"{ens}|{dt:g}|{clip:g}")
                if c is None or "D_mf" not in c:
                    continue
                L.append(f"| {ens} | {dt:g} | {clip:g} | {c['D_mf']:.3f} [{c['U95_mf']:.3f}] | {c['D_dens']:.3f} [{c['U95_dens']:.3f}] | {c['D_mf_vs_dens']:.3f} | "
                         f"{c['Fp_bias_rms']:.2f} | {c['clip_frac']:.2e} | {c.get('D_clip_vs_noclip', float('nan')):.3f} | {c['max_disp']:.3f} | {c['n_nonfinite']} | "
                         f"{', '.join(f'{x:.2f}' for x in c['bond_var_ratio'])} | {max(c['bend_tv']):.3f} |")
    L += ["", "## Gate", "", "| dt | verdict | D_MF [U95] | D_density [U95] | clip frac | D(clip vs no clip) |", "|---|---|---|---|---|---|"]
    for d, g in s["gate"].items():
        L.append(f"| {d} | **{g['verdict']}** | {g['D_mf']:.3f} [{g['U95_mf']:.3f}] | {g['D_dens']:.3f} [{g['U95_dens']:.3f}] | {g['clip_frac']:.2e} | {(g['D_clip_vs_noclip'] or 0):.3f} |")
    L += ["", f"**Selected timestep (largest ADMISSIBLE): {s['selected_dt']}**", "",
          "## Torsional-family populations (A1, unbiased, exact start): drift over 10 time units", "",
          "| dt | clip | " + " | ".join(FAM) + " |", "|---|---|" + "---|" * 4]
    for dt in dts:
        for clip in (200.0, 0.0):
            c = s["cells"].get(f"gibbs|{dt:g}|{clip:g}")
            if c is None:
                continue
            L.append(f"| {dt:g} | {clip:g} | " + " | ".join(f"{e:.4f}: {d:+.4f} ± {se:.4f}" for e, d, se in zip(c["fam_exact"], c["fam_drift_end"], c["fam_drift_end_se"])) + " |")
    L.append("\n(cells: exact Gibbs population: mean drift ± SE over the 8 groups)")
    open(os.path.join(STAGE, "SUMMARY.md"), "w").write("\n".join(L) + "\n")


def plot(s, profiles, dts, ref, exact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    cols = {5e-4: "#c0392b", 2.5e-4: "#e67e22", 1.25e-4: "#2471a3", 6.25e-5: "#17a589"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    for dt in dts:
        k = f"flat|{dt:g}|200"
        if k in profiles:
            ax[0, 0].plot(profiles[k]["R"], profiles[k]["F_mf"], color=cols.get(dt, "k"), lw=1.6, label=f"dt {dt:g}")
            ax[0, 1].plot(profiles[k]["R"], profiles[k]["Fp"], color=cols.get(dt, "k"), lw=1.0, label=f"dt {dt:g}")
    for a_, t in ((ax[0, 0], "(a) F_dt - F_ref, mean-force route (flat-bias ensemble, clip 200)"), (ax[0, 1], "(b) F'_dt - F'_ref (binned, flat-bias ensemble)")):
        a_.axhline(0, color="k", lw=0.6); a_.set_xlabel("R15"); a_.set_title(t, fontsize=10); a_.legend(fontsize=8)
    ax[0, 0].axhspan(-TAU, TAU, color="0.85", zorder=0, label="±tolerance")
    ax[0, 0].set_ylabel("free-energy deviation (energy units; 1 kT = 0.5)"); ax[0, 1].set_ylabel("mean-force deviation")
    x = np.array(dts)
    for route, mk in (("mf", "o"), ("dens", "s")):
        y = [s["cells"][f"flat|{d:g}|200"][f"D_{route}"] for d in dts]; u = [s["cells"][f"flat|{d:g}|200"][f"U95_{route}"] for d in dts]
        ax[0, 2].errorbar(x * (1.03 if route == "dens" else 1.0), y, yerr=[np.zeros(len(y)), np.array(u) - np.array(y)], marker=mk, capsize=3,
                          label={"mf": "mean-force route", "dens": "density route"}[route])
    ax[0, 2].axhline(TAU, color="k", ls="--", lw=1, label=f"tolerance {TAU}")
    ax[0, 2].set_xscale("log"); ax[0, 2].set_yscale("symlog", linthresh=0.01); ax[0, 2].set_xlabel("dt"); ax[0, 2].set_ylabel("debiased windowed RMS D (bar: U95)")
    ax[0, 2].set_title("(c) gate statistic vs dt", fontsize=10); ax[0, 2].legend(fontsize=8)
    cf = [max(s["cells"][f"{e}|{d:g}|200"]["clip_frac"] for e in ("flat", "gibbs")) for d in dts]
    ax[1, 0].plot(x, np.maximum(cf, 1e-9), "o-", color="#7d3c98", label="clipped per-atom force evaluations")
    ax[1, 0].axhline(1e-5, color="k", ls="--", lw=1, label="materiality threshold 1e-5")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("dt"); ax[1, 0].set_title("(d) physical force-clip activity (clip 200)", fontsize=10)
    ax[1, 0].legend(fontsize=8)
    for bi in range(4):
        ax[1, 1].plot(x, [s["cells"][f"gibbs|{d:g}|200"]["bond_var_ratio"][bi] for d in dts], "o-", label=f"bond {bi + 1}-{bi + 2}")
    ax[1, 1].axhline(1, color="k", lw=0.6); ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("dt"); ax[1, 1].set_ylabel("Var_EM / Var_Gibbs")
    ax[1, 1].set_title("(e) bond-length variance (unbiased, exact start)", fontsize=10); ax[1, 1].legend(fontsize=8)
    for fi in range(4):
        for dt in dts:
            c = s["cells"][f"gibbs|{dt:g}|200"]
            t = np.array(c["fam_t"]); m = np.array(c["fam_mean"])[:, fi]
            ax[1, 2].plot(t, m - c["fam_exact"][fi], color=cols.get(dt, "k"), lw=1.0, ls=["-", "--", ":", "-."][fi],
                          label=(f"dt {dt:g}" if fi == 0 else None))
    ax[1, 2].axhline(0, color="k", lw=0.6); ax[1, 2].set_xlabel("time"); ax[1, 2].set_ylabel("population - exact Gibbs")
    ax[1, 2].set_title("(f) torsional families (TT -, TG --, GG same :, GG opp -.)", fontsize=10); ax[1, 2].legend(fontsize=8)
    fig.suptitle("Figure 1. Pentane R15 timestep validity: production EM vs exact Gibbs (beta 2)", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig1_timestep_validity.{ext}"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
