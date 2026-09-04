#!/usr/bin/env python
"""Analyzer for the WCA OT + repair mechanism stage M2 (docs/WCA_OT_REPAIR_MECHANISM.md).

Reads the M2 arms (ot_a{alpha}_c{c}: T0 = c 0, TR = c > 0) and the targeted-relax campaign's W1 /
W1b comparators on the same seeds (abf = A, fr_uniform = F, abf_ptarg1 = R, fr_ptarg1 = F+R) and
reports, per the preregistration:

  1. mechanism (deposit-free): per-bin mean of the first outer deposit after an OT event minus
     F'_ref -- T0's injected bias entering ABF, TR's post-repair residual, TR's pre-repair sample;
     occupancy-weighted RMS over bins with >= MIN_COUNT samples inside the eval window;
  2. error at the frozen read-out h** = 0.00625: paired dI_F and de_F(T) for the preregistered
     contrasts (seeds common to the two arms; descriptive bootstrap CIs, 4 seeds);
  3. compute: C(eps) = outer N x step + inner replica-steps (exact), eps_A = A's final error;
  4. safety / dose counters.

Partial data are analysed as they arrive.  Output: <M2>/analysis.json, <M2>/figures/.
    python scripts/analyze_wca_ot_repair.py
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
from analyze_wca_targeted_relax import compute_axis, first_reach, fmt, paired          # noqa: E402
from analyze_wca_bandwidth_audit import Z_HI, Z_LO, readouts                            # noqa: E402

W1_RAW = os.path.join(ROOT, "results/targeted_relax_campaign/wca/W1/raw")
M2_DIR = os.path.join(ROOT, "results/ot_repair_campaign/wca/M2")
H_STAR = "0.00625"
MIN_COUNT = 200
COMPARATORS = {"abf": "A", "fr_uniform": "F", "abf_ptarg1": "R", "fr_ptarg1": "F+R"}
CONTRASTS = [("T0", "A"), ("TR", "T0"), ("TR", "A"), ("TR", "R"), ("TR", "F"), ("TR", "F+R"), ("T0", "F")]


def load(raw_dir, stage, keep=None):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, f"{stage}__*__*.npz"))):
        d = np.load(f, allow_pickle=True)
        name = str(d["name"])
        if keep is not None and name not in keep:
            continue
        runs.setdefault(name, {})[int(d["seed"])] = {k: d[k] for k in d.files}
    return runs


def mechanism(run, grid, mask, Fp_ref):
    """Occupancy-weighted RMS of (deposit mean - F'_ref) over well-populated bins in the window."""
    out = {}
    for tag in ("pre", "post"):
        C = np.asarray(run.get(f"ot_C_{tag}", np.zeros_like(grid)), float)
        S = np.asarray(run.get(f"ot_Sf_{tag}", np.zeros_like(grid)), float)
        ok = mask & (C >= MIN_COUNT)
        if not ok.any():
            out[tag] = dict(rms=None, rms_Fref=None, n_bins=0, bias=None, counts=None)
            continue
        mean = np.where(ok, S / np.maximum(C, 1.0), np.nan)
        bias = mean - Fp_ref
        w = np.where(ok, C, 0.0); w = w / w.sum()
        out[tag] = dict(rms=float(np.sqrt(np.nansum(w * bias ** 2))), rms_Fref=float(np.sqrt(np.sum(w * Fp_ref ** 2))),
                        mean_signed=float(np.nansum(w * bias)), n_bins=int(ok.sum()), total_samples=float(C[ok].sum()),
                        bias=np.where(ok, bias, np.nan), counts=C)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2", default=M2_DIR)
    ap.add_argument("--w1-raw", default=W1_RAW)
    a = ap.parse_args()
    runs = load(a.w1_raw, "W1", keep=set(COMPARATORS))
    runs = {COMPARATORS[k]: v for k, v in runs.items()}
    m2 = load(os.path.join(a.m2, "raw"), "M2")
    label = {}
    for name in sorted(m2):
        c = float(name.split("_c")[1]); al = float(name.split("_a")[1].split("_")[0])
        lab = "T0" if c == 0 else f"TR{c:g}"
        label[lab] = name; runs[lab] = m2[name]
    if not m2:
        sys.exit("no M2 runs yet")
    any_run = next(iter(runs["A"].values()))
    grid = np.asarray(any_run["grid"], float); ref_F = np.asarray(any_run["reference_free_energy"], float)
    Fp_ref = np.asarray(any_run["reference_mean_force"], float)
    t = np.asarray(any_run["profile_times"], float); mask = (grid >= Z_LO) & (grid <= Z_HI); sigma = float(any_run.get("abf_smooth_sigma", 0.5))
    ro = {m: {s: readouts(r, grid, mask, ref_F, sigma) for s, r in runs[m].items()} for m in runs}
    print(f"arms: {{{', '.join(f'{m}: {sorted(runs[m])}' for m in runs)}}}; read-out h** = {H_STAR}; window [{Z_LO}, {Z_HI}]")

    res = dict(read_out=H_STAR, arms={}, contrasts={}, mechanism={}, compute={})
    # ---- mechanism (deposit-free), per M2 arm, median over its seeds
    print("\nMECHANISM (first outer deposit after an OT event vs F'_ref; occupancy-weighted RMS over bins with >= 200 samples)")
    prof = {}
    for lab in sorted(l for l in runs if l.startswith(("T0", "TR"))):
        per = {s: mechanism(r, grid, mask, Fp_ref) for s, r in runs[lab].items()}
        rec = {}
        for tag in ("pre", "post"):
            vals = [p[tag]["rms"] for p in per.values() if p[tag]["rms"] is not None]
            if vals:
                rec[tag] = dict(rms_median=float(np.median(vals)), rms_per_seed={int(s): p[tag]["rms"] for s, p in per.items() if p[tag]["rms"] is not None},
                                rms_Fref=float(np.median([p[tag]["rms_Fref"] for p in per.values() if p[tag]["rms"] is not None])),
                                mean_signed_median=float(np.median([p[tag]["mean_signed"] for p in per.values() if p[tag]["rms"] is not None])),
                                n_bins=int(np.median([p[tag]["n_bins"] for p in per.values() if p[tag]["rms"] is not None])))
                prof[(lab, tag)] = np.nanmedian(np.stack([p[tag]["bias"] for p in per.values() if p[tag]["bias"] is not None]), 0)
        r0 = next(iter(runs[lab].values()))
        rec["dose"] = dict(alpha=float(r0["ot_alpha"]), dz_max=float(r0["ot_dz_max"]), c=float(r0["ot_c_repair"]),
                           moved_frac=float(np.median([float(r["ot_moved_frac"]) for r in runs[lab].values()])),
                           absdz_mean=float(np.median([float(r["ot_absdz_mean"]) for r in runs[lab].values()])),
                           absdz_max=float(np.median([float(r["ot_absdz_max"]) for r in runs[lab].values()])),
                           capped_frac=float(np.median([float(r["ot_capped_frac"]) for r in runs[lab].values()])),
                           opportunities=int(np.median([int(r["ot_n_opportunities"]) for r in runs[lab].values()])),
                           inner_steps=float(np.median([float(r["relax_steps_total"]) for r in runs[lab].values()])),
                           cost_ratio=float(np.median([float(r["relax_cost_ratio"]) for r in runs[lab].values()])),
                           wall_s=float(np.median([float(r["wall_seconds"]) for r in runs[lab].values()])))
        res["mechanism"][lab] = rec
        d = rec["dose"]
        line = f"  {lab:>5} (alpha {d['alpha']:g}, c {d['c']:g}, {d['opportunities']} events, moved {d['moved_frac']:.3f}, |dz| mean {d['absdz_mean']:.4f} max {d['absdz_max']:.4f}, capped {d['capped_frac']:.3f}, inner {d['inner_steps']:.0f} = {d['cost_ratio']:.2f}x, wall {d['wall_s']:.0f}s)"
        for tag in ("pre", "post"):
            if tag in rec:
                line += f"\n         {tag:>4}: RMS bias {rec[tag]['rms_median']:.3f}  (signed mean {rec[tag]['mean_signed_median']:+.3f}; |F'_ref| RMS on the same bins {rec[tag]['rms_Fref']:.3f}; {rec[tag]['n_bins']} bins)"
        print(line)
    if "T0" in res["mechanism"] and "post" in res["mechanism"]["T0"]:
        t0 = res["mechanism"]["T0"]["post"]
        res["mechanism"]["H1_injected_over_Fref"] = t0["rms_median"] / max(t0["rms_Fref"], 1e-12)
        print(f"  H1: T0 deposit bias / |F'_ref| = {res['mechanism']['H1_injected_over_Fref']:.2f}  (>= 0.5 supports H1)")
        for lab in res["mechanism"]:
            if lab.startswith("TR") and "post" in res["mechanism"][lab]:
                fr = res["mechanism"][lab]["post"]["rms_median"] / max(t0["rms_median"], 1e-12)
                pre = res["mechanism"][lab].get("pre", {}).get("rms_median")
                res["mechanism"][lab]["residual_fraction_vs_T0"] = fr
                res["mechanism"][lab]["residual_fraction_vs_own_pre"] = (res["mechanism"][lab]["post"]["rms_median"] / pre) if pre else None
                print(f"  H2: {lab} post-repair deposit bias / T0's = {fr:.3f}  (own pre -> post {res['mechanism'][lab]['residual_fraction_vs_own_pre']:.3f}); removed {100 * (1 - fr):.0f} %")

    # ---- error and compute at h**
    print(f"\nERROR at h** = {H_STAR} (integrated e_F over the run, final e_F(T); paired on common seeds; descriptive CIs)")
    I = {m: {s: float(np.trapezoid(ro[m][s][H_STAR], t)) for s in ro[m]} for m in ro}
    fin = {m: {s: float(ro[m][s][H_STAR][-1]) for s in ro[m]} for m in ro}
    for m in ro:
        res["arms"][m] = dict(seeds=sorted(ro[m]), I_F_median=float(np.median(list(I[m].values()))), final_median=float(np.median(list(fin[m].values()))))
        print(f"  {m:>5}: seeds {sorted(ro[m])}  I_F {res['arms'][m]['I_F_median']:.4f}  e_F(T) {res['arms'][m]['final_median']:.5f}")
    k = 0
    for arm, ref in CONTRASTS:
        arms_here = [l for l in ro if (l == arm or (arm == "TR" and l.startswith("TR")))]
        for l in arms_here:
            if ref not in ro:
                continue
            common = sorted(set(ro[l]) & set(ro[ref]))
            if not common:
                continue
            cI = paired([I[l][s] for s in common], [I[ref][s] for s in common], k); cF = paired([fin[l][s] for s in common], [fin[ref][s] for s in common], k + 1); k += 2
            res["contrasts"][f"{l} vs {ref}"] = dict(seeds=common, dI_F=cI, d_final=cF)
            print(f"  {l:>5} vs {ref:<4}: dI_F {fmt(cI)}   final {fmt(cF)}")
    # compute
    common_all = sorted(set.intersection(*[set(ro[m]) for m in ro]))
    if common_all and "A" in ro:
        med = {m: np.median([ro[m][s][H_STAR] for s in common_all], axis=0) for m in ro}
        caxis = {m: np.median([compute_axis(runs[m][s]) for s in common_all], axis=0) for m in ro}
        eps_A = float(med["A"][-1]); eps_F = float(med["F"][-1]) if "F" in med else None
        C_A = first_reach(caxis["A"], med["A"], eps_A)
        print(f"\nCOMPUTE (median curves over the {len(common_all)} seeds common to all arms; C = N x step + inner steps): eps_A = {eps_A:.5f}")
        for m in ro:
            C = first_reach(caxis[m], med[m], eps_A)
            rec = dict(C_eps_A=C, ratio_vs_A=(C / C_A if np.isfinite(C_A) and C_A > 0 else None), total_C=float(caxis[m][-1]),
                       cost_ratio=float(caxis[m][-1] / caxis["A"][-1]))
            if eps_F is not None:
                CF = first_reach(caxis[m], med[m], eps_F); rec["C_eps_F"] = CF
            res["compute"][m] = rec
            print(f"  {m:>5}: reaches eps_A at C = {C:.3g} (A: {C_A:.3g}, ratio {rec['ratio_vs_A'] if rec['ratio_vs_A'] is None else round(rec['ratio_vs_A'], 3)}); total compute {rec['cost_ratio']:.2f}x A" +
                  (f"; reaches eps_F at {rec['C_eps_F']:.3g}" if eps_F is not None else ""))
        res["compute"]["common_seeds"] = common_all
    os.makedirs(os.path.join(a.m2, "figures"), exist_ok=True)
    json.dump(res, open(os.path.join(a.m2, "analysis.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    figures(ro, runs, t, grid, mask, Fp_ref, prof, os.path.join(a.m2, "figures"))
    print(f"\nwrote {os.path.join(a.m2, 'analysis.json')}")


def figures(ro, runs, t, grid, mask, Fp_ref, prof, fig_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    order = [m for m in ("A", "F", "R", "F+R", "T0", "TR0.5", "TR1", "TR2") if m in ro]
    cols = dict(A="k", F="tab:blue", R="tab:gray", **{"F+R": "tab:cyan"}, T0="tab:red", **{"TR0.5": "tab:orange", "TR1": "tab:green", "TR2": "tab:olive"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    for m in order:
        seeds = sorted(ro[m]); med = np.median([ro[m][s][H_STAR] for s in seeds], 0)
        axes[0].plot(t, med, color=cols.get(m, None), lw=1.3, label=f"{m} ({len(seeds)})")
        cax = np.median([compute_axis(runs[m][s]) for s in seeds], 0)
        axes[1].plot(cax / 1e6, med, color=cols.get(m, None), lw=1.3)
    for ax in axes[:2]:
        ax.set_yscale("log"); ax.axvline(40 if ax is axes[0] else 40 * 1024 / 1e6, color="gray", lw=0.6, ls=":")
    axes[0].set_xlabel("t (time units)"); axes[0].set_ylabel(f"e_F at h = {H_STAR}"); axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("force evaluations (millions, inner steps charged)"); axes[1].set_ylabel("e_F")
    ax = axes[2]
    ax.plot(grid, Fp_ref, "k-", lw=1.0, label="F'_ref")
    for (lab, tag), b in prof.items():
        ax.plot(grid, b, lw=1.0, ls="-" if tag == "post" else "--", color=cols.get(lab, None), label=f"{lab} {tag}")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlim(Z_LO, Z_HI); ax.set_xlabel("z"); ax.set_ylabel("deposit after OT event - F'_ref")
    ax.legend(fontsize=6, frameon=False); ax.set_title("in-sampler injected / residual bias (median over seeds)", fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "m2_curves_compute_mechanism.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
