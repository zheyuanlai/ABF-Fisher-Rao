#!/usr/bin/env python
"""WCA reference audit (docs/numerical_validation/WCA_REFERENCE_AUDIT.md).

Every historical WCA free-energy reference and every pooled sampler limit, compared with the exact
Gibbs reference (Metropolis MC, results/numerical_validation/wca/mc) on the 160 bin centres inside
[-0.1, 1.1] with the frozen statistic of the validation prereg (debiased RMS D, jackknife noise).

    python scripts/numerical_validation/analyze_wca_reference_audit.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_wca_validation as A  # noqa: E402

ROOT = A.ROOT
LAD = os.path.join(ROOT, "results", "wca_replica_ladder")
OUT = os.path.join(ROOT, "results", "numerical_validation", "wca")


class Fixed:
    """A profile without a sampling-noise estimate (TI references)."""

    def __init__(self, label, F):
        self.label, self.F, self.se, self.n = label, A.centre(F), np.zeros(A.NB), 1
        self.kind = "fixed"

    def boot(self, idx):
        return self.F


def ti_profile(path):
    d = np.load(path, allow_pickle=True)
    return np.interp(A.CEN, np.asarray(d["grid"], float), np.asarray(d["free_energy"], float)), str(d["label"])


def ladder_recs(pattern, arm):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import wca_numba as wn
    out = []
    for f in sorted(glob.glob(os.path.join(LAD, pattern))):
        r = wn.load_result(f)
        a = list(r["arms"]).index(arm)
        out.append(dict(C=np.asarray(r["C_rep"])[a, -1, :A.NB].astype(float), M=np.asarray(r["M_rep"])[a, -1, :A.NB].astype(float)))
    return out


def main():
    mc = A.load_group("mc", "mc")
    ref = A.Ens("MC F_density", mc, "density")
    res = dict(reference="MC F_density (exact Gibbs, %d chains)" % len(mc), rows=[])
    prof = {}

    def add(E, family, note=""):
        c = A.compare(E, ref, nboot=1000)
        c.update(family=family, note=note)
        res["rows"].append(c)
        prof[E.label] = E.F
        print(f"{family:28s} {E.label:58s} D {c['D']:.4f} [up {c['D_upper95']:.4f}] rms {c['rms_obs']:.4f} noise {c['noise']:.4f} max|d| {c['maxabs']:.3f}",
              flush=True)
        return c

    add(A.Ens("MC F_MF (same chains, mean-force route)", mc, "mf"), "exact Gibbs")
    for path, lab in ((os.path.join(ROOT, "cache", "wca_ti_reference.npz"), "v1 TI (51 pts, dt 0.002, smoothed 1 cell)"),
                      (os.path.join(ROOT, "cache", "phase", "wca_ti_b1_h2_w2_n10_a1.5_g160.npz"), "phase TI cache (dt 0.002)"),
                      (os.path.join(ROOT, "cache", "phase_hp_v3", "wca_ti_b1_h2_w2_n10_a1.5_g160.npz"), "v2 TI = accepted reference (dt 0.002)")):
        if os.path.exists(path):
            F, label = ti_profile(path)
            add(Fixed(lab, F), "TI reference", label)
    for tag, dt in (("A_dt0.002", 0.002), ("B_dt0.001", 0.001), ("C_dt0.0005", 0.0005)):
        recs = A.load_old(tag) + A.load_group("em_impl", f"dt{dt:g}")
        if recs:
            add(A.Ens(f"unbiased production EM dt {dt:g}, F_density", recs, "density"), "unbiased EM")
            add(A.Ens(f"unbiased production EM dt {dt:g}, F_MF (production estimator)", recs, "mf"), "unbiased EM")
    for dt in (0.00025, 0.000125):
        recs = A.load_group("em_impl", f"dt{dt:g}")
        if recs:
            add(A.Ens(f"unbiased production EM dt {dt:g}, F_density", recs, "density"), "unbiased EM")
            add(A.Ens(f"unbiased production EM dt {dt:g}, F_MF (production estimator)", recs, "mf"), "unbiased EM")
    for d, dt in (("reference", 0.002), ("reference_dt0.0005", 0.0005)):
        recs = ladder_recs(f"{d}/raw/N1_s*.npz", "hist_abf")
        if recs:
            add(A.Ens(f"dyn-consistent ref: pooled serial ABF dt {dt:g} ({len(recs)} seeds)", recs, "mf"), "dt-consistent ref")
    lim = {}
    for d, dt in (("production", 0.002), ("production_dt0.0005", 0.0005)):
        for N in (1024, 256, 64, 16, 4):
            for arm, short in (("hist_abf", "ABF"), ("hist_fr_uniform", "ABF+FR")):
                recs = ladder_recs(f"{d}/raw/N{N}_s*.npz", arm)
                if recs:
                    E = A.Ens(f"{short} pooled limit dt {dt:g} N {N}", recs, "mf")
                    lim[(dt, N, short)] = E
                    add(E, f"ladder limit dt {dt:g}")
    for rate in ("0.1", "0.03", "0.01"):
        for arm, short in (("hist_abf", "ABF"), ("hist_fr_uniform", "ABF+FR")):
            try:
                recs = ladder_recs(f"fr_dose/raw/r{rate}_s*.npz", arm)
            except ValueError:
                recs = []
            if recs:
                add(A.Ens(f"{short} pooled limit dt 0.002 N 16 FR rate {rate}", recs, "mf"), "FR dose dt 0.002")
    # FR tilt: FR limit vs ABF limit at the same dt and N
    tilt = []
    for (dt, N, short), E in lim.items():
        if short == "ABF+FR" and (dt, N, "ABF") in lim:
            c = A.compare(E, lim[(dt, N, "ABF")], nboot=500)
            tilt.append(dict(dt=dt, N=N, D_FR_vs_ABF=c["D"], up=c["D_upper95"], noise=c["noise"]))
            print(f"FR tilt dt {dt:g} N {N:5d}: D(FR limit, ABF limit) {c['D']:.4f} [up {c['D_upper95']:.4f}] noise {c['noise']:.4f}", flush=True)
    res["fr_tilt"] = tilt
    res["profiles"] = {k: v.tolist() for k, v in prof.items()}
    res["profiles"]["MC F_density"] = ref.F.tolist()
    res["mc_se"] = ref.se.tolist()
    res["centres"] = A.CEN.tolist()
    json.dump(res, open(os.path.join(OUT, "reference_audit.json"), "w"), indent=1)
    figure(res, prof, ref)


def figure(res, prof, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    W = A.WIN
    z = A.CEN[W]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=False)
    ax = axs[0]
    ax.fill_between(z, -2 * ref.se[W], 2 * ref.se[W], color="#dddddd", label="MC +-2 se")
    sty = [("v2 TI = accepted reference (dt 0.002)", "#555555", "-", 2.0), ("v1 TI (51 pts, dt 0.002, smoothed 1 cell)", "#999999", "--", 1.4),
           ("dyn-consistent ref: pooled serial ABF dt 0.002 (32 seeds)", "#c0392b", "-", 1.4),
           ("dyn-consistent ref: pooled serial ABF dt 0.0005 (32 seeds)", "#2a78d6", "-", 1.4),
           ("MC F_MF (same chains, mean-force route)", "#1baf7a", ":", 1.6)]
    for k, c, ls, lw in sty:
        if k in prof:
            ax.plot(z, (prof[k] - ref.F)[W], color=c, ls=ls, lw=lw, label=k)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("z"); ax.set_ylabel("profile - F_MC (kT)"); ax.set_title("(a) references vs exact Gibbs", loc="left")
    ax.legend(fontsize=6.5)
    for ax, dt in ((axs[1], 0.002), (axs[2], 0.0005)):
        ax.fill_between(z, -2 * ref.se[W], 2 * ref.se[W], color="#dddddd")
        for N, c in ((1024, "#6c3483"), (64, "#e67e22"), (16, "#1baf7a")):
            for short, ls in (("ABF", "-"), ("ABF+FR", "--")):
                k = f"{short} pooled limit dt {dt:g} N {N}"
                if k in prof:
                    ax.plot(z, (prof[k] - ref.F)[W], color=c, ls=ls, lw=1.3, label=f"{short} N {N}")
        k = f"unbiased production EM dt {dt:g}, F_density"
        if k in prof:
            ax.plot(z, (prof[k] - ref.F)[W], color="k", lw=1.0, label="unbiased EM, density")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("z"); ax.set_title(f"({'b' if dt == 0.002 else 'c'}) sampler limits at dt {dt:g} minus F_MC", loc="left")
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_wca_reference_audit.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
