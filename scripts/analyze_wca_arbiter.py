#!/usr/bin/env python
"""POST-HOC arbiter for the WCA free energy (configs/wca_replica_ladder/unbiased_arbiter.json).

Compares, on the 160-node grid after additive alignment over the eval window:
  F_density   -kT log(visit counts) of plain UNBIASED dynamics (no mean-force estimator, no FR, no TI)
  F_mf_unb    the integrated mean-force estimate of the same unbiased runs (tests the estimator)
  ABF limit   pooled serial ABF of the independent reference seeds (3400-3431)
  FR limits   pooled ABF+FR of the ladder at N = 16, 64, 256 (production seeds 3300-3315)
  v2 TI       the accepted reference
"""
import glob
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import wca_numba as wn  # noqa: E402

OUT = os.path.join(ROOT, "results", "wca_replica_ladder")


def main():
    torch, core = wn._cpu_torch()
    ref = wn.load_reference()
    sim, _ = wn.accepted_setup(n_bins=160)
    g = np.asarray(ref["grid"])
    m = (g >= sim.eval_z_lo) & (g <= sim.eval_z_hi)
    beta = wn.ACCEPTED_CFG["beta"]
    nb = 160
    delta = (sim.z_max - sim.z_min) / nb
    centres = sim.z_min + (np.arange(nb) + 0.5) * delta

    def centre(F):
        return F - F[m].mean()

    def pmf(M, C):
        est = core.make_abf_estimator(sim, torch.linspace(sim.z_min, sim.z_max, sim.n_grid, dtype=torch.float64))
        est.M, est.C = torch.tensor(M), torch.tensor(C)
        return centre(core.to_numpy(est.pmf_profile()))

    def density_F(C):
        p = np.asarray(C, float) / (np.sum(C) * delta)
        Fc = -np.log(np.maximum(p, 1e-300)) / beta
        return centre(np.interp(g, centres, Fc))

    rms = lambda a, b: float(np.sqrt(np.mean((a - b)[m] ** 2)))

    def load(pattern):
        return [wn.load_result(f) for f in sorted(glob.glob(os.path.join(OUT, pattern)))]

    U = load("unbiased/raw/N1_s*.npz")
    assert len(U) >= 16, len(U)
    sumC = lambda rs, a=0: sum(np.asarray(r["C_prod"])[a, -1, :nb] for r in rs)
    sumM = lambda rs, a=0: sum(np.asarray(r["M_prod"])[a, -1, :nb] for r in rs)
    Fd = density_F(sumC(U))
    Fd_h = [density_F(sumC(h)) for h in (U[: len(U) // 2], U[len(U) // 2:])]
    Fmf = pmf(sumM(U), sumC(U))
    Fmf_h = [pmf(sumM(h), sumC(h)) for h in (U[: len(U) // 2], U[len(U) // 2:])]

    Rr = load("reference/raw/N1_s*.npz")
    ai = list(Rr[0]["arms"]).index("hist_abf")
    Fabf = pmf(sum(np.asarray(r["M_rep"])[ai, -1, :nb] for r in Rr), sum(np.asarray(r["C_rep"])[ai, -1, :nb] for r in Rr))
    prof = {"F_density (unbiased MD)": Fd, "F_mf (unbiased runs, mean-force estimator)": Fmf,
            "ABF long-run limit (32 independent seeds)": Fabf, "v2 TI reference": centre(np.asarray(ref["free_energy"]))}
    for N in (256, 64, 16):
        R = load(f"production/raw/N{N}_s*.npz")
        prof[f"ABF+FR limit, N = {N}"] = pmf(sum(np.asarray(r["M_rep"])[1, -1, :nb] for r in R), sum(np.asarray(r["C_rep"])[1, -1, :nb] for r in R))
        prof[f"ABF limit, N = {N} (ladder)"] = pmf(sum(np.asarray(r["M_rep"])[0, -1, :nb] for r in R), sum(np.asarray(r["C_rep"])[0, -1, :nb] for r in R))

    noise_d = rms(*Fd_h) / 2.0
    noise_mf = rms(*Fmf_h) / 2.0
    res = dict(n_unbiased=len(U), visits_per_seed=float(np.sum(np.asarray(U[0]["C_prod"])[0, -1])),
               min_window_count=float(sumC(U)[(centres >= sim.eval_z_lo) & (centres <= sim.eval_z_hi)].min()),
               noise_F_density=noise_d, noise_F_mf=noise_mf, rms_vs_F_density={k: rms(v, Fd) for k, v in prof.items()})
    print(json.dumps(res, indent=1))
    zs = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1]
    print("profiles minus F_density at z:", zs)
    for k, v in prof.items():
        print(f"  {k:45s}", " ".join(f"{(v - Fd)[np.argmin(abs(g - z))]:+.3f}" for z in zs))
    res["profiles"] = {k: v.tolist() for k, v in prof.items()}
    res["grid"] = g.tolist()
    json.dump(res, open(os.path.join(OUT, "arbiter.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "axes.titleweight": "bold", "grid.color": "#e6e5e0"})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    sty = {"v2 TI reference": ("#8a8984", "-", 2.2), "ABF long-run limit (32 independent seeds)": ("#2a78d6", "-", 1.8),
           "F_mf (unbiased runs, mean-force estimator)": ("#1baf7a", (0, (4, 2)), 1.6), "ABF+FR limit, N = 64": ("#eb6834", "-", 1.8),
           "ABF+FR limit, N = 16": ("#eb6834", (0, (2, 1.5)), 1.4), "ABF+FR limit, N = 256": ("#eb6834", (0, (5, 1.5)), 1.4)}
    ax = axs[0]
    ax.plot(g, Fd, color="#0b0b0b", lw=2.6, label="F_density = -kT log P(z), unbiased MD")
    for k, (c, ls, lw) in sty.items():
        ax.plot(g, prof[k], color=c, ls=ls, lw=lw, label=k)
    ax.axvspan(sim.eval_z_lo, sim.eval_z_hi, color="#f2f1ec", zorder=-1)
    ax.set_xlabel("z  (dimer bond CV)"); ax.set_ylabel("F (kT), aligned on the eval window"); ax.set_title("(a) free-energy profiles", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax = axs[1]
    ax.axhspan(-2 * noise_d, 2 * noise_d, color="#e6e5e0", label="+-2 noise of F_density")
    for k, (c, ls, lw) in sty.items():
        ax.plot(g[m], (prof[k] - Fd)[m], color=c, ls=ls, lw=lw, label=f"{k}  (RMS {rms(prof[k], Fd):.3f})")
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("z"); ax.set_ylabel("profile - F_density (kT)"); ax.set_title("(b) difference from the unbiased-MD free energy", loc="left")
    ax.legend(fontsize=6.5, loc="best"); ax.grid(True)
    fig.suptitle(f"WCA dimer: which free energy is right? Arbiter = plain unbiased dynamics ({len(U)} seeds x 1.2288e8 steps), POST-HOC",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_wca_arbiter.{ext}"), dpi=170)


if __name__ == "__main__":
    main()
