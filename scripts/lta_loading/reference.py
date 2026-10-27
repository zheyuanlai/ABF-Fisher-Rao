"""Reference free energy of the tagged guest at finite loading (docs/lta_loading/PREREGISTRATION.md §4).

Two runs of the same engine, seeds disjoint from every scored run:
  (a) pilot ABF (FR off), 4 seeds x N 1024 x 300k steps -> bias table Gamma_pilot (pooled production bins);
  (b) frozen-bias equilibrium, 8 independent groups x N 1024 x 500k steps (burn-in 50k) under the FIXED
      bias Gamma_pilot.  The bias depends on phi only, so
        gamma_ref  = pooled conditional mean force M/C          (mean-force route; F_ref = histogram_pmf)
        F_density  = A_pilot - kT log C_prod  (+ const)          (density route; independent of the forces)
      with group-jackknife standard errors over the 8 groups.
Writes results/lta_loading/reference/reference.npz (+ .json summary).  Usage:
  python scripts/lta_loading/reference.py run      # (a) and (b), on the GPU
  python scripts/lta_loading/reference.py build    # the reference from the two npz files
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import lta_ladder_numba as E  # noqa: E402

OUT = os.path.join(ROOT, "results", "lta_loading", "reference")
PILOT = dict(seeds="30000-30003", N=1024, n_steps=300_000)
FROZEN = dict(seeds="30100-30107", N=1024, n_steps=500_000, burn=50_000)
T_K = 150.0


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def run():
    os.makedirs(OUT, exist_ok=True)
    py = sys.executable
    rb = os.path.join(HERE, "run_batch.py")
    sh([py, rb, "--method", "abf", "--N", str(PILOT["N"]), "--seeds", PILOT["seeds"], "--n-steps",
        str(PILOT["n_steps"]), "--out", "results/lta_loading/reference/pilot_abf.npz",
        "--save-steps", f"0,{PILOT['n_steps'] // 2},{PILOT['n_steps']}"])
    from lta_loading import engine as LE
    out, _ = LE.load_npz(os.path.join(OUT, "pilot_abf.npz"))
    acc = out["sv_acc"][-1]
    M, C = acc[:, 2].sum(0), acc[:, 3].sum(0)
    if (C == 0).any():
        raise SystemExit(f"pilot left {(C == 0).sum()} empty bins; the frozen bias would be undefined there")
    table = E.mean_force(M[None], C[None])[0]
    np.save(os.path.join(OUT, "table_pilot.npy"), table)
    sh([py, rb, "--method", "frozen", "--N", str(FROZEN["N"]), "--seeds", FROZEN["seeds"], "--n-steps",
        str(FROZEN["n_steps"]), "--table", "results/lta_loading/reference/table_pilot.npy",
        "--out", "results/lta_loading/reference/frozen.npz",
        "--save-steps", f"0,{FROZEN['burn']},{FROZEN['n_steps'] // 2},{FROZEN['n_steps']}",
        "--set", "warmup_steps=1", f"burn_in_steps={FROZEN['burn']}"])
    build()


def build():
    from lta_loading import engine as LE
    out, meta = LE.load_npz(os.path.join(OUT, "frozen.npz"))
    table = np.load(os.path.join(OUT, "table_pilot.npy"))
    kT = E.KB * T_K
    acc = out["sv_acc"][-1]                      # (R, 4, ng)
    Mg, Cg = acc[:, 2], acc[:, 3]
    R, ng = Mg.shape
    dphi = 2 * np.pi / ng
    grid = -np.pi + (np.arange(ng) + 0.5) * dphi
    A = E.histogram_pmf(table[None], dphi)[0]

    def routes(M, C):
        g = E.mean_force(M[None], C[None])[0]
        Fm = E.histogram_pmf(g[None], dphi)[0]
        Fd = A - kT * np.log(np.maximum(C, 1e-300))
        Fd = Fd - Fd.mean()
        return g, Fm, Fd

    g, Fm, Fd = routes(Mg.sum(0), Cg.sum(0))
    jk = [routes(Mg.sum(0) - Mg[r], Cg.sum(0) - Cg[r]) for r in range(R)]

    def jse(k):
        x = np.stack([j[k] for j in jk])
        return np.sqrt((R - 1) / R * ((x - x.mean(0)) ** 2).sum(0))

    g_se, Fm_se, Fd_se = jse(0), jse(1), jse(2)
    d = Fm - Fd
    d = d - d.mean()
    single = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
    i_cage = int(np.argmax(np.abs(grid)))
    i_win = int(np.argmin(np.abs(grid)))
    barrier = float(Fm[i_win] - Fm.min())
    barrier_single = float(single["F_ref"][i_win] - single["F_ref"].min())
    # the scorer convention: F_ref is the histogram_pmf of the reference's own per-bin mean force
    np.savez(os.path.join(OUT, "reference.npz"), F_ref=Fm, gamma_ref=g, F_ref_se=Fm_se, gamma_se=g_se,
             F_density=Fd, F_density_se=Fd_se, grid_phi=grid, T_K=T_K, n_guests=meta["n_guests"],
             counts=Cg.sum(0), table_pilot=table)
    summ = dict(
        T_K=T_K, n_guests=meta["n_guests"], groups=R, samples=float(Cg.sum()),
        min_bin_count=float(Cg.sum(0).min()),
        F_ref_se_rms=float(np.sqrt(np.mean(Fm_se ** 2))), F_ref_se_max=float(Fm_se.max()),
        density_route_se_rms=float(np.sqrt(np.mean(Fd_se ** 2))),
        mf_vs_density_rms=float(np.sqrt(np.mean(d ** 2))), mf_vs_density_maxabs=float(np.abs(d).max()),
        barrier_kJ=barrier, barrier_kT=barrier / kT, barrier_single_guest_kJ=barrier_single,
        barrier_single_guest_kT=barrier_single / kT,
        cage_minus_window_note="barrier = F(window bin) - min F, mean-force route",
        frozen_run_s=meta["timers"]["run_s"], frozen_engine_sha=meta["engine_sha256"],
        mc_acceptance=float(out["sv_i"][-1, :, 2].sum() / max(1, out["sv_i"][-1, :, 3].sum())),
        tagged_crossings_per_walker=float(out["sv_i"][-1, :, 1].sum() / (R * meta["N"])),
    )
    with open(os.path.join(OUT, "reference.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"run": run, "build": build}[sys.argv[1]]()
