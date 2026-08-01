"""Configurational-temperature and dt-ladder check for the frozen alanine integrator.

Why the KINETIC temperature is the wrong gate here.  BAOAB at finite ``dt`` under-thermalises
the stiffest modes' kinetic energy: the fastest X-H stretch has a ~10 fs period, so at
``dt = 1 fs`` the measured kinetic temperature sits a couple of percent below the target.  Our
observable is a CONFIGURATIONAL free energy, and BAOAB is chosen precisely because its
configurational error is ``O(dt^2)`` with a small prefactor.  The diagnostic that matters is
therefore the **configurational temperature**

    kB T_conf = <|grad V|^2> / <laplacian V>

which equals ``T`` exactly when the configurational marginal is the Boltzmann one, and is blind
to how the kinetic energy is distributed.

The Laplacian is computed exactly (66 double-backward passes on the 22-atom system), not by a
stochastic estimator, so the number is not itself noisy in the trace.

Usage: CUDA_VISIBLE_DEVICES=7 python -u scripts/check_alanine_dt_bias.py
       OMP_NUM_THREADS=32 python -u scripts/check_alanine_dt_bias.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import math

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB, KB, make_seed_streams               # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters             # noqa: E402
from alanine.system import PHI_ATOMS, PSI_ATOMS, reference_minimum     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_alanine_reference import dihedral_t                           # noqa: E402


def laplacian_and_gradsq(tff, x):
    """Exact ``<|grad V|^2>`` and ``<laplacian V>`` over a batch ``x (B,A,3)``."""
    B, A, _ = x.shape
    n = A * 3
    xg = x.detach().requires_grad_(True)
    E = tff.energy(xg).sum()
    g, = torch.autograd.grad(E, xg, create_graph=True)
    gflat = g.reshape(B, n)
    lap = torch.zeros(B, device=x.device, dtype=x.dtype)
    for k in range(n):                       # exact trace: one double-backward per coordinate
        gk, = torch.autograd.grad(gflat[:, k].sum(), xg, retain_graph=(k < n - 1))
        lap = lap + gk.reshape(B, n)[:, k]
    return (gflat.detach() ** 2).sum(-1), lap.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alanine/dt_bias")
    ap.add_argument("--walkers", type=int, default=256)
    ap.add_argument("--equil-ps", type=float, default=20.0)
    ap.add_argument("--ps", type=float, default=100.0)
    ap.add_argument("--dts", default="0.0005,0.001,0.002")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--sample-ps", type=float, default=2.0)   # matched TIME interval
    ap.add_argument("--device", default=None)
    ap.add_argument("--rng-seed", type=int, default=20260803)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (never GPUs 0-3)")
    dtype = torch.float64
    system, X0 = reference_minimum()
    P = extract_parameters(system)
    tff = TorchFF(P, device=dev, dtype=dtype)
    rows = []

    for dt in [float(s) for s in a.dts.split(",")]:
        x = torch.as_tensor(np.repeat(X0[None], a.walkers, 0), device=dev, dtype=dtype).contiguous()
        integ = BAOAB(P["masses"], dt, a.gamma, a.temperature,
                      lambda q: tff.forces(q), device=dev, dtype=dtype)
        gen = make_seed_streams(a.rng_seed, 1, dev)[0]        # SAME stream at every dt
        v = integ.maxwell((a.walkers, 22, 3), gen, dev, dtype)
        f = tff.forces(x)
        n_eq, n_pr = int(a.equil_ps / dt), int(a.ps / dt)
        # Sample at a matched physical interval so EVERY dt yields the same number of
        # samples.  Sampling every fixed number of STEPS gives N ~ 1/dt, and the
        # resulting unequal statistics dominate any TV comparison between ladders.
        sample_every = max(1, int(round(a.sample_ps / dt)))
        t0 = time.perf_counter()
        for s in range(n_eq):
            x, v, f = integ.step(x, v, f, gen)
        Tk, num, den, phis, psis = [], [], [], [], []
        for s in range(n_pr):
            x, v, f = integ.step(x, v, f, gen)
            if (s + 1) % sample_every == 0:
                Tk.append(float(integ.kinetic_temperature(v)))
                gs, lp = laplacian_and_gradsq(tff, x)
                num.append(float(gs.sum()))
                den.append(float(lp.sum()))
                phis.append(dihedral_t(x, PHI_ATOMS).cpu().numpy())
                psis.append(dihedral_t(x, PSI_ATOMS).cpu().numpy())
        T_conf = (np.sum(num) / np.sum(den)) / KB
        rows.append(dict(dt_fs=dt * 1000, T_kin=float(np.mean(Tk)), T_conf=float(T_conf),
                         n_samples=len(Tk) * a.walkers,
                         phi=np.concatenate(phis), psi=np.concatenate(psis),
                         wall=time.perf_counter() - t0))
        print(f"dt={dt*1000:.2f} fs : T_kin={np.mean(Tk):7.2f} K ({100*(np.mean(Tk)/a.temperature-1):+.2f}%)  "
              f"T_conf={T_conf:7.2f} K ({100*(T_conf/a.temperature-1):+.2f}%)  "
              f"[{rows[-1]['n_samples']} samples, {rows[-1]['wall']:.0f}s]", flush=True)

    # Configurational agreement across the dt ladder -- interpreted against the SAMPLING NOISE
    # FLOOR, without which the comparison is meaningless.  A TV between two finite histograms of
    # the SAME distribution is already ~0.05 at these sample sizes, so any threshold below that
    # can never be met no matter how correct the physics is.  The floor is measured empirically
    # by splitting each run's own samples in half.
    NB = 36
    cell = (2 * np.pi / NB) ** 2

    def hist(phi, psi):
        return np.histogram2d(phi, psi, bins=NB, range=[[-np.pi, np.pi]] * 2, density=True)[0]

    def tv(h0, h1):
        return 0.5 * np.abs(h0 - h1).sum() * cell

    def split_half_floor(r, n_rep=12, rng=np.random.default_rng(0)):
        n = len(r["phi"])
        out = []
        for _ in range(n_rep):
            idx = rng.permutation(n)
            a_, b_ = idx[: n // 2], idx[n // 2:]
            out.append(tv(hist(r["phi"][a_], r["psi"][a_]), hist(r["phi"][b_], r["psi"][b_])))
        return float(np.mean(out)), float(np.std(out))

    base = rows[0]
    hb = hist(base["phi"], base["psi"])
    print("\nconfigurational marginal agreement (phi,psi, 36x36), against the noise floor:")
    print(f"  {'dt':>8}  {'N':>7}  {'TV vs ref':>10}  {'noise floor':>12}  verdict")
    for r in rows:
        t = tv(hb, hist(r["phi"], r["psi"]))
        # floor for comparing two half-size samples; scale to full-size pair comparison
        f_mean, f_sd = split_half_floor(r)
        floor = f_mean / math.sqrt(2.0)          # half-vs-half -> full-vs-full
        ok = t <= floor + 3 * f_sd or r is base
        r["TV_vs_smallest_dt"] = float(t)
        r["TV_noise_floor"] = float(floor)
        r["TV_consistent_with_noise"] = bool(ok)
        print(f"  {r['dt_fs']:6.2f}fs  {len(r['phi']):7d}  {t:10.4f}  {floor:12.4f}  "
              f"{'consistent with noise' if ok else 'EXCEEDS noise floor'}")

    out = [{k: v for k, v in r.items() if k not in ("phi", "psi")} for r in rows]
    with open(os.path.join(a.out, "dt_bias.json"), "w") as fh:
        json.dump(dict(temperature=a.temperature, walkers=a.walkers, prod_ps=a.ps, rows=out),
                  fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(a.out, 'dt_bias.json')}")
    print("GATE: T_conf within 2% of target at the frozen dt, and TV(dt vs ref) consistent\n      with the measured sampling-noise floor (a fixed TV threshold below that floor is\n      unmeetable regardless of the physics, which is how the first version was wrong).")


if __name__ == "__main__":
    main()
