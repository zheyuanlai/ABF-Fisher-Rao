#!/usr/bin/env python
"""Z1b: why does the fixed-xi constrained ensemble show a wider gate (<A_gate> 2.95 A) than the
umbrella reference (2.86 A) at the same xi?  Four preparations at the window-plane site
[-0.25, 0.25), 256 replicas each, <A_gate>, theta, framework kinetic temperature and <f_xi> in
2.5 ps blocks:

  A  fast pull (2 ps, as Z1) -> constrained BAOAB 60 ps            (does the shifted state relax?)
  B  slow pull (20 ps)       -> constrained BAOAB 40 ps            (is the shift a pull artefact?)
  C  umbrella spring at xi = 0 (the reference's own kappa), free BAOAB from the pool, 60 ps
                                                                    (does the reference protocol give 2.86 here?)
  D  constraint applied to C's final states at their own xi, 20 ps  (does the constraint itself move the gate?)

    CUDA_VISIBLE_DEVICES=1 python -u scripts/zif8_ot_z1b_gate_consistency.py
-> results/ot_repair_campaign/zif8/Z1b/{z1b.json, figures/z1b_gate_consistency.png}
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
REF = os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz")
POOL = os.path.join(ROOT, "cache/zif8/init_pool_T300.npz")
OUT = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z1b")
N = 256; BLOCK = 5000; GATE_STRIDE = 5
KB = 0.008314462618


def main():
    quick = "--quick" in sys.argv
    torch.use_deterministic_algorithms(False)
    try:
        torch._inductor.config.deterministic = False
    except Exception:
        pass
    from alkanes import periodic as per
    from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs
    from zif8.ot_repair_zif8 import ConstrainedBAOAB, local_mean_force_xi, reference_mean_force
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pre = json.load(open(PREREG)); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(pre))
    sim = ZIF8SimConfig(**{k: v for k, v in pre["sampler"].items() if not k.startswith("_")})
    ref = np.load(REF, allow_pickle=True); kappa = float(ref["kappa"]); L = float(ref["period"])
    _, _, Fp_interp = reference_mean_force(ref)
    gate_edges = np.asarray(ref["gate_edges"]); gh = np.asarray(ref["gate_hist_window_xi"])
    pr = gh[3] + gh[4]; pr = pr / pr.sum(); cg = 0.5 * (gate_edges[1:] + gate_edges[:-1]); A_ref = float(np.sum(pr * cg))
    gen = torch.Generator(device=dev).manual_seed(20260907)
    pool = torch.as_tensor(np.load(POOL)["q"], device=dev, dtype=system.dtype); xi_pool = system.xi_value(pool)
    nf = system.n_frame; mf = system.mass[:nf]; dof = 3 * nf - 3
    sc = 0.2 if quick else 1.0
    n_pull_fast, n_pull_slow = int(4000 * sc), int(40000 * sc)
    n_A, n_B, n_C, n_D = int(120000 * sc), int(80000 * sc), int(120000 * sc), int(40000 * sc)
    block = int(BLOCK * sc)
    dyn = ConstrainedBAOAB(system, sim, gen)
    results = {}

    random_pool = "--random-pool" in sys.argv                 # the reference's own draw: random pool configurations
    arms = [a for a in sys.argv if a.startswith("--arms=")]; arms = arms[0].split("=")[1].split(",") if arms else ["A", "B", "C", "D"]
    OUT_ = OUT + ("_randompool" if random_pool else "")

    def draw(n):
        tgt = -0.25 + 0.5 * torch.rand(n, generator=gen, device=dev, dtype=system.dtype)
        if random_pool:
            pick = torch.randint(0, pool.shape[0], (n,), generator=gen, device=dev)
        else:
            d = xi_pool[None, :] - tgt[:, None]; d = d - L * torch.round(d / L)
            order = torch.argsort(d.abs(), dim=1)[:, :8]
            pick = order[torch.arange(n, device=dev), torch.randint(0, 8, (n,), generator=gen, device=dev)]
        q = pool[pick].clone(); shift = torch.round((tgt - xi_pool[pick]) / L) * L
        q[:, nf:] += (shift[:, None] * system.normal[None, :])[:, None, :]
        v = system.pin_frame_com(system.maxwell_velocities((n,), gen))
        return q, v, tgt

    def blocks_constrained(name, q, v, xi_fixed, n_steps, F=None, t_offset=0.0):
        rows = []; acc = dict(A=0.0, th=0.0, T=0.0, f=0.0, ng=0, nfc=0)
        Fp = torch.as_tensor(Fp_interp(xi_fixed.cpu().numpy()), device=dev, dtype=system.dtype)

        def rec(k, qq, vv, FF):
            acc["f"] += float((local_mean_force_xi(system, qq, FF) - Fp).mean()); acc["nfc"] += 1
            if k % GATE_STRIDE == 0:
                ag, th = system.gate_observables(qq); acc["A"] += float(ag.mean()); acc["th"] += float(th.mean())
                ke = (0.5 * mf[None, :, None] * vv[:, :nf] ** 2).sum(dim=(1, 2)); acc["T"] += float((2 * ke / (dof * KB)).mean()); acc["ng"] += 1
            if (k + 1) % block == 0:
                rows.append(dict(t_ps=t_offset + (k + 1) * sim.dt, A=acc["A"] / acc["ng"], theta=acc["th"] / acc["ng"], T_frame=acc["T"] / acc["ng"], b=acc["f"] / acc["nfc"]))
                for kk in acc:
                    acc[kk] = 0.0 if kk not in ("ng", "nfc") else 0
        q, v, F = dyn.run(q, v, xi_fixed, n_steps, F=F, record=rec)
        print(f"  [{name}] " + " ".join(f"t{r['t_ps']:.1f}:A{r['A']:.3f}" for r in rows[::max(len(rows) // 6, 1)]) + f"  T_frame {rows[-1]['T_frame']:.1f} K", flush=True)
        return q, v, F, rows

    t0 = time.time()
    # ---- A: fast pull + long constrained ----
    if "A" in arms:
        q, v, tgt = draw(N); xi0 = system.xi_value(q)
        q, v, F = dyn.run(q, v, tgt, n_pull_fast, xi_schedule=lambda k: xi0 + (tgt - xi0) * (k + 1) / n_pull_fast)
        _, _, _, rows = blocks_constrained("A fast-pull+constrained", q, v, tgt, n_A, F=F)
        results["A_fastpull_constrained"] = rows; print(f"A done {time.time() - t0:.0f}s", flush=True)
    # ---- B: slow pull + constrained ----
    if "B" in arms:
        q, v, tgt = draw(N); xi0 = system.xi_value(q)
        results["B_pull_distance_A"] = dict(mean=float((tgt - xi0).abs().mean()), max=float((tgt - xi0).abs().max()))
        q, v, F = dyn.run(q, v, tgt, n_pull_slow, xi_schedule=lambda k: xi0 + (tgt - xi0) * (k + 1) / n_pull_slow)
        _, _, _, rows = blocks_constrained("B slow-pull+constrained", q, v, tgt, n_B, F=F)
        results["B_slowpull_constrained"] = rows; print(f"B done {time.time() - t0:.0f}s", flush=True)
    # ---- C: umbrella spring (reference protocol), free BAOAB ----
    q, v, _ = draw(N)                                        # pool configs (guest in the cage), lattice-shifted near xi 0
    m = system.mass[None, :, None]; c1 = math.exp(-sim.gamma * sim.dt); c2 = math.sqrt(1 - c1 * c1); vsig = torch.sqrt(system.kT / system.mass)[None, :, None]
    zero = torch.zeros(N, device=dev, dtype=system.dtype)

    def spring(qq):
        phi = system.cv_value(qq).reshape(1, N)
        return system.bias_cartesian(-kappa * per.circular_distance(phi, zero.reshape(1, N)), 1, N), phi.reshape(N)
    F = system.forces(q); Fu, phi = spring(q)
    rows = []; acc = dict(A=0.0, th=0.0, T=0.0, ng=0, Ab=0.0, nb=0, inband=0.0)
    for k in range(n_C):
        v = v + (0.5 * sim.dt) * (F + Fu) / m; q = q + (0.5 * sim.dt) * v
        noise = torch.randn(q.shape, generator=gen, device=dev, dtype=system.dtype)
        v = system.pin_frame_com(c1 * v + c2 * vsig * noise); q = q + (0.5 * sim.dt) * v
        F = system.forces(q); Fu, phi = spring(q); v = v + (0.5 * sim.dt) * (F + Fu) / m
        if k % GATE_STRIDE == 0:
            ag, th = system.gate_observables(q); xi = system.xi_value(q); inb = xi.abs() < 0.25
            acc["A"] += float(ag.mean()); acc["th"] += float(th.mean()); acc["ng"] += 1
            ke = (0.5 * mf[None, :, None] * v[:, :nf] ** 2).sum(dim=(1, 2)); acc["T"] += float((2 * ke / (dof * KB)).mean())
            if int(inb.sum()) > 0:
                acc["Ab"] += float(ag[inb].sum()); acc["nb"] += int(inb.sum())
            acc["inband"] += float(inb.to(torch.float64).mean())
        if (k + 1) % block == 0:
            rows.append(dict(t_ps=(k + 1) * sim.dt, A=acc["A"] / acc["ng"], A_inband=(acc["Ab"] / acc["nb"] if acc["nb"] else float("nan")), frac_inband=acc["inband"] / acc["ng"],
                             theta=acc["th"] / acc["ng"], T_frame=acc["T"] / acc["ng"]))
            acc = dict(A=0.0, th=0.0, T=0.0, ng=0, Ab=0.0, nb=0, inband=0.0)
    print("  [C umbrella spring] " + " ".join(f"t{r['t_ps']:.1f}:A{r['A']:.3f}(band {r['A_inband']:.3f},{100 * r['frac_inband']:.0f}%)" for r in rows[::max(len(rows) // 6, 1)]) + f"  T_frame {rows[-1]['T_frame']:.1f} K", flush=True)
    results["C_umbrella_spring"] = rows; print(f"C done {time.time() - t0:.0f}s", flush=True)
    # ---- D: constrain C's final states at their own xi ----
    xi_now = system.xi_value(q)
    _, _, _, rows = blocks_constrained("D constrained-from-umbrella", q, v, xi_now, n_D, F=F, t_offset=float(n_C * sim.dt))
    results["D_constrained_from_umbrella"] = rows
    results["D_xi_stats"] = dict(mean=float(xi_now.mean()), sd=float(xi_now.std()), frac_inband=float((xi_now.abs() < 0.25).to(torch.float64).mean()))
    print(f"D done {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(OUT_, "figures"), exist_ok=True)
    meta = dict(random_pool=random_pool, arms=arms, N=N, block_steps=block, n_pull_fast=n_pull_fast, n_pull_slow=n_pull_slow, n_A=n_A, n_B=n_B, n_C=n_C, n_D=n_D, kappa=kappa, A_ref_window=A_ref,
                A_empty_framework=2.7957, quick=quick, wall_s=time.time() - t0)
    json.dump(dict(meta=meta, results=results), open(os.path.join(OUT_, "z1b.json"), "w"), indent=1)
    try:
        os.environ.setdefault("MPLBACKEND", "Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
        for name, rows in results.items():
            if not isinstance(rows, list):
                continue
            t = [r["t_ps"] for r in rows]
            axes[0].plot(t, [r["A"] for r in rows], "-o", ms=3, label=name)
            axes[1].plot(t, [r["theta"] for r in rows], "-o", ms=3, label=name)
            axes[2].plot(t, [r["T_frame"] for r in rows], "-o", ms=3, label=name)
        if "C_umbrella_spring" in results:
            rows = results["C_umbrella_spring"]; axes[0].plot([r["t_ps"] for r in rows], [r["A_inband"] for r in rows], "k:", lw=1, label="C, frames with |xi|<0.25")
        axes[0].axhline(A_ref, color="gray", ls="--", lw=1, label=f"umbrella reference {A_ref:.3f}"); axes[0].axhline(2.7957, color="gray", ls=":", lw=1, label="empty framework 2.796")
        axes[0].set_ylabel("<A_gate> (A)"); axes[1].set_ylabel("<theta_gate> (deg)"); axes[2].set_ylabel("framework T_kin (K)"); axes[2].axhline(300, color="gray", lw=0.8)
        for ax in axes:
            ax.set_xlabel("t (ps)")
        axes[0].legend(fontsize=6, frameon=False)
        fig.suptitle("Z1b: gate aperture at the window plane under four preparations", fontsize=9.5)
        fig.savefig(os.path.join(OUT_, "figures", "z1b_gate_consistency.png"), dpi=160)
    except Exception as exc:
        print("plot failed", exc)
    print(f"wrote {OUT_} ({meta['wall_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
