#!/usr/bin/env python
"""Z1c: is the held-guest gate aperture (2.95 A) a slow transient?  Two 150 ps arms at the window,
256 replicas each, random pool frameworks, <A_gate> in the band per 5 ps block:
  --arm umbrella     the library's own run_umbrella (the reference protocol) at one window centred on xi 0
  --arm constrained  slow 20 ps pull + ConstrainedBAOAB at xi' ~ U[-0.25, 0.25)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src"))
arm = sys.argv[sys.argv.index("--arm") + 1]; n_ps = float(sys.argv[sys.argv.index("--ps") + 1]) if "--ps" in sys.argv else 150.0
OUT = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z1c"); os.makedirs(OUT, exist_ok=True)
torch.use_deterministic_algorithms(False)
try:
    torch._inductor.config.deterministic = False
except Exception:
    pass
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs, run_umbrella
from zif8.ot_repair_zif8 import ConstrainedBAOAB
pre = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json"))); dev = torch.device("cuda")
system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(pre)); sim = ZIF8SimConfig(**{k: v for k, v in pre["sampler"].items() if not k.startswith("_")})
ref = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz"), allow_pickle=True)
kappa = float(ref["kappa"]); L = float(ref["period"]); N = 256; n_steps = int(n_ps / sim.dt); block_ps = 5.0
pool_path = os.path.join(ROOT, "cache/zif8/init_pool_T300.npz"); t0 = time.time(); rows = []
if arm == "umbrella":
    phis, us, uhgs, gates, thetas = run_umbrella(system, sim, [0.0], kappa, n_steps, N, burn_in=0, sample_every=50, seed=20260908, init_pool=pool_path, verbose=True)
    xi = phis[:, 0, :] / system.k_phi; A = gates[:, 0, :]; th = thetas[:, 0, :]
    fpb = int(block_ps / (50 * sim.dt))
    for b in range(0, xi.shape[0], fpb):
        sl = slice(b, b + fpb); inb = np.abs(xi[sl]) < 0.25
        rows.append(dict(t_ps=(b + fpb) * 50 * sim.dt, A_all=float(A[sl].mean()), A_inband=float(A[sl][inb].mean()) if inb.any() else float("nan"), frac_inband=float(inb.mean()), theta=float(th[sl].mean()), xi_mean=float(xi[sl].mean())))
else:
    gen = torch.Generator(device=dev).manual_seed(20260909)
    pool = torch.as_tensor(np.load(pool_path)["q"], device=dev, dtype=system.dtype); xi_pool = system.xi_value(pool)
    tgt = -0.25 + 0.5 * torch.rand(N, generator=gen, device=dev, dtype=system.dtype)
    pick = torch.randint(0, pool.shape[0], (N,), generator=gen, device=dev); q = pool[pick].clone()
    shift = torch.round((tgt - xi_pool[pick]) / L) * L; q[:, system.n_frame:] += (shift[:, None] * system.normal[None, :])[:, None, :]
    v = system.pin_frame_com(system.maxwell_velocities((N,), gen)); dyn = ConstrainedBAOAB(system, sim, gen); xi0 = system.xi_value(q)
    n_pull = 40000; q, v, F = dyn.run(q, v, tgt, n_pull, xi_schedule=lambda k: xi0 + (tgt - xi0) * (k + 1) / n_pull)
    acc = dict(A=0.0, th=0.0, n=0); blk = int(block_ps / sim.dt)
    def rec(k, qq, vv, FF):
        if k % 5 == 0:
            ag, t_ = system.gate_observables(qq); acc["A"] += float(ag.mean()); acc["th"] += float(t_.mean()); acc["n"] += 1
        if (k + 1) % blk == 0:
            rows.append(dict(t_ps=(k + 1) * sim.dt, A_all=acc["A"] / acc["n"], A_inband=acc["A"] / acc["n"], frac_inband=1.0, theta=acc["th"] / acc["n"])); acc.update(A=0.0, th=0.0, n=0)
            print(f"  t {rows[-1]['t_ps']:.0f} ps  A {rows[-1]['A_inband']:.4f}", flush=True)
    dyn.run(q, v, tgt, n_steps, F=F, record=rec)
json.dump(dict(arm=arm, n_ps=n_ps, N=N, rows=rows, wall_s=time.time() - t0), open(os.path.join(OUT, f"z1c_{arm}.json"), "w"), indent=1)
print(f"[{arm}] " + " ".join(f"t{r['t_ps']:.0f}:{r['A_inband']:.3f}" for r in rows[::3]) + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)
