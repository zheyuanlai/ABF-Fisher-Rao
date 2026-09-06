#!/usr/bin/env python
"""Corrected hidden-gate reference p_ref(A_gate | xi sub-bin) for ethane/ZIF-8 at 300 K.

Defect in the stored reference (found 2026-09-06, docs/ZIF8_OT_REPAIR.md §Z1b): the CV is periodic
on the circle, so a guest at the PERIODIC IMAGE of the window (unwrapped xi ~ +-L) has phi ~ 0 and is
"in band", but gate_observables always measures the one indexed window, which is then EMPTY.  The
umbrella reference and the production runs draw pool frameworks with the guest anywhere in cage A
(xi in [-11.2, -1.7]); 54 % of them are nearer the image window, so the stored p_ref(A_gate | band)
is a 46/54 mixture of the held-guest gate (2.95 A) and the empty gate (2.80 A) -- mean 2.866, sd 0.094
predicted, 2.857-2.867 / 0.09 observed.

This builder runs the SAME umbrella protocol (spring kappa of the reference, free BAOAB) but
lattice-shifts each replica's guest so that it is pulled into the indexed window, and bins the gate by
the UNWRAPPED xi.  8 windows at the sub-bin centres of |xi| < 1 A, 128 replicas each.

    CUDA_VISIBLE_DEVICES=1 python -u scripts/zif8_build_gate_reference_v2.py
-> cache/zif8/gate_reference_v2_T300.npz, results/ot_repair_campaign/zif8/gate_reference_v2.json
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src"))
torch.use_deterministic_algorithms(False)
try:
    torch._inductor.config.deterministic = False
except Exception:
    pass
from alkanes import periodic as per
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs, js_divergence
pre = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json"))); dev = torch.device("cuda")
system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(pre)); sim = ZIF8SimConfig(**{k: v for k, v in pre["sampler"].items() if not k.startswith("_")})
ref = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz"), allow_pickle=True)
kappa = float(ref["kappa"]); L = float(ref["period"]); gate_edges = np.asarray(ref["gate_edges"]); xedges = np.asarray(ref["gate_xi_edges"])
quick = "--quick" in sys.argv
W, N = 8, 128; burn, n_steps, every = (6000, 18000, 50) if quick else (60000, 180000, 50)
centres = 0.5 * (xedges[1:] + xedges[:-1])
gen = torch.Generator(device=dev).manual_seed(20260910)
pool = torch.as_tensor(np.load(os.path.join(ROOT, "cache/zif8/init_pool_T300.npz"))["q"], device=dev, dtype=system.dtype); xi_pool = system.xi_value(pool)
B = W * N; c_xi = torch.as_tensor(np.repeat(centres, N), device=dev, dtype=system.dtype)
pick = torch.randint(0, pool.shape[0], (B,), generator=gen, device=dev); q = pool[pick].clone()
shift = torch.round((c_xi - xi_pool[pick]) / L) * L                      # bring the guest within L/2 of the INDEXED window
q[:, system.n_frame:] += (shift[:, None] * system.normal[None, :])[:, None, :]
assert float((system.xi_value(q) - c_xi).abs().max()) <= L / 2 + 1e-6
v = system.pin_frame_com(system.maxwell_velocities((B,), gen)); m = system.mass[None, :, None]
c1 = math.exp(-sim.gamma * sim.dt); c2 = math.sqrt(1 - c1 * c1); vsig = torch.sqrt(system.kT / system.mass)[None, :, None]
c_phi = (c_xi * system.k_phi).reshape(1, B)
def spring(qq):
    phi = system.cv_value(qq).reshape(1, B)
    return system.bias_cartesian(-kappa * per.circular_distance(phi, c_phi), 1, B)
F = system.forces(q); Fu = spring(q); t0 = time.time()
xis, gates, thetas = [], [], []
for k in range(n_steps):
    v = v + 0.5 * sim.dt * (F + Fu) / m; q = q + 0.5 * sim.dt * v
    v = system.pin_frame_com(c1 * v + c2 * vsig * torch.randn(q.shape, generator=gen, device=dev, dtype=system.dtype)); q = q + 0.5 * sim.dt * v
    F = system.forces(q); Fu = spring(q); v = v + 0.5 * sim.dt * (F + Fu) / m
    if k >= burn and k % every == 0:
        ag, th = system.gate_observables(q); xis.append(system.xi_value(q).cpu().numpy()); gates.append(ag.cpu().numpy()); thetas.append(th.cpu().numpy())
    if (k + 1) % 20000 == 0:
        print(f"  step {k + 1}/{n_steps} ({(time.time() - t0) / 60:.1f} min)", flush=True)
xi = np.stack(xis); A = np.stack(gates); th = np.stack(thetas)                 # (frames, B) UNWRAPPED xi
wrong = np.mean(np.abs(xi) > L / 2); band = np.abs(xi) < 1.0
h, _, _ = np.histogram2d(xi[band], A[band], bins=[xedges, gate_edges])
T2 = xi.shape[0] // 2
ha, _, _ = np.histogram2d(xi[:T2][band[:T2]], A[:T2][band[:T2]], bins=[xedges, gate_edges]); hb, _, _ = np.histogram2d(xi[T2:][band[T2:]], A[T2:][band[T2:]], bins=[xedges, gate_edges])
js = float(np.mean(js_divergence(ha, hb))); cg = 0.5 * (gate_edges[1:] + gate_edges[:-1])
means = [(float(np.sum(h[k] / h[k].sum() * cg)) if h[k].sum() else float("nan")) for k in range(W)]
old = np.asarray(ref["gate_hist_window_xi"]); old_means = [float(np.sum(old[k] / old[k].sum() * cg)) for k in range(W)]
tv_old = [0.5 * float(np.abs(h[k] / h[k].sum() - old[k] / old[k].sum()).sum()) for k in range(W)]
print(f"corrected gate reference: per-sub-bin <A_gate> {np.round(means, 3).tolist()} (stored reference {np.round(old_means, 3).tolist()}); TV new-vs-stored {np.round(tv_old, 3).tolist()}; "
      f"split-half JS {js:.2e}; frames at the image window {wrong:.4f}; in-band samples per sub-bin {h.sum(1).astype(int).tolist()}; theta band {th[band].mean():.2f}", flush=True)
out = os.path.join(ROOT, "cache/zif8/gate_reference_v2_T300.npz")
np.savez(out, gate_hist_window_xi=h, gate_xi_edges=xedges, gate_edges=gate_edges, gate_hist_split=np.stack([ha, hb]), split_half_js=js, means=np.asarray(means),
         stored_means=np.asarray(old_means), theta_band_mean=float(th[band].mean()), frames_at_image_window=wrong, protocol=json.dumps(dict(W=W, N=N, burn=burn, n_steps=n_steps, every=every, kappa=kappa, seed=20260910, quick=quick)))
json.dump(dict(means=means, stored_means=old_means, tv_new_vs_stored=tv_old, split_half_js=js, frames_at_image_window=wrong, samples=h.sum(1).tolist(), theta_band=float(th[band].mean()), wall_min=(time.time() - t0) / 60),
          open(os.path.join(ROOT, "results/ot_repair_campaign/zif8/gate_reference_v2.json"), "w"), indent=1)
print(f"wrote {out} ({(time.time() - t0) / 60:.1f} min)")
