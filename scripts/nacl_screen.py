"""Stage III -- the ABF-only fixed-compute regime map (SPEC_nacl_water.md §7).

**No Fisher-Rao anywhere in this stage**, and no gate verdict is computed here: `T_hit`,
`T_est`, the bias-aware targets and every classification are derived afterwards from the saved
traces by ``nacl_gates.py``.

Why the cells run PACKED
------------------------
The frozen map is ``N*T = 100 ns`` per ensemble over ``N in {8,16,32,64}``, 8 seeds each --
3200 ns of MD whose *physics* is fixed but whose *scheduling* is not.  Run cell-by-cell, the
``N = 8`` cell is 64 concurrent trajectories on an H200, which is a fraction of the device.
Run packed, every cell's ensembles are steps of ONE batch:

    t <= 1.5625 ns   960 walkers   (8+16+32+64) x 8 seeds
    t <= 3.125  ns   448 walkers   (N=64 retired)
    t <= 6.25   ns   192 walkers   (N=32 retired)
    t <= 12.5   ns    64 walkers   (N=16 retired; N=8 alone)

Identical physics, identical seeds, identical per-ensemble estimators -- each ensemble keeps
its own ABF bias and never sees another's samples -- and the whole map lands in one process,
which the WCA cross-process determinism finding requires of a comparable block anyway.
``--cells`` still allows a single cell for a partial map.

Initial conditions (SPEC §7): every walker starts in the CIP basin -- the published
``equilibrate.coor`` sits at r = 3.0 A -- with an independently equilibrated solvent shell
(>= 50 ps restrained, outside the ABF budget), refused if cloned.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_screen.py --out results/nacl/screen
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alkanes import interval as iv                               # noqa: E402
from methane.cv import PeriodicDistanceCV                        # noqa: E402
from methane.dynamics import (BAOAB, CompositeConstraints, PairConstraint,   # noqa: E402
                              RigidWaterConstraints)
from nacl import system as nsys                                  # noqa: E402
from nacl.core import (NaClSimConfig, assert_distinct_solvent, colvars_trust,  # noqa: E402
                       masked_bin_sum, wall_force)
from nacl.nonbonded import NaClNonbonded                         # noqa: E402
from nacl.observables import HydrationDescriptors                # noqa: E402

SEEDS = list(range(4000, 4008))          #: frozen by §5 of the preregistration
B_MD_NS = 100.0                          #: the published ABF budget, per ensemble
CELLS = (8, 16, 32, 64)
R_CIP_NM = 0.30                          #: published equilibrate.coor separation


def build_population(ff, L, start, n_total, r_hold, prep_ps, dt, chunk, seed, dev):
    """``n_total`` walkers from ``start``, each propagated under independent noise at fixed r."""
    x = torch.tensor(np.repeat(start[None], n_total, axis=0), device=dev, dtype=torch.float32)
    cons = CompositeConstraints(
        [RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                               ff.params["mass"], device=dev, dtype=torch.float32),
         PairConstraint(0, 1, np.full(n_total, r_hold), ff.params["mass"],
                        device=dev, dtype=torch.float32)],
        atom_sets=[ff.params["waters"], [0, 1]])
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=chunk), ff.params["mass"], cons,
                  dt, nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cons.apply_positions(x, x.clone())
    v = integ.maxwell_velocities(x, generator=gen)
    _, f = ff.energy_forces(x, chunk=chunk)
    for _ in range(int(round(prep_ps / dt))):
        _, f = integ.step(x, v, f, generator=gen)
    return x.cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nacl/screen")
    ap.add_argument("--baths", default="results/nacl/baths")
    ap.add_argument("--cells", default=",".join(str(c) for c in CELLS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--prep-ps", type=float, default=50.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--triton", action="store_true", help="fused pair kernel (gated)")
    ap.add_argument("--save-every-ps", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cells = [int(c) for c in args.cells.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    S = len(seeds)

    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    L = float(box["L_nm"])
    R_hi = float(box["finite_size_gate"]["R_hi_nm"])
    gate = json.load(open(nsys.REPO / "results/nacl/stage1/dynamics_gate.json"))
    dt = float(gate["dt_chosen_ps"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bz = np.load(os.path.join(args.baths, "baths.npz"))
    key = f"start_b0_f0_r{R_CIP_NM:.4f}"
    if key not in bz:
        raise SystemExit(f"no minimised CIP start {key}; run scripts/nacl_baths.py --per-r")
    start = bz[key].astype(np.float64)

    ff = NaClNonbonded(L, device=dev, dtype=torch.float32)
    if args.triton:
        ff.enable_triton()
    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
    print(f"[engine] torch {dev} float32 compiled{' +triton' if args.triton else ''}; "
          f"L = {L:.6f} nm, dt = {dt} ps, domain [{nsys.R_LO_NM}, {R_hi}] nm", flush=True)

    # ---- the packed plan -------------------------------------------------------------------
    plan = []
    for N in cells:
        T_ns = B_MD_NS / N
        plan.append(dict(N=N, T_ns=T_ns, n_steps=int(round(T_ns * 1000.0 / dt))))
    plan.sort(key=lambda c: c["n_steps"])          # shortest first: retire early, shrink batch
    total_walkers = sum(S * c["N"] for c in plan)
    total_ns = sum(S * c["N"] * c["T_ns"] for c in plan)
    for c in plan:
        print(f"[plan] N = {c['N']:3d}  T = {c['T_ns']:8.4f} ns  {c['n_steps']:8d} steps  "
              f"x {S} seeds", flush=True)
    print(f"[plan] {total_walkers} walkers packed, {total_ns:.0f} ns aggregate MD", flush=True)

    # ---- populations: one independent solvent per walker, all cells at once ----------------
    t_start = time.time()
    pop_path = os.path.join(args.out, "populations.npz")
    if os.path.exists(pop_path):
        pops = dict(np.load(pop_path))
        print(f"[pop] cached {pop_path}", flush=True)
    else:
        print(f"[pop] building {total_walkers} independent CIP-basin walkers "
              f"({args.prep_ps} ps each) ...", flush=True)
        t0 = time.time()
        allpop = build_population(ff, L, start, total_walkers, R_CIP_NM, args.prep_ps, dt,
                                  args.chunk, 740000, dev)
        spread = assert_distinct_solvent(allpop)
        pops, off = {}, 0
        for c in plan:
            n = S * c["N"]
            pops[f"N{c['N']}"] = allpop[off:off + n].astype(np.float32)
            off += n
        np.savez_compressed(pop_path, **pops)
        print(f"[pop] built in {(time.time()-t0)/60:.1f} min; min solvent deviation "
              f"{spread:.3e} nm -> {pop_path}", flush=True)

    # ---- per-cell state --------------------------------------------------------------------
    beta = nsys.beta_per_kJ()
    grid, dz = iv.interval_grid(nsys.N_GRID, nsys.R_LO_NM, R_hi, device=dev, dtype=torch.float32)
    sim0 = NaClSimConfig(n_ensembles=S, box_nm=L, dt=dt, R_hi=R_hi, wall_hi=R_hi,
                         save_every=int(round(args.save_every_ps / dt)))
    K_abf = iv.gaussian_kernel_matrix(grid, sim0.abf_bandwidth)
    K_kde = iv.reflected_kernel_matrix(grid, sim0.kde_bandwidth, nsys.R_LO_NM, R_hi)
    cv = PeriodicDistanceCV(0, 1, L)
    hyd = HydrationDescriptors(ff.params["waters"], L, device=dev)
    cons = RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                                 ff.params["mass"], device=dev, dtype=torch.float32)
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=args.chunk), ff.params["mass"], cons,
                  dt, nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(740001)

    st = []
    for c in plan:
        n = S * c["N"]
        x = torch.tensor(pops[f"N{c['N']}"], device=dev, dtype=torch.float32)
        st.append(dict(cell=c, n=n,
                       x=x,
                       fsum=torch.zeros(S, nsys.N_GRID, device=dev, dtype=torch.float32),
                       csum=torch.zeros(S, nsys.N_GRID, device=dev, dtype=torch.float32),
                       diag={k: [] for k in ("steps", "times", "mean_force", "pmf", "p_hat",
                                             "eff_counts", "occupancy", "out_of_domain")},
                       xi_trace=[], xi_steps=[], y_trace=[], y_steps=[], done=False))

    q = torch.cat([s["x"] for s in st], dim=0)
    for s in st:
        del s["x"]
    bounds, off = [], 0
    for s in st:
        bounds.append((off, off + s["n"]))
        off += s["n"]
    v = integ.maxwell_velocities(q, generator=gen)
    _, f = ff.energy_forces(q, chunk=args.chunk)

    max_steps = max(c["n_steps"] for c in plan)
    xi_every = max(1, int(round(0.5 / dt)))
    y_every = max(1, int(round(1.0 / dt)))
    t0 = time.time()
    print(f"\n[run] {max_steps} steps max, packed batch {q.shape[0]}", flush=True)

    for step in range(max_steps + 1):
        f_loc, r_flat, _ = cv.local_mean_force(q, f, beta)
        f_loc = torch.clamp(f_loc, -8.0 * sim0.abf_force_clip, 8.0 * sim0.abf_force_clip)

        bias_parts = []
        for (lo, hi), s in zip(bounds, st):
            c = s["cell"]
            r = r_flat[lo:hi].view(S, c["N"])
            if step <= c["n_steps"]:
                in_dom = ((r >= nsys.R_LO_NM) & (r <= R_hi)).to(torch.float32)
                s["fsum"] += masked_bin_sum(r, f_loc[lo:hi].view(S, c["N"]), in_dom,
                                            nsys.N_GRID, nsys.R_LO_NM, R_hi)
                s["csum"] += masked_bin_sum(r, torch.ones_like(r), in_dom,
                                            nsys.N_GRID, nsys.R_LO_NM, R_hi)
            mf = iv.mean_force_profile(s["fsum"], s["csum"], K_abf)
            eff = iv.effective_counts(s["csum"], K_abf)
            s["mf"], s["eff"] = mf, eff
            bias_parts.append(mf * colvars_trust(eff, sim0.full_samples))

            if step <= c["n_steps"]:
                if step % xi_every == 0:
                    s["xi_trace"].append(r.to(torch.float32).cpu().numpy())
                    s["xi_steps"].append(step)
                if step % y_every == 0:
                    s["y_trace"].append(hyd.Y(q[lo:hi]).to(torch.float32).cpu().numpy())
                    s["y_steps"].append(step)
                if step % sim0.save_every == 0 or step == c["n_steps"]:
                    A = iv.free_energy_from_mean_force(bias_parts[-1], grid, dz)
                    p = iv.kde_marginal(r, K_kde, nsys.N_GRID, dz, nsys.R_LO_NM, R_hi)
                    s["diag"]["steps"].append(step)
                    s["diag"]["times"].append(step * dt)
                    s["diag"]["mean_force"].append(mf.cpu().numpy())
                    s["diag"]["pmf"].append(A.cpu().numpy())
                    s["diag"]["p_hat"].append(p.cpu().numpy())
                    s["diag"]["eff_counts"].append(eff.cpu().numpy())
                    # masked, not clamped: a walker outside the domain is not evidence for the
                    # edge bin, and Gate C reads basin occupancy off this array
                    s["diag"]["occupancy"].append(
                        masked_bin_sum(r, torch.ones_like(r), in_dom,
                                       nsys.N_GRID, nsys.R_LO_NM, R_hi).cpu().numpy())
                    s["diag"]["out_of_domain"].append(
                        float((1.0 - in_dom).mean()))

        if step % sim0.save_every == 0:
            if not bool((r_flat < 0.995 * 0.5 * L).all()):
                raise RuntimeError(f"an ion pair reached 99.5% of L/2 "
                                   f"(r_max = {float(r_flat.max()):.4f} nm); xi degenerate")
            if step % (20 * sim0.save_every) == 0:
                el = time.time() - t0
                ns_done = sum(min(step, s['cell']['n_steps']) * dt * 1e-3 * s["n"] for s in st)
                print(f"  step {step:8d}/{max_steps}  t = {step*dt:9.2f} ps  "
                      f"T = {float(integ.temperature(v).mean()):6.1f} K  "
                      f"batch {q.shape[0]:4d}  {ns_done:7.1f} ns  "
                      f"({ns_done/max(el,1)*86400:7.0f} ns/day)  ({el:7.0f}s)", flush=True)

        # ---- retire finished cells: write, then shrink the batch ---------------------------
        retire = [i for i, s in enumerate(st)
                  if (not s["done"]) and step >= s["cell"]["n_steps"]]
        for i in retire:
            s = st[i]
            lo, hi = bounds[i]
            c = s["cell"]
            A = iv.free_energy_from_mean_force(
                s["mf"] * colvars_trust(s["eff"], sim0.full_samples), grid, dz)
            g64 = grid.to(torch.float64)
            from methane.cv import W_from_F, Wprime_from_Fprime
            out = dict(N=c["N"], T_ns=c["T_ns"], n_steps=c["n_steps"], dt_ps=dt,
                       seed_labels=np.asarray(seeds), box_L_nm=L, R_hi_nm=R_hi,
                       grid=grid.cpu().numpy(), dz=dz,
                       mean_force=s["mf"].cpu().numpy(), pmf=A.cpu().numpy(),
                       W_pmf=W_from_F(A.to(torch.float64), g64, beta).cpu().numpy(),
                       W_mean_force=Wprime_from_Fprime(s["mf"].to(torch.float64), g64, beta)
                       .cpu().numpy(),
                       eff_counts=s["eff"].cpu().numpy(),
                       xi_trace=np.asarray(s["xi_trace"]), xi_steps=np.asarray(s["xi_steps"]),
                       y_trace=np.asarray(s["y_trace"]), y_steps=np.asarray(s["y_steps"]),
                       final_positions=q[lo:hi].cpu().numpy().astype(np.float32))
            for k, val in s["diag"].items():
                out[f"diag_{k}"] = np.asarray(val)
            path = os.path.join(args.out, f"cell_N{c['N']}.npz")
            np.savez_compressed(path, **out)
            s["done"] = True
            print(f"[retire] N = {c['N']} at step {step} ({step*dt/1000:.4f} ns) -> {path}",
                  flush=True)
        if retire:
            survivors = [i for i, s in enumerate(st) if not s["done"]]
            if not survivors:
                break
            keep = torch.cat([torch.arange(bounds[i][0], bounds[i][1], device=dev)
                              for i in survivors])
            q = q[keep].contiguous(); v = v[keep].contiguous(); f = f[keep].contiguous()
            # the per-cell estimators are (S, n_grid) and are untouched by the batch shrink;
            # only the walker-axis bounds change, so the biases carry over by re-slicing
            bias_parts = [bias_parts[i] for i in survivors]
            st = [st[i] for i in survivors]
            bounds, off = [], 0
            for s in st:
                bounds.append((off, off + s["n"]))
                off += s["n"]
            torch.cuda.empty_cache()
            print(f"[batch] shrunk to {q.shape[0]} walkers", flush=True)

        def _bias_at(q_new, _parts=bias_parts, _bounds=list(bounds), _st=list(st)):
            r_new, grad_new, _ = cv.geometry(q_new)
            g = torch.empty_like(r_new)
            for (lo, hi), s, prof in zip(_bounds, _st, _parts):
                rr = r_new[lo:hi].view(S, s["cell"]["N"])
                mf_new = iv.interval_interp(prof, grid, rr).clamp(-sim0.abf_force_clip,
                                                                 sim0.abf_force_clip)
                g[lo:hi] = mf_new.reshape(-1)
            return cv.bias_force(grad_new, g + wall_force(r_new, sim0))

        _, f = integ.step(q, v, f, bias_fn=_bias_at, generator=gen)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="nacl_abf_screen", spec="docs/SPEC_nacl_water.md §7",
                       amendment="V2_PREREGISTRATION.md Amendment 14",
                       cells=cells, seeds=seeds, B_MD_ns=B_MD_NS, dt_ps=dt,
                       prep_ps=args.prep_ps, r_cip_nm=R_CIP_NM, box_L_nm=L, R_hi_nm=R_hi,
                       packed=True, triton=bool(args.triton),
                       aggregate_ns=total_ns,
                       gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                       wall_hours=(time.time() - t_start) / 3600.0,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True,
                                                 text=True).stdout.strip()), fh, indent=2)
    print(f"\n[done] {(time.time()-t_start)/3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
