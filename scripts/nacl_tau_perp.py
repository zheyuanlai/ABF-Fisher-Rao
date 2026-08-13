"""tau_perp at fixed r: the family-TV estimate and the clone-twin estimate (SPEC §6).

Both are computed and **both are reported**; a disagreement beyond 2x is an open finding, not
something to resolve by picking the friendlier number (Amendment 10 / methane §5.2).

  family estimate   tau_perp(r_k) = inf{ t : max_{a,b} TV[p_t(Y|r_k,a), p_t(Y|r_k,b)] <= 0.2 }
  clone estimate    tau_clone(r_k) = inf{ t : Corr(Y_i(t), Y_j(t)) <= 1/e }  over duplicated
                    pairs propagated under independent noise

`r_k` comes from the ACCEPTED reference: the CIP minimum, the barrier, the SSIP minimum and one
outer point.  The hydration descriptor `Y = (n_NaO, n_ClH, n_bridge)` is the frozen one.

Gate D consumes `tau_perp = max_k tau_perp(r_k)` as the ceiling `lambda_rep <= 0.1 / tau_perp`.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_tau_perp.py --out results/nacl/tau_perp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane.dynamics import (BAOAB, CompositeConstraints, PairConstraint,   # noqa: E402
                              RigidWaterConstraints)
from nacl import system as nsys                                  # noqa: E402
from nacl.nonbonded import NaClNonbonded                         # noqa: E402
from nacl.observables import HydrationDescriptors                # noqa: E402

TV_THRESHOLD = 0.2
N_REPLICAS = 48          #: per family per r
N_TWINS = 48             #: duplicated pairs per r
EQUIL_PS = 50.0
TRACK_PS = 200.0
SAMPLE_PS = 1.0
N_BINS = 24


def tv(a, b, edges):
    ha = np.histogram(a, bins=edges)[0].astype(float)
    hb = np.histogram(b, bins=edges)[0].astype(float)
    ha /= max(ha.sum(), 1); hb /= max(hb.sum(), 1)
    return 0.5 * float(np.abs(ha - hb).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nacl/tau_perp")
    ap.add_argument("--baths", default="results/nacl/baths")
    ap.add_argument("--ref", default="results/nacl/reference")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--triton", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    L = float(box["L_nm"])
    gate = json.load(open(nsys.REPO / "results/nacl/stage1/dynamics_gate.json"))
    dt = float(gate["dt_chosen_ps"])
    rep = json.load(open(os.path.join(args.ref, "reference_report.json")))
    basins = rep["basins"]

    r_points = []
    for b in basins:
        if b.get("r_min_nm"):
            r_points.append(round(float(b["r_min_nm"]), 4))
    for b in basins[:-1]:
        r_points.append(round(float(b["r_hi_nm"]), 4))          # basin bound == barrier
    r_points = sorted(set(r_points))[:4]
    print(f"[plan] r points from the accepted reference: {r_points}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bz = np.load(os.path.join(args.baths, "baths.npz"))
    ff = NaClNonbonded(L, device=dev, dtype=torch.float32)
    if args.triton:
        ff.enable_triton()
    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
    hyd = HydrationDescriptors(ff.params["waters"], L, device=dev)

    # ---- assemble: 3 families x N_REPLICAS + N_TWINS pairs, per r --------------------------
    fams = (0, 1, 2)
    recs, starts = [], []
    for r_nm in r_points:
        for fam in fams:
            x0 = bz[f"start_b0_f{fam}_r{r_nm:.4f}"].astype(np.float64)
            for k in range(N_REPLICAS):
                recs.append((r_nm, fam, k, 0))
                starts.append(x0)
        # twins: pairs sharing a start (family 2 = dissociated-derived, the neutral one)
        x0 = bz[f"start_b0_f2_r{r_nm:.4f}"].astype(np.float64)
        for k in range(N_TWINS):
            for tw in (0, 1):
                recs.append((r_nm, -1, k, tw))
                starts.append(x0)
    recs = np.asarray(recs)
    x = torch.tensor(np.asarray(starts, dtype=np.float32), device=dev)
    print(f"[plan] {len(recs)} trajectories", flush=True)

    cons = CompositeConstraints(
        [RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                               ff.params["mass"], device=dev, dtype=torch.float32),
         PairConstraint(0, 1, recs[:, 0].astype(np.float64), ff.params["mass"],
                        device=dev, dtype=torch.float32)],
        atom_sets=[ff.params["waters"], [0, 1]])
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=args.chunk), ff.params["mass"], cons,
                  dt, nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=dev, dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(760001)
    cons.apply_positions(x, x.clone())
    v = integ.maxwell_velocities(x, generator=gen)
    _, f = ff.energy_forces(x, chunk=args.chunk)

    t0 = time.time()
    for _ in range(int(EQUIL_PS / dt)):
        _, f = integ.step(x, v, f, generator=gen)
    print(f"[equil] {EQUIL_PS} ps in {(time.time()-t0)/60:.1f} min", flush=True)

    n_track = int(TRACK_PS / dt)
    every = int(SAMPLE_PS / dt)
    times, Ys = [], []
    for step in range(n_track + 1):
        if step % every == 0:
            Ys.append(hyd.Y(x).cpu().numpy())
            times.append(step * dt)
        if step < n_track:
            _, f = integ.step(x, v, f, generator=gen)
    Ys = np.asarray(Ys)                                   # (T, n_traj, 3)
    times = np.asarray(times)
    np.savez_compressed(os.path.join(args.out, "traces.npz"),
                        recs=recs, times_ps=times, Y=Ys.astype(np.float32),
                        r_points=np.asarray(r_points))

    # ---- family TV -------------------------------------------------------------------------
    results = {}
    for r_nm in r_points:
        tau_fam = {}
        for comp, name in enumerate(("n_NaO", "n_ClH", "n_bridge")):
            vals = Ys[:, :, comp]
            lo, hi = float(vals.min()), float(vals.max()) + 1e-9
            edges = np.linspace(lo, hi, N_BINS + 1)
            tv_t = []
            for ti in range(len(times)):
                worst = 0.0
                for a in range(len(fams)):
                    for b in range(a + 1, len(fams)):
                        ma = (recs[:, 0] == r_nm) & (recs[:, 1] == fams[a])
                        mb = (recs[:, 0] == r_nm) & (recs[:, 1] == fams[b])
                        worst = max(worst, tv(vals[ti, ma], vals[ti, mb], edges))
                tv_t.append(worst)
            tv_t = np.asarray(tv_t)
            below = np.flatnonzero(tv_t <= TV_THRESHOLD)
            tau_fam[name] = dict(
                tau_ps=(float(times[below[0]]) if below.size else None),
                tv_trace=tv_t.tolist())
        # ---- clone twins ------------------------------------------------------------------
        mt = (recs[:, 0] == r_nm) & (recs[:, 1] == -1)
        idx0 = np.flatnonzero(mt & (recs[:, 3] == 0))
        idx1 = np.flatnonzero(mt & (recs[:, 3] == 1))
        tau_clone = {}
        for comp, name in enumerate(("n_NaO", "n_ClH", "n_bridge")):
            c_t = []
            for ti in range(len(times)):
                a, b = Ys[ti, idx0, comp], Ys[ti, idx1, comp]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    c_t.append(1.0)
                else:
                    c_t.append(float(np.corrcoef(a, b)[0, 1]))
            c_t = np.asarray(c_t)
            below = np.flatnonzero(c_t <= np.exp(-1.0))
            tau_clone[name] = dict(tau_ps=(float(times[below[0]]) if below.size else None),
                                   corr_trace=c_t.tolist())
        results[f"r{r_nm:.4f}"] = dict(family=tau_fam, clone=tau_clone)
        f_taus = [v["tau_ps"] for v in tau_fam.values() if v["tau_ps"] is not None]
        c_taus = [v["tau_ps"] for v in tau_clone.values() if v["tau_ps"] is not None]
        print(f"  r = {r_nm:.3f} nm: family tau = {max(f_taus) if f_taus else '>track'} ps, "
              f"clone tau = {max(c_taus) if c_taus else '>track'} ps", flush=True)

    all_fam = [v["tau_ps"] for r in results.values() for v in r["family"].values()
               if v["tau_ps"] is not None]
    all_clone = [v["tau_ps"] for r in results.values() for v in r["clone"].values()
                 if v["tau_ps"] is not None]
    tau_family = max(all_fam) if all_fam else None
    tau_clone_max = max(all_clone) if all_clone else None
    disagree = (tau_family and tau_clone_max
                and max(tau_family, tau_clone_max) / min(tau_family, tau_clone_max) > 2.0)
    summary = dict(r_points=r_points, per_point=results,
                   tau_perp_family_ps=tau_family, tau_perp_clone_ps=tau_clone_max,
                   track_ps=TRACK_PS, tv_threshold=TV_THRESHOLD,
                   disagreement_over_2x=bool(disagree),
                   note=("family and clone estimates disagree by >2x -- reported as an open "
                         "finding, NOT resolved by choosing one (Amendment 10)")
                   if disagree else "family and clone estimates agree within 2x",
                   gate_D_lambda_rep_ceiling_per_ps=(0.1 / max(tau_family or 0, tau_clone_max or 0)
                                                     if (tau_family or tau_clone_max) else None),
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    with open(os.path.join(args.out, "tau_perp.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\ntau_perp: family {tau_family} ps, clone {tau_clone_max} ps "
          f"-> {args.out}/tau_perp.json")


if __name__ == "__main__":
    main()
