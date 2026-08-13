"""Stage 3 -- the ABF-only screen (SPEC §6.3, Amendment 12.5: `N = 512` first).

**No Fisher-Rao anywhere in this stage.**  This is the stage that decides the study, and it
computes no regime classification: `T_hit`, `T_est`, `tau_perp` and every gate verdict are derived
afterwards from the saved traces by ``methane_gates.py``.

Initial conditions (SPEC §6.1)
-----------------------------
All walkers start in the **contact** basin, so discovery is a real question rather than defined
out of existence (the Amendment 4 lesson: a distributed start makes `T_hit = 0` everywhere and
Gate B can never fail).

Each walker gets an **independently equilibrated solvent environment** -- built here by holding
the pair at the contact minimum under a rigid constraint and propagating every walker under its
own noise for ``--prep-ps``.  Without this, "many walkers" would mean "many clones" at `t = 0`
and the mechanism test would be contaminated at exactly the point that matters.
``assert_distinct_solvent`` refuses a cloned population.

Declared bias: a contact start makes discovery *harder*, so it can only push the classification
toward discovery-limited.  It cannot manufacture an establishment-limited verdict, which is the
direction that would license an mFR arm.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_screen.py --out results/methane/screen_N512
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

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.core import MethaneSimConfig, assert_distinct_solvent, run_screen  # noqa: E402
from methane.dynamics import (BAOAB, CompositeConstraints, PairConstraint,      # noqa: E402
                              RigidWaterConstraints, water_molecules)
from methane.nonbonded import MethaneNonbonded                   # noqa: E402

SEEDS = list(range(5000, 5008))          #: frozen by §5 of the preregistration


def build_population(ff, topology, L, start, n_walkers, r_contact, prep_ps, chunk, seed, dev):
    """`n_walkers` copies of ``start`` propagated under independent noise at fixed contact `r`."""
    mols = water_molecules(topology)
    mi = ff.params["methane_index"]
    x = torch.tensor(np.repeat(start[None], n_walkers, axis=0), device=dev, dtype=torch.float32)
    cons = CompositeConstraints(
        [RigidWaterConstraints(mols, [msys.R_OH_NM, msys.R_OH_NM, msys.r_HH_nm()],
                               ff.params["mass"], device=dev, dtype=torch.float32),
         PairConstraint(int(mi[0]), int(mi[1]), np.full(n_walkers, r_contact),
                        ff.params["mass"], device=dev, dtype=torch.float32)],
        atom_sets=[mols, mi])
    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=chunk), ff.params["mass"], cons,
                  msys.DT_PS, msys.TEMPERATURE_K, msys.GAMMA_PS, device=dev,
                  dtype=torch.float32)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cons.apply_positions(x, x.clone())
    v = integ.maxwell_velocities(x, generator=gen)
    _, f = ff.energy_forces(x, chunk=chunk)
    for _ in range(int(round(prep_ps / msys.DT_PS))):
        _, f = integ.step(x, v, f, generator=gen)
    return x.cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/screen_N512")
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--baths", default="results/methane/baths")
    ap.add_argument("--n-walkers", type=int, default=512)
    ap.add_argument("--run-ps", type=float, default=200.0)
    ap.add_argument("--prep-ps", type=float, default=50.0)
    ap.add_argument("--r-contact", type=float, default=0.38)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--checkpoint-every", type=int, default=20_000,
                    help="steps between full-state checkpoints (10 ps at dt=0.5 fs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    system = msys.build_system(mod.topology)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)

    bz = np.load(os.path.join(args.baths, "baths.npz"))
    key = f"start_b0_f0_r{args.r_contact:.4f}"
    if key not in bz:
        raise SystemExit(f"no minimised contact start {key}; run scripts/methane_baths.py --per-r")
    start = bz[key].astype(np.float64)

    ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
    print(f"[engine] torch on {dev}, float32, compiled; L = {L:.6f} nm", flush=True)

    n_steps = int(round(args.run_ps / msys.DT_PS))
    t_start = time.time()
    for si, seed in enumerate(seeds):
        cell = os.path.join(args.out, f"seed{seed}.npz")
        if os.path.exists(cell):
            print(f"[seed {seed}] cached", flush=True)
            continue
        t0 = time.time()
        print(f"\n[seed {seed}] building {args.n_walkers} independent contact-basin walkers "
              f"({args.prep_ps} ps each) ...", flush=True)
        init = build_population(ff, mod.topology, L, start, args.n_walkers, args.r_contact,
                                args.prep_ps, args.chunk, 700000 + seed, dev)
        spread = assert_distinct_solvent(init, ff.params["methane_index"])
        print(f"[seed {seed}] population built in {(time.time()-t0)/60:.1f} min; "
              f"min solvent deviation between walkers = {spread:.3e} nm", flush=True)

        sim = MethaneSimConfig(n_walkers=args.n_walkers, n_steps=n_steps, box_nm=L,
                               rng_seed=seed, chunk=args.chunk)
        ckpt = os.path.join(args.out, f"seed{seed}.ckpt")
        out = run_screen(ff, sim, init, mod.topology, device=dev, dtype=torch.float32,
                         verbose=True, progress_every=20_000,
                         checkpoint_path=ckpt, checkpoint_every=args.checkpoint_every)
        np.savez_compressed(cell, seed=seed, **out)
        if os.path.exists(ckpt):        # seed is complete; the resume state is now dead weight
            os.remove(ckpt)
        print(f"[seed {seed}] done in {(time.time()-t0)/60:.1f} min -> {cell}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(dict(stage="abf_screen", amendment="Amendment 12.5 (N=512 first)",
                       n_walkers=args.n_walkers, run_ps=args.run_ps, prep_ps=args.prep_ps,
                       r_contact_nm=args.r_contact, seeds=seeds, box_L_nm=L,
                       wall_hours=(time.time() - t_start) / 3600.0,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True,
                                                 text=True).stdout.strip()), fh, indent=2)
    print(f"\n[done] {(time.time()-t_start)/3600:.2f} h -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
