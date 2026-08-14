"""Constrained mean-force reference for C60 -- SPEC_c60_water.md §5, one build per invocation.

The paper's instrument (fixed cages, average force over the run, integrate) with the
campaign's structure on top: 68 uniform windows x 4 solvent families x 3 replicas, batched as
ONE torch ensemble of 816 systems on the pinned GPU, in one process (the determinism rule).

Families (SPEC §6 recipe, prepared per build with build-specific seeds):
  bulk  frozen-box waters, cages at d_k, thermal jiggle, relax          (control)
  wet   snapshot equilibrated at 2.428 nm, cages teleported to d_k      (water-rich gap)
  dry   snapshot equilibrated at 0.968 nm (paper's contact), teleported (water-poor gap)
  hot   frozen-box waters, 0.05 nm noise, relax                        (destroyed interface)

Phases:  A) anchor ensembles at 2.428 / 0.968 nm produce the wet/dry source snapshots;
         B) 816 starts relaxed (clipped SD, cages fixed, SHAKE + vsites each step),
            100 ps equilibration, 250 ps production with 5 ps block means of
            f = (1/2)(F_A,z - F_B,z) and n_gap / n_shell traces.

Output: results/c60/reference/build<k>/windows.npz + manifest.json.  Full checkpoint/resume.

Usage:  CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference.py --build 1
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import geometry, system as csys  # noqa: E402
from c60.dynamics import C60Dynamics  # noqa: E402
from c60.nonbonded import C60Nonbonded  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "reference")
BOX = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "box", "frozen_box.npz")

N_WINDOWS = 68
D_GRID = np.linspace(csys.XI_LO_NM, csys.XI_HI_NM, N_WINDOWS)
FAMILIES = ("bulk", "wet", "dry", "hot")
N_REP = 3
B_TOTAL = N_WINDOWS * len(FAMILIES) * N_REP          # 816

ANCHOR_WET_NM = csys.D_REF_NM                        # 2.428, the PMF anchor state
ANCHOR_DRY_NM = 0.968                                # the paper's contact minimum (declared)
N_ANCHOR_STREAMS = 12
ANCHOR_SETTLE_PS = 30.0
ANCHOR_RUN_PS = 120.0
ANCHOR_SNAP_PS = (60.0, 90.0, 120.0)

EQUIL_PS = 100.0
PROD_PS = 250.0
BLOCK_PS = 5.0
NGAP_EVERY_PS = 2.0
CHUNK = 256
SD_STEPS = 300


def _dt_ps():
    with open(os.path.join(os.path.dirname(REF), "parity", "dt_gate.json")) as fh:
        return float(json.load(fh)["decision_dt_ps"])


def _load_engine(dtype, device="cuda"):
    import openmm as mm
    import openmm.unit as u

    fz = np.load(BOX)
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    base = np.asarray(fz["positions"], dtype=np.float64)
    mod = csys.build_modeller()
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box, pme_params=csys.pme_params())
    alpha, nx, ny, nz = csys.pme_params()
    eng = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha, (nx, ny, nz),
                       device=device, dtype=dtype)
    return eng, base, lx, lz


def _place_batch(eng, base, lx, lz, d_values, device, dtype):
    """(B, N, 3) with cages at per-walker separations and the base waters."""
    B = len(d_values)
    x = torch.as_tensor(base, device=device, dtype=dtype)[None].repeat(B, 1, 1).contiguous()
    center = (0.5 * lx, 0.5 * lx, 0.5 * lz)
    cage = torch.as_tensor(geometry.c60_cage(), device=device, dtype=dtype)
    c = torch.as_tensor(center, device=device, dtype=dtype)
    d = torch.as_tensor(np.asarray(d_values), device=device, dtype=dtype)
    x[:, eng.cage_a, :] = cage[None] + c[None, None, :]
    x[:, eng.cage_a, 2] += -0.5 * d[:, None]
    x[:, eng.cage_b, :] = cage[None] + c[None, None, :]
    x[:, eng.cage_b, 2] += +0.5 * d[:, None]
    return x


def _relax(eng, dyn, x, n_steps=SD_STEPS):
    """Clipped steepest descent on the waters; cages pinned, constraints re-applied."""
    for _ in range(n_steps):
        _, f_raw = eng.energy_forces(x, chunk=CHUNK)
        f = eng.redistribute(f_raw)
        step = (0.5e-5 * f).clamp(-2e-4, 2e-4)
        step[:, eng.cage_a, :] = 0.0
        step[:, eng.cage_b, :] = 0.0
        x_ref = x.clone()
        x += step
        dyn.cons.apply_positions(x, x_ref)
        eng.compute_vsites(x)
    return x


def _ngap_torch(eng, x, lx, lz, r_cyl=0.62):
    """Batched smooth n_gap (SPEC §4), water oxygens in the inter-cage cylinder."""
    o = eng.waters[:, 0]
    com_a = x[:, eng.cage_a, :].mean(dim=1)
    com_b = x[:, eng.cage_b, :].mean(dim=1)
    xi = com_b[:, 2] - com_a[:, 2]
    center = 0.5 * (com_a + com_b)
    off = x[:, o, :] - center[:, None, :]
    L = torch.tensor([lx, lx, lz], device=x.device, dtype=x.dtype)
    off = off - L * torch.round(off / L)
    u = off[..., 2]
    w = torch.sqrt(off[..., 0] ** 2 + off[..., 1] ** 2)
    s_ax = 1.0 / (1.0 + (u.abs() / (0.5 * xi[:, None])).pow(6))
    s_rad = 1.0 / (1.0 + (w / r_cyl).pow(6))
    return (s_ax * s_rad).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--smoke", action="store_true",
                    help="1/50th durations, no outputs frozen -- ladder step only")
    a = ap.parse_args()

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev != "3":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev!r}; SPEC §11 pins this study to GPU 3")
    assert torch.cuda.device_count() == 1

    scale = 0.02 if a.smoke else 1.0
    equil_ps = EQUIL_PS * scale
    prod_ps = PROD_PS * scale
    anchor_run = ANCHOR_RUN_PS * scale
    anchor_settle = ANCHOR_SETTLE_PS * scale
    snap_ps = tuple(t * scale for t in ANCHOR_SNAP_PS)

    out_dir = os.path.join(REF, f"build{a.build}" + ("_smoke" if a.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    dt = _dt_ps()
    dtype = torch.float32
    eng, base, lx, lz = _load_engine(dtype)
    ef_compiled = torch.compile(eng.energy_forces, dynamic=False)
    dyn = C60Dynamics(eng, dt, device="cuda", dtype=dtype,
                      force_fn=lambda q: ef_compiled(q, chunk=CHUNK))
    seed0 = 20260814 + a.build * 1000
    gen = torch.Generator(device="cuda").manual_seed(seed0)
    rng = np.random.default_rng(seed0)

    def force_of(x):
        _, f_raw = ef_compiled(x, chunk=CHUNK)
        return eng.redistribute(f_raw), f_raw

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=os.path.dirname(__file__)).stdout.strip()

    t0 = time.perf_counter()

    # ------------------------------------------------------------------ phase A: anchors
    anchors_path = os.path.join(out_dir, "anchors.npz")
    if os.path.exists(anchors_path):
        anc = np.load(anchors_path)
        snaps_wet = anc["wet"]
        snaps_dry = anc["dry"]
        print(f"[phase A] loaded anchors ({snaps_wet.shape[0]} wet, "
              f"{snaps_dry.shape[0]} dry snapshots)", flush=True)
    else:
        snaps = {}
        for name, d_anchor in (("wet", ANCHOR_WET_NM), ("dry", ANCHOR_DRY_NM)):
            x = _place_batch(eng, base, lx, lz, [d_anchor] * N_ANCHOR_STREAMS, "cuda", dtype)
            noise = torch.as_tensor(
                rng.normal(0.0, 0.003, x.shape), device="cuda", dtype=dtype)
            noise[:, eng.cage_a, :] = 0.0
            noise[:, eng.cage_b, :] = 0.0
            x += noise
            x_ref = x.clone()
            dyn.cons.apply_positions(x, x_ref)
            eng.compute_vsites(x)
            _relax(eng, dyn, x)
            v = dyn.maxwell_velocities(x, generator=gen)
            f, _ = force_of(x)
            store = []
            n_steps = int(round((anchor_settle + anchor_run) / dt))
            snap_steps = {int(round((anchor_settle + t) / dt)) for t in snap_ps}
            for step in range(1, n_steps + 1):
                _, f = dyn.step(x, v, f, generator=gen)
                if step in snap_steps:
                    store.append(x.detach().cpu().numpy().astype(np.float32))
            snaps[name] = np.concatenate(store, axis=0)      # (12*3, N, 3)
            print(f"[phase A] {name} anchors done: {snaps[name].shape[0]} snapshots "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        np.savez(anchors_path, wet=snaps["wet"], dry=snaps["dry"])
        snaps_wet, snaps_dry = snaps["wet"], snaps["dry"]

    # ------------------------------------------------------------------ phase B: windows
    # flat layout: i = w * 12 + fam * 3 + rep
    d_values = np.repeat(D_GRID, 12)
    fam_idx = np.tile(np.repeat(np.arange(4), 3), N_WINDOWS)
    rep_idx = np.tile(np.arange(3), N_WINDOWS * 4)

    n_equil = int(round(equil_ps / dt))
    n_prod = int(round(prod_ps / dt))
    block_steps = int(round(BLOCK_PS * scale / dt))
    n_blocks = max(1, n_prod // block_steps)
    ngap_every = max(1, int(round(NGAP_EVERY_PS * scale / dt)))

    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        x = ck["x"]; v = ck["v"]; f = ck["f"]
        fsum = ck["fsum"]; fcount = ck["fcount"]
        ngap_rows = list(ck["ngap_rows"]); ngap_steps_l = list(ck["ngap_steps"])
        gen.set_state(ck["rng"].cpu())
        start_step = int(ck["step"])
        print(f"[resume] step {start_step}", flush=True)
    else:
        starts = np.empty((B_TOTAL, base.shape[0], 3), dtype=np.float32)
        n_wet, n_dry = snaps_wet.shape[0], snaps_dry.shape[0]
        for i in range(B_TOTAL):
            w, fam, rep = i // 12, (i % 12) // 3, i % 3
            if FAMILIES[fam] == "bulk":
                starts[i] = base
            elif FAMILIES[fam] == "wet":
                starts[i] = snaps_wet[(w * 3 + rep) % n_wet]
            elif FAMILIES[fam] == "dry":
                starts[i] = snaps_dry[(w * 3 + rep) % n_dry]
            else:                                            # hot
                starts[i] = base + rng.normal(0.0, 0.05, base.shape)
        x = torch.as_tensor(starts, device="cuda", dtype=dtype)
        # cages to window separations; per-replica jiggle on waters; relax
        cage = torch.as_tensor(geometry.c60_cage(), device="cuda", dtype=dtype)
        c = torch.tensor([0.5 * lx, 0.5 * lx, 0.5 * lz], device="cuda", dtype=dtype)
        d_t = torch.as_tensor(d_values, device="cuda", dtype=dtype)
        x[:, eng.cage_a, :] = cage[None] + c[None, None, :]
        x[:, eng.cage_a, 2] += -0.5 * d_t[:, None]
        x[:, eng.cage_b, :] = cage[None] + c[None, None, :]
        x[:, eng.cage_b, 2] += +0.5 * d_t[:, None]
        jig = torch.as_tensor(rng.normal(0.0, 0.003, x.shape), device="cuda", dtype=dtype)
        jig[:, eng.cage_a, :] = 0.0
        jig[:, eng.cage_b, :] = 0.0
        x += jig
        x_ref = x.clone()
        dyn.cons.apply_positions(x, x_ref)
        eng.compute_vsites(x)
        print(f"[phase B] relaxing {B_TOTAL} starts...", flush=True)
        _relax(eng, dyn, x)
        v = dyn.maxwell_velocities(x, generator=gen)
        f, _ = force_of(x)
        fsum = torch.zeros(B_TOTAL, n_blocks, device="cuda", dtype=torch.float64)
        fcount = torch.zeros(n_blocks, device="cuda", dtype=torch.float64)
        ngap_rows, ngap_steps_l = [], []

    total_steps = n_equil + n_prod
    ckpt_every = max(2000, int(round(10.0 / dt)))
    for step in range(start_step, total_steps):
        _, f = dyn.step(x, v, f, generator=gen)
        s_prod = step - n_equil
        if s_prod >= 0:
            b = min(s_prod // block_steps, n_blocks - 1)
            # the estimator wants cage z-forces, which redistribution does not touch --
            # the dynamics force is reused, no second energy evaluation
            fsum[:, b] += eng.local_mean_force(f).to(torch.float64)
            fcount[b] += 1
            if s_prod % ngap_every == 0:
                ngap_rows.append(_ngap_torch(eng, x, lx, lz).detach().cpu().numpy()
                                 .astype(np.float32))
                ngap_steps_l.append(step)
        if step % 2000 == 0:
            el = time.perf_counter() - t0
            agg = (step - start_step + 1) * dt * B_TOTAL / 1000.0
            print(f"  step {step:8d}/{total_steps}  t={step*dt:8.2f} ps  "
                  f"T={float(dyn.temperature(v).mean()):6.1f} K  "
                  f"agg {agg:8.2f} ns  ({el:6.0f}s)", flush=True)
        if (step + 1) % ckpt_every == 0:
            tmp = ckpt_path + ".tmp"
            torch.save(dict(step=step + 1, x=x, v=v, f=f, fsum=fsum, fcount=fcount,
                            ngap_rows=ngap_rows, ngap_steps=ngap_steps_l,
                            rng=gen.get_state()), tmp)
            os.replace(tmp, ckpt_path)

    f_blocks = (fsum / fcount.clamp_min(1.0)[None, :]).cpu().numpy()
    np.savez(os.path.join(out_dir, "windows.npz"),
             d_grid=D_GRID, d_values=d_values, family=fam_idx, replica=rep_idx,
             f_block_means=f_blocks, block_ps=BLOCK_PS * scale,
             ngap=np.asarray(ngap_rows), ngap_steps=np.asarray(ngap_steps_l),
             equil_ps=equil_ps, prod_ps=prod_ps, dt_ps=dt)
    manifest = dict(csys.manifest(), stage=f"reference_build{a.build}",
                    smoke=bool(a.smoke), seed0=seed0, dt_ps=dt,
                    n_windows=N_WINDOWS, families=FAMILIES, n_rep=N_REP, B=B_TOTAL,
                    equil_ps=equil_ps, prod_ps=prod_ps, block_ps=BLOCK_PS * scale,
                    anchors=dict(wet_nm=ANCHOR_WET_NM, dry_nm=ANCHOR_DRY_NM,
                                 streams=N_ANCHOR_STREAMS, snap_ps=list(snap_ps)),
                    commit=commit, cuda_visible_devices=dev,
                    wall_hours=(time.perf_counter() - t0) / 3600.0)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"build {a.build} done: {manifest['wall_hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
