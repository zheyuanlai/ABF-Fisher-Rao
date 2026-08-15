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
from c60.prep import assert_relaxed, drag_cages, push_waters_off_cages  # noqa: E402

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
DRAG_RATE_NM_PS = 0.04          #: Amendment 16.9: constant-rate cage drag, rate is physical
DRAG_CLAMP = 5.0e4              #: per-site force clamp DURING the drag only (kJ/mol/nm)


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
    """Clash-push then clipped steepest descent; cages pinned, constraints re-applied.

    The pusher is load-bearing: SD's total reach (n_steps x clamp = 0.06 nm) cannot clear
    teleport overlaps of ~0.25 nm, and the first ladder smoke died on exactly that (singular
    M-SHAKE after a water blew apart).  The force guard at the end refuses to return a state
    dynamics cannot integrate.
    """
    left = push_waters_off_cages(x, eng)
    if left:
        raise RuntimeError(f"{left} waters still clashing after push iterations; prep defect")
    x_ref = x.clone()
    dyn.cons.apply_positions(x, x_ref)
    eng.compute_vsites(x)
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
    assert_relaxed(eng, x, chunk=CHUNK)
    return x


def _drag(eng, dyn, x, xi_from, xi_to, center, gen, rate_nm_ps=DRAG_RATE_NM_PS):
    """Amendment 16.9: move the cages linearly in xi while the water propagates.

    ``xi_from``/``xi_to``: per-walker (B,) tensors.  Wall duration is set by the LONGEST
    traverse at ``rate_nm_ps``; shorter traverses finish early and hold.  A per-site force
    clamp is active during the drag only; the settle/equilibration that follows at fixed d
    (unclamped) is what sets the ensemble.
    """
    return drag_cages(eng, dyn, x, xi_from, xi_to, center, gen,
                      rate_nm_ps=rate_nm_ps, clamp=DRAG_CLAMP, chunk=CHUNK)


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
    # RATE SHORTCUTS ARE BANNED (two ladder deaths: the 2x spot shortcut jammed a water,
    # the 8x smoke shortcut exploded a replica at the B=816 tail).  The smoke shrinks the
    # WORKLOAD instead: 6 windows spanning the domain, dragged at the production rate, so
    # the smoke exercises exactly the production code path.
    drag_rate = DRAG_RATE_NM_PS
    equil_ps = EQUIL_PS * scale
    prod_ps = PROD_PS * scale
    anchor_run = ANCHOR_RUN_PS * scale
    anchor_settle = ANCHOR_SETTLE_PS * scale
    snap_ps = tuple(t * scale for t in ANCHOR_SNAP_PS)

    n_windows = 6 if a.smoke else N_WINDOWS
    d_grid_run = (D_GRID[np.linspace(0, N_WINDOWS - 1, n_windows).round().astype(int)]
                  if a.smoke else D_GRID)
    b_total = n_windows * len(FAMILIES) * N_REP
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
    anchors_ok = False
    if os.path.exists(anchors_path):
        # cache hits skip build-branch checks (the NaCl lesson): re-assert on load, at the
        # JAM threshold (1e4), not the explosion threshold -- a cached jammed snapshot from
        # an earlier code state must invalidate the cache, not resume as poisoned prep
        anc = np.load(anchors_path)
        snaps_wet = anc["wet"]
        snaps_dry = anc["dry"]
        anchors_ok = True
        for name, arr, d_a in (("wet", snaps_wet, ANCHOR_WET_NM),
                               ("dry", snaps_dry, ANCHOR_DRY_NM)):
            if not np.isfinite(arr).all():
                anchors_ok = False
                break
            xa = torch.as_tensor(arr, device="cuda", dtype=dtype)
            if float((eng.xi(xa) - d_a).abs().max()) > 1e-4:
                anchors_ok = False
                del xa
                break
            _, f_chk = eng.energy_forces(xa, chunk=CHUNK)
            if float(f_chk.abs().amax()) > 1.0e4:
                anchors_ok = False
                del xa, f_chk
                break
            del xa, f_chk
        if anchors_ok:
            print(f"[phase A] loaded anchors ({snaps_wet.shape[0]} wet, "
                  f"{snaps_dry.shape[0]} dry snapshots), revalidated at jam threshold",
                  flush=True)
        else:
            print("[phase A] cached anchors FAILED revalidation; rebuilding", flush=True)
            os.remove(anchors_path)
    if not anchors_ok:
        snaps = {}
        center_t = torch.tensor([0.5 * lx, 0.5 * lx, 0.5 * lz], device="cuda", dtype=dtype)
        for name, d_anchor in (("wet", ANCHOR_WET_NM), ("dry", ANCHOR_DRY_NM)):
            # both anchors START from the frozen 2.428 nm box (no teleport anywhere);
            # the dry anchor is DRAGGED to 0.968 (Amendment 16.9)
            x = _place_batch(eng, base, lx, lz, [ANCHOR_WET_NM] * N_ANCHOR_STREAMS,
                             "cuda", dtype)
            noise = torch.as_tensor(
                rng.normal(0.0, 0.003, x.shape), device="cuda", dtype=dtype)
            noise[:, eng.cage_a, :] = 0.0
            noise[:, eng.cage_b, :] = 0.0
            x += noise
            x_ref = x.clone()
            dyn.cons.apply_positions(x, x_ref)
            eng.compute_vsites(x)
            _relax(eng, dyn, x)
            if abs(d_anchor - ANCHOR_WET_NM) > 1e-9:
                B = x.shape[0]
                _drag(eng, dyn, x,
                      torch.full((B,), ANCHOR_WET_NM, device="cuda", dtype=dtype),
                      torch.full((B,), d_anchor, device="cuda", dtype=dtype),
                      center_t, gen, rate_nm_ps=drag_rate)
                print(f"[phase A] {name}: dragged to {d_anchor} nm "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
            v = dyn.maxwell_velocities(x, generator=gen)
            f, _ = force_of(x)
            store = []
            n_steps = int(round((anchor_settle + anchor_run) / dt))
            snap_steps = {int(round((anchor_settle + t) / dt)) for t in snap_ps}
            for step in range(1, n_steps + 1):
                _, f = dyn.step(x, v, f, generator=gen)
                if step in snap_steps:
                    store.append(x.detach().cpu().numpy().astype(np.float32))
            # anchor jam census: a jammed stream would poison every descendant window start.
            # Snapshots are taken only from clean streams; >= 2/3 must survive.
            _, f_chk = ef_compiled(x, chunk=CHUNK)
            per_stream = f_chk.abs().amax(dim=(1, 2))
            clean = (per_stream < 1.0e4).cpu().numpy()
            n_clean = int(clean.sum())
            if n_clean < (2 * N_ANCHOR_STREAMS) // 3:
                raise RuntimeError(f"{name} anchors: only {n_clean}/{N_ANCHOR_STREAMS} "
                                   "clean streams; prep defect")
            arr = np.concatenate(store, axis=0)              # (12*n_snap, N, 3)
            keep = np.tile(clean, len(store))
            snaps[name] = arr[keep]
            if n_clean < N_ANCHOR_STREAMS:
                print(f"[phase A] {name}: excluded {N_ANCHOR_STREAMS - n_clean} jammed "
                      f"streams", flush=True)
            print(f"[phase A] {name} anchors done: {snaps[name].shape[0]} snapshots "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        np.savez(anchors_path + ".tmp.npz", wet=snaps["wet"], dry=snaps["dry"])
        os.replace(anchors_path + ".tmp.npz", anchors_path)      # atomic: no torn cache
        snaps_wet, snaps_dry = snaps["wet"], snaps["dry"]

    # ------------------------------------------------------------------ phase B: windows
    # flat layout: i = w * 12 + fam * 3 + rep
    d_values = np.repeat(d_grid_run, 12)
    fam_idx = np.tile(np.repeat(np.arange(4), 3), n_windows)
    rep_idx = np.tile(np.arange(3), n_windows * 4)

    n_equil = int(round(equil_ps / dt))
    n_prod = int(round(prod_ps / dt))
    block_steps = int(round(BLOCK_PS * scale / dt))
    n_blocks = max(1, n_prod // block_steps)
    ngap_every = max(1, int(round(NGAP_EVERY_PS * scale / dt)))

    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        x = ck["x"]; v = ck["v"]; f = ck["f"]
        if x.shape[0] != b_total or not (torch.isfinite(x).all() and torch.isfinite(v).all()
                                         and torch.isfinite(f).all()):
            raise RuntimeError(f"checkpoint invalid: shape {tuple(x.shape)} vs B={b_total} "
                               "or non-finite state; refusing to resume")
        fsum = ck["fsum"]; fcount = ck["fcount"]
        ngap_rows = list(ck["ngap_rows"]); ngap_steps_l = list(ck["ngap_steps"])
        gen.set_state(ck["rng"].cpu())
        start_step = int(ck["step"])
        print(f"[resume] step {start_step}", flush=True)
    else:
        # Amendment 16.9: every start reaches its window separation by DRAG, never teleport.
        # wet: drag DOWN from a 2.428 snapshot; dry: drag UP from a 0.968 anchor snapshot;
        # bulk: independent wet snapshot at HALF rate (most adiabatic control);
        # hot: dragged wet snapshot + water noise + clash push + SD.
        starts_path = os.path.join(out_dir, "starts_dragged.npz")
        starts = np.empty((b_total, base.shape[0], 3), dtype=np.float32)
        src_xi = np.empty(b_total)
        n_wet, n_dry = snaps_wet.shape[0], snaps_dry.shape[0]
        for i in range(b_total):
            w, fam, rep = i // 12, (i % 12) // 3, i % 3
            if FAMILIES[fam] == "dry":
                starts[i] = snaps_dry[(w * 3 + rep) % n_dry]
                src_xi[i] = ANCHOR_DRY_NM
            else:                                            # wet, bulk, hot from wet pool
                starts[i] = snaps_wet[(w * 3 + rep + 7 * fam) % n_wet]
                src_xi[i] = ANCHOR_WET_NM
        starts_ok = False
        if os.path.exists(starts_path):
            # load-path revalidation at the JAM threshold; a poisoned cache invalidates
            x = torch.as_tensor(np.load(starts_path)["x"], device="cuda", dtype=dtype)
            xi_err = float((eng.xi(x) - torch.as_tensor(d_values, device="cuda",
                                                        dtype=dtype)).abs().max())
            _, f_chk = eng.energy_forces(x, chunk=CHUNK)
            n_jam = int((f_chk.abs().amax(dim=(1, 2)) > 1.0e4).sum())
            if torch.isfinite(x).all() and xi_err < 1e-4 and n_jam == 0:
                starts_ok = True
                print("[phase B] loaded dragged starts, revalidated at jam threshold",
                      flush=True)
            else:
                print(f"[phase B] cached starts FAILED revalidation "
                      f"(xi_err {xi_err:.1e}, {n_jam} jammed); rebuilding", flush=True)
                os.remove(starts_path)
            del f_chk
        if not starts_ok:
            x = torch.as_tensor(starts, device="cuda", dtype=dtype)
            c = torch.tensor([0.5 * lx, 0.5 * lx, 0.5 * lz], device="cuda", dtype=dtype)
            d_t = torch.as_tensor(d_values, device="cuda", dtype=dtype)
            src_t = torch.as_tensor(src_xi, device="cuda", dtype=dtype)
            fam_t = torch.as_tensor(fam_idx, device="cuda")
            print(f"[phase B] dragging {b_total} starts to window separations...", flush=True)
            fast = fam_t != 0                                # bulk (fam 0) drags at half rate
            for mask, rate in ((fast, drag_rate), (~fast, 0.5 * drag_rate)):
                idx = torch.nonzero(mask, as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                xs = x[idx].contiguous()
                _drag(eng, dyn, xs, src_t[idx], d_t[idx], c, gen, rate_nm_ps=rate)
                x[idx] = xs
                print(f"[phase B] dragged {idx.numel()} starts at rate {rate:.3f} nm/ps "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
            # hot family: destroy the interface AFTER the drag, then push + SD + guard
            hot = torch.nonzero(fam_t == 3, as_tuple=True)[0]
            xh = x[hot].contiguous()
            noise = torch.as_tensor(rng.normal(0.0, 0.05, xh.shape), device="cuda",
                                    dtype=dtype)
            noise[:, eng.cage_a, :] = 0.0
            noise[:, eng.cage_b, :] = 0.0
            xh += noise
            x_ref = xh.clone()
            dyn.cons.apply_positions(xh, x_ref)
            eng.compute_vsites(xh)
            _relax(eng, dyn, xh)
            x[hot] = xh
            np.savez(starts_path + ".tmp.npz",
                     x=x.detach().cpu().numpy().astype(np.float32))
            os.replace(starts_path + ".tmp.npz", starts_path)    # atomic: no torn cache
        v = dyn.maxwell_velocities(x, generator=gen)
        f, _ = force_of(x)
        fsum = torch.zeros(b_total, n_blocks, device="cuda", dtype=torch.float64)
        fcount = torch.zeros(n_blocks, device="cuda", dtype=torch.float64)
        ngap_rows, ngap_steps_l = [], []

    total_steps = n_equil + n_prod
    ckpt_every = max(2000, int(round(10.0 / dt)))
    max_f_post_equil = None
    for step in range(start_step, total_steps):
        _, f = dyn.step(x, v, f, generator=gen)
        s_prod = step - n_equil
        if s_prod == 0:
            # per-walker jam census at the equilibration/production boundary: a wedged water
            # reads 2-5e4 per-site force (thermal ceiling ~2.5e3); recorded, not raised --
            # the analysis excludes jammed replicas with the count reported (a raise would
            # kill an 816-replica build for a couple of walkers)
            _, f_raw_chk = ef_compiled(x, chunk=CHUNK)
            max_f_post_equil = f_raw_chk.abs().amax(dim=(1, 2)).detach().cpu().numpy()
            n_jam = int((max_f_post_equil > 1.0e4).sum())
            print(f"  [jam census] {n_jam}/{b_total} replicas above 1e4 kJ/mol/nm "
                  f"at production start", flush=True)
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
            agg = (step - start_step + 1) * dt * b_total / 1000.0
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
             d_grid=d_grid_run, d_values=d_values, family=fam_idx, replica=rep_idx,
             f_block_means=f_blocks, block_ps=BLOCK_PS * scale,
             max_force_post_equil=(max_f_post_equil if max_f_post_equil is not None
                                   else np.full(b_total, np.nan)),
             ngap=np.asarray(ngap_rows), ngap_steps=np.asarray(ngap_steps_l),
             equil_ps=equil_ps, prod_ps=prod_ps, dt_ps=dt)
    manifest = dict(csys.manifest(), stage=f"reference_build{a.build}",
                    smoke=bool(a.smoke), seed0=seed0, dt_ps=dt,
                    n_windows=n_windows, families=FAMILIES, n_rep=N_REP, B=b_total,
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
