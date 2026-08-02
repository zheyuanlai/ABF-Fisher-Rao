#!/usr/bin/env python
"""A COARSE, SCREENING-ONLY reference free energy F_pilot(phi, chi1) for Ace-Val-Nme.

This is deliberately **not** publication ground truth.  The full 24x24 Stage-4 reference is
premature: gate V3 decides whether an mFR arm is worth running at all, and V3's establishment
metric needs only a provisional F to define each state's target population.  Building a
576-window reference for a CV that then fails V3 would be wasted, so the ordering is
V3 first, reference second.

What it must be good enough for
-------------------------------
* connected MBAR overlap across the window lattice (no disconnected component);
* no major state missing relative to the S1 state map;
* qualitative stability under split-half analysis;
* approximate agreement between the two independent psi starts.

Three things here differ from `run_alanine_reference.py`, and each is forced by a measurement.

**dt = 0.5 fs, not 1 fs.**  Stage 0 measured the restrained system 6.8 K (~10 sigma) below
300 K at 1 fs, recovering at 0.5 and 0.25 fs -- the O(dt^2) signature of an under-integrated
stiff mode -- while the unrestrained system sits within 0.6 sigma of 300 K at every step size.
MBAR removes the restraint using its ANALYTIC potential, so a discretisation error in the
sampled distribution is not unwound by it.  `valine.accepted` enforces this.

**psi is free, and is started from several well-separated values.**  psi is the coordinate the
selected CV omits.  Running every start and comparing them pairwise is the global version of the
sec.32 check, which was made only at six anchors: if a substantial region of (phi, chi1) held two
slowly interconverting psi states, the starts would disagree there.  Note what this catches that
a split-half over COPIES cannot -- copies of one window share its psi start, so that test is
blind to psi by construction, and in the first pilot it read 0.38 kT while the psi starts
disagreed by 3.22 kT.

**A softer restraint, because the spacing is coarser.**  Overlap depends on the ratio of the
restrained width sqrt(kT/kappa) to the window spacing, not on kappa alone.  The alanine
reference ran 24 windows (15 deg) at kappa = 200, a ratio of 0.43; the default here reproduces
that ratio at 18 windows (20 deg).  A stiffer restraint at coarser spacing breaks neighbour
overlap; a softer one lets walkers slide off the 11-18 kT chi1 ridges and leaves them unsampled.

Usage
-----
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_pilot_reference.py --benchmark
    CUDA_VISIBLE_DEVICES=7 python -u scripts/run_valine_pilot_reference.py \
        --out results/valine/pilot_reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine import reference as ref                                          # noqa: E402
from alanine.dynamics import BAOAB                                            # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from alanine.projection import require_odd_grid                              # noqa: E402
from valine import accepted                                                   # noqa: E402
from valine.system import (CHI1_ATOMS, PHI_ATOMS, PSI_ATOMS, make_seed,       # noqa: E402
                           make_system, restrained_minimise, seed_lattice, validate_seed)
from valine.umbrella import dihedrals_iupac                                    # noqa: E402

ALLOWED_GPUS = {"4", "5", "6", "7"}
KB = 0.008314462618
TWO_PI = 2.0 * math.pi
QUADS = (PHI_ATOMS, PSI_ATOMS, CHI1_ATOMS)

#: Default psi starts.  psi is the coordinate the selected CV omits, and it is NOT restrained
#: here, so each window has to equilibrate it on its own.  The FIRST pilot used two starts
#: (+120, -40) and 150 ps, and its two starts disagreed by a median 1.4-1.9 kT in the populated
#: region -- against 0.38 kT for a split-half over copies, which shares a psi start and is
#: therefore blind to exactly this.  Walkers do cross (38.8 % changed psi basin in 150 ps) but
#: retain start memory (final basin fraction 0.551 vs 0.463), so the failure is incomplete
#: EQUILIBRATION, not trapping.  Four starts spread around the circle plus a longer production
#: attack both halves of that: more independent initial conditions to average over, and more
#: time for each to forget.
PSI_STARTS_DEG = (150.0, 60.0, -40.0, -140.0)


def enforce_gpu_policy(est_peak_gib):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (allowed: 4,5,6,7)")
    cvd = cvd.strip()
    if cvd not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} is not an absolute index in "
                         f"{sorted(ALLOWED_GPUS)}; GPUs 0-3 belong to another user")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly 1 visible device, saw {torch.cuda.device_count()}")
    free = torch.cuda.mem_get_info()[0] / 2 ** 30
    if free < 1.5 * est_peak_gib:
        raise SystemExit(f"only {free:.1f} GiB free, need 1.5 x {est_peak_gib:.1f} GiB")
    return cvd


def git_info():
    def sh(*a):
        try:
            return subprocess.check_output(a, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:                                            # noqa: BLE001
            return "unknown"
    return {"commit": sh("git", "rev-parse", "HEAD"),
            "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(sh("git", "status", "--porcelain"))}


def build_windows(system, n_win, verbose=True):
    """Seed every ``(phi, chi1)`` window from each psi start.

    Returns ``(q0 (M,28,3) nm, centres (M,2) rad, psi0 (M,), window (M,), dropped)`` where a
    window may appear once per surviving psi start.  A window is kept if AT LEAST ONE psi start
    validates: dropping the whole window because one start is sterically impossible would delete
    regions of (phi, chi1) that are perfectly accessible from the other backbone conformation,
    and those regions are exactly where the psi-start comparison is informative.
    """
    X, e = make_seed((-80.0, 80.0, 180.0), system=system)
    validate_seed(system, X[None], np.radians([[-80.0, 80.0, 180.0]]), energy=[e])
    if verbose:
        print(f"  parent E = {e:.2f} kJ/mol", flush=True)

    axis = np.linspace(-180.0, 180.0, n_win, endpoint=False)
    q0, cen, psi0, wid, dropped = [], [], [], [], []
    t0 = time.time()
    for i, phi_c in enumerate(axis):
        for j, chi_c in enumerate(axis):
            w = i * n_win + j
            for p in PSI_STARTS_DEG:
                tgt = (float(phi_c), float(p), float(chi_c))
                rot = seed_lattice(X, np.radians([tgt]))[0]
                rel, _ = restrained_minimise(system, rot * 10.0, tgt)
                try:
                    validate_seed(system, rel[None] * 0.1, np.radians([tgt]), cv_tol_deg=5.0)
                except ValueError as exc:
                    dropped.append({"window": w, "target_deg": list(tgt),
                                    "reason": str(exc)})
                    continue
                q0.append(rel * 0.1)
                cen.append((math.radians(phi_c), math.radians(chi_c)))
                psi0.append(math.radians(p))
                wid.append(w)
        if verbose and (i + 1) % 4 == 0:
            print(f"  seeded {(i + 1) * n_win}/{n_win * n_win} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if verbose:
        lost = sorted(set(range(n_win * n_win)) - set(wid))
        print(f"  {len(q0)}/{2 * n_win * n_win} (window, psi-start) seeds validated; "
              f"{len(lost)} windows lost entirely, {len(dropped)} seeds dropped", flush=True)
    return (np.stack(q0), np.array(cen), np.array(psi0), np.array(wid), dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/valine/pilot_reference")
    ap.add_argument("--windows", type=int, default=18, help="per axis; 18 -> 324 windows")
    ap.add_argument("--copies", type=int, default=8, help="per (window, psi-start)")
    ap.add_argument("--psi-starts", type=float, nargs="+", default=None,
                    help="override the psi start values (deg)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="kJ/mol/rad^2; default reproduces alanine's width/spacing ratio 0.43")
    ap.add_argument("--randomize-ps", type=float, default=5.0)
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--prod-ps", type=float, default=150.0)
    ap.add_argument("--save-every", type=int, default=200)      # 0.1 ps at dt = 0.5 fs
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--n-grid", type=int, default=accepted.N_GRID)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--mbar-per-window", type=int, default=400)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    global PSI_STARTS_DEG
    if a.psi_starts:
        PSI_STARTS_DEG = tuple(float(x) for x in a.psi_starts)
    dt = accepted.DT_RESTRAINED_PS                    # 0.5 fs -- see module docstring
    accepted.assert_accepted(dt_ps=dt, n_grid=a.n_grid, restrained=True)
    require_odd_grid(a.n_grid)

    spacing = TWO_PI / a.windows
    if a.kappa is None:
        # sigma / spacing = 0.427, the ratio the accepted alanine reference achieved
        a.kappa = round(KB * a.temperature / (0.427 * spacing) ** 2, 1)
    sigma = math.sqrt(KB * a.temperature / a.kappa)
    print(f"pilot reference: {a.windows}x{a.windows} = {a.windows ** 2} windows on (phi, chi1), "
          f"psi FREE, starts {PSI_STARTS_DEG} deg")
    print(f"  kappa {a.kappa} kJ/mol/rad^2 -> sigma {math.degrees(sigma):.2f} deg, "
          f"spacing {math.degrees(spacing):.1f} deg, ratio {sigma / spacing:.3f}")
    print(f"  dt {dt * 1000:.1f} fs (restrained: the 1 fs clamp is under-integrated)")

    est_peak = 6.0e-5 * a.windows ** 2 * len(PSI_STARTS_DEG) * a.copies
    if a.cpu:
        device, cvd = "cpu", None
    else:
        cvd = enforce_gpu_policy(est_peak)
        device = "cuda"
    dtype = torch.float64
    beta = 1.0 / (KB * a.temperature)

    _, _, system = make_system()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    if phash != accepted.PARAM_HASH:
        raise SystemExit(f"param_hash {phash} != accepted {accepted.PARAM_HASH}")
    tff = TorchFF(P, device=device, dtype=dtype)
    print(f"  param_hash {phash}  device {device}  CUDA_VISIBLE_DEVICES={cvd}", flush=True)

    print("building and validating window seeds ...", flush=True)
    q0_np, cen_np, psi0_np, wid_np, dropped = build_windows(system, a.windows)

    x = torch.as_tensor(np.repeat(q0_np, a.copies, axis=0), device=device,
                        dtype=dtype).contiguous()
    centres = torch.as_tensor(np.repeat(cen_np, a.copies, axis=0), device=device, dtype=dtype)
    psi0 = np.repeat(psi0_np, a.copies)
    wid = np.repeat(wid_np, a.copies)
    B = x.shape[0]
    kappa = float(a.kappa)
    print(f"  batch {B} = {q0_np.shape[0]} seeds x {a.copies} copies", flush=True)

    def total_force(q):
        with torch.enable_grad():
            qg = q.detach().requires_grad_(True)
            th = dihedrals_iupac(qg, QUADS)
            E = tff.energy(qg) + ref.restraint_energy(th[:, 0], th[:, 2], centres, kappa)
            g, = torch.autograd.grad(E.sum(), qg)
        return -g

    integ = BAOAB(P["masses"], dt, a.gamma, a.temperature, total_force, device=device,
                  dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(int(a.seed))
    v = integ.maxwell(x.shape, gen, device, dtype)
    f = total_force(x)

    def run(n_steps, label, gamma=None, save_every=0):
        nonlocal x, v, f, integ
        if gamma is not None:
            integ = BAOAB(P["masses"], dt, gamma, a.temperature, total_force,
                          device=device, dtype=dtype)
            f = total_force(x)
        out, temps = [], []
        t0 = time.perf_counter()
        for s in range(n_steps):
            x, v, f = integ.step(x, v, f, gen)
            if save_every and (s + 1) % save_every == 0:
                out.append(dihedrals_iupac(x, QUADS).to(torch.float32).cpu())
                temps.append(float(integ.kinetic_temperature(v)))
            if n_steps >= 5 and (s + 1) % max(n_steps // 5, 1) == 0:
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite positions in {label} at step {s + 1}")
                el = time.perf_counter() - t0
                print(f"  {label} {s + 1}/{n_steps}  {(s + 1) / el:.0f} steps/s  "
                      f"T={float(integ.kinetic_temperature(v)):.1f} K  "
                      f"eta {(n_steps - s - 1) / ((s + 1) / el) / 60:.1f} min", flush=True)
        return (torch.stack(out, 1).numpy() if out else None,
                np.array(temps) if temps else np.array([]))

    if a.benchmark:
        t0 = time.perf_counter()
        run(200, "bench")
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 200 * 1e3
        total = (a.randomize_ps + a.equil_ps + a.prod_ps) / dt * ms / 1e3 / 60
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
        print(f"\n{ms:.2f} ms/step at B={B} -> projected {total:.1f} min;  peak {peak:.2f} GiB")
        return

    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    run(int(a.randomize_ps / dt), "randomize", gamma=20.0)
    run(int(a.equil_ps / dt), "equil", gamma=a.gamma)
    traj, temps = run(int(a.prod_ps / dt), "prod", save_every=a.save_every)
    wall = time.time() - t_start
    print(f"sampling done in {wall / 60:.1f} min; traj {traj.shape}; "
          f"mean T = {temps.mean():.2f} K", flush=True)
    # The restraint-induced kinetic bias is why dt was halved; verify it actually went away.
    t_dev = abs(temps.mean() - a.temperature) / a.temperature
    print(f"kinetic temperature deviation {100 * t_dev:.2f} % "
          f"({'OK' if t_dev < 0.02 else 'EXCEEDS the 2 % guard'})", flush=True)

    # How much does a window still remember which psi it started from?  This is the quantity
    # the first pilot failed on, so it is measured directly rather than only inferred from the
    # per-start FES comparison further down.
    psi_tr = traj[:, :, 1].astype(np.float64)
    late = psi_tr[:, psi_tr.shape[1] // 2:]
    in_beta = ((late > math.radians(60)) | (late < math.radians(-150))).mean(1)
    memory = {}
    for p0 in sorted(set(psi0.tolist())):
        m = np.isclose(psi0, p0)
        memory[f"{math.degrees(p0):+.0f}"] = float(in_beta[m].mean())
    spread = max(memory.values()) - min(memory.values())
    print(f"psi start memory (fraction in the beta/PPII basin over the last half): "
          + ", ".join(f"{k} deg -> {v:.3f}" for k, v in memory.items()))
    print(f"  spread across starts {spread:.3f} "
          f"({'OK' if spread < 0.10 else 'LARGE -- psi is not equilibrated within windows'})",
          flush=True)

    np.savez_compressed(os.path.join(a.out, "samples.npz"),
                        theta=traj.astype(np.float32), window=wid, psi_start=psi0,
                        centres=np.repeat(cen_np, a.copies, axis=0), temperature=temps)
    print(f"raw samples written to {a.out}/samples.npz", flush=True)

    # ------------------------------------------------------------------ MBAR
    frames = traj.shape[1]
    dev_t = torch.device(device)
    phi_all = torch.as_tensor(traj[:, :, 0].reshape(-1), device=dev_t, dtype=dtype)
    psi_all = torch.as_tensor(traj[:, :, 1].reshape(-1), device=dev_t, dtype=dtype)
    chi_all = torch.as_tensor(traj[:, :, 2].reshape(-1), device=dev_t, dtype=dtype)
    seed_of_sample = np.repeat(np.arange(B), frames)

    # One MBAR "state" per (window, psi-start) SEED, not per window: the two psi starts are
    # different initial conditions of the same restraint, so pooling them into one state would
    # hide any disagreement between them -- which is precisely what this run is meant to detect.
    n_seeds = q0_np.shape[0]
    sub = max(1, (a.copies * frames) // a.mbar_per_window)
    keep = np.zeros(B * frames, dtype=bool)
    keep[::sub] = True
    idx = np.flatnonzero(keep)
    sel_seed = seed_of_sample[idx] // a.copies
    order = np.argsort(sel_seed, kind="stable")
    idx, sel_seed = idx[order], sel_seed[order]
    N_k = torch.as_tensor(np.bincount(sel_seed, minlength=n_seeds), device=dev_t)
    if int((N_k == 0).sum()) > 0:
        raise SystemExit(f"{int((N_k == 0).sum())} MBAR states have no samples after subsampling")
    it = torch.as_tensor(idx, device=dev_t)
    centres_k = torch.as_tensor(cen_np, device=dev_t, dtype=dtype)
    print(f"MBAR: {n_seeds} states, {len(idx):,} samples "
          f"({int(N_k.min())}-{int(N_k.max())} per state) ...", flush=True)
    f_k, n_it, resid = ref.mbar_solve(phi_all[it], chi_all[it], centres_k, kappa, beta, N_k,
                                      verbose=True)
    print(f"  converged in {n_it} iterations, residual {resid:.2e}", flush=True)

    O = ref.overlap_matrix(phi_all[it], chi_all[it], centres_k, kappa, beta, N_k, f_k)
    # Neighbour overlap on the WINDOW lattice: a pair counts if any seed of one window
    # neighbours any seed of the other.
    seed_win = wid_np
    nn = []
    for i in range(a.windows):
        for j in range(a.windows):
            w = i * a.windows + j
            for di, dj in ((1, 0), (0, 1)):
                w2 = ((i + di) % a.windows) * a.windows + (j + dj) % a.windows
                A = np.flatnonzero(seed_win == w)
                Bx = np.flatnonzero(seed_win == w2)
                if A.size and Bx.size:
                    nn.append(float(O[np.ix_(A, Bx)].sum()))
    nn = np.asarray(nn)
    print(f"  nearest-neighbour overlap: min {nn.min():.4f}  p1 {np.percentile(nn, 1):.4f}  "
          f"median {np.median(nn):.4f}  n<0.03 {int((nn < 0.03).sum())}/{nn.size}", flush=True)

    logw = ref.mbar_log_weights(phi_all[it], chi_all[it], centres_k, kappa, beta, N_k, f_k)
    F, counts, _ = ref.fes_from_weights(phi_all[it], chi_all[it], logw, a.n_grid, beta)
    F_np = F.cpu().numpy()
    kT = KB * a.temperature
    filled = int(np.isfinite(F_np).sum())
    print(f"  F_pilot on a {a.n_grid}^2 grid: {filled}/{a.n_grid ** 2} cells filled "
          f"({100 * filled / a.n_grid ** 2:.1f} %), range 0 - {np.nanmax(F_np[np.isfinite(F_np)]) / kT:.1f} kT",
          flush=True)

    # Split-half over COPIES, and per-psi-start.  These need different treatment and conflating
    # them is a silent failure: a copies split keeps every MBAR state populated, but restricting
    # to ONE psi start empties every state belonging to the other -- each MBAR state here is a
    # (window, psi-start) pair.  A subset must therefore also RESTRICT AND REINDEX the state
    # list, not just the samples.  Without that the psi comparison returns "not computable" and
    # the run silently loses the check that globalises sec.32.
    def fes_subset(mask_samples, seed_mask=None):
        states = np.arange(n_seeds) if seed_mask is None else np.flatnonzero(seed_mask)
        if states.size == 0:
            return None
        remap = np.full(n_seeds, -1, dtype=np.int64)
        remap[states] = np.arange(states.size)
        sel = remap[seed_of_sample[idx][mask_samples] // a.copies]
        if sel.size == 0 or (sel < 0).any():
            return None
        Nk2 = np.bincount(sel, minlength=states.size)
        if (Nk2 == 0).any():
            return None
        m = torch.as_tensor(np.flatnonzero(mask_samples), device=dev_t)
        ck = torch.as_tensor(cen_np[states], device=dev_t, dtype=dtype)
        Nk2 = torch.as_tensor(Nk2, device=dev_t)
        fk2, _, _ = ref.mbar_solve(phi_all[it][m], chi_all[it][m], ck, kappa, beta, Nk2)
        lw2 = ref.mbar_log_weights(phi_all[it][m], chi_all[it][m], ck, kappa, beta, Nk2, fk2)
        return ref.fes_from_weights(phi_all[it][m], chi_all[it][m], lw2, a.n_grid,
                                    beta)[0].cpu().numpy()

    copy_of_sample = seed_of_sample[idx] % a.copies
    print("  split-half over copies ...", flush=True)
    F_a = fes_subset(copy_of_sample < a.copies // 2)
    F_b = fes_subset(copy_of_sample >= a.copies // 2)
    print("  per-psi-start ...", flush=True)
    psi_of_sample = psi0[seed_of_sample[idx]]
    starts = sorted(set(psi0_np.tolist()))
    F_per_start = {}
    for p0 in starts:
        F_per_start[p0] = fes_subset(np.isclose(psi_of_sample, p0),
                                     seed_mask=np.isclose(psi0_np, p0))
    # With more than two starts the honest summary is the WORST pair, not an average: a single
    # start that fails to equilibrate is exactly the failure mode this check exists to catch,
    # and averaging it against three agreeing ones would hide it.
    F_p1 = F_per_start[starts[0]]
    F_p2 = F_per_start[starts[-1]]

    def compare(A, Bm, tag):
        if A is None or Bm is None:
            print(f"  {tag}: not computable")
            return None
        m = np.isfinite(A) & np.isfinite(Bm) & np.isfinite(F_np) & (F_np < 8.0 * kT)
        if m.sum() < 10:
            print(f"  {tag}: too few shared cells ({int(m.sum())})")
            return None
        d = (A[m] - A[m].mean()) - (Bm[m] - Bm[m].mean())
        r = dict(n_cells=int(m.sum()), rmse_kT=float(np.sqrt((d ** 2).mean()) / kT),
                 max_kT=float(np.abs(d).max() / kT))
        print(f"  {tag}: RMSE {r['rmse_kT']:.2f} kT, max {r['max_kT']:.2f} kT "
              f"over {r['n_cells']} cells below 8 kT")
        return r

    split = compare(F_a, F_b, "split-half (copies)")
    pair_cmp = {}
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            r = compare(F_per_start[starts[i]], F_per_start[starts[j]],
                        f"psi {math.degrees(starts[i]):+.0f} vs {math.degrees(starts[j]):+.0f}")
            if r is not None:
                pair_cmp[f"{math.degrees(starts[i]):+.0f}_vs_{math.degrees(starts[j]):+.0f}"] = r
    psicmp = (max(pair_cmp.values(), key=lambda r: r["rmse_kT"]) if pair_cmp else None)
    if psicmp is not None:
        print(f"  WORST psi-start pair: RMSE {psicmp['rmse_kT']:.2f} kT")

    np.savez_compressed(
        os.path.join(a.out, "pilot_reference.npz"),
        F=F_np, counts=counts.cpu().numpy(), f_k=f_k.cpu().numpy(),
        overlap=O.cpu().numpy(), nn_overlap=nn, centres=cen_np, psi_start_seed=psi0_np,
        window_of_seed=wid_np, F_split_a=F_a if F_a is not None else np.zeros(0),
        F_split_b=F_b if F_b is not None else np.zeros(0),
        F_psi_pos=F_p1 if F_p1 is not None else np.zeros(0),
        F_psi_neg=F_p2 if F_p2 is not None else np.zeros(0),
        # The MBAR-weighted sample set itself.  This is the only Boltzmann-weighted (phi, psi,
        # chi1) ensemble the study has, so it -- not the relaxation cloud of the state map -- is
        # what the GLOBAL p(psi | phi, chi1) check must be run on.
        mbar_phi=phi_all[it].cpu().numpy().astype(np.float32),
        mbar_psi=psi_all[it].cpu().numpy().astype(np.float32),
        mbar_chi1=chi_all[it].cpu().numpy().astype(np.float32),
        mbar_logw=logw.cpu().numpy(), mbar_seed=sel_seed.astype(np.int32),
        mbar_psi_start=psi0[seed_of_sample[idx]].astype(np.float32),
        window_of_walker=wid, psi_start_of_walker=psi0)

    meta = {
        "stage": "S-pilot: coarse screening reference F_pilot(phi, chi1)",
        "IS_NOT_PUBLICATION_QUALITY": True,
        "purpose": "provisional target populations for the V3 establishment metric only",
        "param_hash": phash, "cuda_visible_devices": cvd, "device": device,
        "physical_model": accepted.PHYSICAL_MODEL,
        "dt_ps": dt, "dt_reason": "restrained; 1 fs under-integrates the dihedral clamp",
        "windows_per_axis": a.windows, "n_windows": a.windows ** 2,
        "n_seeds": int(n_seeds), "copies_per_seed": a.copies, "batch": int(B),
        "kappa": kappa, "sigma_deg": math.degrees(sigma),
        "sigma_over_spacing": sigma / spacing,
        "psi_starts_deg": list(PSI_STARTS_DEG), "psi_restrained": False,
        "randomize_ps": a.randomize_ps, "equil_ps": a.equil_ps, "prod_ps": a.prod_ps,
        "save_every": a.save_every, "frames": int(frames), "seed": a.seed,
        "n_grid": a.n_grid, "mbar_iterations": int(n_it), "mbar_residual": float(resid),
        "mbar_samples": int(len(idx)),
        "nn_overlap": {"min": float(nn.min()), "p1": float(np.percentile(nn, 1)),
                       "median": float(np.median(nn)),
                       "n_below_0p03": int((nn < 0.03).sum()), "n_pairs": int(nn.size)},
        "grid_cells_filled": filled, "grid_cells": a.n_grid ** 2,
        "mean_temperature_K": float(temps.mean()),
        "temperature_deviation_frac": float(t_dev),
        "split_half": split, "psi_start_agreement": psicmp,
        "psi_start_pairwise": pair_cmp,
        "psi_start_memory": memory, "psi_start_memory_spread": float(spread),
        "n_dropped_seeds": len(dropped), "dropped_seeds": dropped[:50],
        "wall_seconds": wall, "git": git_info(),
    }
    meta["config_hash"] = hashlib.md5(
        json.dumps({k: v for k, v in meta.items() if k not in ("git", "wall_seconds")},
                   sort_keys=True, default=str).encode()).hexdigest()[:12]
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"\nwrote {a.out}/meta.json, samples.npz and pilot_reference.npz")


if __name__ == "__main__":
    main()
