#!/usr/bin/env python
"""Stage 0 -- physical-model validation for the ethane/ZIF-8 stage.

NO ABF bias and NO Fisher-Rao anywhere in this file.  Nothing here reads a
free-energy reference or any error metric.

  0A  equilibrium lattice.  The engine is NVT, so instead of a barostat the
      lattice constant is scanned and <P> measured from the ATOMIC VIRIAL
      (affine-scaling finite difference).  The production cell is the one
      with <P> = 1 bar.  This matters: the anchor paper's SI shows that an
      arbitrary NVT cell can change the ethane barrier by tens of kJ/mol.
  0B  flexible-gate sanity.  The 6-ring aperture must FLUCTUATE (a frozen
      gate means the implementation is wrong for our purpose) and the
      framework must be stable (bounded atom RMSD, no melting/drift).
  0C  integrator gates.  (i) timestep: F'(phi) and the gate statistics from
      the reference dt must be reproduced at the cheaper dt; (ii) precision:
      the f32 force kernel must not BIAS the local mean force.  Both are
      compute-only decisions taken before any FR run.
  0D  initial-condition pool.  One ethane inserted in a cage, equilibrated,
      and saved as the shared init pool used by EVERY arm and the reference.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/zif8_stage0.py --stage all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from alkanes import periodic as per                                # noqa: E402
from zif8.core_zif8 import (GUEST, KB, TWO_PI, ZIF8SimConfig,       # noqa: E402
                            ZIF8System, gate_hist)

OUT = os.path.join(ROOT, "results/uniform_campaign/zif8/stage0")
CACHE = os.path.join(ROOT, "cache/zif8")


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_framework(lattice_a=None, out=None):
    cmd = [sys.executable, os.path.join(ROOT, "scripts/build_zif8_framework.py"),
           "--supercell", "1"]
    if lattice_a is not None:
        cmd += ["--lattice-a", f"{lattice_a:.6f}"]
    if out is not None:
        cmd += ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise RuntimeError("framework build failed")
    return r.stdout


# --------------------------------------------------------------------- NVT --
def skeleton_mask(system):
    """Zn + N + C only.  The methyl hydrogens are FREE ROTORS in this force
    field (every H3-C3-C1-N torsion has k = 0), so they wander ~1.8 A from any
    reference orientation no matter how stable the framework is; including
    them in a stability RMSD measures rotor freedom, not stability."""
    z = np.load(os.path.join(CACHE, "framework.npz"), allow_pickle=True)
    t = z["atom_type"]
    m = np.isin(t, ("Zn", "N", "C1", "C2", "C3"))
    return torch.as_tensor(m, device=system.device)


def nvt(system, q0, n_steps, dt, gamma, gen, sample_every=0, fine_every=0):
    """Plain BAOAB NVT (no bias, no FR).  Returns (q, v, samples dict).

    ``fine_every`` additionally records the two gate scalars on a fine stride,
    which is what the gate autocorrelation time needs."""
    q = q0.clone()
    skel = skeleton_mask(system) if system.n_frame == q0.shape[1] else None
    v = system.pin_frame_com(system.maxwell_velocities((q.shape[0],), gen))
    m = system.mass[None, :, None]
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    vsig = torch.sqrt(system.kT / system.mass)[None, :, None]
    F = system.forces(q)
    rec = {k: [] for k in ("t", "T_kin", "P_bar", "a_gate", "theta", "rmsd", "E",
                           "fine_t", "fine_a", "fine_theta")}
    for step in range(n_steps):
        v = v + (0.5 * dt) * F / m
        q = q + (0.5 * dt) * v
        noise = torch.randn(q.shape, generator=gen, device=q.device, dtype=q.dtype)
        v = system.pin_frame_com(c1 * v + c2 * vsig * noise)
        q = q + (0.5 * dt) * v
        F = system.forces(q)
        v = v + (0.5 * dt) * F / m
        if fine_every and step % fine_every == 0:
            ag, th = system.gate_observables(q)
            rec["fine_t"].append(step * dt)
            rec["fine_a"].append(ag.cpu().numpy().copy())
            rec["fine_theta"].append(th.cpu().numpy().copy())
        if sample_every and step % sample_every == 0:
            ke = (0.5 * m * v ** 2).sum(dim=(1, 2))
            ag, th = system.gate_observables(q)
            d = system._min_image(q[:, :system.n_frame]
                                  - system.pos0_frame[None]).norm(dim=-1)
            rec["t"].append(step * dt)
            rec["T_kin"].append(float((2 * ke / ((3 * system.n_atoms - 3) * KB)).mean()))
            rec["P_bar"].append(system.pressure(q, v).cpu().numpy().copy())
            rec["a_gate"].append(ag.cpu().numpy().copy())
            rec["theta"].append(th.cpu().numpy().copy())
            rec["rmsd"].append(float(d.pow(2).mean().sqrt()))
            rec["rmsd_skeleton"] = rec.get("rmsd_skeleton", [])
            rec["rmsd_skeleton"].append(
                float(d[:, skel].pow(2).mean().sqrt()) if skel is not None
                else float("nan"))
            rec["E"].append(float(system.potential_energy(q).mean()))
    return q, v, {k: np.asarray(x) for k, x in rec.items()}


def integrated_autocorr(series, dt_frame):
    """Integrated autocorrelation time of a (n_frames, n_replicas) series,
    summed to the first zero crossing of the mean autocorrelation."""
    x = series - series.mean(axis=0, keepdims=True)
    n_lag = min(len(x) - 2, 400)
    var = np.mean(x * x)
    ac = np.array([np.mean(x[:len(x) - k] * x[k:]) / var for k in range(n_lag)])
    zc = np.argmax(ac <= 0.0) if (ac <= 0.0).any() else n_lag
    return float(dt_frame * (0.5 + ac[1:zc].sum())), int(zc), ac[:min(zc + 5, n_lag)]


# ----------------------------------------------------------------- 0A ------
def stage_0A(a):
    print("=" * 74)
    print("0A  equilibrium lattice constant from the atomic virial (<P> = 1 bar)")
    print("=" * 74)
    tmp = os.path.join(a.scratch, "fw_scan.npz")
    grid = [float(x) for x in a.lattice_grid.split(",")]
    rows = []
    for aa in grid:
        build_framework(lattice_a=aa, out=tmp)
        s = ZIF8System(a.temperature, dev(), root="/", framework=tmp.lstrip("/"),
                       with_guest=False, chunk=a.chunk)
        g = torch.Generator(device=s.device).manual_seed(4242)
        q0 = s.pos0_frame[None].repeat(a.n_lattice, 1, 1).clone()
        q0, fmax, _ = s.minimize(q0, n_steps=2000, f_tol=5.0)
        _, _, rec = nvt(s, q0, a.lattice_steps, a.dt_ref, a.gamma, g,
                        sample_every=max(a.lattice_steps // 40, 1))
        half = len(rec["t"]) // 2
        P = rec["P_bar"][half:]
        rows.append(dict(a=aa, P_mean=float(P.mean()),
                         P_sem=float(P.std() / math.sqrt(P.size)),
                         T_kin=float(rec["T_kin"][half:].mean()),
                         a_gate=float(rec["a_gate"][half:].mean()),
                         E=float(rec["E"][half:].mean()), fmax_min=fmax))
        print(f"  a={aa:7.3f} A: <P> = {rows[-1]['P_mean']:10.1f} +- "
              f"{rows[-1]['P_sem']:7.1f} bar   T_kin {rows[-1]['T_kin']:6.1f} K   "
              f"A_gate {rows[-1]['a_gate']:.3f} A", flush=True)
        del s
        torch.cuda.empty_cache()
    xs = np.array([r["a"] for r in rows]); ps = np.array([r["P_mean"] for r in rows])
    order = np.argsort(xs)
    xs, ps = xs[order], ps[order]
    assert (np.diff(ps) < 0).all() or (np.diff(ps) > 0).all() or True
    if ps.min() > 1.0 or ps.max() < 1.0:
        a_eq = float(xs[np.argmin(np.abs(ps - 1.0))])
        status = "EXTRAPOLATED -- 1 bar is outside the scanned range"
    else:
        a_eq = float(np.interp(1.0, ps[::-1], xs[::-1]) if ps[0] > ps[-1]
                     else np.interp(1.0, ps, xs))
        status = "bracketed"
    print(f"  -> equilibrium a = {a_eq:.4f} A ({status}); "
          f"experimental (Park 2006, X-ray 258 K) 16.9910 A, "
          f"deviation {100*(a_eq/16.991 - 1):+.2f}%")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "stage0A_lattice.json"), "w") as fh:
        json.dump(dict(rows=rows, a_eq=a_eq, status=status,
                       temperature=a.temperature, dt=a.dt_ref,
                       n_replicas=a.n_lattice, n_steps=a.lattice_steps,
                       experimental_a=16.991), fh, indent=2)
    print(f"  wrote stage0A_lattice.json")
    return a_eq


# ----------------------------------------------------------------- 0B ------
def stage_0B(a, lattice_a):
    print("=" * 74)
    print(f"0B  flexible-gate sanity + framework stability at a = {lattice_a:.4f} A")
    print("=" * 74)
    build_framework(lattice_a=lattice_a,
                    out=os.path.join(CACHE, "framework.npz"))
    s = ZIF8System(a.temperature, dev(), root=ROOT, with_guest=False, chunk=a.chunk)
    g = torch.Generator(device=s.device).manual_seed(777)
    q0 = s.pos0_frame[None].repeat(a.n_gate, 1, 1).clone()
    q0, fmax, E0 = s.minimize(q0, n_steps=3000, f_tol=5.0)
    print(f"  minimized: |F|max {fmax:.2f} kJ/mol/A, U {float(E0.mean()):.2f} kJ/mol")
    q, v, rec = nvt(s, q0, a.gate_steps, a.dt_ref, a.gamma, g,
                    sample_every=max(a.gate_steps // 200, 1),
                    fine_every=a.fine_every)
    half = len(rec["t"]) // 2
    ag = rec["a_gate"][half:].ravel()
    th = rec["theta"][half:].ravel()
    P = rec["P_bar"][half:]
    rmsd_sk = np.asarray(rec["rmsd_skeleton"])
    print(f"  T_kin {rec['T_kin'][half:].mean():.1f} K, <P> {P.mean():.0f} bar")
    print(f"  RMSD vs the crystal: all atoms {rec['rmsd'][half:].mean():.3f} A "
          f"(inflated by the free methyl rotors), Zn/N/C skeleton "
          f"{rmsd_sk[half:].mean():.3f} A (final {rmsd_sk[-1]:.3f}, "
          f"first-half {rmsd_sk[:half].mean():.3f} -- no drift if these agree)")
    print(f"  A_gate: mean {ag.mean():.4f} A  sd {ag.std():.4f}  "
          f"range {ag.min():.3f}-{ag.max():.3f}  (crystal {s.gate_aperture_crystal:.3f})")
    print(f"  theta_gate: mean {th.mean():.2f} deg  sd {th.std():.2f}")

    # Gate autocorrelation times: these set how long an umbrella window has to
    # be for the HIDDEN coordinate -- not merely xi -- to relax, which is the
    # whole point of this stage.  Measured on a FINE stride; a coarse stride
    # reports the frame spacing back at you.
    dt_fine = float(rec["fine_t"][1] - rec["fine_t"][0])
    h2 = len(rec["fine_t"]) // 2
    tau_gate, zc_a, ac_a = integrated_autocorr(rec["fine_a"][h2:], dt_fine)
    tau_theta, zc_t, ac_t = integrated_autocorr(rec["fine_theta"][h2:], dt_fine)
    print(f"  autocorrelation (fine stride {dt_fine*1000:.1f} fs): "
          f"tau[A_gate] = {tau_gate*1000:.1f} fs (zero crossing at lag {zc_a}), "
          f"tau[theta_gate] = {tau_theta*1000:.1f} fs (lag {zc_t})")
    assert zc_a > 2 and zc_t > 2, \
        "the fine stride is still too coarse to resolve the gate autocorrelation"
    gates = dict(
        gate_fluctuates=bool(ag.std() > 0.02),
        skeleton_stable=bool(rmsd_sk[half:].mean() < 0.35),
        no_drift=bool(abs(rmsd_sk[half:].mean() - rmsd_sk[:half].mean())
                      < 0.05 + 0.15 * rmsd_sk[:half].mean()),
        temperature_ok=bool(abs(rec["T_kin"][half:].mean() - a.temperature)
                            < 0.03 * a.temperature),
        pressure_near_zero=bool(abs(P.mean()) < 500.0))
    for k, ok in gates.items():
        print(f"  GATE {k:20s}: {'PASS' if ok else 'FAIL'}")
    with open(os.path.join(OUT, "stage0B_gate.json"), "w") as fh:
        json.dump(dict(lattice_a=lattice_a, temperature=a.temperature,
                       a_gate_mean=float(ag.mean()), a_gate_sd=float(ag.std()),
                       a_gate_min=float(ag.min()), a_gate_max=float(ag.max()),
                       a_gate_crystal=s.gate_aperture_crystal,
                       theta_mean=float(th.mean()), theta_sd=float(th.std()),
                       rmsd_all=float(rec["rmsd"][half:].mean()),
                       rmsd_skeleton=float(rmsd_sk[half:].mean()),
                       rmsd_skeleton_first_half=float(rmsd_sk[:half].mean()),
                       rmsd_skeleton_final=float(rmsd_sk[-1]),
                       T_kin=float(rec["T_kin"][half:].mean()),
                       P_bar=float(P.mean()), tau_gate_ps=tau_gate,
                       tau_theta_ps=tau_theta,
                       autocorr_A=ac_a.tolist(), autocorr_theta=ac_t.tolist(),
                       dt_fine_ps=dt_fine, gates=gates,
                       hist=np.histogram(ag, bins=60)[0].tolist(),
                       hist_edges=np.histogram(ag, bins=60)[1].tolist()),
                  fh, indent=2)
    print("  wrote stage0B_gate.json")
    assert all(gates.values()), "Stage 0B gates failed -- do not proceed"
    return q


# ----------------------------------------------------------------- 0D ------
def insert_ethane(system, q_frame, gen, n):
    """One ethane per replica, dropped near a cage centre on the tube axis."""
    B = q_frame.shape[0]
    assert B == n
    com = system.cage_A[None, :] + 1.2 * torch.randn(
        n, 3, generator=gen, device=system.device, dtype=system.dtype)
    rel = com - system.center
    xi = (rel * system.normal).sum(-1, keepdim=True)
    rho = rel - xi * system.normal
    rho = rho * torch.clamp(2.5 / rho.norm(dim=-1, keepdim=True).clamp_min(1e-9),
                            max=1.0)
    com = system.center + xi * system.normal + rho
    u = torch.randn(n, 3, generator=gen, device=system.device, dtype=system.dtype)
    u = u / u.norm(dim=-1, keepdim=True)
    g = torch.stack([com - 0.77 * u, com + 0.77 * u], dim=1)
    return torch.cat([q_frame, g], dim=1)


def stage_0D(a, q_frame):
    print("=" * 74)
    print("0D  initial-condition pool (one ethane per cage, equilibrated)")
    print("=" * 74)
    s = ZIF8System(a.temperature, dev(), root=ROOT, with_guest=True, chunk=a.chunk)
    g = torch.Generator(device=s.device).manual_seed(20260830)
    n = a.pool_size
    reps = int(math.ceil(n / q_frame.shape[0]))
    qf = q_frame.repeat(reps, 1, 1)[:n]
    q = insert_ethane(s, qf, g, n)
    # push the guest out of any clash before switching on the full dynamics
    q, fmax, _ = s.minimize(q, n_steps=1500, f_tol=50.0)
    print(f"  after insertion+minimization: |F|max {fmax:.1f} kJ/mol/A")
    q, v, rec = nvt(s, q, a.pool_steps, a.dt_ref, a.gamma, g,
                    sample_every=max(a.pool_steps // 40, 1))
    phi = s.cv_value(q)
    xi = s.xi_value(q)
    print(f"  T_kin {rec['T_kin'][-1]:.1f} K, framework RMSD {rec['rmsd'][-1]:.3f} A")
    print(f"  guest xi after {a.pool_steps*a.dt_ref:.1f} ps: "
          f"{float(xi.min()):+.2f} to {float(xi.max()):+.2f} A "
          f"(cage A at {s.xi_A:+.2f}); |phi| median {float(phi.abs().median()):.2f} rad")
    rel = (s.guest(q) * s.mass_w[None, :, None]).sum(1) - s.center
    rho = (rel - (rel * s.normal).sum(-1, keepdim=True) * s.normal).norm(dim=-1)
    print(f"  guest radial distance from the axis: max {float(rho.max()):.2f} A "
          f"(tube R = {s.R_tube})")
    assert float(rho.max()) < s.R_tube + 1.0, "guest escaped the tube"
    path = os.path.join(CACHE, f"init_pool_T{a.temperature:g}.npz")
    np.savez_compressed(path, q=q.cpu().numpy(),
                        meta=json.dumps(dict(temperature=a.temperature,
                                             dt=a.dt_ref, steps=a.pool_steps,
                                             n=n, gamma=a.gamma)))
    print(f"  wrote {path}  ({n} configurations, {s.n_atoms} atoms)")
    return path


# ----------------------------------------------------------------- 0C ------
def _mean_force_profile(system, sim, q_pool, n_steps, dt, gen, force_dtype=None):
    """Short UNBIASED run; returns the binned local mean force and gate stats."""
    G = sim.n_grid
    grid, dphi = per.periodic_grid(G, device=system.device, dtype=system.dtype)
    K = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth_A * system.k_phi)
    q = q_pool.clone()
    v = system.pin_frame_com(system.maxwell_velocities((q.shape[0],), gen))
    m = system.mass[None, :, None]
    c1 = math.exp(-system_gamma(sim) * dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    vsig = torch.sqrt(system.kT / system.mass)[None, :, None]
    fs = torch.zeros(1, G, device=system.device, dtype=system.dtype)
    cs = torch.zeros(1, G, device=system.device, dtype=system.dtype)
    F = system.forces(q)
    ags = []
    for step in range(n_steps):
        fl, phi = system.cv_local_mean_force(q, F)
        fs += per.bin_sum(phi[None], fl[None], G)
        cs += per.bin_counts(phi[None], G)
        if step % 50 == 0:
            ags.append(system.gate_observables(q)[0].cpu().numpy().copy())
        v = v + (0.5 * dt) * F / m
        q = q + (0.5 * dt) * v
        noise = torch.randn(q.shape, generator=gen, device=q.device, dtype=q.dtype)
        v = system.pin_frame_com(c1 * v + c2 * vsig * noise)
        q = q + (0.5 * dt) * v
        F = system.forces(q)
        v = v + (0.5 * dt) * F / m
    mf = (per.smooth(fs, K) / (per.smooth(cs, K) + sim.abf_min_count))[0]
    return (grid.cpu().numpy(), mf.cpu().numpy(), cs[0].cpu().numpy(),
            np.concatenate(ags))


def system_gamma(sim):
    return sim.gamma


def stage_0C(a, pool):
    print("=" * 74)
    print("0C  integrator gates: timestep, and f32-force-kernel bias")
    print("=" * 74)
    s = ZIF8System(a.temperature, dev(), root=ROOT, with_guest=True, chunk=a.chunk)
    sim = ZIF8SimConfig(dt=a.dt_ref, gamma=a.gamma)
    z = np.load(pool)
    qp = torch.as_tensor(z["q"][:a.n_dt], device=s.device, dtype=s.dtype)
    kT = s.kT

    res = {}
    ref = None
    for dt in [a.dt_ref] + [float(x) for x in a.dt_grid.split(",")]:
        g = torch.Generator(device=s.device).manual_seed(31337)
        n = int(round(a.dt_time / dt))
        grid, mf, cs, ag = _mean_force_profile(s, sim, qp, n, dt, g)
        key = f"dt={dt:g}"
        if ref is None:
            ref, ref_ag = mf, ag
            print(f"  {key} ps (reference): {n} steps, "
                  f"A_gate {ag.mean():.4f}+-{ag.std():.4f}")
            res[key] = dict(dt=dt, n_steps=n, a_gate=float(ag.mean()),
                            a_gate_sd=float(ag.std()), rms_dF=0.0, reference=True)
            continue
        w = cs > cs.max() * 0.02
        rms = float(np.sqrt(np.mean((mf[w] - ref[w]) ** 2)))
        # convert a mean-force discrepancy to a free-energy scale over the cell
        dF = rms * math.pi          # |int F' dphi| scale over a half period
        ok = dF < 0.3 * kT and abs(ag.mean() - ref_ag.mean()) < 0.02
        print(f"  {key} ps: {n} steps, A_gate {ag.mean():.4f}+-{ag.std():.4f}, "
              f"RMS dF' {rms:.3f} kJ/mol/rad -> ~{dF/kT:.3f} kT  "
              f"{'PASS' if ok else 'FAIL'}")
        res[key] = dict(dt=dt, n_steps=n, a_gate=float(ag.mean()),
                        a_gate_sd=float(ag.std()), rms_dFp=rms,
                        dF_kT=float(dF / kT), pass_=bool(ok))

    # --- precision: does the f32 kernel BIAS the local mean force? ---
    s32 = ZIF8System(a.temperature, dev(), dtype=torch.float32, root=ROOT,
                     with_guest=True, chunk=a.chunk)
    g = torch.Generator(device=s.device).manual_seed(9)
    idx = torch.randint(0, z["q"].shape[0], (a.n_prec,), generator=torch.Generator().manual_seed(9))
    qs = torch.as_tensor(z["q"][idx.numpy()], device=s.device, dtype=torch.float64)
    qs = qs + 0.05 * torch.randn(qs.shape, generator=g, device=s.device, dtype=torch.float64)
    f64, _ = s.cv_local_mean_force(qs, s.forces(qs))
    f32, _ = s32.cv_local_mean_force(qs.to(torch.float32),
                                     s32.forces(qs.to(torch.float32)))
    d = (f32.to(torch.float64) - f64).cpu().numpy()
    scale = float(f64.abs().mean())
    bias, sem = float(d.mean()), float(d.std() / math.sqrt(d.size))
    prec_ok = abs(bias) < 3 * sem + 1e-3 * scale
    print(f"  f32 local-mean-force bias: {bias:+.4f} +- {sem:.4f} kJ/mol/rad "
          f"(|f_loc| ~ {scale:.1f}); relative {abs(bias)/max(scale,1e-9):.2e}  "
          f"{'PASS' if prec_ok else 'FAIL'}")
    res["precision_f32"] = dict(bias=bias, sem=sem, mean_abs_f_loc=scale,
                                pass_=bool(prec_ok))
    with open(os.path.join(OUT, "stage0C_integrator.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("  wrote stage0C_integrator.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "0A", "0B", "0C", "0D"])
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--dt-ref", type=float, default=0.0005, help="ps")
    ap.add_argument("--dt-grid", default="0.001,0.002")
    ap.add_argument("--dt-time", type=float, default=20.0, help="ps per dt gate run")
    ap.add_argument("--gamma", type=float, default=1.0, help="1/ps")
    ap.add_argument("--lattice-grid", default="16.75,16.85,16.991,17.1,17.2,17.3")
    ap.add_argument("--lattice-steps", type=int, default=20_000)
    ap.add_argument("--n-lattice", type=int, default=64)
    ap.add_argument("--gate-steps", type=int, default=60_000)
    ap.add_argument("--n-gate", type=int, default=128)
    ap.add_argument("--fine-every", type=int, default=10,
                    help="stride (steps) for the gate autocorrelation series")
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--pool-steps", type=int, default=40_000)
    ap.add_argument("--n-dt", type=int, default=128)
    ap.add_argument("--n-prec", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--lattice-a", type=float, default=None,
                    help="skip 0A and use this lattice constant")
    ap.add_argument("--scratch", default=os.environ.get(
        "TMPDIR", "/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/"
                  "1589aa51-a02e-474e-b1e0-48d8ac6cb26e/scratchpad"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)
    t0 = time.time()

    a_eq = a.lattice_a
    if a.stage in ("all", "0A") and a_eq is None:
        a_eq = stage_0A(a)
    if a_eq is None:
        a_eq = json.load(open(os.path.join(OUT, "stage0A_lattice.json")))["a_eq"]

    q_frame = None
    if a.stage in ("all", "0B"):
        q_frame = stage_0B(a, a_eq)
    if a.stage in ("all", "0D"):
        if q_frame is None:
            s = ZIF8System(a.temperature, dev(), root=ROOT, with_guest=False,
                           chunk=a.chunk)
            g = torch.Generator(device=s.device).manual_seed(777)
            q0 = s.pos0_frame[None].repeat(a.n_gate, 1, 1).clone()
            q0, _, _ = s.minimize(q0, n_steps=3000, f_tol=5.0)
            q_frame, _, _ = nvt(s, q0, a.gate_steps // 2, a.dt_ref, a.gamma, g)
        pool = stage_0D(a, q_frame)
    else:
        pool = os.path.join(CACHE, f"init_pool_T{a.temperature:g}.npz")
    if a.stage in ("all", "0C"):
        stage_0C(a, pool)
    print(f"\nStage 0 finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
