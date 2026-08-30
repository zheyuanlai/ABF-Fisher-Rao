#!/usr/bin/env python
"""Stage-0E: how much of this stage's barrier is a 1x1x1 FINITE-SIZE artifact?

The production cell is the 1x1x1 conventional cell, chosen on compute.  In it
the guest's periodic images sit at the SAME crystallographic gate, so the gate
deformation the guest induces is repeated coherently in every cell -- every
symmetry-equivalent gate in the crystal opens at once.  At real dilution only
one opens.  Opening them all coherently costs more elastic energy per gate, so
the 1x1x1 barrier should be an OVERESTIMATE.  The anchor paper used 2x2x2,
where 7 of the 8 gates of that class stay shut.

This measures the effect directly and cheaply: relax the framework around a
guest held fixed, once at the window and once in the cage, in both cells at the
SAME cutoff, and compare the window-minus-cage energies.  Nothing here touches
the sampling question; it exists so the barrier can be reported with its
finite-size caveat quantified instead of asserted.

    CUDA_VISIBLE_DEVICES=3 python -u scripts/zif8_finite_size.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import KB, ZIF8System  # noqa: E402

OUT = os.path.join(ROOT, "results/uniform_campaign/zif8/stage0")


def relax_at_fixed_com(system, q, n_steps=1500, step0=1e-5, frozen_frame=False):
    """Steepest descent with the guest's CENTRE OF MASS pinned.

    The guest is free to ROTATE and stretch; only its COM -- and hence xi and
    the radial position -- is held.  A fixed random orientation instead of a
    relaxed one is not usable here: at the window the energy is violently
    orientation-dependent, so a lab-frame orientation set samples a DIFFERENT
    set relative to each cell's own gate frame, and the difference between
    cells is then orientation-sampling noise rather than finite size.  That is
    exactly what this script's null control caught on its first run.
    """
    q = q.clone()
    w = system.mass_w[None, :, None]
    com0 = (q[:, system.n_frame:] * w).sum(dim=1, keepdim=True)

    def repin(x):
        g = x[:, system.n_frame:]
        x = x.clone()
        x[:, system.n_frame:] = g + (com0 - (g * w).sum(dim=1, keepdim=True))
        return x
    E = system.potential_energy(q)
    step = torch.full((q.shape[0], 1, 1), step0, device=q.device, dtype=q.dtype)
    for _ in range(n_steps):
        F = system.forces(q)
        if frozen_frame:
            F[:, :system.n_frame] = 0.0     # guest orientation only
        q_try = repin(q + step * F)
        E_try = system.potential_energy(q_try)
        ok = (E_try < E)[:, None, None]
        q = torch.where(ok, q_try, q)
        E = torch.where(ok[:, 0, 0], E_try, E)
        step = torch.where(ok, step * 1.15, step * 0.5).clamp(1e-12, 1e-3)
    return q, E


def place(system, xi_values, n_orient=24, seed=0):
    """Guest on the gate axis at each xi, best of n_orient orientations."""
    dev, dt = system.device, system.dtype
    n = len(xi_values)
    g = torch.Generator(device=dev).manual_seed(seed)
    best_q, best_E = None, torch.full((n,), 1e30, device=dev, dtype=dt)
    for _ in range(n_orient):
        u = torch.randn(3, generator=g, device=dev, dtype=dt)
        u = u / u.norm()
        q = torch.zeros(n, system.n_atoms, 3, device=dev, dtype=dt)
        q[:, :system.n_frame] = system.pos0_frame[None]
        com = (system.center[None, :]
               + torch.as_tensor(xi_values, device=dev, dtype=dt)[:, None]
               * system.normal[None, :])
        q[:, system.n_frame + 0] = com - 0.77 * u
        q[:, system.n_frame + 1] = com + 0.77 * u
        E = system.potential_energy(q)
        if best_q is None:
            best_q, best_E = q, E
        else:
            take = (E < best_E)[:, None, None]
            best_q = torch.where(take, q, best_q)
            best_E = torch.minimum(E, best_E)
    return best_q, best_E


def build(supercell, lattice_a, rc, out):
    cmd = [sys.executable, os.path.join(ROOT, "scripts/build_zif8_framework.py"),
           "--supercell", str(supercell), "--lattice-a", f"{lattice_a:.6f}",
           "--rc", f"{rc:.6f}", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise RuntimeError("framework build failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice-a", type=float, default=16.5068)
    ap.add_argument("--rc", type=float, default=None,
                    help="cutoff held FIXED across cells; default = the 1x1x1 value")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-xi", type=int, default=25)
    ap.add_argument("--cutoffs", type=float, nargs="*", default=[8.088, 10.0, 12.0])
    ap.add_argument("--null-tol", type=float, default=1.0,
                    help="kJ/mol tolerance on the rigid-framework null control")
    ap.add_argument("--scratch", default=os.environ.get(
        "TMPDIR", "/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/"
                  "1589aa51-a02e-474e-b1e0-48d8ac6cb26e/scratchpad"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rc = a.rc if a.rc is not None else 0.49 * a.lattice_a
    kT = KB * 300.0
    print(f"Stage 0E: 1x1x1 vs 2x2x2 at a = {a.lattice_a:.4f} A, "
          f"rc = {rc:.3f} A held FIXED across both cells\n")

    rows = {}
    for S in (1, 2):
        fw = os.path.join(a.scratch, f"fw_fs_{S}.npz")
        build(S, a.lattice_a, rc, fw)
        sysm = ZIF8System(300.0, dev, root="/", framework=fw.lstrip("/"),
                          chunk=a.chunk, force_dtype=torch.float32)
        # A profile over the whole period, so the barrier is PEAK MINUS MIN --
        # polarity-independent.  The two cells' builders may pick different
        # (symmetry-equivalent) <111> window variants whose gate polarity is
        # reversed, so any barrier defined by the SIGN of xi is not comparable
        # across cells; a peak-minus-min is.
        xis = np.linspace(-sysm.period / 2, sysm.period / 2, a.n_xi)
        q, _ = place(sysm, xis)
        # NULL CONTROL on the same footing as the measurement: relax the guest
        # ORIENTATION with the framework FROZEN.  Comparing raw random-orientation
        # energies instead leaves orientation-sampling noise in the control but
        # not in the measurement, so the control fails for a reason that has
        # nothing to do with cell size -- which is what happened on the first two
        # runs of this script.
        _, E_rigid = relax_at_fixed_com(sysm, q, n_steps=a.steps, frozen_frame=True)
        qr, E_relax = relax_at_fixed_com(sysm, q, n_steps=a.steps)
        disp = (qr - q)[:, :sysm.n_frame].norm(dim=-1)
        b_rigid = float(E_rigid.max() - E_rigid.min())
        b_relax = float(E_relax.max() - E_relax.min())
        rows[S] = dict(
            n_atoms=int(sysm.n_atoms), box=float(sysm.box[0]),
            barrier_rigid=b_rigid, barrier_relaxed=b_relax,
            xi_peak_relaxed=float(xis[int(E_relax.argmax())]),
            max_frame_disp_at_peak=float(disp[int(E_relax.argmax())].max()),
            profile_relaxed=(E_relax - E_relax.min()).cpu().numpy().tolist(),
            xis=xis.tolist())
        r = rows[S]
        print(f"  {S}x{S}x{S}  ({r['n_atoms']} atoms, box {r['box']:.2f} A)")
        print(f"    rigid-framework barrier (peak-min)   {b_rigid:8.2f} kJ/mol "
              f"= {b_rigid/kT:6.2f} kT")
        print(f"    with the framework relaxed           {b_relax:8.2f} kJ/mol "
              f"= {b_relax/kT:6.2f} kT   (peak at xi={r['xi_peak_relaxed']:+.2f})")
        print(f"    max framework displacement at the peak "
              f"{r['max_frame_disp_at_peak']:.3f} A\n", flush=True)
        del sysm
        torch.cuda.empty_cache()

    d_rigid = rows[1]["barrier_rigid"] - rows[2]["barrier_rigid"]
    d_relax = rows[1]["barrier_relaxed"] - rows[2]["barrier_relaxed"]
    print(f"  FINITE-SIZE EFFECT (1x1x1 minus 2x2x2), same cutoff:")
    print(f"    NULL CONTROL, rigid framework  {d_rigid:+8.2f} kJ/mol "
          f"({d_rigid/kT:+.2f} kT)")
    print(f"      A rigid 2x2x2 cell IS eight copies of the rigid 1x1x1 cell, so "
          f"this must be ~0.\n      If it is not, the two cells are not being "
          f"compared like for like and the\n      flexible number below means "
          f"nothing.")
    ctrl_ok = abs(d_rigid) < a.null_tol
    print(f"      -> {'PASS' if ctrl_ok else 'FAIL'} against a {a.null_tol} kJ/mol "
          f"tolerance")
    print(f"    flexible                       {d_relax:+8.2f} kJ/mol "
          f"({d_relax/kT:+.2f} kT)")
    if ctrl_ok:
        print(f"      -> the coherent gate-opening penalty the 1x1x1 cell pays, "
              f"energy only")
    else:
        print(f"      -> NOT REPORTABLE: the null control failed")
    # --- what does the CUTOFF cost?  Only the 2x2x2 cell can hold 12 A, the
    # value the anchor paper used; the 1x1x1 minimum-image limit is 8.5 A.
    cut = {}
    print("  CUTOFF, measured in the 2x2x2 cell (only it can hold 12 A):")
    for rcx in a.cutoffs:
        fw = os.path.join(a.scratch, f"fw_rc_{rcx:g}.npz")
        build(2, a.lattice_a, rcx, fw)
        sysm = ZIF8System(300.0, dev, root="/", framework=fw.lstrip("/"),
                          chunk=a.chunk, force_dtype=torch.float32)
        xis = np.linspace(-sysm.period / 2, sysm.period / 2, a.n_xi)
        q, _ = place(sysm, xis)
        _, E = relax_at_fixed_com(sysm, q, n_steps=a.steps)
        cut[f"{rcx:g}"] = float(E.max() - E.min())
        print(f"    2x2x2 rc = {rcx:5.2f} A -> relaxed barrier "
              f"{cut[f'{rcx:g}']:7.2f} kJ/mol = {cut[f'{rcx:g}']/kT:5.2f} kT", flush=True)
        del sysm
        torch.cuda.empty_cache()
    ks = sorted(cut, key=float)
    print(f"    cutoff effect {ks[0]} -> {ks[-1]} A: "
          f"{cut[ks[-1]] - cut[ks[0]]:+.2f} kJ/mol")

    res = dict(lattice_a=a.lattice_a, rc=rc, cells=rows, cutoff_sweep_2x2x2=cut,
               null_control_kJmol=d_rigid, null_control_passed=bool(ctrl_ok),
               finite_size_flexible_kJmol=(d_relax if ctrl_ok else None),
               note=("Energy-only (0 K relaxation), so this bounds the ENERGETIC "
                     "part of the finite-size effect, not the entropic part. "
                     "Reported so the 1x1x1 barrier carries a measured caveat "
                     "rather than an asserted one."))
    with open(os.path.join(OUT, "stage0E_finite_size.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote stage0E_finite_size.json")


if __name__ == "__main__":
    main()
