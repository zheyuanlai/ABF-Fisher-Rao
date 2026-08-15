"""Start-configuration preparation: clearing cage-water overlaps after a cage teleport.

Why this exists (measured 2026-08-15, ladder smoke + dt-gate first read):
teleporting the cages into water equilibrated at another separation buries waters deep inside
the carbon spheres.  A clipped steepest descent (300 steps x 2e-4 nm clamp = 0.06 nm total
reach) cannot clear overlaps of ~0.25 nm; the first dynamics steps then see ~1e9 kJ/mol
forces, a water blows apart, and M-SHAKE's Newton matrix goes singular (reference smoke), or
the trajectory goes NaN and poisons a verdict file (the dt gate's 0.968/1.20 nm spots, whose
NaN means forced the 1 fs fallback while equipartition itself passed at 0.15 K).

The geometric fix is the same one the parity test uses, batched: translate every clashing
water *as a whole molecule* radially off its nearest carbon to a safe distance, iterate (a
push can create a shallow secondary overlap with the other cage), then let the usual clipped
SD and the thermostat handle the remaining soft contacts.  A force guard afterwards refuses
to hand a still-pathological state to dynamics -- exploded states must raise, never sample.
"""
from __future__ import annotations

import torch

MIN_DIST_NM = 0.30       #: a water O closer than this to any carbon is a clash
TARGET_NM = 0.33         #: pushed out to here (sigma_CO is 0.319 nm)
MAX_SAFE_FORCE = 1.0e6   #: kJ/mol/nm; anything above this after relaxation is a defect


def push_waters_off_cages(x, eng, n_iter=3, chunk=256):
    """Translate clashing waters (whole molecule) radially off their nearest carbon.

    ``x``: (B, N, 3), modified in place.  Minimum image is applied per axis with the engine's
    cell.  Iterated ``n_iter`` times; returns the number of pushed waters in the last pass so
    callers can assert convergence (0 on the final iteration for a healthy prep).
    """
    L = eng.pair.L                                         # (3,)
    carbons = torch.cat([eng.cage_a, eng.cage_b])
    o, h1, h2, m = (eng.waters[:, k] for k in range(4))
    last = 0
    for _ in range(n_iter):
        last = 0
        xc = x[:, carbons, :]                              # (B, 120, 3)
        for lo in range(0, o.numel(), chunk):
            hi = min(lo + chunk, o.numel())
            xo = x[:, o[lo:hi], :]                         # (B, c, 3)
            d = xo[:, :, None, :] - xc[:, None, :, :]
            d = d - L * torch.round(d / L)
            r = d.norm(dim=-1)                             # (B, c, 120)
            rmin, jmin = r.min(dim=-1)                     # (B, c)
            clash = rmin < MIN_DIST_NM
            if not bool(clash.any()):
                continue
            last += int(clash.sum())
            dnear = torch.gather(d, 2, jmin[:, :, None, None].expand(-1, -1, 1, 3)).squeeze(2)
            unit = dnear / rmin.clamp_min(1e-6)[:, :, None]
            shift = torch.where(clash[:, :, None],
                                (TARGET_NM - rmin).clamp_min(0.0)[:, :, None] * unit,
                                torch.zeros_like(unit))
            for sites in (o, h1, h2, m):
                x[:, sites[lo:hi], :] += shift
    return last


def assert_relaxed(eng, x, chunk=256, max_force=MAX_SAFE_FORCE):
    """Refuse to hand a pathological state to dynamics: finite forces below the guard."""
    _, f = eng.energy_forces(x, chunk=chunk)
    worst = float(f.abs().max())
    if not torch.isfinite(f).all():
        raise RuntimeError("non-finite forces after preparation; prep defect")
    if worst > max_force:
        raise RuntimeError(f"max |force| {worst:.3e} kJ/mol/nm after preparation exceeds "
                           f"{max_force:.0e}; overlaps not cleared -- prep defect")
    return worst


def drag_cages(eng, dyn, x, xi_from, xi_to, center, gen,
               rate_nm_ps=0.04, clamp=5.0e4, chunk=256):
    """Amendment 16.9: move the cages linearly in xi while the water propagates.

    ``xi_from``/``xi_to``: per-walker (B,) tensors.  Wall duration is set by the longest
    traverse at ``rate_nm_ps``; shorter traverses finish early and hold.  A per-site force
    clamp is active during the drag only; the settle at fixed d afterwards (unclamped) is
    what sets the ensemble.  Ends with the force guard: an unintegrable state raises.
    """
    dt = dyn.dt
    # two-phase schedule: full rate until 0.1 nm remain, then rate/4 for the final approach.
    # Jamming happens in the last stretch of a CLOSING drag, where the remaining waters must
    # escape through a narrowing annulus (measured: 1/16 replicas trapped at the uniform
    # production rate; max|F| 2.98e4 after settle).  Slowing only the approach costs ~15 ps.
    dist = (xi_to - xi_from).abs()
    d_final = torch.minimum(dist, torch.full_like(dist, 0.1))
    n_main = int(torch.ceil((dist - d_final).max() / (rate_nm_ps * dt)).item())
    n_final = int(torch.ceil(d_final.max() / (0.25 * rate_nm_ps * dt)).item())
    v = dyn.maxwell_velocities(x, generator=gen)
    _, f_raw = eng.energy_forces(x, chunk=chunk)
    f = eng.redistribute(f_raw).clamp(-clamp, clamp)
    sgn = torch.sign(xi_to - xi_from)
    xi_break = xi_to - sgn * d_final
    for phase_steps, a, b in ((n_main, xi_from, xi_break), (n_final, xi_break, xi_to)):
        for k in range(1, max(1, phase_steps) + 1):
            frac = min(1.0, k / max(1, phase_steps))
            xi_k = a + (b - a) * frac
            dyn.place_cages(x, xi_k, center)
            eng.compute_vsites(x)
            _, f = dyn.step(x, v, f, generator=gen)
            f = f.clamp(-clamp, clamp)
    dyn.place_cages(x, xi_to, center)
    eng.compute_vsites(x)
    assert_relaxed(eng, x, chunk=chunk)
    return x
