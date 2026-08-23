"""Load the unbiased-MD reference and put it on the campaign grids.

The reference carries its own error bars: statistics are accumulated in
independent blocks, so the block spread gives the precision of F_ref directly
and the campaign can refuse to claim anything below it.
"""
from __future__ import annotations

import math, os

import numpy as np
import torch

from ..grid import Grid1D, cumtrapz


def _periodic_interp(centers, vals, xq):
    """Linear interpolation of a periodic profile sampled at bin centers."""
    n = centers.size
    dx = centers[1] - centers[0]
    pos = (xq - centers[0]) / dx
    i0 = np.floor(pos).astype(int)
    f = pos - i0
    return (1 - f) * vals[i0 % n] + f * vals[(i0 + 1) % n]


def _lin_interp(c, v, xq, periodic):
    """Interpolate a table sampled at uniform centres `c` (last axis) at `xq`."""
    n = c.size
    dx = c[1] - c[0]
    pos = (xq - c[0]) / dx
    i0 = np.floor(pos).astype(int)
    f = pos - i0
    if periodic:
        a, b = v[..., i0 % n], v[..., (i0 + 1) % n]
    else:
        a = v[..., np.clip(i0, 0, n - 1)]
        b = v[..., np.clip(i0 + 1, 0, n - 1)]
        f = np.where((i0 < 0) | (i0 + 1 > n - 1), 0.0, f)
    return (1 - f) * a + f * b


def _from_stratified(Hj, z_grid: Grid1D, nz_out, ny_out, y_lo=-math.pi):
    """Resample a stratified (z, y) table onto (nz_out x ny_out) cells.

    The table already lives in CAMPAIGN coordinates -- z on the campaign grid,
    y in whatever convention the engines report -- so no CV offset is applied.
    """
    nz, ny = Hj.shape
    zc = z_grid.xmin + (np.arange(nz) + 0.5) * z_grid.volume / nz
    yc = y_lo + (np.arange(ny) + 0.5) * 2 * math.pi / ny
    zq = z_grid.xmin + (np.arange(nz_out) + 0.5) * z_grid.volume / nz_out
    yq = y_lo + (np.arange(ny_out) + 0.5) * 2 * math.pi / ny_out
    tmp = _lin_interp(yc, Hj, yq, True)                       # (nz, ny_out)
    return _lin_interp(zc, tmp.T, zq, False).T                # (nz_out, ny_out)


def load_reference(path, grid: Grid1D, y_grid: Grid1D, device, dtype, k=0,
                   cv_shift=0.0, ti_path=None, cond_path=None):
    """Return F_ref, F'_ref, the reference conditional table and the block spread.

    `cv_shift` is the constant offset baked into the CV: a grid node `zeta`
    corresponds to raw dihedral `zeta - cv_shift`, and every table is read there.
    `ti_path`, when present, REPLACES the histogram free energy by a stratified
    constrained-TI one -- needed when the CV has basins unbiased dynamics cannot
    connect (see scripts/mol_ti_reference.py).  The conditionals still come from
    the unbiased run, which samples them correctly wherever it goes.
    """
    d = np.load(path)
    beta = float(d["beta"])
    ctr = d["centers"]
    H1 = d["H1"][:, k]                      # (blocks, nb)
    S0, S1 = d["S0"][:, k], d["S1"][:, k]
    xg = np.linspace(grid.xmin, grid.xmax, grid.n) - cv_shift

    def prof(h):
        p = h / max(h.sum(), 1.0)
        F = -np.log(np.maximum(p, 1e-300)) / beta
        F = _periodic_interp(ctr, F, xg)
        return F - F.mean()

    Fb = np.stack([prof(H1[b]) for b in range(H1.shape[0])])
    F_ref = prof(H1.sum(0))
    blk_sd = Fb.std(0, ddof=1) / math.sqrt(H1.shape[0])

    mf = S1.sum(0) / np.maximum(S0.sum(0), 1e-9)
    Fp_ref = _periodic_interp(ctr, mf, xg)

    T = lambda a: torch.as_tensor(a, device=device, dtype=dtype)
    lp = path.replace("_ref.npz", "_conflib.npz")
    out = {"F_ref": T(F_ref).unsqueeze(0), "Fp_ref": T(Fp_ref).unsqueeze(0),
           "F_blocks": T(Fb), "F_sd": T(blk_sd), "beta": beta,
           "centers": ctr, "gz": grid}
    if os.path.exists(lp):
        L = np.load(lp)
        out["conflib"] = torch.as_tensor(L["lib"], device=device, dtype=torch.float32)
        out["conffill"] = torch.as_tensor(L["fill"], device=device, dtype=torch.long)
    if ti_path is not None and os.path.exists(ti_path):
        ti = np.load(ti_path)
        Fti = ti["F"][-1]                                    # (rows, G_ti)
        cti = np.linspace(-math.pi, math.pi, Fti.shape[-1])   # the TI run's grid
        Fi = np.stack([_periodic_interp(cti[:-1], f[:-1], xg) for f in Fti])
        Fi = Fi - Fi.mean(-1, keepdims=True)
        out["F_ref"] = T(Fi.mean(0)).unsqueeze(0)
        out["F_sd"] = T(Fi.std(0, ddof=1) / math.sqrt(Fi.shape[0]))
        out["F_blocks"] = T(Fi)
        out["F_hist"] = T(F_ref).unsqueeze(0)                # the unbiased one, for cross-check
    if cond_path is not None and os.path.exists(cond_path):
        # a STRATIFIED conditional: every window sampled equally, so the table is
        # populated at the high-F edges of the domain where an unbiased run is
        # thin and its conditional is little better than noise
        Hc = np.load(cond_path)["Hjoint"]
        out["H2"] = T(_from_stratified(Hc, grid, grid.n, y_grid.n))
        out["Hdiag"] = T(np.stack([_from_stratified(Hc, grid, 36, 36)]))
        out["cond_source"] = "stratified"
    elif d["H2"].size:
        H2 = d["H2"].sum(0)                 # (nb, nb) on the raw bin grid
        out["H2"] = T(_regrid2d(H2, ctr, grid, y_grid, cv_shift))
        out["H2_raw"] = T(H2)
        # the coarse diagnostic table the engines compare against
        out["Hdiag"] = T(np.stack([_regrid_coarse(H2, ctr, grid, 36, cv_shift)]))
        out["cond_source"] = "unbiased"
    if "Hpair" in d and d["Hpair"].size:
        # per-mode reference conditionals p(y_k | z), on the campaign grids.
        # These, not the full joint, are what the single-mode Metropolis proposal
        # and the z-resolved conditional diagnostic read.
        Hp = d["Hpair"].sum(0)                       # (n_fib, nb, nb)
        out["Hcond"] = T(np.stack([_regrid2d(Hp[k], ctr, grid, y_grid, cv_shift)
                                   for k in range(Hp.shape[0])]))
        out["Hdiag"] = T(np.stack([_regrid_coarse(Hp[k], ctr, grid, 36, cv_shift)
                                   for k in range(Hp.shape[0])]))
    if "Hjoint" in d and d["Hjoint"].size:
        Hj = d["Hjoint"].sum(0)
        out["Hjoint"] = T(Hj)
        nt = Hj.ndim
        if nt > 2:
            # per-mode reference conditionals for the z-resolved diagnostic
            outs = []
            for kf in range(nt - 1):
                H = Hj
                for ax in reversed(range(nt - 1)):
                    if ax != kf:
                        H = H.sum(axis=ax + 1)
                outs.append(_coarsen(H, 30))
            out["Hdiag"] = T(np.stack(outs))
    return out


def _regrid2d(H2, ctr, z_grid: Grid1D, y_grid: Grid1D, cv_shift=0.0):
    """Reference joint count table onto (gz nodes) x (gy nodes), both periodic.

    `cv_shift` is subtracted on BOTH axes: TorsionCV carries one offset for every
    dihedral it holds, so the secondary CV the engines report is shifted too, and
    reading the reference at the unshifted y silently compares two different
    coordinates (it showed up as D_cond ~ 3.5 nats on alanine).
    """
    yg = np.linspace(y_grid.xmin, y_grid.xmax, y_grid.n) - cv_shift
    zg = np.linspace(z_grid.xmin, z_grid.xmax, z_grid.n) - cv_shift
    tmp = np.stack([_periodic_interp(ctr, H2[i], yg) for i in range(H2.shape[0])])
    return np.stack([_periodic_interp(ctr, tmp[:, j], zg)
                     for j in range(tmp.shape[1])], axis=1)


def _regrid_coarse(H2, ctr, z_grid: Grid1D, nb, cv_shift=0.0, sub=5):
    """(nb x nb) diagnostic table: z on the CAMPAIGN grid's domain, y periodic.

    The run-side histogram counts samples per coarse CELL, so the reference has
    to be INTEGRATED over the same cell, not sampled at its centre -- sampling
    compares a cell average against a point value and inflates the conditional
    KL threefold.
    """
    dz, dy = z_grid.volume / nb, 2 * math.pi / nb
    off = (np.arange(sub) + 0.5) / sub - 0.5
    zc = (z_grid.xmin + (np.arange(nb)[:, None] + 0.5) * dz
          + off[None] * dz - cv_shift).reshape(-1)
    yc = (-math.pi + (np.arange(nb)[:, None] + 0.5) * dy
          + off[None] * dy - cv_shift).reshape(-1)
    tmp = np.stack([_periodic_interp(ctr, H2[i], yc) for i in range(H2.shape[0])])
    out = np.stack([_periodic_interp(ctr, tmp[:, j], zc) for j in range(tmp.shape[1])],
                   axis=1)
    return out.reshape(nb, sub, nb, sub).sum((1, 3))


def _coarsen(H2, nb):
    f = H2.shape[0] // nb
    return H2[: nb * f, : nb * f].reshape(nb, f, nb, f).sum((1, 3))
