"""PNAS-2005-type ERI / united-atom ethane model: parameters, framework neighbour lists, numba force kernels, CV.

Model (docs/eri_campaign/MODEL_REPORT.md, review/LIT_CROSSCHECK.md Sec. 2; a LABELLED SURROGATE of Dubbeldam,
Beerdsen, Calero & Smit, PNAS 102, 12317 (2005)):

    U(q) = sum_molecules (k/2)(|q_m0 - q_m1| - r0)^2
         + sum_beads sum_framework atoms [V_t(r) - V_t(rc)] 1{r < rc}             (t = O or Si)
         + sum_{bead pairs of DIFFERENT molecules} [V_CC(r) - V_CC(rc)] 1{r < rc}
    V(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6], minimum image in the orthorhombic box (every width > 2 rc)

    eps(O-CH3)/kB 142 K, sigma 3.17 A;  eps(Si-CH3)/kB 82 K, sigma 2.12 A;  eps(CH3-CH3)/kB 104 K, sigma 3.78 A
    (Schuering et al. JCP 116, 10890 (2002) Table I, as quoted by PNAS 2005); rc 12 A, energy-shifted, no tail;
    harmonic bond k/kB 96 500 K/A^2, r0 1.54 A (Dubbeldam et al. 2004); no charges; rigid all-silica framework.

Units: kJ/mol, Angstrom; overdamped time unit t.u. = A^2 mol/kJ (unit mobility, D = kT).

Reaction coordinate (the tagged molecule m = 0, COM with equal bead masses):
    phi = wrap(2 pi (X_com - a/4) / P),   P = a/2,   X along the hexagonal a1 axis,
so the between-row 8-ring windows (fractional (0, 1/2, z), (1/2, 1/2, z)) sit at phi = 0 and the ERI-cage rows at
phi = +-pi.  Linear CV: grad phi = (pi / P) e_x on each bead, |grad phi|^2 = 2 (pi/P)^2, local mean force
f = -(P / 2 pi)(F_0x + F_1x), no geometric (divergence) term.

Framework neighbour lists: the orthorhombic unit cell (a, sqrt(3) a, c) is cut into voxels of side <= h; voxel v lists
every framework atom (per type) whose minimum-image distance to the voxel centre is < rc + half the voxel diagonal, at
that image's absolute coordinate, so every atom within rc of any point of the voxel is in the list (EXACT, no
approximation; tested against brute force).  The inner loop is branch-free (cutoff as a 0/1 weight) and its sums
use fastmath 'reassoc' only, so results are bitwise reproducible per numba compile target (as lta_ladder_numba).
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit, prange

from . import framework as fwm

KB = 0.008314462618
TWO_PI = 2.0 * math.pi

PARAMS = dict(
    T_K=600.0,
    eps_O_K=142.0, sig_O=3.17,
    eps_Si_K=82.0, sig_Si=2.12,
    eps_CC_K=104.0, sig_CC=3.78,
    rc=12.0, shift=True,
    k_bond_K=96500.0, r0=1.54,
)


def lj_shift(eps, sig, rc):
    s6 = (sig / rc) ** 6
    return 4.0 * eps * (s6 * s6 - s6)


def pack_pair_params(p=PARAMS):
    """pp[t] = (4 eps, sigma^2, V(rc)) for t = 0 (O), 1 (Si), 2 (CH3-CH3)."""
    out = np.zeros((3, 3))
    for t, (e, s) in enumerate(((p["eps_O_K"], p["sig_O"]), (p["eps_Si_K"], p["sig_Si"]),
                                (p["eps_CC_K"], p["sig_CC"]))):
        eps = e * KB
        out[t] = (4.0 * eps, s * s, lj_shift(eps, s, p["rc"]) if p["shift"] else 0.0)
    return out


@njit(cache=True, error_model="numpy")
def _build_cell_voxels(P, T, box, cell, nvox, rcut, counts_only, start, X, Y, Z):
    """Voxels of the ORTHORHOMBIC UNIT CELL ``cell`` (a, sqrt(3) a, c).  Voxel v lists every framework atom of type
    ``T`` whose minimum-image distance to the voxel centre is < rcut, stored at that image's absolute coordinate
    (centre + d), so a bead reduced into the unit cell needs no minimum image: d = bead - X[k]."""
    hx, hy, hz = cell[0] / nvox[0], cell[1] / nvox[1], cell[2] / nvox[2]
    rc2 = rcut * rcut
    for ix in range(nvox[0]):
        cx = (ix + 0.5) * hx
        for iy in range(nvox[1]):
            cy = (iy + 0.5) * hy
            for iz in range(nvox[2]):
                cz = (iz + 0.5) * hz
                v = (ix * nvox[1] + iy) * nvox[2] + iz
                n0 = 0 if counts_only else start[v]
                c = 0
                for j in range(P.shape[0]):
                    dx = P[j, 0] - cx
                    dx -= box[0] * np.rint(dx / box[0])
                    dy = P[j, 1] - cy
                    dy -= box[1] * np.rint(dy / box[1])
                    dz = P[j, 2] - cz
                    dz -= box[2] * np.rint(dz / box[2])
                    if dx * dx + dy * dy + dz * dz < rc2:
                        if not counts_only:
                            X[n0 + c] = cx + dx
                            Y[n0 + c] = cy + dy
                            Z[n0 + c] = cz + dz
                        c += 1
                if counts_only:
                    start[v + 1] = c


class Framework:
    """Framework atoms + exact unit-cell voxel neighbour lists (per atom type), packed for the kernels.

    Exactness: the box is a (3, 2, 3) supercell of the orthorhombic unit cell (a, sqrt(3) a, c) and rc < L/2, so the
    minimum-image bead-atom sum over the box equals the sum over the infinite crystal within rc, which is what the
    unit-cell voxel lists evaluate (every atom within rc of a point of voxel v is within rc + half-diagonal of its
    centre; the stored image is the one within rc of the bead because L > 2 (rc + half-diagonal))."""

    def __init__(self, variant, rc=PARAMS["rc"], h=1.0, include_si=True):
        fw = fwm.load(variant)
        self.variant = variant
        self.a = float(fw["a"])
        # self.c is used ONLY as the length of the hidden-variable diagnostic eta = cos(4 pi Z_com / c) (ERI: the
        # crystal c, so eta = +1 at window heights); a generic framework may set "eta_len" in its npz
        self.c = float(fw["eta_len"]) if "eta_len" in fw else float(fw["c"])
        self.box = np.ascontiguousarray(fw["box"], dtype=np.float64)
        self.pos = np.ascontiguousarray(fw["pos"], dtype=np.float64)
        self.types = np.ascontiguousarray(fw["types"], dtype=np.int64)
        self.cages = fw["cages"]
        self.windows = fw["windows"]
        reps = np.asarray(fw["reps"], dtype=np.int64)
        self.cell = np.ascontiguousarray(self.box / reps)
        if "cell" in fw:                           # generic orthorhombic framework (overnight extension)
            assert np.allclose(self.cell, fw["cell"])
        else:                                      # ERI (hexagonal a1, sqrt(3) a, c orthorhombic cell)
            assert np.allclose(self.cell, [self.a, math.sqrt(3.0) * self.a, self.c])
        assert rc < 0.5 * self.box.min(), "cutoff must fit the minimum image"
        self.nvox = np.ascontiguousarray(np.ceil(self.cell / h).astype(np.int64))
        hv = self.cell / self.nvox
        hd = 0.5 * math.sqrt(float((hv * hv).sum()))
        assert self.box.min() > 2.0 * (rc + hd) + 1e-6, "stored images must be unique within rc"
        rcut = rc + hd + 1e-9
        nv = int(np.prod(self.nvox))
        self.lists = []
        for t in (0, 1):
            Pt = np.ascontiguousarray(self.pos[self.types == t])
            if t == 1 and not include_si:          # O-only force fields: no Si candidates at all
                Pt = np.zeros((0, 3))
            start = np.zeros(nv + 1, dtype=np.int64)
            dummy = np.zeros(1)
            _build_cell_voxels(Pt, t, self.box, self.cell, self.nvox, rcut, True, start, dummy, dummy, dummy)
            start = np.cumsum(start)
            X = np.zeros(int(start[-1]))
            Y = np.zeros(int(start[-1]))
            Z = np.zeros(int(start[-1]))
            _build_cell_voxels(Pt, t, self.box, self.cell, self.nvox, rcut, False, start, X, Y, Z)
            self.lists.append((start, X, Y, Z))
        self.P = float(fw["P"]) if "P" in fw else self.a / 2.0                    # CV period
        self.x_window = float(fw["x_window"]) if "x_window" in fw else self.a / 4.0  # X of the phi = 0 windows

    def packed(self):
        (s0, x0, y0, z0), (s1, x1, y1, z1) = self.lists
        return (self.cell, self.nvox, s0, x0, y0, z0, s1, x1, y1, z1)


# ----------------------------------------------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------------------------------------------
@njit(cache=True, fastmath={"reassoc", "contract"}, error_model="numpy")
def _type_sum(xr, yr, zr, k0, k1, X, Y, Z, a4, s2t, vsh, rc2):
    e = 0.0
    fx = 0.0
    fy = 0.0
    fz = 0.0
    for k in range(k0, k1):
        dx = xr - X[k]
        dy = yr - Y[k]
        dz = zr - Z[k]
        r2 = dx * dx + dy * dy + dz * dz
        inv = 1.0 / r2
        s2 = s2t * inv
        s6 = s2 * s2 * s2
        w = 1.0 if r2 < rc2 else 0.0
        e += w * (a4 * (s6 * s6 - s6) - vsh)
        c = w * a4 * (12.0 * s6 * s6 - 6.0 * s6) * inv
        fx += c * dx
        fy += c * dy
        fz += c * dz
    return e, fx, fy, fz


@njit(cache=True, error_model="numpy")
def bead_fw(x, y, z, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2):
    """(energy, fx, fy, fz) of one CH3 bead with the framework (O + Si, energy-shifted LJ, exact within rc)."""
    xr = x - cell[0] * math.floor(x / cell[0])
    yr = y - cell[1] * math.floor(y / cell[1])
    zr = z - cell[2] * math.floor(z / cell[2])
    ix = int(xr * nvox[0] / cell[0])
    iy = int(yr * nvox[1] / cell[1])
    iz = int(zr * nvox[2] / cell[2])
    if ix >= nvox[0]:
        ix = nvox[0] - 1
    if iy >= nvox[1]:
        iy = nvox[1] - 1
    if iz >= nvox[2]:
        iz = nvox[2] - 1
    v = (ix * nvox[1] + iy) * nvox[2] + iz
    e0, fx0, fy0, fz0 = _type_sum(xr, yr, zr, s0[v], s0[v + 1], x0, y0, z0, pp[0, 0], pp[0, 1], pp[0, 2], rc2)
    e1, fx1, fy1, fz1 = _type_sum(xr, yr, zr, s1[v], s1[v + 1], x1, y1, z1, pp[1, 0], pp[1, 1], pp[1, 2], rc2)
    return e0 + e1, fx0 + fx1, fy0 + fy1, fz0 + fz1


@njit(cache=True, error_model="numpy")
def config_energy_force(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F):
    """Total U (kJ/mol) of one replica configuration q (M, 2, 3) (unwrapped), forces into F (M, 2, 3).
    Returns (U_total, U_framework, U_bond, U_guestguest)."""
    M = q.shape[0]
    ufw = 0.0
    ub = 0.0
    ugg = 0.0
    for m in range(M):
        for b in range(2):
            e, fx, fy, fz = bead_fw(q[m, b, 0], q[m, b, 1], q[m, b, 2], cell, nvox, s0, x0, y0, z0, s1, x1, y1,
                                    z1, pp, rc2)
            ufw += e
            F[m, b, 0] = fx
            F[m, b, 1] = fy
            F[m, b, 2] = fz
        dx = q[m, 0, 0] - q[m, 1, 0]
        dy = q[m, 0, 1] - q[m, 1, 1]
        dz = q[m, 0, 2] - q[m, 1, 2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        ub += 0.5 * kb * (r - r0) * (r - r0)
        c = -kb * (r - r0) / r
        F[m, 0, 0] += c * dx
        F[m, 0, 1] += c * dy
        F[m, 0, 2] += c * dz
        F[m, 1, 0] -= c * dx
        F[m, 1, 1] -= c * dy
        F[m, 1, 2] -= c * dz
    # guest-guest CH3-CH3 (different molecules only)
    a4 = pp[2, 0]
    s2c = pp[2, 1]
    vsh = pp[2, 2]
    for m in range(M):
        for b in range(2):
            for n in range(m + 1, M):
                for d in range(2):
                    dx = q[m, b, 0] - q[n, d, 0]
                    dx -= box[0] * np.rint(dx / box[0])
                    dy = q[m, b, 1] - q[n, d, 1]
                    dy -= box[1] * np.rint(dy / box[1])
                    dz = q[m, b, 2] - q[n, d, 2]
                    dz -= box[2] * np.rint(dz / box[2])
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < rc2:
                        s2 = s2c / r2
                        s6 = s2 * s2 * s2
                        ugg += a4 * (s6 * s6 - s6) - vsh
                        c = a4 * (12.0 * s6 * s6 - 6.0 * s6) / r2
                        F[m, b, 0] += c * dx
                        F[m, b, 1] += c * dy
                        F[m, b, 2] += c * dz
                        F[n, d, 0] -= c * dx
                        F[n, d, 1] -= c * dy
                        F[n, d, 2] -= c * dz
    return ufw + ub + ugg, ufw, ub, ugg


@njit(cache=True, error_model="numpy")
def cv_phi(q, P, xw):
    """phi of the tagged molecule (m = 0), in [-pi, pi)."""
    x = 0.5 * (q[0, 0, 0] + q[0, 1, 0])
    p = TWO_PI * (x - xw) / P
    return (p + math.pi) - TWO_PI * math.floor((p + math.pi) / TWO_PI) - math.pi


@njit(cache=True, error_model="numpy")
def local_mean_force(F, P):
    """f = -(F . grad phi)/|grad phi|^2 for the linear CV: -(P / 2 pi)(F_0x + F_1x) of the tagged molecule."""
    return -(P / TWO_PI) * (F[0, 0, 0] + F[0, 1, 0])


@njit(cache=True, parallel=True, error_model="numpy")
def batch_energy_force(Q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F, U):
    """Q (N, M, 2, 3) -> F (N, M, 2, 3), U (N, 4) (total, framework, bond, guest-guest); replicas in parallel
    (each replica is independent: the result does not depend on the thread count)."""
    for i in prange(Q.shape[0]):
        ut, uf, ub, ug = config_energy_force(Q[i], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb,
                                             r0, F[i])
        U[i, 0] = ut
        U[i, 1] = uf
        U[i, 2] = ub
        U[i, 3] = ug


class Model:
    """Convenience wrapper: framework + parameters + kernels."""

    def __init__(self, variant="schlenker", T_K=None, params=None, h=1.0):
        self.p = dict(PARAMS if params is None else params)
        if T_K is not None:
            self.p["T_K"] = float(T_K)
        self.fw = Framework(variant, rc=self.p["rc"], h=h, include_si=self.p["eps_Si_K"] > 0.0)
        self.pp = pack_pair_params(self.p)
        self.rc2 = self.p["rc"] ** 2
        self.kb = self.p["k_bond_K"] * KB
        self.r0 = self.p["r0"]
        self.kT = KB * self.p["T_K"]

    def args(self):
        f = self.fw
        return (f.box,) + f.packed() + (self.pp, self.rc2, self.kb, self.r0)

    def energy_force(self, q):
        q = np.ascontiguousarray(q, dtype=np.float64)
        F = np.zeros_like(q)
        U = config_energy_force(q, *self.args(), F)
        return U, F

    def batch(self, Q):
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        F = np.zeros_like(Q)
        U = np.zeros((Q.shape[0], 4))
        batch_energy_force(Q, *self.args(), F, U)
        return U, F

    def phi(self, q):
        return cv_phi(np.ascontiguousarray(q, dtype=np.float64), self.fw.P, self.fw.x_window)


# ----------------------------------------------------------------------------------------------------------------
# independent brute-force reference (numpy, no neighbour lists) -- used only by the Gate 0 tests
# ----------------------------------------------------------------------------------------------------------------
def brute_energy(q, fw_pos, fw_types, box, p=PARAMS):
    """U of one configuration q (M, 2, 3) by direct minimum-image sums over every atom (numpy)."""
    q = np.asarray(q, float)
    rc = p["rc"]
    U = 0.0
    eps = {0: p["eps_O_K"] * KB, 1: p["eps_Si_K"] * KB}
    sig = {0: p["sig_O"], 1: p["sig_Si"]}
    for m in range(q.shape[0]):
        for b in range(2):
            d = q[m, b][None, :] - fw_pos
            d -= box * np.round(d / box)
            r = np.sqrt((d * d).sum(1))
            for t in (0, 1):
                rr = r[(fw_types == t) & (r < rc)]
                sr6 = (sig[t] / rr) ** 6
                U += (4 * eps[t] * (sr6 ** 2 - sr6)).sum() - len(rr) * lj_shift(eps[t], sig[t], rc)
        r = np.linalg.norm(q[m, 0] - q[m, 1])
        U += 0.5 * p["k_bond_K"] * KB * (r - p["r0"]) ** 2
    ecc, scc = p["eps_CC_K"] * KB, p["sig_CC"]
    for m in range(q.shape[0]):
        for n in range(m + 1, q.shape[0]):
            for b in range(2):
                for d_ in range(2):
                    d = q[m, b] - q[n, d_]
                    d -= box * np.round(d / box)
                    r = math.sqrt((d * d).sum())
                    if r < rc:
                        sr6 = (scc / r) ** 6
                        U += 4 * ecc * (sr6 ** 2 - sr6) - lj_shift(ecc, scc, rc)
    return U
