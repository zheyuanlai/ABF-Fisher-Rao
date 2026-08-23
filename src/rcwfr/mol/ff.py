"""Batched molecular mechanics in PyTorch: united-atom alkanes (TraPPE) and
anything else that can be written as bonds + angles + cosine torsions + LJ pairs.

Everything is batched over arbitrary leading dimensions, so one kernel launch
serves (rows x replicas) configurations at once.  On an H200 the inner loop is
launch-bound, not flop-bound, so widening the batch is close to free -- 2^18
pentane walkers cost the same wall-clock per step as 2^10.

Units: kcal/mol, Angstrom, radian, amu.  Coordinates are (..., A, 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

KB = 0.0019872041          # kcal / (mol K)


def _wrap(x):
    """Wrap an angle into [-pi, pi)."""
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi


def dihedral(q, idx, shift=True):
    """Dihedral, optionally rotated by a constant offset.

    `shift=True` adds pi, which puts an alkane's trans minimum at 0 and turns the
    TraPPE/OPLS series into `c1(1-cos) + c2(1-cos2) + c3(1-cos3)`.  A float shifts
    by that many radians instead -- used to move a CV's inaccessible arc to the
    domain boundary, so a cumulative TI integral never has to cross it.

    idx: (T, 4) long.  q: (..., A, 3).  Returns (..., T) in [-pi, pi).

    The shift is phi <- phi_IUPAC + pi, which turns the TraPPE/OPLS torsion
    series into the familiar `c1(1-cos) + c2(1-cos2) + c3(1-cos3)` with the
    trans minimum at the origin -- convenient for a periodic CV grid centred
    on the dominant basin.
    """
    r = q[..., idx, :]                                    # (..., T, 4, 3)
    b1 = r[..., 1, :] - r[..., 0, :]
    b2 = r[..., 2, :] - r[..., 1, :]
    b3 = r[..., 3, :] - r[..., 2, :]
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True)
    m1 = torch.cross(n1, b2n, dim=-1)
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)
    a = torch.atan2(y, x)
    if shift is True:
        return _wrap(a + torch.pi)
    if shift is False or shift == 0.0:
        return _wrap(a)
    return _wrap(a + float(shift))


def angle(q, idx):
    """Bond angle in radians.  idx: (A3, 3)."""
    r = q[..., idx, :]
    v1 = r[..., 0, :] - r[..., 1, :]
    v2 = r[..., 2, :] - r[..., 1, :]
    c = (v1 * v2).sum(-1)
    s = torch.linalg.norm(torch.cross(v1, v2, dim=-1), dim=-1)
    return torch.atan2(s, c)


def bond(q, idx):
    r = q[..., idx, :]
    return torch.linalg.norm(r[..., 1, :] - r[..., 0, :], dim=-1)


def rotate_about_bond(q, j, k, movers, delta):
    """Rotate `movers` about the axis r_k - r_j by `delta`.  q: (..., A, 3).

    For a torsion (i, j, k, l) with `movers` the atoms on the far side of the
    j-k bond, this changes THAT dihedral by exactly delta and leaves every bond
    length, bond angle and other dihedral untouched -- it is a rigid motion of
    the distal fragment.  A SHAKE projection along M^{-1} grad xi does not have
    that property: for a large displacement it buys the constraint by bending
    bonds, and the resulting configuration can be hundreds of kcal/mol uphill.
    """
    axis = q[..., k, :] - q[..., j, :]
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    org = q[..., k, :].unsqueeze(-2)
    v = q[..., movers, :] - org
    a = axis.unsqueeze(-2)
    c = torch.cos(delta).unsqueeze(-1).unsqueeze(-1)
    sn = torch.sin(delta).unsqueeze(-1).unsqueeze(-1)
    dot = (v * a).sum(-1, keepdim=True)
    rot = v * c + torch.cross(a.expand_as(v), v, dim=-1) * sn + a * dot * (1.0 - c)
    out = q.clone()
    out[..., movers, :] = rot + org
    return out


@dataclass
class Topology:
    """A molecular mechanics model.  All tensors live on one device."""
    name: str
    mass: torch.Tensor                 # (A,)
    bond_idx: torch.Tensor             # (B, 2)
    bond_r0: torch.Tensor              # (B,)
    bond_k: torch.Tensor               # (B,)   U = k/2 (r - r0)^2
    ang_idx: torch.Tensor              # (N, 3)
    ang_t0: torch.Tensor
    ang_k: torch.Tensor                # U = k/2 (theta - t0)^2
    tor_idx: torch.Tensor              # (T, 4)
    tor_c: torch.Tensor                # (T, 4): c0..c3, U = sum_n c_n (1 - cos n phi)
    lj_idx: torch.Tensor               # (P, 2)
    lj_eps: torch.Tensor
    lj_sig: torch.Tensor
    lj_scale: torch.Tensor             # (P,) 1-4 scaling
    charge: torch.Tensor = None        # (A,) elementary charge; None = uncharged
    coul_idx: torch.Tensor = None      # (P2, 2)
    coul_scale: torch.Tensor = None

    @property
    def n_atoms(self):
        return self.mass.numel()

    @property
    def device(self):
        return self.mass.device

    @property
    def dtype(self):
        return self.mass.dtype

    # -- energy -------------------------------------------------------------
    def energy(self, q):
        """q: (..., A, 3) -> (...)"""
        e = torch.zeros(q.shape[:-2], device=q.device, dtype=q.dtype)
        if self.bond_idx.numel():
            r = bond(q, self.bond_idx)
            e = e + (0.5 * self.bond_k * (r - self.bond_r0) ** 2).sum(-1)
        if self.ang_idx.numel():
            t = angle(q, self.ang_idx)
            e = e + (0.5 * self.ang_k * (t - self.ang_t0) ** 2).sum(-1)
        if self.tor_idx.numel():
            p = dihedral(q, self.tor_idx)
            n = torch.arange(self.tor_c.shape[-1], device=q.device, dtype=q.dtype)
            e = e + (self.tor_c * (1.0 - torch.cos(n * p.unsqueeze(-1)))).sum(-1).sum(-1)
        if self.lj_idx.numel():
            d = bond(q, self.lj_idx)
            sr6 = (self.lj_sig / d) ** 6
            e = e + (self.lj_scale * 4.0 * self.lj_eps * (sr6 * sr6 - sr6)).sum(-1)
        if self.coul_idx is not None and self.coul_idx.numel():
            d = bond(q, self.coul_idx)
            qq = self.charge[self.coul_idx[:, 0]] * self.charge[self.coul_idx[:, 1]]
            e = e + (self.coul_scale * 332.0637 * qq / d).sum(-1)
        return e

    def grad(self, q):
        """-force.  One backward pass for the whole batch.

        `torch.func.grad` rather than `autograd.grad`, because the latter forces
        a dynamo graph break and the inner loop then costs 4 ms of kernel
        launches instead of 0.3 ms.
        """
        if getattr(self, "_gradfn", None) is None:
            self._gradfn = torch.func.grad(lambda x: self.energy(x).sum())
        return self._gradfn(q)


# ---------------------------------------------------------------------------
# TraPPE-UA alkanes
# ---------------------------------------------------------------------------
_TRAPPE = dict(
    m_ch3=15.0350, m_ch2=14.0270,
    eps_ch3=98.0 * KB, sig_ch3=3.75,
    eps_ch2=46.0 * KB, sig_ch2=3.95,
    r0=1.54, kb=520.0,                    # OPLS-UA C-C: U = K(r-r0)^2, K = 260
    t0=114.0 * torch.pi / 180.0, kt=62500.0 * KB,
    c1=355.03 * KB, c2=-68.19 * KB, c3=791.32 * KB,
)


def ua_alkane(n_c: int, device, dtype) -> Topology:
    """Linear united-atom alkane with n_c pseudo-atoms (n_c=4 butane, 5 pentane).

    Flexible bonds and angles (TraPPE is rigid; a flexible fiber is the point
    here -- a rigid model would leave nothing on Sigma(z) to sample).  Torsions
    are the TraPPE cosine series; intramolecular LJ acts on 1-4 and beyond,
    which is what produces the pentane effect.
    """
    P = _TRAPPE
    t = lambda x: torch.tensor(x, device=device, dtype=dtype)
    L = lambda x: torch.tensor(x, device=device, dtype=torch.long)
    mass = t([P["m_ch3"]] + [P["m_ch2"]] * (n_c - 2) + [P["m_ch3"]])
    eps = t([P["eps_ch3"]] + [P["eps_ch2"]] * (n_c - 2) + [P["eps_ch3"]])
    sig = t([P["sig_ch3"]] + [P["sig_ch2"]] * (n_c - 2) + [P["sig_ch3"]])
    bidx = L([[i, i + 1] for i in range(n_c - 1)])
    aidx = L([[i, i + 1, i + 2] for i in range(n_c - 2)])
    tidx = L([[i, i + 1, i + 2, i + 3] for i in range(n_c - 3)])
    # TraPPE-UA computes intramolecular LJ only for sites separated by FOUR or
    # more bonds; the 1-4 interaction is already inside the torsion series, and
    # adding it as well roughly doubles the cis barrier.  Butane is then left
    # with no pair at all; pentane with exactly one, the 1-5 CH3...CH3 contact
    # that produces the pentane effect -- the reason phi2 is a slow fiber mode
    # strongly coupled to phi1.
    pairs = [(i, j) for i in range(n_c) for j in range(i + 4, n_c)]
    pidx = L(pairs)
    if pidx.numel() == 0:
        pidx = L([]).reshape(0, 2)
        pe = ps = t([])
    else:
        pe = torch.sqrt(eps[pidx[:, 0]] * eps[pidx[:, 1]])
        ps = 0.5 * (sig[pidx[:, 0]] + sig[pidx[:, 1]])
    return Topology(
        name=f"UA-C{n_c}", mass=mass,
        bond_idx=bidx, bond_r0=t([P["r0"]] * (n_c - 1)), bond_k=t([P["kb"]] * (n_c - 1)),
        ang_idx=aidx, ang_t0=t([P["t0"]] * (n_c - 2)), ang_k=t([P["kt"]] * (n_c - 2)),
        tor_idx=tidx,
        tor_c=t([[0.0, P["c1"], P["c2"], P["c3"]]] * (n_c - 3)),
        lj_idx=pidx, lj_eps=pe, lj_sig=ps,
        lj_scale=torch.ones(pidx.shape[0], device=device, dtype=dtype),
    )


def ideal_alkane(top: Topology, n_c: int, phis, device, dtype):
    """Build coordinates from ideal bonds/angles and the given torsions.

    phis: (..., n_c-3).  Returns (..., n_c, 3).  Standard NeRF placement.
    """
    P = _TRAPPE
    b, th = P["r0"], P["t0"]
    batch = phis.shape[:-1]
    q = torch.zeros(batch + (n_c, 3), device=device, dtype=dtype)
    q[..., 1, 0] = b
    q[..., 2, 0] = b - b * torch.cos(torch.tensor(th, device=device, dtype=dtype))
    q[..., 2, 1] = b * torch.sin(torch.tensor(th, device=device, dtype=dtype))
    for k in range(3, n_c):
        # place atom k from atoms k-3,k-2,k-1 with bond b, angle th, torsion phi
        a, bb, c = q[..., k - 3, :], q[..., k - 2, :], q[..., k - 1, :]
        bc = c - bb
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True)
        n = torch.cross(bb - a, bc, dim=-1)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
        m = torch.cross(n, bc, dim=-1)
        phi = -(phis[..., k - 3] + torch.pi)     # undo the trans-at-zero shift
        d2 = torch.stack([-b * torch.cos(torch.tensor(th, device=device, dtype=dtype))
                          * torch.ones_like(phi),
                          b * torch.sin(torch.tensor(th, device=device, dtype=dtype))
                          * torch.cos(phi),
                          b * torch.sin(torch.tensor(th, device=device, dtype=dtype))
                          * torch.sin(phi)], dim=-1)
        q[..., k, :] = c + (d2[..., 0:1] * bc + d2[..., 1:2] * m + d2[..., 2:3] * n)
    return q
