"""Atomistic Ace-Ala-Nme (22 atoms, AMBER ff14SB, vacuum): mollified SHUS (+ FR)
on xi = phi (periodic 1D, psi hidden) or xi = (phi, psi) (periodic 2D).

Physical model ported FROZEN from the closed ABF-Fisher-Rao campaign (see
docs/PROVENANCE.md): torch ff14SB re-implementation constructed from the cached
parameter artifact (param_hash 6ffd00dc241f, verified against OpenMM at
extraction; parity fixtures stored), BAOAB Langevin (dt = 1 fs, gamma = 1 ps^-1,
T = 300 K, float64, no constraints), IUPAC dihedral convention. Units: kJ/mol,
nm, ps, amu, radians.

Reference: the campaign's accepted umbrella + MBAR occupancy FES on the 97x97
cell-centred torus (results/alanine/reference, trimmed artifact vendored here);
the evaluation mask is mask8 (F - Fmin <= 8 kT). The engine grid IS the
reference's cell-centre lattice, so no resampling of the reference ever happens.
The 1D reference F_phi and the conditional p_ref(psi | phi) are derived from the
same 2D surface by exact column sums.

One replica = one molecule. An FR clone copies (q, cached f), inherits ancestry,
and draws FRESH Maxwell momenta (the closed campaign's validated full-state
cloning; exact because rho(q,p) factorizes). Estimator protection as everywhere:
events gather walker arrays only.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..events1p import fr_event1p
from ..events2d import fr_event2
from ..grid import DEVICE, DTYPE, EPS
from ..grid1p import (Grid1P, ShusAccumulator1P, binned_density1p, integral1p,
                      kl_to_uniform1p, tv_to_uniform1p)
from ..grid2d import (GridT2, binned_density2, integral2, kl_to_uniform2,
                      periodic_gaussian_kernel, tv_to_uniform2, wrap_periodic)
from ..resampling import ancestor_stats, surviving_ancestors
from ..shus2d import ShusAccumulator2
from .gateway import Method, _fires_at_block, _schedule_source

REFERENCE_ID = "ala-mbar-umbrella-97 (param 6ffd00dc241f)"
KB = 0.008314462618                  # kJ/mol/K
PI = math.pi
N_GRID = 97
DZ = 2.0 * PI / N_GRID
X0MIN = -PI + 0.5 * DZ               # grid nodes = reference cell centres
ALA_GRID2 = GridT2(x1min=X0MIN, L1=2 * PI, n1=N_GRID,
                   x2min=X0MIN, L2=2 * PI, n2=N_GRID)
ALA_GRID1 = Grid1P(xmin=X0MIN, L=2 * PI, n=N_GRID)

PHI_ATOMS = (4, 6, 8, 14)            # C(ACE) N CA C(ALA)   -- frozen indices
PSI_ATOMS = (6, 8, 14, 16)           # N CA C(ALA) N(NME)
N_ATOMS = 22
REF_DIR = os.path.join(os.path.dirname(__file__), "..", "references")


# -----------------------------------------------------------------------------
# force field (ported TorchFF, built from the cached artifact -- no OpenMM)
# -----------------------------------------------------------------------------
ONE_4PI_EPS0 = 138.935456


class TorchFF:
    """Batched ff14SB energy/forces from the cached parameter artifact."""

    def __init__(self, device=DEVICE, dtype=DTYPE):
        z = np.load(os.path.join(REF_DIR, "ala_ff14sb_vacuum.npz"))
        assert str(z["param_hash"]) == "6ffd00dc241f", "wrong ff parameter artifact"
        t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        L = lambda a: torch.as_tensor(np.asarray(a), device=device,
                                      dtype=torch.long)
        self.bi, self.b0, self.bk = L(z["bonds_idx"]), t(z["bonds_r0"]), t(z["bonds_k"])
        self.ai, self.a0, self.ak = L(z["angles_idx"]), t(z["angles_t0"]), t(z["angles_k"])
        self.ti, self.tn = L(z["torsions_idx"]), t(z["torsions_n"])
        self.tp, self.tk = t(z["torsions_phase"]), t(z["torsions_k"])
        q, sig, eps = z["nb_q"], z["nb_sigma"], z["nb_eps"]
        n = len(q)
        iu, ju = np.triu_indices(n, 1)
        exmap = {(min(int(i), int(j)), max(int(i), int(j))): (qq, ss, ee)
                 for (i, j), qq, ss, ee in zip(z["exc_idx"], z["exc_qq"],
                                              z["exc_sigma"], z["exc_eps"])}
        qq = q[iu] * q[ju]
        ss = 0.5 * (sig[iu] + sig[ju])
        ee = np.sqrt(eps[iu] * eps[ju])
        for k, (i, j) in enumerate(zip(iu, ju)):
            if (int(i), int(j)) in exmap:
                qq[k], ss[k], ee[k] = exmap[(int(i), int(j))]
        keep = ~((qq == 0) & (ee == 0))
        self.pi, self.pj = L(iu[keep]), L(ju[keep])
        self.pq, self.ps, self.pe = t(qq[keep] * ONE_4PI_EPS0), t(ss[keep]), t(ee[keep])
        self.masses = t(z["masses"])
        self.X0 = t(z["X0"])                          # minimised C7eq structure

    def energy(self, x):
        d = x[:, self.bi[:, 0]] - x[:, self.bi[:, 1]]
        E = (0.5 * self.bk * (d.norm(dim=-1) - self.b0) ** 2).sum(-1)
        v1 = x[:, self.ai[:, 0]] - x[:, self.ai[:, 1]]
        v2 = x[:, self.ai[:, 2]] - x[:, self.ai[:, 1]]
        th = torch.atan2(torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1),
                         (v1 * v2).sum(-1))
        E = E + (0.5 * self.ak * (th - self.a0) ** 2).sum(-1)
        p0, p1, p2, p3 = (x[:, self.ti[:, k]] for k in range(4))
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1n = b1 / b1.norm(dim=-1, keepdim=True)
        vv = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
        ww = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
        ang = torch.atan2((torch.linalg.cross(b1n, vv, dim=-1) * ww).sum(-1),
                          (vv * ww).sum(-1))
        E = E + (self.tk * (1.0 + torch.cos(self.tn * ang - self.tp))).sum(-1)
        rp = (x[:, self.pi] - x[:, self.pj]).norm(dim=-1).clamp_min(1e-8)
        sr6 = (self.ps / rp) ** 6
        return E + (4.0 * self.pe * (sr6 * sr6 - sr6) + self.pq / rp).sum(-1)

    def forces(self, x):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            g, = torch.autograd.grad(self.energy(xg).sum(), xg)
        return -g

    def forces_analytic(self, x):
        """Closed-form forces: identical physics to :meth:`forces` (autograd),
        validated against it to ~1e-10 in tests; no autograd graph in the hot
        loop, and safe inside CUDA-graph capture."""
        B = x.shape[0]
        F = torch.zeros_like(x)

        # bonds: E = k/2 (r - r0)^2
        d = x[:, self.bi[:, 0]] - x[:, self.bi[:, 1]]
        r = d.norm(dim=-1).clamp_min(1e-12)
        fb = (-self.bk * (r - self.b0) / r).unsqueeze(-1) * d
        F.index_add_(1, self.bi[:, 0], fb)
        F.index_add_(1, self.bi[:, 1], -fb)

        # angles: E = k/2 (theta - theta0)^2 with theta = atan2(|v1 x v2|, v1.v2)
        v1 = x[:, self.ai[:, 0]] - x[:, self.ai[:, 1]]
        v2 = x[:, self.ai[:, 2]] - x[:, self.ai[:, 1]]
        r1 = v1.norm(dim=-1).clamp_min(1e-12)
        r2 = v2.norm(dim=-1).clamp_min(1e-12)
        u = v1 / r1.unsqueeze(-1)
        w = v2 / r2.unsqueeze(-1)
        cth = (u * w).sum(-1).clamp(-1.0, 1.0)
        sth = torch.sqrt((1.0 - cth * cth).clamp_min(1e-12))
        th = torch.atan2(torch.linalg.cross(v1, v2, dim=-1).norm(dim=-1),
                         (v1 * v2).sum(-1))
        dEdth = self.ak * (th - self.a0)
        dth0 = -(w - cth.unsqueeze(-1) * u) / (r1 * sth).unsqueeze(-1)
        dth2 = -(u - cth.unsqueeze(-1) * w) / (r2 * sth).unsqueeze(-1)
        F.index_add_(1, self.ai[:, 0], -dEdth.unsqueeze(-1) * dth0)
        F.index_add_(1, self.ai[:, 2], -dEdth.unsqueeze(-1) * dth2)
        F.index_add_(1, self.ai[:, 1], dEdth.unsqueeze(-1) * (dth0 + dth2))

        # torsions: E = k (1 + cos(n ang - phase))
        p0, p1, p2, p3 = (x[:, self.ti[:, k]] for k in range(4))
        ang, g0, g1, g2, g3 = dihedral_value_grad_analytic(p0, p1, p2, p3)
        dEdphi = (-self.tk * self.tn * torch.sin(self.tn * ang - self.tp)
                  ).unsqueeze(-1)
        F.index_add_(1, self.ti[:, 0], -dEdphi * g0)
        F.index_add_(1, self.ti[:, 1], -dEdphi * g1)
        F.index_add_(1, self.ti[:, 2], -dEdphi * g2)
        F.index_add_(1, self.ti[:, 3], -dEdphi * g3)

        # pairs: E = 4 eps ((s/r)^12 - (s/r)^6) + qq/r
        dp = x[:, self.pi] - x[:, self.pj]
        rp = dp.norm(dim=-1).clamp_min(1e-8)
        sr6 = (self.ps / rp) ** 6
        dEdr = (4.0 * self.pe * (-12.0 * sr6 * sr6 + 6.0 * sr6) / rp
                - self.pq / rp ** 2)
        fp = (-dEdr / rp).unsqueeze(-1) * dp
        F.index_add_(1, self.pi, fp)
        F.index_add_(1, self.pj, -fp)
        return F


# -----------------------------------------------------------------------------
# collective variables (IUPAC convention; values + Cartesian gradients)
# -----------------------------------------------------------------------------
def dihedral_torch(p0, p1, p2, p3):
    """Signed dihedral, IUPAC (trans at pi); inputs (..., 3) -> (...)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.linalg.cross(b1n, v, dim=-1) * w).sum(-1),
                       (v * w).sum(-1))


def dihedral_value_grad_analytic(p0, p1, p2, p3):
    """Signed dihedral (same convention as :func:`dihedral_torch`) and its four
    Cartesian gradients, closed form (Blondel-Karplus). Inputs (..., 3).

    Returns (phi, g0, g1, g2, g3); validated against autograd in tests."""
    Fv = p0 - p1
    G = p1 - p2
    H = p3 - p2
    A = torch.linalg.cross(Fv, G, dim=-1)
    Bv = torch.linalg.cross(H, G, dim=-1)
    Gn = G.norm(dim=-1).clamp_min(1e-12)
    A2 = (A * A).sum(-1).clamp_min(1e-24)
    B2 = (Bv * Bv).sum(-1).clamp_min(1e-24)
    phi = dihedral_torch(p0, p1, p2, p3)
    t0 = -(Gn / A2).unsqueeze(-1) * A
    t3 = (Gn / B2).unsqueeze(-1) * Bv
    fg = ((Fv * G).sum(-1) / (Gn * Gn)).unsqueeze(-1)
    hg = ((H * G).sum(-1) / (Gn * Gn)).unsqueeze(-1)
    t1 = -t0 - fg * t0 - hg * t3
    t2 = -t3 + fg * t0 + hg * t3
    return phi, t0, t1, t2, t3


def cv_values(q):
    """(phi, psi) of a (B, 22, 3) batch, each (B,)."""
    phi = dihedral_torch(*(q[:, i] for i in PHI_ATOMS))
    psi = dihedral_torch(*(q[:, i] for i in PSI_ATOMS))
    return phi, psi


def cv_value_grad_analytic(q, atoms):
    """Analytic counterpart of :func:`cv_value_grad`: (val (B,), grad (B, 4, 3))."""
    phi, g0, g1, g2, g3 = dihedral_value_grad_analytic(
        q[:, atoms[0]], q[:, atoms[1]], q[:, atoms[2]], q[:, atoms[3]])
    return phi, torch.stack([g0, g1, g2, g3], dim=1)


def cv_value_grad(q, atoms):
    """Dihedral value and its Cartesian gradient on the 4 CV atoms.

    q: (B, 22, 3) -> (val (B,), grad (B, 4, 3)); exact autograd, detached.
    """
    sub = q[:, list(atoms), :].detach().requires_grad_(True)
    with torch.enable_grad():
        val = dihedral_torch(sub[:, 0], sub[:, 1], sub[:, 2], sub[:, 3])
        g, = torch.autograd.grad(val.sum(), sub)
    return val.detach(), g


def scatter_cv_force(q_like, coeff, grad, atoms):
    """coeff (B,) times the CV gradient, scattered to (B, 22, 3)."""
    out = torch.zeros_like(q_like)
    out[:, list(atoms), :] = coeff[:, None, None] * grad
    return out


# -----------------------------------------------------------------------------
# reference (trimmed MBAR artifact; 1D/conditional objects by exact column sums)
# -----------------------------------------------------------------------------
def load_reference(device=DEVICE, dtype=DTYPE):
    z = np.load(os.path.join(REF_DIR, "ala_reference_mbar.npz"), allow_pickle=True)
    F2 = torch.as_tensor(z["F"], device=device, dtype=dtype)        # kJ/mol, min 0
    mask8 = torch.as_tensor(z["mask8"], device=device, dtype=torch.bool)
    meta = z["meta"].item() if z["meta"].shape == () else str(z["meta"])
    kT = KB * 300.0
    beta = 1.0 / kT
    rho2 = torch.where(torch.isfinite(F2), torch.exp(-beta * F2),
                       torch.zeros_like(F2))
    col = rho2.sum(dim=1) * DZ                                       # (97,)
    F1 = -kT * torch.log(torch.clamp(col, min=EPS))
    F1 = F1 - F1.min()
    mask1 = F1 <= 8.0 * kT
    p_cond = rho2 / torch.clamp(rho2.sum(dim=1, keepdim=True), min=EPS) / DZ
    rho2n = rho2 / (rho2.sum() * DZ * DZ)
    rho1n = col / (col.sum() * DZ)
    return dict(F2=F2, mask8=mask8, F1=F1, mask1=mask1, p_cond=p_cond,
                rho2=rho2n, rho1=rho1n, kT=kT, meta=meta)


def basin_labels(F2, mask8, n_basins=4):
    """Watershed labels (97, 97) int64: flood from the n deepest reference minima
    (min-max water level, periodic 8-neighbour); -1 outside mask8.

    Ported convention from the closed campaign's alanine/basins.py (basins are
    regions of the ACCEPTED reference, fixed before any run)."""
    import heapq
    F = F2.detach().cpu().numpy()
    m = mask8.detach().cpu().numpy()
    n = F.shape[0]
    minima = []
    for i in range(n):
        for j in range(n):
            if not m[i, j] or not np.isfinite(F[i, j]):
                continue
            nb = [F[(i + di) % n, (j + dj) % n]
                  for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
            if all((not np.isfinite(x)) or F[i, j] <= x for x in nb):
                minima.append((float(F[i, j]), i, j))
    minima.sort()
    seeds = minima[:n_basins]
    lab = np.full((n, n), -1, dtype=np.int64)
    pq = []
    for k, (fv, i, j) in enumerate(seeds):
        heapq.heappush(pq, (fv, i, j, k))
    while pq:
        lvl, i, j, k = heapq.heappop(pq)
        if lab[i, j] != -1:
            continue
        lab[i, j] = k
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if (di, dj) == (0, 0):
                    continue
                a, b = (i + di) % n, (j + dj) % n
                if m[a, b] and lab[a, b] == -1 and np.isfinite(F[a, b]):
                    heapq.heappush(pq, (max(lvl, float(F[a, b])), a, b, k))
    return torch.as_tensor(lab), [(i, j) for _, i, j in seeds]


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AlaConfig:
    temperature: float = 300.0
    dt: float = 1e-3                 # ps (1 fs) -- reference-validated
    gamma: float = 1.0               # ps^-1
    K: int = 128
    n_steps: int = 1_000_000
    block: int = 20
    eps_bw: float = 0.12             # mollifier bandwidth (radians)
    eta_bw: float = 0.25             # KDE bandwidth (marginal / FR score)
    n_saves: int = 400
    profile_every: int = 8
    ess_window_steps: int = 4000
    cv: str = "phipsi"               # "phipsi" | "phi"

    @property
    def beta(self) -> float:
        return 1.0 / (KB * self.temperature)

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt


# -----------------------------------------------------------------------------
# the batched simulation
# -----------------------------------------------------------------------------
def simulate_batch(configs, seeds, methods, batch_seed=12345, device=DEVICE,
                   dtype=DTYPE, progress=None):
    """Runs with torch deterministic algorithms enabled (restored on exit): the
    force scatters (index_add_/scatter_add_) are then order-stable, so GPU runs
    are bitwise reproducible per (device, seed) — at ~1.7x the nondeterministic
    graph-replay speed, still ~4x faster than the autograd engine.

    Arm pairing on GPU is same-noise-stream pairing: two IDENTICAL arms in one
    batch agree bitwise on CPU but can differ at the last bit on CUDA (row-
    position-dependent reduction order), which underdamped MD then amplifies
    chaotically — the regime every GPU campaign in this project operated in."""
    was_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _simulate_batch(configs, seeds, methods, batch_seed, device,
                               dtype, progress)
    finally:
        torch.use_deterministic_algorithms(was_det)


def _simulate_batch(configs, seeds, methods, batch_seed, device, dtype, progress):
    cfgs, methods = list(configs), list(methods)
    assert len(cfgs) == len(seeds)
    B, M = len(cfgs), len(methods)
    R = B * M

    c0 = cfgs[0]
    for c in cfgs:
        for a in ("temperature", "dt", "gamma", "K", "n_steps", "block", "eps_bw",
                  "eta_bw", "n_saves", "profile_every", "ess_window_steps", "cv"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    K, dt, n_steps, block = c0.K, c0.dt, c0.n_steps, c0.block
    assert n_steps % block == 0
    n_blocks = n_steps // block
    is2d = c0.cv == "phipsi"
    assert c0.cv in ("phipsi", "phi")

    by_name = {m.name: m for m in methods}
    scheds = [_schedule_source(m, by_name) for m in methods]
    name_col = {m.name: j for j, m in enumerate(methods)}
    partner_col = [name_col[m.shadows] if m.sham else j for j, m in enumerate(methods)]
    partner = torch.tensor([b * M + partner_col[j] for b in range(B) for j in range(M)],
                           device=device, dtype=torch.long)
    event_blocks = [k for k in range(1, n_blocks + 1)
                    if any(_fires_at_block(s, k, n_blocks) for s in scheds)]
    n_events = len(event_blocks)
    fires = torch.tensor(
        [[_fires_at_block(scheds[j], k, n_blocks) for b in range(B) for j in range(M)]
         for k in event_blocks], device=device, dtype=torch.bool
    ).reshape(n_events, R) if n_events else torch.zeros((0, R), device=device,
                                                        dtype=torch.bool)
    is_fr_row = torch.tensor([m.use_fr and not m.sham for m in methods],
                             device=device).repeat(B)
    is_sham_row = torch.tensor([m.sham for m in methods], device=device).repeat(B)
    is_coarse_row = torch.tensor([m.coarse_bins > 0 for m in methods],
                                 device=device).repeat(B)
    coarse_nb = torch.tensor([m.coarse_bins for m in methods], device=device,
                             dtype=torch.long).repeat(B)
    theta0 = torch.tensor([m.theta if (m.use_fr and not m.sham) else 0.0
                           for m in methods], device=device, dtype=dtype).repeat(B)
    alpha_ess = torch.tensor([m.alpha_ess for m in methods], device=device,
                             dtype=dtype).repeat(B)
    assert all(m.g_shus > 0 for m in methods), "g_shus must be positive"
    gain = torch.tensor([m.g_shus for m in methods], device=device,
                        dtype=dtype).repeat(B)

    tff = TorchFF(device, dtype)
    ref = load_reference(device, dtype)
    labels_t, _seeds = basin_labels(ref["F2"], ref["mask8"])
    labels_flat = labels_t.to(device).reshape(-1)
    n_basins = int(labels_t.max()) + 1
    beta = c0.beta
    kT = 1.0 / beta

    # integrator constants (BAOAB)
    m_col = tff.masses.reshape(-1, 1)                       # (A, 1)
    c1 = math.exp(-c0.gamma * dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    sigma_v = math.sqrt(kT) / m_col.sqrt()                  # (A, 1)

    beta_rows = torch.full((R, 1), beta, device=device, dtype=dtype)
    if is2d:
        shus = ShusAccumulator2(R, ALA_GRID2, beta_rows, c0.eps_bw, device, dtype,
                                gain=gain)
        k1e, r1e = periodic_gaussian_kernel(c0.eta_bw, ALA_GRID2.dx1, N_GRID,
                                            device, dtype)
        k2e, r2e = k1e, r1e
        F_ref = ref["F2"]
        emask = ref["mask8"]
        rho_ref = ref["rho2"].unsqueeze(0)
    else:
        shus = ShusAccumulator1P(R, ALA_GRID1, beta_rows, c0.eps_bw, device, dtype,
                                 gain=gain)
        k1e, r1e = periodic_gaussian_kernel(c0.eta_bw, ALA_GRID1.dx, N_GRID,
                                            device, dtype)
        k2e, r2e = k1e, r1e
        F_ref = ref["F1"]
        emask = ref["mask1"]
        rho_ref = ref["rho1"].unsqueeze(0)

    gen_n = torch.Generator(device=device)
    gen_n.manual_seed(2000 + batch_seed)
    gen_f = torch.Generator(device=device)
    gen_f.manual_seed(3000 + batch_seed)

    # ---- static state (shared by the eager path and the CUDA-graph replay) ----
    q_s = tff.X0.unsqueeze(0).expand(R * K, N_ATOMS, 3).contiguous().clone()
    v0 = torch.randn((B, K, N_ATOMS, 3), device=device, dtype=dtype,
                     generator=gen_n) * sigma_v
    v_s = v0.repeat_interleave(M, dim=0).reshape(R * K, N_ATOMS, 3).contiguous()
    del v0
    f_s = torch.zeros_like(q_s)
    z1_s = torch.zeros((R, K), device=device, dtype=dtype)
    z2_s = torch.zeros((R, K), device=device, dtype=dtype)
    noise_s = torch.zeros((block, B, K, N_ATOMS, 3), device=device, dtype=dtype)
    if is2d:
        R_s = shus.R.clone()
        Fp1_s = shus.Fp1.clone()
        Fp2_s = shus.Fp2.clone()
    else:
        R_s = shus.R.clone()
        Fp1_s = shus.Fp.clone()
        Fp2_s = None
    anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    anc_g = anc.clone()

    def wrapg(a):
        return wrap_periodic(a, X0MIN, 2 * PI)

    from ..grid1p import interp1p, nearest_bin1p
    from ..grid2d import interp2, nearest_bin2

    phi_idx = torch.tensor(PHI_ATOMS, device=device, dtype=torch.long)
    psi_idx = torch.tensor(PSI_ATOMS, device=device, dtype=torch.long)

    def compute_f_and_cv():
        """Force (physical + bias from the static mirrors) and CV at q_s, into
        f_s/z1_s/z2_s. Analytic throughout -- capture-safe (device indices only)."""
        f = tff.forces_analytic(q_s)
        phi, gphi = cv_value_grad_analytic(q_s, PHI_ATOMS)
        z1 = wrapg(phi).view(R, K)
        z1_s.copy_(z1)
        if is2d:
            psi, gpsi = cv_value_grad_analytic(q_s, PSI_ATOMS)
            z2 = wrapg(psi).view(R, K)
            z2_s.copy_(z2)
            cphi = interp2(z1, z2, Fp1_s, ALA_GRID2).reshape(R * K)
            cpsi = interp2(z1, z2, Fp2_s, ALA_GRID2).reshape(R * K)
            f.index_add_(1, phi_idx, cphi[:, None, None] * gphi)
            f.index_add_(1, psi_idx, cpsi[:, None, None] * gpsi)
        else:
            cphi = interp1p(z1, Fp1_s, ALA_GRID1).reshape(R * K)
            f.index_add_(1, phi_idx, cphi[:, None, None] * gphi)
        f_s.copy_(f)

    def deposit():
        """Block-frozen deposit with weight R_n(xi) read from the static mirror,
        scattered into the accumulator's (storage-stable) buffer."""
        if is2d:
            w = interp2(z1_s, z2_s, R_s, ALA_GRID2)
            idx = nearest_bin2(z1_s, z2_s, ALA_GRID2)
        else:
            w = interp1p(z1_s, R_s, ALA_GRID1)
            idx = nearest_bin1p(z1_s, ALA_GRID1)
        shus.buf.scatter_add_(1, idx, w)

    def propagate_block():
        """block BAOAB steps + deposits, reading noise_s. No RNG, no Python state
        beyond loop constants: identical for eager execution and graph replay."""
        for i in range(block):
            v_s.add_((0.5 * dt) * f_s / m_col)
            q_s.add_((0.5 * dt) * v_s)
            v_s.mul_(c1).add_(
                c2 * noise_s[i].repeat_interleave(M, dim=0).reshape(
                    R * K, N_ATOMS, 3) * sigma_v)
            q_s.add_((0.5 * dt) * v_s)
            compute_f_and_cv()
            v_s.add_((0.5 * dt) * f_s / m_col)
            deposit()

    def refresh_statics():
        R_s.copy_(shus.R)
        if is2d:
            Fp1_s.copy_(shus.Fp1)
            Fp2_s.copy_(shus.Fp2)
        else:
            Fp1_s.copy_(shus.Fp)

    def fill_noise():
        for i in range(block):
            noise_s[i].copy_(torch.randn((B, K, N_ATOMS, 3), device=device,
                                         dtype=dtype, generator=gen_n))

    compute_f_and_cv()                     # initial force/CV at the start state

    use_graph = (device.type == "cuda") if hasattr(device, "type") else False
    graph = None
    if use_graph:
        snap = [t.clone() for t in (q_s, v_s, f_s, z1_s, z2_s, shus.buf)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                propagate_block()          # warmup (zero noise, discarded)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            propagate_block()
        for t, sn in zip((q_s, v_s, f_s, z1_s, z2_s, shus.buf), snap):
            t.copy_(sn)
        del snap

    # ---- saves aligned to adaptation-block boundaries -------------------------
    blk_per_save = max(1, (n_steps // c0.n_saves) // block)
    save_blocks = sorted({*range(blk_per_save, n_blocks + 1, blk_per_save),
                          n_blocks})
    save_steps = [b * block for b in save_blocks]
    n_saves = len(save_steps)
    prof_blocks = save_blocks[:: c0.profile_every]
    if save_blocks[-1] not in prof_blocks:
        prof_blocks = prof_blocks + [save_blocks[-1]]
    prof_steps = [b * block for b in prof_blocks]
    n_prof = len(prof_steps)
    prof_set = set(prof_blocks)
    save_set = set(save_blocks)

    ts = {k: torch.zeros((R, n_saves), device=device, dtype=dtype) for k in
          ("l2_f", "kl_u", "tv_u", "ess_anc", "wmax", "ess_anc_glob", "wmax_glob",
           "n_anc", "dep_ref", "dep_self", "e_cond", "temp_kin")}
    ts["P"] = torch.zeros((R, n_saves, n_basins + 1), device=device, dtype=dtype)
    gshape = (N_GRID, N_GRID) if is2d else (N_GRID,)
    prof = {"pmf": torch.zeros((R, n_prof) + gshape, device=device, dtype=dtype),
            "marg": torch.zeros((R, n_prof) + gshape, device=device, dtype=dtype)}
    if not is2d:   # xi = phi: keep the joint (phi, psi) KDE for conditional scoring
        prof["marg2"] = torch.zeros((R, n_prof, N_GRID, N_GRID), device=device,
                                    dtype=dtype)
    ev = {k: torch.zeros((R, max(n_events, 1)), device=device, dtype=dtype)
          for k in ("theta", "ess_fr", "turnover")}
    tot_turn = torch.zeros(R, device=device, dtype=dtype)
    dep_ref_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)
    dep_self_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)
    save_ptr, prof_ptr, event_ptr = 0, 0, 0
    ar = torch.arange(K, device=device).unsqueeze(0).expand(R, K)
    p_cond_ref = ref["p_cond"].unsqueeze(0)          # (1, 97, 97)

    ess_blk = max(1, c0.ess_window_steps // block)

    for blk in range(1, n_blocks + 1):
        if c0.ess_window_steps > 0 and (blk - 1) % ess_blk == 0:
            anc = ar.clone()

        # ---- propagation: one adaptation block (graph replay on cuda) ----------
        fill_noise()
        if graph is not None:
            graph.replay()
        else:
            propagate_block()

        # ---- SHUS update + deposition diagnostics ------------------------------
        if is2d:
            r_n = shus.R / integral2(shus.R, ALA_GRID2).reshape(R, 1, 1)
            inc = shus.update(dt, K)
            d_n = inc / torch.clamp(integral2(inc, ALA_GRID2),
                                    min=EPS).reshape(R, 1, 1)
            dep_ref_cur = torch.sqrt(((d_n - rho_ref) ** 2).mean(dim=(1, 2)))
            dep_self_cur = torch.sqrt(((d_n - r_n) ** 2).mean(dim=(1, 2)))
        else:
            r_n = shus.R / integral1p(shus.R, ALA_GRID1).unsqueeze(1)
            inc = shus.update(dt, K)
            d_n = inc / torch.clamp(integral1p(inc, ALA_GRID1),
                                    min=EPS).unsqueeze(1)
            dep_ref_cur = torch.sqrt(((d_n - rho_ref) ** 2).mean(dim=1))
            dep_self_cur = torch.sqrt(((d_n - r_n) ** 2).mean(dim=1))
        refresh_statics()

        # ---- (maybe) an FR event ----------------------------------------------
        if event_ptr < n_events and event_blocks[event_ptr] == blk:
            active = fires[event_ptr]
            if is2d:
                sel, turn, theta_used, essf = fr_event2(
                    z1_s, z2_s, active & is_fr_row, active & is_sham_row,
                    is_coarse_row, coarse_nb, partner, theta0, alpha_ess,
                    k1e, r1e, k2e, r2e, ALA_GRID2, gen_f)
            else:
                sel, turn, theta_used, essf = fr_event1p(
                    z1_s, active & is_fr_row, active & is_sham_row,
                    is_coarse_row, coarse_nb, partner, theta0, alpha_ess,
                    k1e, r1e, ALA_GRID1, gen_f)
            ev["theta"][:, event_ptr] = theta_used
            ev["ess_fr"][:, event_ptr] = essf
            ev["turnover"][:, event_ptr] = turn.to(dtype)
            tot_turn += turn.to(dtype)
            # full-state clone into the STATIC buffers: gather (q, f, ancestry,
            # CV caches); fresh Maxwell momenta for replaced slots (fixed-size)
            replaced = (sel != ar)
            gi = torch.arange(R, device=device).unsqueeze(1)
            q_s.copy_(q_s.view(R, K, N_ATOMS, 3)[gi, sel].reshape(
                R * K, N_ATOMS, 3))
            f_s.copy_(f_s.view(R, K, N_ATOMS, 3)[gi, sel].reshape(
                R * K, N_ATOMS, 3))
            v_new = torch.randn((R, K, N_ATOMS, 3), device=device, dtype=dtype,
                                generator=gen_f) * sigma_v
            v_s.copy_(torch.where(replaced.unsqueeze(-1).unsqueeze(-1), v_new,
                                  v_s.view(R, K, N_ATOMS, 3)).reshape(
                R * K, N_ATOMS, 3))
            z1_s.copy_(torch.gather(z1_s, 1, sel))
            if is2d:
                z2_s.copy_(torch.gather(z2_s, 1, sel))
            anc = torch.gather(anc, 1, sel)
            anc_g = torch.gather(anc_g, 1, sel)
            event_ptr += 1

        # ---- checkpoints -------------------------------------------------------
        if blk in save_set:
            F_hat = shus.f_estimate()
            d = (F_hat - F_ref.unsqueeze(0))[:, emask]
            d = d - d.mean(dim=1, keepdim=True)
            ts["l2_f"][:, save_ptr] = torch.sqrt((d * d).mean(dim=1))
            if is2d:
                p_hat = binned_density2(z1_s, z2_s, k1e, r1e, k2e, r2e, ALA_GRID2)
                ts["kl_u"][:, save_ptr] = kl_to_uniform2(p_hat, ALA_GRID2)
                ts["tv_u"][:, save_ptr] = tv_to_uniform2(p_hat, ALA_GRID2)
                zz1, zz2 = z1_s, z2_s
            else:
                p_hat = binned_density1p(z1_s, k1e, r1e, ALA_GRID1)
                ts["kl_u"][:, save_ptr] = kl_to_uniform1p(p_hat, ALA_GRID1)
                ts["tv_u"][:, save_ptr] = tv_to_uniform1p(p_hat, ALA_GRID1)
                phiv, psiv = cv_values(q_s)
                zz1 = wrapg(phiv).view(R, K)
                zz2 = wrapg(psiv).view(R, K)
                p2 = binned_density2(zz1, zz2, k1e, r1e, k1e, r1e, ALA_GRID2)
                p_phi = p2.sum(dim=2) * DZ
                p_c = p2 / torch.clamp(p2.sum(dim=2, keepdim=True), min=EPS) / DZ
                tv_col = 0.5 * ((p_c - p_cond_ref).abs().sum(dim=2) * DZ)
                ts["e_cond"][:, save_ptr] = (p_phi * tv_col).sum(dim=1) * DZ
                if blk in prof_set:
                    prof["marg2"][:, prof_ptr] = p2
            e_, w_ = ancestor_stats(anc, K)
            ts["ess_anc"][:, save_ptr] = e_
            ts["wmax"][:, save_ptr] = w_
            eg_, wg_ = ancestor_stats(anc_g, K)
            ts["ess_anc_glob"][:, save_ptr] = eg_
            ts["wmax_glob"][:, save_ptr] = wg_
            ts["n_anc"][:, save_ptr] = surviving_ancestors(anc_g, K)
            ts["dep_ref"][:, save_ptr] = dep_ref_cur
            ts["dep_self"][:, save_ptr] = dep_self_cur
            ke = 0.5 * (m_col * v_s * v_s).view(R, K, -1).sum(dim=2)
            ts["temp_kin"][:, save_ptr] = (2.0 * ke / (3 * N_ATOMS * KB)).mean(dim=1)
            i1 = torch.remainder(torch.round((zz1 - X0MIN) / DZ).long(), N_GRID)
            i2 = torch.remainder(torch.round((zz2 - X0MIN) / DZ).long(), N_GRID)
            lab = labels_flat[(i1 * N_GRID + i2).reshape(-1)].reshape(R, K)
            for k in range(n_basins):
                ts["P"][:, save_ptr, k] = (lab == k).to(dtype).mean(dim=1)
            ts["P"][:, save_ptr, n_basins] = (lab < 0).to(dtype).mean(dim=1)
            if blk in prof_set:
                prof["pmf"][:, prof_ptr] = F_hat
                prof["marg"][:, prof_ptr] = p_hat
                prof_ptr += 1
            save_ptr += 1
        if progress is not None and (blk * block) % progress < block:
            print(f"    step {blk * block}/{n_steps}", flush=True)

    # ---- finalize ---------------------------------------------------------------
    t_axis = np.array([s * dt for s in save_steps])
    prof_t = np.array([s * dt for s in prof_steps])
    ev_t = np.array([k * block * dt for k in event_blocks])

    def npy(t):
        return t.detach().cpu().numpy()

    recs = []
    for b in range(B):
        for mm in range(M):
            r = b * M + mm
            l2 = npy(ts["l2_f"][r])
            rec = dict(
                config=asdict(cfgs[b]), seed=int(seeds[b]),
                method=asdict(methods[mm]), batch_seed=batch_seed,
                reference_id=REFERENCE_ID, cv=c0.cv,
                eval_window="mask8" if is2d else "mask1",
                time=t_axis, profile_time=prof_t,
                x1_grid=npy(ALA_GRID2.x1(device, dtype)),
                x2_grid=npy(ALA_GRID2.x2(device, dtype)),
                F_ref=npy(F_ref), eval_mask=npy(emask),
                basin_labels=npy(labels_t),
                pmf_t=npy(prof["pmf"][r]), marginal_t=npy(prof["marg"][r]),
                l2_f_t=l2, kl_u_t=npy(ts["kl_u"][r]), tv_u_t=npy(ts["tv_u"][r]),
                e_cond_t=npy(ts["e_cond"][r]),
                temp_kin_t=npy(ts["temp_kin"][r]),
                ess_anc_t=npy(ts["ess_anc"][r]), wmax_t=npy(ts["wmax"][r]),
                ess_anc_glob_t=npy(ts["ess_anc_glob"][r]),
                wmax_glob_t=npy(ts["wmax_glob"][r]), n_anc_t=npy(ts["n_anc"][r]),
                dep_ref_l2_t=npy(ts["dep_ref"][r]),
                dep_self_l2_t=npy(ts["dep_self"][r]),
                P_regions=npy(ts["P"][r]),
                event_time=ev_t, event_theta=npy(ev["theta"][r]),
                event_ess_fr=npy(ev["ess_fr"][r]),
                event_turnover=npy(ev["turnover"][r]),
                final_l2_f=float(l2[-1]),
                int_l2_f=float(np.trapezoid(l2, t_axis)),
                total_turnover=float(tot_turn[r]),
            )
            if not is2d:
                rec["marginal2_t"] = npy(prof["marg2"][r])
            recs.append(rec)
    return recs
