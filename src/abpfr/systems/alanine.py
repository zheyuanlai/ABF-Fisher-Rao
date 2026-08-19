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


def cv_values(q):
    """(phi, psi) of a (B, 22, 3) batch, each (B,)."""
    phi = dihedral_torch(*(q[:, i] for i in PHI_ATOMS))
    psi = dihedral_torch(*(q[:, i] for i in PSI_ATOMS))
    return phi, psi


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

    # initial conditions: every walker starts at the minimised C7eq structure with
    # fresh Maxwell momenta; per-seed noise streams are shared across arms (paired)
    q = tff.X0.unsqueeze(0).expand(R * K, N_ATOMS, 3).clone()
    v0 = torch.randn((B, K, N_ATOMS, 3), device=device, dtype=dtype,
                     generator=gen_n) * sigma_v
    v = v0.repeat_interleave(M, dim=0).reshape(R * K, N_ATOMS, 3).clone()
    del v0
    anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    anc_g = anc.clone()

    def wrapg(a):
        return wrap_periodic(a, X0MIN, 2 * PI)

    def bias_and_cv(qq):
        """Physical+bias force and wrapped CV values at qq (R*K, A, 3)."""
        f = tff.forces(qq)
        phi, gphi = cv_value_grad(qq, PHI_ATOMS)
        phi_w = wrapg(phi)
        if is2d:
            psi, gpsi = cv_value_grad(qq, PSI_ATOMS)
            psi_w = wrapg(psi)
            z1 = phi_w.view(R, K)
            z2 = psi_w.view(R, K)
            from ..grid2d import interp2
            cphi = interp2(z1, z2, shus.Fp1, ALA_GRID2).reshape(R * K)
            cpsi = interp2(z1, z2, shus.Fp2, ALA_GRID2).reshape(R * K)
            f = f + scatter_cv_force(qq, cphi, gphi, PHI_ATOMS)
            f = f + scatter_cv_force(qq, cpsi, gpsi, PSI_ATOMS)
            return f, z1, z2
        cphi = shus.bias_force_at(phi_w.view(R, K)).reshape(R * K)
        f = f + scatter_cv_force(qq, cphi, gphi, PHI_ATOMS)
        return f, phi_w.view(R, K), None

    f, z1, z2 = bias_and_cv(q)

    save_steps = sorted({*range(0, n_steps, max(1, n_steps // c0.n_saves)),
                         n_steps - 1})
    n_saves = len(save_steps)
    save_set = set(save_steps)
    prof_steps = save_steps[:: c0.profile_every]
    if save_steps[-1] not in prof_steps:
        prof_steps = prof_steps + [save_steps[-1]]
    n_prof = len(prof_steps)
    prof_set = set(prof_steps)

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

    for step in range(n_steps):
        if c0.ess_window_steps > 0 and step % c0.ess_window_steps == 0:
            anc = ar.clone()

        # ---- BAOAB step (paired noise across arms) -----------------------------
        v = v + (0.5 * dt) * f / m_col
        q = q + (0.5 * dt) * v
        noise = torch.randn((B, K, N_ATOMS, 3), device=device, dtype=dtype,
                            generator=gen_n).repeat_interleave(M, dim=0)
        v = c1 * v + c2 * noise.reshape(R * K, N_ATOMS, 3) * sigma_v
        q = q + (0.5 * dt) * v
        f, z1, z2 = bias_and_cv(q)
        v = v + (0.5 * dt) * f / m_col

        # ---- SHUS deposit -------------------------------------------------------
        if is2d:
            shus.deposit(z1, z2)
        else:
            shus.deposit(z1)

        # ---- block boundary: update, then (maybe) an FR event -------------------
        if (step + 1) % block == 0:
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
            blk = (step + 1) // block
            if event_ptr < n_events and event_blocks[event_ptr] == blk:
                active = fires[event_ptr]
                if is2d:
                    sel, turn, theta_used, essf = fr_event2(
                        z1, z2, active & is_fr_row, active & is_sham_row,
                        is_coarse_row, coarse_nb, partner, theta0, alpha_ess,
                        k1e, r1e, k2e, r2e, ALA_GRID2, gen_f)
                else:
                    sel, turn, theta_used, essf = fr_event1p(
                        z1, active & is_fr_row, active & is_sham_row,
                        is_coarse_row, coarse_nb, partner, theta0, alpha_ess,
                        k1e, r1e, ALA_GRID1, gen_f)
                ev["theta"][:, event_ptr] = theta_used
                ev["ess_fr"][:, event_ptr] = essf
                ev["turnover"][:, event_ptr] = turn.to(dtype)
                tot_turn += turn.to(dtype)
                # full-state clone: gather (q, f, ancestry); FRESH Maxwell momenta
                # for replaced slots (fixed-size draw -> pairing preserved)
                replaced = (sel != ar)
                gi = torch.arange(R, device=device).unsqueeze(1)
                q = q.view(R, K, N_ATOMS, 3)[gi, sel].reshape(R * K, N_ATOMS, 3)
                f = f.view(R, K, N_ATOMS, 3)[gi, sel].reshape(R * K, N_ATOMS, 3)
                v_new = torch.randn((R, K, N_ATOMS, 3), device=device, dtype=dtype,
                                    generator=gen_f) * sigma_v
                v = torch.where(replaced.unsqueeze(-1).unsqueeze(-1),
                                v_new, v.view(R, K, N_ATOMS, 3)).reshape(
                    R * K, N_ATOMS, 3)
                anc = torch.gather(anc, 1, sel)
                anc_g = torch.gather(anc_g, 1, sel)
                event_ptr += 1

        # ---- checkpoints ---------------------------------------------------------
        if step in save_set:
            F_hat = shus.f_estimate()
            d = (F_hat - F_ref.unsqueeze(0))[:, emask]
            d = d - d.mean(dim=1, keepdim=True)
            ts["l2_f"][:, save_ptr] = torch.sqrt((d * d).mean(dim=1))
            if is2d:
                p_hat = binned_density2(z1, z2, k1e, r1e, k2e, r2e, ALA_GRID2)
                ts["kl_u"][:, save_ptr] = kl_to_uniform2(p_hat, ALA_GRID2)
                ts["tv_u"][:, save_ptr] = tv_to_uniform2(p_hat, ALA_GRID2)
            else:
                p_hat = binned_density1p(z1, k1e, r1e, ALA_GRID1)
                ts["kl_u"][:, save_ptr] = kl_to_uniform1p(p_hat, ALA_GRID1)
                ts["tv_u"][:, save_ptr] = tv_to_uniform1p(p_hat, ALA_GRID1)
                # conditional diagnostic E_cond(t): TV(p_t(psi|phi), p_ref) weighted
                # by the sampled phi-marginal (psi tracked even though unbiased)
                phiv, psiv = cv_values(q)
                zz1 = wrapg(phiv).view(R, K)
                zz2 = wrapg(psiv).view(R, K)
                p2 = binned_density2(zz1, zz2, k1e, r1e, k1e, r1e, ALA_GRID2)
                p_phi = p2.sum(dim=2) * DZ                       # (R, 97)
                p_c = p2 / torch.clamp(p2.sum(dim=2, keepdim=True), min=EPS) / DZ
                tv_col = 0.5 * ((p_c - p_cond_ref).abs().sum(dim=2) * DZ)
                ts["e_cond"][:, save_ptr] = (p_phi * tv_col).sum(dim=1) * DZ
                if step in prof_set:
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
            ke = 0.5 * (m_col * v * v).view(R, K, -1).sum(dim=2)
            ts["temp_kin"][:, save_ptr] = (2.0 * ke / (3 * N_ATOMS * KB)).mean(dim=1)
            # basin occupancies via cell lookup on the reference lattice
            # (for cv="phi" the psi coordinate is tracked even though unbiased;
            # zz1/zz2 were computed in the conditional-diagnostic branch above)
            if is2d:
                zz1, zz2 = z1, z2
            i1 = torch.remainder(torch.round((zz1 - X0MIN) / DZ).long(), N_GRID)
            i2 = torch.remainder(torch.round((zz2 - X0MIN) / DZ).long(), N_GRID)
            lab = labels_flat[(i1 * N_GRID + i2).reshape(-1)].reshape(R, K)
            for k in range(n_basins):
                ts["P"][:, save_ptr, k] = (lab == k).to(dtype).mean(dim=1)
            ts["P"][:, save_ptr, n_basins] = (lab < 0).to(dtype).mean(dim=1)
            if step in prof_set:
                prof["pmf"][:, prof_ptr] = F_hat
                prof["marg"][:, prof_ptr] = p_hat
                prof_ptr += 1
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

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
