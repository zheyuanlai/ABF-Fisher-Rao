"""Independent float64 numba model of ethane in rigid LTA for NUMERICAL VALIDATION (2026-10-09).

docs/numerical_validation/LTA_VALIDATION.md.  Never imported by the production engine
(src/lta/core_lta.py), which it re-implements from the physical definition:

    U(q) = (k_b/2)(|q_0 - q_1| - r_0)^2 + sum_{beads b} sum_{O} [V_LJ(|q_b - o|_mi) - V_LJ(r_c)] 1{r < r_c},
    V_LJ(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6],    eps = 93 K k_B, sigma 3.48 A, r_c 10 A,
    k_b 400 kJ/mol/A^2, r_0 1.54 A; 384 framework O, cubic box L = 23.838 A, minimum image on bead-O
    displacements only; coordinates unwrapped (the bond needs no minimum image);
    phi = wrap((2 pi / a) x_COM) in [-pi, pi), x_COM = (x_0 + x_1)/2, a = 11.919 A;
    local mean force f = -(F . grad phi)/|grad phi|^2 = -(a/2 pi)(F_0x + F_1x) (linear CV: no geometric term).

Frozen-bias sampling.  Every chain samples exp(-beta [U(q) - A(phi(q))]) with a FIXED smooth periodic
bias A (a truncated Fourier series fitted to the existing reference, so phi is nearly uniform).  Then
    F_density(phi) = A(phi) - kT log p(phi)      and      F_MF(phi) = int <f | phi>
both estimate the free energy of the UNBIASED model (the bias depends on phi only, so the conditional
law at fixed phi -- and hence <f | phi> -- is unchanged).  Chains:
    EM   q' = q + dt (F + A'(phi) grad phi) + sqrt(2 dt kT) xi        (the production integrator)
    LM   Leimkuhler-Matthews with the same drift (second order for the invariant measure)
    MC   Metropolis: single-bead uniform displacements and rigid-molecule translations (exact; energy only)
"""
from __future__ import annotations

import math
import os

import numpy as np
from numba import njit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KB = 0.008314462618
EM, LM, MC = 0, 1, 2
SCHEMES = {"EM": EM, "LM": LM, "MC": MC}
NBIN = 180


def framework():
    z = np.load(os.path.join(ROOT, "cache", "lta", "framework.npz"), allow_pickle=True)
    return np.ascontiguousarray(z["o_pos"], dtype=np.float64), float(z["a_pseudo"]), float(z["box"])


def params(T):
    """[kT, eps, sigma, rc, r0, k_bond, v_rc, a, L]"""
    _, a, L = framework()
    eps, sig, rc = 93.0 * KB, 3.48, 10.0
    sr6 = (sig / rc) ** 6
    v_rc = 4.0 * eps * (sr6 * sr6 - sr6)
    return np.array([KB * T, eps, sig, rc, 1.54, 400.0, v_rc, a, L], dtype=np.float64)


@njit(cache=True)
def bead_lj(x, y, z, O, pp):
    """(energy, fx, fy, fz) of one bead with every framework O (shifted LJ, minimum image)."""
    eps, sig, rc, v_rc, L = pp[1], pp[2], pp[3], pp[6], pp[8]
    rc2 = rc * rc
    s2 = sig * sig
    e = 0.0
    fx = 0.0
    fy = 0.0
    fz = 0.0
    invL = 1.0 / L
    for k in range(O.shape[0]):
        dx = x - O[k, 0]
        dx -= L * np.rint(dx * invL)
        dy = y - O[k, 1]
        dy -= L * np.rint(dy * invL)
        dz = z - O[k, 2]
        dz -= L * np.rint(dz * invL)
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < rc2:
            ir2 = s2 / r2
            sr6 = ir2 * ir2 * ir2
            e += 4.0 * eps * (sr6 * sr6 - sr6) - v_rc
            c = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2
            fx += c * dx
            fy += c * dy
            fz += c * dz
    return e, fx, fy, fz


@njit(cache=True)
def energy_force(q, O, pp, F):
    """U(q) (no bias) and F = -grad U into F (2, 3)."""
    r0, kb = pp[4], pp[5]
    U = 0.0
    for b in range(2):
        e, fx, fy, fz = bead_lj(q[b, 0], q[b, 1], q[b, 2], O, pp)
        U += e
        F[b, 0] = fx
        F[b, 1] = fy
        F[b, 2] = fz
    dx = q[0, 0] - q[1, 0]
    dy = q[0, 1] - q[1, 1]
    dz = q[0, 2] - q[1, 2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    U += 0.5 * kb * (r - r0) ** 2
    c = -kb * (r - r0) / r
    F[0, 0] += c * dx
    F[0, 1] += c * dy
    F[0, 2] += c * dz
    F[1, 0] -= c * dx
    F[1, 1] -= c * dy
    F[1, 2] -= c * dz
    return U


@njit(cache=True)
def phi_of(q, a):
    x = 0.5 * (q[0, 0] + q[1, 0])
    p = (2.0 * math.pi / a) * x
    return (p + math.pi) - 2.0 * math.pi * math.floor((p + math.pi) / (2.0 * math.pi)) - math.pi


@njit(cache=True)
def bias(phi, ac, bc):
    """A(phi) = sum_k ac[k] cos(k phi) + bc[k] sin(k phi) (k >= 1) and dA/dphi."""
    A = 0.0
    dA = 0.0
    for k in range(1, ac.shape[0]):
        c = math.cos(k * phi)
        s = math.sin(k * phi)
        A += ac[k] * c + bc[k] * s
        dA += k * (-ac[k] * s + bc[k] * c)
    return A, dA


@njit(cache=True)
def pbin(phi):
    j = int(math.floor((phi + math.pi) / (2.0 * math.pi) * NBIN))
    if j < 0:
        j = 0
    if j > NBIN - 1:
        j = NBIN - 1
    return j


@njit(cache=True)
def run_molecules(scheme, q0s, O, pp, ac, bc, kappa, centre, dt, n_steps, burn_in, n_pre_mc, rng, d_bead, d_trans,
                  sample_every, nfine):
    """Independent molecules, one after the other, each from q0s[m]: ``n_pre_mc`` Metropolis steps
    (exact relaxation of the start, no clamps), then ``burn_in`` + ``n_steps`` steps of ``scheme``.
    Bias: kappa > 0 -> umbrella U + (kappa/2) wrap(phi - centre)^2 (i.e. A = -(kappa/2) wrap^2);
    kappa == 0 -> Fourier bias U - A(phi).  Deposits every ``sample_every`` steps after burn-in:
    fine phi histogram (nfine bins, for WHAM), coarse (180) counts and sums of the local mean force
    (physical force only), bond-length and energy moments.  Returns a tuple (keys ``_KEYS``)."""
    kT, a = pp[0], pp[7]
    Cf = np.zeros(nfine)
    C = np.zeros(NBIN)
    Mf = np.zeros(NBIN)
    Mf2 = np.zeros(NBIN)
    sb = 0.0; sb2 = 0.0; sU = 0.0; nd = 0.0
    n_acc = 0.0; n_prop = 0.0; n_tacc = 0.0; n_tprop = 0.0
    cross = 0.0
    max_disp = 0.0
    F = np.zeros((2, 3))
    Fn = np.zeros((2, 3))
    q = np.zeros((2, 3))
    qn = np.zeros((2, 3))
    R = np.zeros((2, 3))
    ns = math.sqrt(2.0 * dt * kT)
    gpx = math.pi / a                      # d phi / d x_bead
    twopi = 2.0 * math.pi
    for m in range(q0s.shape[0]):
        for b in range(2):
            for k in range(3):
                q[b, k] = q0s[m, b, k]
        U = energy_force(q, O, pp, F)
        ph = phi_of(q, a)
        if kappa > 0.0:
            dd = ph - centre
            dd = dd - twopi * math.floor((dd + math.pi) / twopi)
            A = -0.5 * kappa * dd * dd; dA = -kappa * dd
        else:
            A, dA = bias(ph, ac, bc)
        cell = math.floor(0.5 * (q[0, 0] + q[1, 0]) / a)
        for b in range(2):
            for k in range(3):
                R[b, k] = rng.standard_normal()
        total = n_pre_mc + burn_in + n_steps
        for step in range(total):
            sch = MC if step < n_pre_mc else scheme
            st = step - n_pre_mc - burn_in
            if st >= 0 and st % sample_every == 0:
                j = pbin(ph)
                jf = int(math.floor((ph + math.pi) / twopi * nfine))
                if jf < 0:
                    jf = 0
                if jf > nfine - 1:
                    jf = nfine - 1
                Cf[jf] += 1.0
                f = -(a / twopi) * (F[0, 0] + F[1, 0])
                C[j] += 1.0
                Mf[j] += f
                Mf2[j] += f * f
                dx = q[0, 0] - q[1, 0]; dy = q[0, 1] - q[1, 1]; dz = q[0, 2] - q[1, 2]
                rb = math.sqrt(dx * dx + dy * dy + dz * dz)
                sb += rb; sb2 += rb * rb; sU += U; nd += 1.0
            if sch == EM or sch == LM:
                for b in range(2):
                    for k in range(3):
                        dr = F[b, k] * dt
                        if k == 0:
                            dr += dA * gpx * dt
                        if sch == EM:
                            dr += ns * rng.standard_normal()
                        else:
                            rn = rng.standard_normal()
                            dr += ns * 0.5 * (R[b, k] + rn)
                            R[b, k] = rn
                        if abs(dr) > max_disp:
                            max_disp = abs(dr)
                        q[b, k] += dr
                U = energy_force(q, O, pp, F)
                ph = phi_of(q, a)
                if kappa > 0.0:
                    dd = ph - centre
                    dd = dd - twopi * math.floor((dd + math.pi) / twopi)
                    A = -0.5 * kappa * dd * dd; dA = -kappa * dd
                else:
                    A, dA = bias(ph, ac, bc)
            else:
                # one MC step = one single-bead move (random bead) + one rigid translation; both symmetric
                for move in range(2):
                    for k in range(3):
                        qn[0, k] = q[0, k]
                        qn[1, k] = q[1, k]
                    if move == 0:
                        b = 0 if rng.random() < 0.5 else 1
                        for k in range(3):
                            qn[b, k] = q[b, k] + d_bead * (2.0 * rng.random() - 1.0)
                    else:
                        for k in range(3):
                            t = d_trans * (2.0 * rng.random() - 1.0)
                            qn[0, k] = q[0, k] + t
                            qn[1, k] = q[1, k] + t
                    Un = energy_force(qn, O, pp, Fn)
                    phn = phi_of(qn, a)
                    if kappa > 0.0:
                        dd = phn - centre
                        dd = dd - twopi * math.floor((dd + math.pi) / twopi)
                        An = -0.5 * kappa * dd * dd; dAn = -kappa * dd
                    else:
                        An, dAn = bias(phn, ac, bc)
                    la = -((Un - An) - (U - A)) / kT
                    if move == 0:
                        n_prop += 1.0
                    else:
                        n_tprop += 1.0
                    if la >= 0.0 or rng.random() < math.exp(la):
                        if move == 0:
                            n_acc += 1.0
                        else:
                            n_tacc += 1.0
                        for k in range(3):
                            q[0, k] = qn[0, k]
                            q[1, k] = qn[1, k]
                            F[0, k] = Fn[0, k]
                            F[1, k] = Fn[1, k]
                        U = Un; ph = phn; A = An; dA = dAn
            if st >= 0:
                c2 = math.floor(0.5 * (q[0, 0] + q[1, 0]) / a)
                if c2 != cell:
                    cross += 1.0
                cell = c2
            else:
                cell = math.floor(0.5 * (q[0, 0] + q[1, 0]) / a)
    return Cf, C, Mf, Mf2, sb, sb2, sU, nd, n_acc, n_prop, n_tacc, n_tprop, cross, max_disp


_KEYS = ("C_fine", "C", "Mf", "Mf2", "sum_bond", "sum_bond2", "sum_U", "n_dep", "n_acc", "n_prop", "n_tacc", "n_tprop",
         "cell_crossings", "max_step_component")


def fourier_fit(phi_grid, F, K=24):
    """Least-squares truncated Fourier series (k = 1..K, constant dropped) of a periodic profile."""
    X = [np.cos(k * phi_grid) for k in range(1, K + 1)] + [np.sin(k * phi_grid) for k in range(1, K + 1)]
    X = np.stack([np.ones_like(phi_grid)] + X, 1)
    coef, *_ = np.linalg.lstsq(X, F, rcond=None)
    ac = np.zeros(K + 1); bc = np.zeros(K + 1)
    ac[1:] = coef[1:K + 1]
    bc[1:] = coef[K + 1:]
    return ac, bc


def initial_states(n, seed, a, L, r0=1.54):
    """Molecules at random alpha-cage centres (a/2 + i a, ...) with jitter 0.5 A, random orientation."""
    g = np.random.default_rng(int(seed))
    S = int(round(L / a))
    cages = (np.array([[i + 0.5, j + 0.5, k + 0.5] for i in range(S) for j in range(S) for k in range(S)])) * a
    com = cages[g.integers(0, len(cages), n)] + 0.5 * g.standard_normal((n, 3))
    u = g.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.stack([com + 0.5 * r0 * u, com - 0.5 * r0 * u], 1)


def window_states(n, seed, a, L, centre, r0=1.54, jitter=0.2):
    """Molecules with COM on the cage-window-cage axis (y, z at alpha-cage centres) at phi = centre,
    random cell, random orientation -- the existing reference's start (COM x moved onto the window centre)."""
    g = np.random.default_rng(int(seed))
    S = int(round(L / a))
    x = a * centre / (2.0 * np.pi) + a * g.integers(0, S, n)
    y = a * (g.integers(0, S, n) + 0.5)
    z = a * (g.integers(0, S, n) + 0.5)
    com = np.stack([x, y, z], 1) + jitter * g.standard_normal((n, 3))
    u = g.standard_normal((n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return np.stack([com + 0.5 * r0 * u, com - 0.5 * r0 * u], 1)


def run(scheme, T, n_mol, n_steps, burn_in, seed, dt=2e-4, kappa=0.0, centre=0.0, ac=None, bc=None, n_pre_mc=5000,
        d_bead=0.08, d_trans=0.3, sample_every=5, nfine=1800):
    O, a, L = framework()
    pp = params(T)
    q0 = window_states(n_mol, seed, a, L, centre) if kappa > 0 else initial_states(n_mol, seed, a, L)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    ac = np.zeros(2) if ac is None else np.asarray(ac, float)
    bc = np.zeros(2) if bc is None else np.asarray(bc, float)
    out = run_molecules(SCHEMES[scheme], q0, O, pp, ac, bc, float(kappa), float(centre), float(dt), int(n_steps),
                        int(burn_in), int(n_pre_mc), rng, float(d_bead), float(d_trans), int(sample_every), int(nfine))
    return {k: (np.asarray(v) if isinstance(v, np.ndarray) else float(v)) for k, v in zip(_KEYS, out)}


def wham(H, centres, kappa, kT, nfine=None, tol=1e-12, max_iter=200000, mode="fine"):
    """Binned periodic WHAM.  H: (K, nb) per-window histograms on nb equal bins of [-pi, pi).
    mode 'fine': bias evaluated at bin centres of the given (fine) bins; returns F on these bins
    (kJ/mol, -kT log p, mean zero over bins with p > 0).  Bin probabilities p_j sum to 1."""
    H = np.asarray(H, float)
    K, nb = H.shape
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    d = mid[None, :] - np.asarray(centres)[:, None]
    d = d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))
    w = 0.5 * kappa * d ** 2 / kT                      # beta * bias
    N = H.sum(1)
    num = H.sum(0)
    f = np.zeros(K)
    for it in range(max_iter):
        den = (N[:, None] * np.exp(f[:, None] - w)).sum(0)
        p = np.where(num > 0, num / np.maximum(den, 1e-300), 0.0)
        p /= p.sum()
        fn = -np.log((p[None, :] * np.exp(-w)).sum(1))
        fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol:
            f = fn
            break
        f = fn
    return p, it
