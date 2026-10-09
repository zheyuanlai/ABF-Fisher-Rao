"""Independent float64 model of the WCA dimer for NUMERICAL VALIDATION (2026-10-09).

Written for docs/numerical_validation/ (WCA_FORCE_AUDIT.md, WCA_LANGEVIN_VALIDATION.md,
WCA_REFERENCE_AUDIT.md).  Nothing in the production engines imports this module; it never
modifies them.  It provides

* ``energy_force``: the INTENDED potential, written from the physical definition and not from the
  production code (own minimum image, own pair loop, no clamps, no clips):

      U(q) = sum_{i<j, (i,j) != (0,1), r_ij < r_c} V_WCA(r_ij) + V_dim(r_01) + U_wall(z),
      V_WCA(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps,   r_c = 2^(1/6) sigma,
      V_dim(r) = h (1 - u^2)^2,  u = (r - r_c - w)/w,
      z = xi(q) = (r_01 - r_c)/(2 w),
      U_wall(z) = (k_w/2) [ (z - z_max)_+^2 + (z_min - z)_+^2 ]   (zero on [z_min, z_max]),

  in a periodic square box of side L = n_dim * a, minimum image; F = -grad U analytically.
  ``U_wall`` is the potential whose gradient the production RC wall force is (it is zero inside the
  evaluation window, so it does not change F(z) there).

* chains that sample (approximately or exactly) exp(-beta U):
    EM_IMPL   Euler-Maruyama with the PRODUCTION drift (``wca_numba.wca_force`` + the per-particle
              force clip + the RC wall, exactly the unbiased branch of ``wca_numba.simulate``);
    EM_TRUE   Euler-Maruyama with -grad U (no clip, no clamp);
    LM_TRUE   Leimkuhler-Matthews: q' = q + dt F(q) + sqrt(2 dt/beta) (R_n + R_{n+1})/2
              (the high-friction limit of BAOAB; second order for the invariant measure);
    BAOAB     underdamped Langevin, unit mass, friction gamma, B-A-O-A-B splitting, -grad U;
    MALA      EM proposal with a Metropolis-Hastings accept/reject on exp(-beta U): EXACT;
    MC        Metropolis Monte Carlo (single-particle displacements + a dimer-stretch move with
              the 2-D Jacobian r'/r): EXACT, and uses ONLY the energy -- never a force.

  Every chain accumulates on the 160 production bins: visit counts C_j, the local mean force with
  the TRUE force (M_true_j, no sample clip) and, for EM_IMPL, with the production-clipped force and
  the production sample clip (M_impl_j); energy moments; counters for every regularisation (force
  clip binding, pair inside min_r, mean-force sample clip); a coarse z time series.

The local mean force is the Lelievre-Rousset-Stoltz expression for xi = (r_01 - r_c)/(2w) in 2-D:
G = grad xi / |grad xi|^2 = (w d/r, -w d/r) with d = q_0 - q_1, div G = 2 w / r, so
f = grad U . G - beta^{-1} div G = (w/r) d . (F_1 - F_0) - 2 w / (beta r).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from numba import njit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

# scheme ids
EM_IMPL, EM_TRUE, LM_TRUE, BAOAB, MALA, MC = 0, 1, 2, 3, 4, 5
SCHEMES = {"EM_IMPL": EM_IMPL, "EM_TRUE": EM_TRUE, "LM_TRUE": LM_TRUE, "BAOAB": BAOAB, "MALA": MALA, "MC": MC}

#: the accepted physical cell (wca_numba.ACCEPTED_CFG): beta 1, 100 particles, a 1.5, h 2, w 2
DEFAULT_PHYS = dict(n_dim=10, a=1.5, sigma=1.0, epsilon=1.0, h=2.0, w=2.0, beta=1.0,
                    wall=80.0, z_min=-0.2, z_max=1.2, min_r=0.65, force_clip=250.0, mf_clip=500.0)
NB = 160            # production bins on [z_min, z_max]
NRMIN = 64          # bins of the per-step minimum solvent-pair distance on [0.5, r_c]


def phys_array(p):
    """Pack: [L, sigma, eps, h, w, beta, r_c, wall, z_min, z_max, min_r, force_clip, mf_clip, P]."""
    L = p["n_dim"] * p["a"]
    rc = 2.0 ** (1.0 / 6.0) * p["sigma"]
    return np.array([L, p["sigma"], p["epsilon"], p["h"], p["w"], p["beta"], rc, p["wall"], p["z_min"],
                     p["z_max"], p["min_r"] * p["sigma"], p["force_clip"], p["mf_clip"], p.get("P", p["n_dim"] ** 2)],
                    dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# The intended potential (independent implementation)
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def mimg(d, L):
    """Minimum image by floor (the production code uses rint; the two differ only at |d| = L/2)."""
    return d - L * math.floor(d / L + 0.5)


@njit(cache=True)
def wca_pair(r2, sig2, eps, rc2):
    """(V_WCA, s) for squared distance r2 with F_i = s * (q_i - q_j); zero beyond the cutoff."""
    if r2 >= rc2:
        return 0.0, 0.0
    if r2 < 1e-300:
        return 1e300, 0.0
    x2 = sig2 / r2
    x6 = x2 * x2 * x2
    x12 = x6 * x6
    V = 4.0 * eps * (x12 - x6) + eps
    s = 4.0 * eps * (12.0 * x12 - 6.0 * x6) / r2           # -V'(r)/r
    return V, s


@njit(cache=True)
def dimer_terms(r, h, w, rc, wall, z_min, z_max, beta):
    """(V_dim + U_wall, dU/dr) of the dimer bond length r."""
    u = (r - rc - w) / w
    V = h * (1.0 - u * u) ** 2
    dV = -4.0 * h * u * (1.0 - u * u) / w
    z = (r - rc) / (2.0 * w)
    if z > z_max:
        e = z - z_max
        V += 0.5 * wall * e * e
        dV += wall * e / (2.0 * w)
    elif z < z_min:
        e = z - z_min
        V += 0.5 * wall * e * e
        dV += wall * e / (2.0 * w)
    return V, dV


@njit(cache=True)
def energy_force_ref(q, ph, F):
    """U(q) and F = -grad U (into F, shape (P, 2)) of the intended potential: the plain double loop
    (the readable definition; ``energy_force`` is the screened fast version, tested equal).
    Returns (U, U_wca, min solvent-pair distance^2 over pairs != (0,1))."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max = ph[7], ph[8], ph[9]
    P = q.shape[0]
    sig2 = sig * sig
    rc2 = rc * rc
    for p in range(P):
        F[p, 0] = 0.0
        F[p, 1] = 0.0
    Uw = 0.0
    m2 = 1e300
    for i in range(P - 1):
        xi = q[i, 0]
        yi = q[i, 1]
        for j in range(i + 1, P):
            if i == 0 and j == 1:
                continue
            dx = mimg(xi - q[j, 0], L)
            dy = mimg(yi - q[j, 1], L)
            r2 = dx * dx + dy * dy
            if r2 < m2:
                m2 = r2
            if r2 < rc2:
                V, s = wca_pair(r2, sig2, eps, rc2)
                Uw += V
                F[i, 0] += s * dx
                F[i, 1] += s * dy
                F[j, 0] -= s * dx
                F[j, 1] -= s * dy
    dx = mimg(q[0, 0] - q[1, 0], L)
    dy = mimg(q[0, 1] - q[1, 1], L)
    r = math.sqrt(dx * dx + dy * dy)
    Vd, dV = dimer_terms(r, h, w, rc, wall, z_min, z_max, beta)
    c = -dV / r
    F[0, 0] += c * dx
    F[0, 1] += c * dy
    F[1, 0] -= c * dx
    F[1, 1] -= c * dy
    return Uw + Vd, Uw, m2


@njit(cache=True)
def energy_only(q, ph):
    """U(q) without forces (used by MC/MALA and the finite-difference tests)."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max = ph[7], ph[8], ph[9]
    P = q.shape[0]
    sig2 = sig * sig
    rc2 = rc * rc
    U = 0.0
    for i in range(P - 1):
        for j in range(i + 1, P):
            if i == 0 and j == 1:
                continue
            dx = mimg(q[i, 0] - q[j, 0], L)
            dy = mimg(q[i, 1] - q[j, 1], L)
            r2 = dx * dx + dy * dy
            if r2 < rc2:
                U += wca_pair(r2, sig2, eps, rc2)[0]
    dx = mimg(q[0, 0] - q[1, 0], L)
    dy = mimg(q[0, 1] - q[1, 1], L)
    U += dimer_terms(math.sqrt(dx * dx + dy * dy), h, w, rc, wall, z_min, z_max, beta)[0]
    return U


@njit(cache=True)
def particle_energy_ref(q, i, x, y, ph):
    """Energy of particle i placed at (x, y) with every other particle (pair + dimer terms): the
    plain loop (``particle_energy`` is the screened fast version, tested equal)."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max = ph[7], ph[8], ph[9]
    P = q.shape[0]
    sig2 = sig * sig
    rc2 = rc * rc
    U = 0.0
    for j in range(P):
        if j == i:
            continue
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            dx = mimg(x - q[j, 0], L)
            dy = mimg(y - q[j, 1], L)
            U += dimer_terms(math.sqrt(dx * dx + dy * dy), h, w, rc, wall, z_min, z_max, beta)[0]
            continue
        dx = mimg(x - q[j, 0], L)
        dy = mimg(y - q[j, 1], L)
        r2 = dx * dx + dy * dy
        if r2 < rc2:
            U += wca_pair(r2, sig2, eps, rc2)[0]
    return U


@njit(cache=True)
def energy_force(q, ph, F):
    """``energy_force_ref`` with a branch-free distance screen (vectorisable) before the pair terms.
    Same arithmetic per active pair, same pair order; tested equal to round-off."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max = ph[7], ph[8], ph[9]
    P = q.shape[0]
    sig2 = sig * sig
    rc2 = rc * rc
    invL = 1.0 / L
    r2b = np.empty(P)
    dxb = np.empty(P)
    dyb = np.empty(P)
    for p in range(P):
        F[p, 0] = 0.0
        F[p, 1] = 0.0
    Uw = 0.0
    m2 = 1e300
    for i in range(P - 1):
        xi = q[i, 0]
        yi = q[i, 1]
        for j in range(i + 1, P):
            dx = xi - q[j, 0]
            dx = dx - L * np.rint(dx * invL)
            dy = yi - q[j, 1]
            dy = dy - L * np.rint(dy * invL)
            dxb[j] = dx
            dyb[j] = dy
            r2b[j] = dx * dx + dy * dy
        j0 = 2 if i == 0 else i + 1
        for j in range(j0, P):
            r2 = r2b[j]
            if r2 < m2:
                m2 = r2
            if r2 < rc2:
                V, s = wca_pair(r2, sig2, eps, rc2)
                dx = dxb[j]
                dy = dyb[j]
                Uw += V
                F[i, 0] += s * dx
                F[i, 1] += s * dy
                F[j, 0] -= s * dx
                F[j, 1] -= s * dy
    dx = mimg(q[0, 0] - q[1, 0], L)
    dy = mimg(q[0, 1] - q[1, 1], L)
    r = math.sqrt(dx * dx + dy * dy)
    Vd, dV = dimer_terms(r, h, w, rc, wall, z_min, z_max, beta)
    c = -dV / r
    F[0, 0] += c * dx
    F[0, 1] += c * dy
    F[1, 0] -= c * dx
    F[1, 1] -= c * dy
    return Uw + Vd, Uw, m2


@njit(cache=True)
def particle_energy(q, i, x, y, ph):
    """``particle_energy_ref`` with a branch-free distance screen; tested equal to round-off."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max = ph[7], ph[8], ph[9]
    P = q.shape[0]
    sig2 = sig * sig
    rc2 = rc * rc
    invL = 1.0 / L
    r2b = np.empty(P)
    for j in range(P):
        dx = x - q[j, 0]
        dx = dx - L * np.rint(dx * invL)
        dy = y - q[j, 1]
        dy = dy - L * np.rint(dy * invL)
        r2b[j] = dx * dx + dy * dy
    U = 0.0
    for j in range(P):
        if j == i:
            continue
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            U += dimer_terms(math.sqrt(r2b[j]), h, w, rc, wall, z_min, z_max, beta)[0]
            continue
        r2 = r2b[j]
        if r2 < rc2:
            U += wca_pair(r2, sig2, eps, rc2)[0]
    return U


@njit(cache=True)
def dimer_geom(q, L, rc, w):
    dx = mimg(q[0, 0] - q[1, 0], L)
    dy = mimg(q[0, 1] - q[1, 1], L)
    r = math.sqrt(dx * dx + dy * dy)
    return dx, dy, r, (r - rc) / (2.0 * w)


@njit(cache=True)
def local_mf(dx, dy, r, F, w, beta):
    """(w/r) d.(F_1 - F_0) - 2w/(beta r)."""
    return (w / r) * (dx * (F[1, 0] - F[0, 0]) + dy * (F[1, 1] - F[0, 1])) - 2.0 * w / (beta * r)


@njit(cache=True)
def zbin(z, z_min, z_max, nb):
    """Bin of z on nb equal bins of [z_min, z_max]; -1 below, nb above."""
    if z < z_min:
        return -1
    if z >= z_max:
        return nb
    j = int((z - z_min) / (z_max - z_min) * nb)
    return j if j < nb else nb - 1


@njit(cache=True)
def clip2(fx, fy, c):
    n2 = fx * fx + fy * fy
    if n2 <= c * c:
        return fx, fy, False
    s = c / math.sqrt(n2)
    return fx * s, fy * s, True


# ---------------------------------------------------------------------------------------------
# The production drift (EM_IMPL): wca_numba kernels, unbiased branch of simulate()
# ---------------------------------------------------------------------------------------------
def _prod_kernels():
    import wca_numba as wn
    return wn.wca_force, wn._clip_vec


_wca_force, _clip_vec = _prod_kernels()


@njit(cache=True)
def prod_drift(q, ph, P, Fraw, Fdr, xs, ys, r2b, pl_i, pl_j, pl_fx, pl_fy):
    """The production unbiased drift: Fraw = wca_force (min_r clamp inside), Fdr = clip(clip(Fraw) +
    wall) exactly as wca_numba.simulate with abf_scale = 0 (the zero ABF term and the second clip of
    an already clipped vector are kept for fidelity).  Returns the number of particles whose RAW
    force exceeded the clip (the clip binding)."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max, rmin, fclip = ph[7], ph[8], ph[9], ph[10], ph[11]
    _wca_force(q, P, L, sig, 4.0 * eps, rc, rmin, h, w, rc, xs, ys, r2b, pl_i, pl_j, pl_fx, pl_fy, Fraw, 1)
    dx = q[0, 0] - q[1, 0]
    dx = dx - L * np.rint(dx / L)
    dy = q[0, 1] - q[1, 1]
    dy = dy - L * np.rint(dy / L)
    r01 = math.sqrt(dx * dx + dy * dy)
    z = (r01 - rc) / (2.0 * w)
    up = z - z_max
    if up < 0.0:
        up = 0.0
    lo = z - z_min
    if lo > 0.0:
        lo = 0.0
    mw = -(wall * (up + lo))
    den4 = 2.0 * w * r01
    gx = dx / den4
    gy = dy / den4
    nclip = 0
    for p in range(P):
        if Fraw[p, 0] * Fraw[p, 0] + Fraw[p, 1] * Fraw[p, 1] > fclip * fclip:
            nclip += 1
        tx, ty = _clip_vec(Fraw[p, 0], Fraw[p, 1], fclip, 1)
        tx, ty = _clip_vec(tx + 0.0, ty + 0.0, fclip, 1)
        if p == 0:
            tx = tx + mw * gx
            ty = ty + mw * gy
        elif p == 1:
            tx = tx + mw * (-gx)
            ty = ty + mw * (-gy)
        tx, ty = _clip_vec(tx, ty, fclip, 1)
        Fdr[p, 0] = tx
        Fdr[p, 1] = ty
    return nclip


# ---------------------------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def _wrap(x, L):
    if 0.0 <= x < L:
        return x
    return x - L * math.floor(x / L)


@njit(cache=True)
def run_chain(scheme, q0, ph, dt, gamma, n_steps, rng, mc_delta, mc_dimer_delta, mc_dimer_moves,
              z_stride, burn_in, nb, diag_stride):
    """One chain of ``n_steps`` steps (MC: sweeps).  Structure of every step: DEPOSIT the current
    state (the state after ``step`` completed steps; only if step >= burn_in), then MOVE.  So the
    deposits are the states after burn_in, ..., n_steps - 1 completed steps, as in the production
    engine (which deposits at state ``step`` before moving).  Returns a tuple (keys ``_OUT_KEYS``).

    EM_IMPL: the drift is the production one (``prod_drift``); its RAW production force equals
    -grad U whenever no solvent pair is inside min_r, so M_true uses it (and the steps where a pair
    is inside min_r are counted at the diagnostic stride); M_impl uses the clipped force with the
    production sample clip.  Energies and the minimum pair distance are evaluated every
    ``diag_stride`` steps for EM_IMPL (every step for the other schemes, which need the true force)."""
    L, sig, eps, h, w, beta, rc = ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]
    wall, z_min, z_max, rmin, fclip, mfclip = ph[7], ph[8], ph[9], ph[10], ph[11], ph[12]
    P = q0.shape[0]
    q = q0.copy()
    F = np.zeros((P, 2))       # force at q (EM_IMPL: RAW production force)
    Fd = np.zeros((P, 2))      # EM_IMPL drift
    Fn = np.zeros((P, 2))
    qn = np.zeros((P, 2))
    p_mom = np.zeros((P, 2))
    Rprev = np.zeros((P, 2))
    xs = np.empty(P); ys = np.empty(P); r2b = np.empty(P)
    npm = P * (P - 1) // 2 + 1
    pl_i = np.empty(npm, dtype=np.int64); pl_j = np.empty(npm, dtype=np.int64)
    pl_fx = np.empty(npm); pl_fy = np.empty(npm)

    C = np.zeros(nb + 2)                 # [below z_min, bins..., at/above z_max]
    Mt = np.zeros(nb + 2)                # local mean force, TRUE force, no sample clip
    Mi = np.zeros(nb + 2)                # local mean force, production clipped force + sample clip (EM_IMPL)
    Mt2 = np.zeros(nb + 2)               # second moment of the true local mean force
    rmin_h = np.zeros(NRMIN + 2)         # minimum solvent-pair distance histogram on [0.5, r_c] (+ under/over)
    n_z = n_steps // z_stride + 2
    zts = np.zeros(n_z, dtype=np.float32)
    nzs = 0
    sU = 0.0; sU2 = 0.0; sUw = 0.0; nU = 0.0
    n_clip_part = 0.0                    # particle-steps with |F_raw| > force_clip (EM_IMPL)
    n_clip_steps = 0.0                   # steps with >= 1 clipped particle (EM_IMPL)
    n_minr = 0.0                         # diagnostic samples with a solvent pair inside min_r
    n_mf_clip = 0.0                      # deposits with |f_impl| > mf_clip (EM_IMPL)
    max_disp = 0.0                       # largest single-particle displacement in one step (squared, then sqrt)
    n_acc = 0.0; n_prop = 0.0            # MALA / MC single-particle acceptance
    n_dacc = 0.0; n_dprop = 0.0          # MC dimer-stretch acceptance
    sp = np.zeros(4)                     # BAOAB: sum p, p^2, p^3, p^4 over all dof at full steps
    sK = 0.0
    nonfinite = 0
    noise_scale = math.sqrt(2.0 * dt / beta)
    sq_kT = math.sqrt(1.0 / beta)
    c1 = math.exp(-gamma * dt)
    c3 = math.sqrt(1.0 - c1 * c1) * sq_kT
    U = 0.0; Uw = 0.0; m2 = 1e300

    # ---------------- force at the initial state
    if scheme == EM_IMPL:
        nclip = prod_drift(q, ph, P, F, Fd, xs, ys, r2b, pl_i, pl_j, pl_fx, pl_fy)
    else:
        U, Uw, m2 = energy_force(q, ph, F)
    if scheme == BAOAB:
        for p in range(P):
            p_mom[p, 0] = sq_kT * rng.standard_normal()
            p_mom[p, 1] = sq_kT * rng.standard_normal()
    if scheme == LM_TRUE:
        for p in range(P):
            Rprev[p, 0] = rng.standard_normal()
            Rprev[p, 1] = rng.standard_normal()

    for step in range(n_steps):
        # ================================================ deposit the current state
        if step >= burn_in:
            if scheme == EM_IMPL:
                if nclip > 0:
                    n_clip_part += nclip
                    n_clip_steps += 1.0
                diag = (step - burn_in) % diag_stride == 0
                if diag:
                    U, Uw, m2 = energy_force(q, ph, Fn)
            else:
                diag = True
            if not math.isfinite(U):
                nonfinite += 1
                break
            dx, dy, r, z = dimer_geom(q, L, rc, w)
            j = zbin(z, z_min, z_max, nb) + 1
            fl = local_mf(dx, dy, r, F, w, beta)
            C[j] += 1.0
            Mt[j] += fl
            Mt2[j] += fl * fl
            if scheme == EM_IMPL:
                f0x, f0y = _clip_vec(F[0, 0], F[0, 1], fclip, 1)
                f1x, f1y = _clip_vec(F[1, 0], F[1, 1], fclip, 1)
                fi = (w / r) * (dx * (f1x - f0x) + dy * (f1y - f0y)) - 2.0 * w / (beta * r)
                if fi > mfclip:
                    fi = mfclip
                    n_mf_clip += 1.0
                elif fi < -mfclip:
                    fi = -mfclip
                    n_mf_clip += 1.0
                Mi[j] += fi
            if diag:
                sU += U; sU2 += U * U; sUw += Uw; nU += 1.0
                rmn = math.sqrt(m2)
                if rmn < rmin:
                    n_minr += 1.0
                kb = int(math.floor((rmn - 0.5) / (rc - 0.5) * NRMIN)) + 1
                if kb < 0:
                    kb = 0
                if kb > NRMIN + 1:
                    kb = NRMIN + 1
                rmin_h[kb] += 1.0
            if scheme == BAOAB:
                for p in range(P):
                    for k in range(2):
                        v = p_mom[p, k]
                        sp[0] += v; sp[1] += v * v; sp[2] += v * v * v; sp[3] += v * v * v * v
                        sK += 0.5 * v * v
            if (step - burn_in) % z_stride == 0 and nzs < n_z:
                zts[nzs] = z
                nzs += 1

        # ================================================ move
        if scheme == EM_IMPL:
            for p in range(P):
                # the production update, same association and wrap (wca_numba.simulate)
                nx = (q[p, 0] + dt * Fd[p, 0]) + noise_scale * rng.standard_normal()
                ny = (q[p, 1] + dt * Fd[p, 1]) + noise_scale * rng.standard_normal()
                ex = nx - q[p, 0]
                ey = ny - q[p, 1]
                d2 = ex * ex + ey * ey
                if d2 > max_disp:
                    max_disp = d2
                q[p, 0] = nx if 0.0 < nx < L else nx % L
                q[p, 1] = ny if 0.0 < ny < L else ny % L
            nclip = prod_drift(q, ph, P, F, Fd, xs, ys, r2b, pl_i, pl_j, pl_fx, pl_fy)
        elif scheme == EM_TRUE:
            for p in range(P):
                ex = dt * F[p, 0] + noise_scale * rng.standard_normal()
                ey = dt * F[p, 1] + noise_scale * rng.standard_normal()
                d2 = ex * ex + ey * ey
                if d2 > max_disp:
                    max_disp = d2
                q[p, 0] = _wrap(q[p, 0] + ex, L)
                q[p, 1] = _wrap(q[p, 1] + ey, L)
            U, Uw, m2 = energy_force(q, ph, F)
        elif scheme == LM_TRUE:
            for p in range(P):
                rx = rng.standard_normal()
                ry = rng.standard_normal()
                ex = dt * F[p, 0] + noise_scale * 0.5 * (Rprev[p, 0] + rx)
                ey = dt * F[p, 1] + noise_scale * 0.5 * (Rprev[p, 1] + ry)
                Rprev[p, 0] = rx
                Rprev[p, 1] = ry
                d2 = ex * ex + ey * ey
                if d2 > max_disp:
                    max_disp = d2
                q[p, 0] = _wrap(q[p, 0] + ex, L)
                q[p, 1] = _wrap(q[p, 1] + ey, L)
            U, Uw, m2 = energy_force(q, ph, F)
        elif scheme == BAOAB:
            for p in range(P):
                p_mom[p, 0] += 0.5 * dt * F[p, 0]                                    # B
                p_mom[p, 1] += 0.5 * dt * F[p, 1]
                qx = q[p, 0] + 0.5 * dt * p_mom[p, 0]                                # A
                qy = q[p, 1] + 0.5 * dt * p_mom[p, 1]
                p_mom[p, 0] = c1 * p_mom[p, 0] + c3 * rng.standard_normal()          # O
                p_mom[p, 1] = c1 * p_mom[p, 1] + c3 * rng.standard_normal()
                qx = qx + 0.5 * dt * p_mom[p, 0]                                     # A
                qy = qy + 0.5 * dt * p_mom[p, 1]
                ex = qx - q[p, 0]
                ey = qy - q[p, 1]
                d2 = ex * ex + ey * ey
                if d2 > max_disp:
                    max_disp = d2
                q[p, 0] = _wrap(qx, L)
                q[p, 1] = _wrap(qy, L)
            U, Uw, m2 = energy_force(q, ph, F)
            for p in range(P):
                p_mom[p, 0] += 0.5 * dt * F[p, 0]                                    # B
                p_mom[p, 1] += 0.5 * dt * F[p, 1]
        elif scheme == MALA:
            for p in range(P):
                qn[p, 0] = q[p, 0] + dt * F[p, 0] + noise_scale * rng.standard_normal()
                qn[p, 1] = q[p, 1] + dt * F[p, 1] + noise_scale * rng.standard_normal()
            Un, Uwn, m2n = energy_force(qn, ph, Fn)
            # log alpha = -beta dU + log q(x|y) - log q(y|x); q(y|x) ~ exp(-beta |y - x - dt F(x)|^2 / (4 dt))
            lf = 0.0
            lb = 0.0
            for p in range(P):
                for k in range(2):
                    dfw = (qn[p, k] - q[p, k]) - dt * F[p, k]
                    dbw = (q[p, k] - qn[p, k]) - dt * Fn[p, k]
                    lf += dfw * dfw
                    lb += dbw * dbw
            la = -beta * (Un - U) - (lb - lf) * beta / (4.0 * dt)
            n_prop += 1.0
            if math.isfinite(la) and (la >= 0.0 or rng.random() < math.exp(la)):
                n_acc += 1.0
                for p in range(P):
                    ex = qn[p, 0] - q[p, 0]
                    ey = qn[p, 1] - q[p, 1]
                    d2 = ex * ex + ey * ey
                    if d2 > max_disp:
                        max_disp = d2
                    q[p, 0] = _wrap(qn[p, 0], L)
                    q[p, 1] = _wrap(qn[p, 1], L)
                    F[p, 0] = Fn[p, 0]
                    F[p, 1] = Fn[p, 1]
                U = Un
                Uw = Uwn
                m2 = m2n
        else:   # MC sweep: P single-particle moves at random sites + mc_dimer_moves dimer stretches
            for t in range(P):
                i = int(rng.random() * P)
                if i >= P:
                    i = P - 1
                x_new = _wrap(q[i, 0] + mc_delta * (2.0 * rng.random() - 1.0), L)
                y_new = _wrap(q[i, 1] + mc_delta * (2.0 * rng.random() - 1.0), L)
                dU = particle_energy(q, i, x_new, y_new, ph) - particle_energy(q, i, q[i, 0], q[i, 1], ph)
                n_prop += 1.0
                if dU <= 0.0 or rng.random() < math.exp(-beta * dU):
                    q[i, 0] = x_new
                    q[i, 1] = y_new
                    n_acc += 1.0
            for t in range(mc_dimer_moves):
                dx = mimg(q[0, 0] - q[1, 0], L)
                dy = mimg(q[0, 1] - q[1, 1], L)
                r = math.sqrt(dx * dx + dy * dy)
                rn = r + mc_dimer_delta * (2.0 * rng.random() - 1.0)
                n_dprop += 1.0
                if rn <= 0.0 or rn >= 0.5 * L:
                    continue
                mx = q[1, 0] + 0.5 * dx
                my = q[1, 1] + 0.5 * dy
                ux = dx / r
                uy = dy / r
                x0 = _wrap(mx + 0.5 * rn * ux, L); y0 = _wrap(my + 0.5 * rn * uy, L)
                x1 = _wrap(mx - 0.5 * rn * ux, L); y1 = _wrap(my - 0.5 * rn * uy, L)
                # both dimer atoms move: E_old/E_new = their energies with everyone, the (0,1) term once
                Uold = (particle_energy(q, 0, q[0, 0], q[0, 1], ph) + particle_energy(q, 1, q[1, 0], q[1, 1], ph)
                        - dimer_terms(r, h, w, rc, wall, z_min, z_max, beta)[0])
                ox0 = q[0, 0]; oy0 = q[0, 1]; ox1 = q[1, 0]; oy1 = q[1, 1]
                q[0, 0] = x0; q[0, 1] = y0; q[1, 0] = x1; q[1, 1] = y1
                Unew = (particle_energy(q, 0, x0, y0, ph) + particle_energy(q, 1, x1, y1, ph)
                        - dimer_terms(rn, h, w, rc, wall, z_min, z_max, beta)[0])
                la = -beta * (Unew - Uold) + math.log(rn / r)          # 2-D Jacobian r'/r
                if la >= 0.0 or rng.random() < math.exp(la):
                    n_dacc += 1.0
                else:
                    q[0, 0] = ox0; q[0, 1] = oy0; q[1, 0] = ox1; q[1, 1] = oy1
            U, Uw, m2 = energy_force(q, ph, F)
    return (C, Mt, Mi, Mt2, rmin_h, zts[:nzs], sU, sU2, sUw, nU, n_clip_part, n_clip_steps, n_minr,
            n_mf_clip, math.sqrt(max_disp), n_acc, n_prop, n_dacc, n_dprop, sp, sK, nonfinite, q)


_OUT_KEYS = ("C", "M_true", "M_impl", "M2_true", "rmin_hist", "z_series", "sum_U", "sum_U2", "sum_Uwca", "n_U",
             "n_clip_particle_steps", "n_clip_steps", "n_minr_diag", "n_mf_clip", "max_step_displacement",
             "n_acc", "n_prop", "n_dimer_acc", "n_dimer_prop", "sum_p_moments", "sum_K", "nonfinite", "q_final")


def chain(scheme, q0, dt=0.0005, n_steps=1000, seed=0, phys=None, gamma=1.0, mc_delta=0.15, mc_dimer_delta=0.3,
          mc_dimer_moves=10, z_stride=100, burn_in=0, diag_stride=1):
    """Run one chain; returns a dict of numpy arrays/scalars (keys ``_OUT_KEYS``)."""
    p = dict(DEFAULT_PHYS, **(phys or {}))
    ph = phys_array(p)
    sid = SCHEMES[scheme] if isinstance(scheme, str) else int(scheme)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    q0 = np.ascontiguousarray(q0, dtype=np.float64)
    out = run_chain(sid, q0, ph, float(dt), float(gamma), int(n_steps), rng, float(mc_delta), float(mc_dimer_delta),
                    int(mc_dimer_moves), int(z_stride), int(burn_in), NB, int(diag_stride))
    return {k: (np.asarray(v) if isinstance(v, np.ndarray) else float(v)) for k, v in zip(_OUT_KEYS, out)}


def lattice_q0(seed, phys=None, jitter=0.015):
    """A lattice start like the production one (dimer at r = r_c along +y from particle 0), own RNG."""
    p = dict(DEFAULT_PHYS, **(phys or {}))
    n, a = p["n_dim"], p["a"]
    L = n * a
    g = np.random.default_rng(int(seed))
    base = np.array([((0.5 + i) * a, (0.5 + j) * a) for i in range(n) for j in range(n)], dtype=np.float64)
    q = (base + g.random(2) * L) % L
    q[2:] = (q[2:] + jitter * g.standard_normal(q[2:].shape)) % L
    rc = 2.0 ** (1.0 / 6.0) * p["sigma"]
    q[1] = (q[0] + np.array([0.0, rc])) % L
    return q


def implemented_energy_torch(q, phys=None):
    """The energy the PRODUCTION torch engine reports (``WCADimerEngine.force(compute_energy=True)``,
    float64 CPU): V_WCA evaluated at r_safe = max(r, min_r) -- flat below min_r -- plus V_dim; no wall."""
    import wca_numba as wn
    torch, core = wn._cpu_torch()
    p = dict(DEFAULT_PHYS, **(phys or {}))
    params = core.DimerWCAParams(n_dim=p["n_dim"], a=p["a"], sigma=p["sigma"], epsilon=p["epsilon"], h=p["h"],
                                 w=p["w"], beta=p["beta"], min_r=p["min_r"], force_clip=p["force_clip"])
    eng = core.WCADimerEngine(params, torch.device("cpu"), torch.float64)
    U, F = eng.force(torch.as_tensor(np.asarray(q)[None], dtype=torch.float64), compute_energy=True)
    return float(U[0]), F[0].numpy().copy()


def production_force(q, phys=None, fma=True):
    """The raw production force (``wca_numba.wca_force``: min_r clamp, no clip) of one configuration."""
    import wca_numba as wn
    p = dict(DEFAULT_PHYS, **(phys or {}))
    ph = phys_array(p)
    q = np.ascontiguousarray(q, dtype=np.float64)
    P = q.shape[0]
    F = np.zeros((P, 2))
    npm = P * (P - 1) // 2 + 1
    wn.wca_force(q, P, ph[0], ph[1], 4.0 * ph[2], ph[6], ph[10], ph[3], ph[4], ph[6], np.empty(P), np.empty(P),
                 np.empty(P), np.empty(npm, dtype=np.int64), np.empty(npm, dtype=np.int64), np.empty(npm),
                 np.empty(npm), F, 1 if fma else 0)
    return F


def production_clipped_force(q, phys=None):
    """clip_forces(raw production force, force_clip) -- the physical force every production sampler moves with."""
    p = dict(DEFAULT_PHYS, **(phys or {}))
    F = production_force(q, phys)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F * np.minimum(1.0, p["force_clip"] / np.maximum(n, 1e-12))
