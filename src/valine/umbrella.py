"""Batched dihedral-restrained MD for the conditional chi1 profiles (plan Stage 2).

Stage 2 asks for ``F(chi1 | phi_j, psi_j)`` at a handful of representative backbone points.
That is a 1-D umbrella calculation per backbone point, and there are only 6 x 24 = 144 windows
in total -- small enough that **every window is one row of a single batched GPU run**, sharing
one force evaluation per step.  This is why Stage 2 costs minutes rather than the hours a
per-window loop would take.

Restraints are harmonic in the *periodic* dihedral difference::

    U = sum_d  0.5 * k_d * wrap(theta_d(q) - theta0_d)^2

with ``wrap`` the shortest signed arc, so the restraint is continuous and differentiable
everywhere except the antipode, which no equilibrated window visits.

Gradients reuse the validated ``alkanes.cv._grad_phi4`` primitive (``vmap(grad(...))`` of the
scalar dihedral) rather than a new autodiff path, so the restraint force is exact by
construction and is checked against finite differences in ``tests/test_valine_umbrella.py``.

The convention is RB internally (``_grad_phi4`` differentiates the RB dihedral) but the value
returned by :func:`dihedrals` is IUPAC, matching `valine.system.angles_np` and
`alanine.cv2d.BackboneCV2D`.  Because the two conventions differ by a constant shift of pi,
the gradient is identical and only the reported angle moves.
"""
from __future__ import annotations

import numpy as np
import torch

from alkanes.cv import _grad_phi4

PI = float(np.pi)


def wrap_to_pi(a):
    """Shortest signed arc, for tensors or arrays."""
    if torch.is_tensor(a):
        return (a + PI) % (2 * PI) - PI
    return (a + PI) % (2 * PI) - PI


def _dihedral_rb(q, idx):
    """RB-convention dihedral (trans at 0) of ``q[:, idx]``; ``q`` is ``(B, A, 3)``."""
    p0, p1, p2, p3 = (q[:, i, :] for i in idx)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    x = (v * w).sum(-1)
    return torch.atan2(-y, -x)


def dihedral_grad_analytic(q, idx):
    """Closed-form ``d(theta)/dq`` for one dihedral; returns ``(B, 4, 3)``.

    The `alkanes` primitive ``_grad_phi4`` is ``vmap(grad(...))`` over `torch.func`, whose
    dispatch cost is ~6 ms per dihedral and **flat in batch** -- the same bottleneck the
    alanine handoff measured for the Hessian path.  With three restrained dihedrals that
    dominated the step (25.4 ms/step, against ~5 ms for the physical force).  The closed form
    below is a dozen fused vector ops and removes the dispatch entirely.

    Standard result for atoms ``i-j-k-l`` with ``b1 = rj-ri``, ``b2 = rk-rj``, ``b3 = rl-rk``,
    ``n1 = b1 x b2``, ``n2 = b2 x b3``.  Agreement with ``_grad_phi4`` is asserted to 1e-12 in
    `tests/test_valine_umbrella.py`; this is an optimisation of an exactly known quantity, not
    a re-derivation of the physics.

    The convention shift between RB and IUPAC is a constant, so this gradient serves both.
    """
    i, j, k, l = idx
    b1 = q[:, j] - q[:, i]
    b2 = q[:, k] - q[:, j]
    b3 = q[:, l] - q[:, k]
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2.norm(dim=-1, keepdim=True)
    n1sq = (n1 * n1).sum(-1, keepdim=True).clamp_min(1e-30)
    n2sq = (n2 * n2).sum(-1, keepdim=True).clamp_min(1e-30)
    b2sq = (b2 * b2).sum(-1, keepdim=True).clamp_min(1e-30)

    g_i = -(b2n / n1sq) * n1
    g_l = (b2n / n2sq) * n2
    c12 = (b1 * b2).sum(-1, keepdim=True) / b2sq
    c32 = (b3 * b2).sum(-1, keepdim=True) / b2sq
    # Middle-atom coefficients fixed against `_grad_phi4` rather than copied from a textbook
    # whose b-vector signs differ.  WARNING for anyone editing this: sum(g) = 0 holds for
    # several *wrong* sign combinations too, so translation invariance is a vacuous check
    # here -- two earlier sign choices passed it while being off by O(1).  The only valid
    # test is the elementwise comparison against `_grad_phi4` in tests/test_valine_umbrella.py.
    g_j = -(c12 + 1.0) * g_i + c32 * g_l
    g_k = c12 * g_i - (c32 + 1.0) * g_l
    return torch.stack([g_i, g_j, g_k, g_l], dim=1)


class DihedralRestraint:
    """Harmonic restraints on a fixed list of dihedrals, with per-walker centres.

    ``quads``   -- list of ``d`` atom 4-tuples
    ``centers`` -- ``(B, d)`` target angles in radians, IUPAC convention
    ``kappas``  -- ``(B, d)`` or ``(d,)`` force constants in kJ/mol/rad^2

    A zero ``kappa`` disables that restraint for that walker, which is how a window can pin
    the backbone while leaving chi1 free (or vice versa) inside the same batch.
    """

    def __init__(self, quads, centers, kappas, n_atoms, device="cpu", dtype=torch.float64):
        self.quads = [tuple(int(i) for i in q) for q in quads]
        self.d = len(self.quads)
        self.n_atoms = int(n_atoms)
        self.centers = torch.as_tensor(centers, device=device, dtype=dtype).reshape(-1, self.d)
        k = torch.as_tensor(kappas, device=device, dtype=dtype)
        self.kappas = k.expand_as(self.centers).contiguous() if k.ndim <= 1 else k
        self.device, self.dtype = device, dtype
        # RB centres: IUPAC = RB + pi, so RB centre = IUPAC centre - pi
        self._centers_rb = wrap_to_pi(self.centers - PI)

    def dihedrals(self, q):
        """``(B, d)`` IUPAC dihedral values."""
        return torch.stack([wrap_to_pi(_dihedral_rb(q, idx) + PI) for idx in self.quads], -1)

    def energy_and_force(self, q):
        """Return ``(U (B,), F (B, A, 3), theta_iupac (B, d))``.

        ``F`` is the restraint force, i.e. ``-dU/dq``, ready to be added to the physical force.
        """
        B = q.shape[0]
        U = q.new_zeros(B)
        F = q.new_zeros(B, self.n_atoms, 3)
        theta = q.new_zeros(B, self.d)
        for a, idx in enumerate(self.quads):
            th_rb = _dihedral_rb(q, idx)
            dth = wrap_to_pi(th_rb - self._centers_rb[:, a])
            k = self.kappas[:, a]
            U = U + 0.5 * k * dth * dth
            g = dihedral_grad_analytic(q, idx)                   # d(theta_rb)/dq
            F[:, idx, :] -= (k * dth)[:, None, None] * g
            theta[:, a] = wrap_to_pi(th_rb + PI)
        return U, F, theta


def run_restrained(tff, restraint, q0, n_steps, dt=0.001, gamma=1.0, temperature=300.0,
                   seed=0, save_every=100, burn_in=0, progress=None):
    """BAOAB with an added restraint force.  Frozen physical model; only the bias differs.

    Returns ``dict`` with ``theta`` ``(n_saved, B, d)`` in radians and diagnostics.
    """
    from alanine.dynamics import BAOAB
    device, dtype = q0.device, q0.dtype
    masses = tff.masses if hasattr(tff, "masses") else None
    if masses is None:
        raise ValueError("TorchFF instance must expose .masses")

    def total_force(q):
        return tff.forces(q) + restraint.energy_and_force(q)[1]

    integ = BAOAB(masses, dt=dt, gamma=gamma, temperature=temperature,
                  force_fn=total_force, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    q = q0.clone()
    v = integ.maxwell(q.shape, gen, device, dtype)
    f = total_force(q)

    out, temps = [], []
    for step in range(n_steps):
        q, v, f = integ.step(q, v, f, gen)
        if not torch.isfinite(q).all():
            raise RuntimeError(f"non-finite positions at step {step}")
        if step >= burn_in and (step - burn_in) % save_every == 0:
            out.append(restraint.dihedrals(q).cpu().numpy().copy())
            temps.append(float(integ.kinetic_temperature(v)))
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}  T = {temps[-1] if temps else float('nan'):.1f} K",
                  flush=True)
    return {
        "theta": np.stack(out),                       # (n_saved, B, d)
        "temperature": np.array(temps),
        "q_final": q.detach(),
    }


# --------------------------------------------------------------------------- 1-D MBAR
def mbar_1d_periodic(samples, centers, kappa, beta, n_bins=72, subsample=1):
    """Periodic 1-D MBAR/WHAM for umbrella windows sharing one restrained coordinate.

    ``samples`` -- ``(K, n)`` sampled angle values (radians) for K windows
    ``centers`` -- ``(K,)`` umbrella centres, ``kappa`` scalar or ``(K,)``
    Returns ``(bin_centers, F)`` with ``F`` in kJ/mol, zeroed at its minimum.

    Uses pymbar when available (it is, in the `abffr` env) and otherwise falls back to a
    self-consistent WHAM iteration; both are exercised by the tests.
    """
    S = np.asarray(samples)[:, ::subsample]
    K, n = S.shape
    c = np.asarray(centers, dtype=float).reshape(K)
    kap = np.broadcast_to(np.asarray(kappa, dtype=float), (K,))

    x = S.reshape(-1)                                            # (K*n,)
    d = wrap_to_pi(x[None, :] - c[:, None])                      # (K, K*n)
    u_kn = beta * 0.5 * kap[:, None] * d * d                     # reduced restraint energies
    N_k = np.full(K, n, dtype=int)

    from scipy.special import logsumexp

    def unbiased_log_weights(f_k):
        """MBAR weight of each sample in the UNBIASED ensemble.

        ``log w_n = -logsumexp_k [ log N_k + f_k - u_kn ]``.

        Note this is *not* ``pymbar.MBAR.weights()[:, 0]``: that column is the weight in
        umbrella window 0, i.e. still carrying window 0's restraint.  Using it silently
        returns the biased distribution of the first window -- which is what an earlier
        version of this function did, and it is why `test_mbar_recovers_a_known_periodic_
        profile` exists.
        """
        return -logsumexp(np.log(N_k)[:, None] + f_k[:, None] - u_kn, axis=0)

    f = None
    try:
        from pymbar import MBAR
        mb = MBAR(u_kn, N_k, solver_protocol="robust")
        f = np.asarray(mb.f_k, dtype=float)
    except Exception:
        f = np.zeros(K)
        for _ in range(10_000):
            logw = unbiased_log_weights(f)
            f_new = -logsumexp(logw[None, :] - u_kn, axis=1)
            f_new -= f_new[0]
            if np.max(np.abs(f_new - f)) < 1e-11:
                f = f_new
                break
            f = f_new
    logw = unbiased_log_weights(f)

    edges = np.linspace(-PI, PI, n_bins + 1)
    idx = np.clip(np.digitize(wrap_to_pi(x), edges) - 1, 0, n_bins - 1)
    logw = logw - logw.max()
    w = np.exp(logw)
    hist = np.bincount(idx, weights=w, minlength=n_bins)
    with np.errstate(divide="ignore"):
        F = -np.log(np.where(hist > 0, hist, np.nan)) / beta
    F -= np.nanmin(F)
    return 0.5 * (edges[:-1] + edges[1:]), F
def count_states(F, beta, kT, sep_kT=3.0, min_pop=0.02):
    """Metastable states of a periodic 1-D profile, with NaN bins treated as impassable.

    Returns ``{"n_states", "populations", "has_gap"}``.  A state is a connected run of finite
    bins, split further wherever an interior ridge exceeds ``sep_kT`` above the lower of the
    two wells it separates; states below ``min_pop`` of the total Boltzmann weight are ignored.

    The NaN handling is the point.  ``mbar_1d_periodic`` returns NaN where no umbrella window
    sampled, which here means the window was rejected as physically inaccessible.  Treating
    those bins as separators (rather than as absent) is what keeps the sec.32 verdict honest.
    """
    n = len(F)
    finite = np.isfinite(F)
    has_gap = bool((~finite).any())
    if not finite.any():
        return {"n_states": 0, "populations": [], "has_gap": has_gap}

    # Connected runs of finite bins on the circle.  The traversal must START AT A SEPARATOR,
    # otherwise linearising the circle cuts through whatever happens to sit at index 0 -- and
    # a well straddling +/-pi would be reported as two states.  With a gap, any NaN is a valid
    # cut; without one, cut at the highest ridge, which is a separator by construction.
    runs, cur = [], []
    start = int(np.argmax(~finite)) if has_gap else int(np.nanargmax(F))
    for k in range(n):
        i = (start + k) % n
        if finite[i]:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)

    # split each run at interior ridges taller than sep_kT above the lower flanking well
    segs = []
    for run in runs:
        v = F[run]
        cut = [0]
        for a in range(1, len(run) - 1):
            left, right = v[:a + 1].min(), v[a:].min()
            if v[a] - max(left, right) >= sep_kT * kT:
                cut.append(a)
        cut.append(len(run))
        for a, b in zip(cut[:-1], cut[1:]):
            if b > a:
                segs.append([run[i] for i in range(a, b)])

    w = np.zeros(n)
    w[finite] = np.exp(-beta * (F[finite] - np.nanmin(F)))
    tot = w.sum()
    pops = sorted((float(w[s].sum() / tot) for s in segs), reverse=True)
    keep = [p for p in pops if p >= min_pop]
    return {"n_states": len(keep), "populations": [round(p, 4) for p in keep],
            "has_gap": has_gap}


