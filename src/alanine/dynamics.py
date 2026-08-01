"""BAOAB Langevin, full-state birth--death, per-seed RNG, and fail-fast non-finite containment.

Category-A correctness work for the atomistic study.  Deliberately does NOT contain any
Category-C algorithmic redesign (no exponential forgetting, no weighted projection, no
clone-discounted counts, no lagged/tempered targets).

Why each piece exists
---------------------
**BAOAB with masses.**  The alkane samplers integrate overdamped Brownian dynamics with unit
mobility.  For ff14SB alanine the stiffest bond constant is 476976 kJ/mol/nm^2 and the largest
Cartesian Hessian eigenvalue 1.368e6 kJ/mol/nm^2, so explicit-Euler overdamped stability needs
``dt < 2/lam_max = 1.46e-6 ps`` -- 685x smaller than the 1 fs that BAOAB handles comfortably.
Unit mobility is also unphysical here (H and O would diffuse identically).

**Full-state cloning.**  ``alkanes.core._birth_death`` copies positions only, which is correct
for an overdamped sampler that has no momenta.  With BAOAB a position-only clone would leave the
child carrying the *killed* replica's momentum.  Here a birth copies the parent's position and
cached physical force, inherits its genealogy label, and draws a **fresh Maxwell momentum**;
the parent is left completely untouched.  Fresh momenta are exact: the canonical density
factorises as ``rho(q,p) ~ exp(-beta V(q)) exp(-beta sum p^2/2m)``, so resampling ``p`` from the
Maxwell distribution at fixed ``q`` is a draw from the conditional and preserves the target.
(Copying the parent's momentum is *also* unbiased; fresh momenta additionally decorrelate the
child's velocity immediately, at no cost.)

**Per-seed RNG streams.**  ``alkanes.core._birth_death`` consumes one shared generator
sequentially over seeds by a *data-dependent* amount (``randperm(n)``, ``multinomial(., n)``), so
changing anything about seed 0 re-rolls the birth sources of every later seed.  That does not
invalidate any published comparison -- ``gen_dyn`` draws a fixed-size block per step
independently of method, so ABF-vs-FR matching is intact -- but it does prevent reproducing a
single seed in isolation and makes a seed's result depend on the batch it was run in.  Here each
seed owns a generator and every draw is fixed-size (two ``rand(N)`` calls, inverse-CDF selection
via ``searchsorted``), so a seed is reproducible alone and independent of ``R``.

**Fail fast on non-finite.**  A single NaN mean force contaminates an entire seed's torus through
the separable smoothing and the FFT, permanently, because the estimator accumulators are running
sums; ``torch.clamp`` sanitises infinities but passes NaN through.  Here non-finite states are
detected *before* any accumulation or projection, the failing state and step are saved, and the
seed/job aborts.  A bad replica is never silently replaced by a sibling.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

KB = 0.008314462618          # kJ/mol/K


class SeedFailure(RuntimeError):
    """Raised when a seed produces a non-finite configuration, force or mean force."""

    def __init__(self, message, seed_index, step, dump_path=None):
        super().__init__(message)
        self.seed_index = int(seed_index)
        self.step = int(step)
        self.dump_path = dump_path


# --------------------------------------------------------------------------- RNG
def make_seed_streams(rng_seed, n_seeds, device, offset=987654321, stride=1000):
    """One independent generator per seed.

    Fixed offsets mean seed ``r``'s stream depends only on ``(rng_seed, r)`` -- never on ``R``,
    on the other seeds' event counts, or on execution order.
    """
    return [torch.Generator(device=device).manual_seed(int(rng_seed) + offset + stride * r)
            for r in range(n_seeds)]


# --------------------------------------------------------------------------- non-finite guard
def check_finite(step, *named_tensors, dump_dir=None, tag="state"):
    """Raise :class:`SeedFailure` if any tensor has a non-finite entry.

    Tensors are ``(name, x)`` with a leading seed dimension ``R``.  The check is a single
    device-side reduction per tensor; the host sync happens only on the (rare) failure path.
    """
    bad_seed = None
    bad_name = None
    for name, x in named_tensors:
        if x is None:
            continue
        finite = torch.isfinite(x.reshape(x.shape[0], -1)).all(dim=1)
        if not bool(finite.all()):
            bad_seed = int((~finite).nonzero()[0].item())
            bad_name = name
            break
    if bad_seed is None:
        return
    dump_path = None
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        dump_path = os.path.join(dump_dir, f"FAILED_seed{bad_seed}_step{step}_{tag}.npz")
        np.savez_compressed(dump_path, step=step, seed_index=bad_seed, failing_tensor=bad_name,
                            **{n: x[bad_seed].detach().cpu().numpy()
                               for n, x in named_tensors if x is not None})
    raise SeedFailure(
        f"non-finite {bad_name} in seed {bad_seed} at step {step}; state saved to {dump_path}",
        bad_seed, step, dump_path)


# --------------------------------------------------------------------------- BAOAB
class BAOAB:
    """Batched BAOAB Langevin on ``(R, N, A, 3)`` with a per-atom mass vector.

    Splitting (Leimkuhler--Matthews):  B (half kick) A (half drift) O (OU) A (half drift)
    B (half kick).  BAOAB has ``O(dt^2)`` configurational error with an unusually small
    prefactor, which is why it is the right choice when the observable is a *configurational*
    free energy.
    """

    def __init__(self, masses, dt, gamma, temperature, force_fn, device=None, dtype=torch.float64):
        # Shape (A, 1) so every operation broadcasts against the trailing ``(A, 3)`` of ANY
        # leading batch shape -- ``(R, N, A, 3)`` for the sampler and ``(B, A, 3)`` for the
        # umbrella reference.  A ``(1, 1, A, 1)`` mass would silently promote a 3-D state to 4-D
        # inside the first kick and corrupt every downstream shape.
        self.m = torch.as_tensor(masses, device=device, dtype=dtype).reshape(-1, 1)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.T = float(temperature)
        self.kT = KB * self.T
        self.force_fn = force_fn
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt(1.0 - self.c1 ** 2)
        # Shape (A, 1) so it broadcasts against the trailing ``(A, 3)`` of ANY leading batch
        # shape -- both ``(R, N, A, 3)`` in :meth:`step` and ``(n, A, 3)`` when drawing momenta
        # for a batch of clones.  A ``(1, 1, A, 1)`` sigma would silently return a 4-D tensor
        # for a 3-tuple request and mis-shape the cloned momenta.
        self.sigma = math.sqrt(self.kT) / self.m.sqrt()          # (A,1), same reason

    def maxwell(self, shape, generator, device, dtype):
        """Draw velocities from the Maxwell distribution at ``T``; returns exactly ``shape``."""
        v = torch.randn(shape, generator=generator, device=device, dtype=dtype) * self.sigma
        if tuple(v.shape) != tuple(shape):
            raise AssertionError(f"maxwell produced {tuple(v.shape)}, expected {tuple(shape)}")
        return v

    def kinetic_temperature(self, v, per_batch=False):
        """Instantaneous kinetic temperature.  COM motion is not removed, so dof = 3 * n_atoms.

        Shape-agnostic: ``v`` may be ``(..., A, 3)`` with any leading dims.  ``per_batch=True``
        returns one temperature per leading element, otherwise a single pooled scalar.
        """
        ke = 0.5 * (self.m * v * v)
        if per_batch:
            return 2.0 * ke.sum(dim=(-2, -1)) / (3 * v.shape[-2] * KB)
        return 2.0 * ke.sum() / (v.numel() * KB)

    def step(self, q, v, f, generator, bias_fn=None):
        """One BAOAB step.  ``bias_fn(q) -> extra force`` is added to the physical force."""
        dt, m = self.dt, self.m
        v = v + (0.5 * dt) * f / m
        q = q + (0.5 * dt) * v
        v = self.c1 * v + self.c2 * torch.randn(v.shape, generator=generator,
                                                device=v.device, dtype=v.dtype) * self.sigma
        q = q + (0.5 * dt) * v
        f = self.force_fn(q)
        if bias_fn is not None:
            f = f + bias_fn(q)
        v = v + (0.5 * dt) * f / m
        return q, v, f


# --------------------------------------------------------------------------- birth--death
def birth_death_full_state(q, v, f, score, ancestors, gens, fr_rate, dt_eff,
                           max_event_fraction, integrator):
    """Fixed-population kill-and-clone of the FULL dynamical state.

    A death at slot ``i`` with parent ``j``:  ``q_i <- q_j``, ``f_i <- f_j``,
    ``ancestors_i <- ancestors_j``, and ``v_i ~ Maxwell(T)`` freshly drawn.  **The parent slot
    ``j`` is not modified at all.**

    Every random draw is fixed-size and per-seed, so seed ``r``'s realisation depends only on its
    own generator: two ``rand(N)`` draws plus one ``randn`` for the child momenta.  Birth parents
    are selected by inverse-CDF (``searchsorted`` on the row cumsum of the birth weights) rather
    than ``multinomial``, whose consumption depends on the number of events.

    Returns ``(q, v, f, ancestors, n_events (R,), deaths list, births list)``.
    """
    R, N = score.shape
    q, v, f, anc = q.clone(), v.clone(), f.clone(), ancestors.clone()
    n_events = torch.zeros(R, dtype=torch.long)
    deaths, births = [None] * R, [None] * R
    max_events = int(max_event_fraction * N)
    if max_events < 1 or fr_rate <= 0.0:
        return q, v, f, anc, n_events, deaths, births

    death_w = torch.clamp(score, min=0.0)
    birth_w = torch.clamp(-score, min=0.0)
    p_die = torch.where(death_w > 0, 1.0 - torch.exp(-fr_rate * death_w * dt_eff),
                        torch.zeros_like(death_w))

    for r in range(R):
        g = gens[r]
        u_fire = torch.rand(N, generator=g, device=q.device, dtype=q.dtype)      # fixed size
        u_pick = torch.rand(N, generator=g, device=q.device, dtype=q.dtype)      # fixed size
        if float(birth_w[r].sum()) <= 0.0 or float(death_w[r].sum()) <= 0.0:
            continue
        fire = u_fire < p_die[r]
        di = torch.nonzero(fire, as_tuple=False).flatten()
        if di.numel() == 0:
            continue
        if di.numel() > max_events:                      # deterministic cap: highest score first
            order = torch.argsort(score[r, di], descending=True)
            di = di[order[:max_events]]
        n = int(di.numel())
        cdf = torch.cumsum(birth_w[r], 0)
        cdf = cdf / cdf[-1].clamp_min(1e-30)
        src = torch.searchsorted(cdf, u_pick[:n].clamp(0, 1 - 1e-12)).clamp_(0, N - 1)
        q[r, di] = q[r, src]
        f[r, di] = f[r, src]
        anc[r, di] = ancestors[r, src]
        v[r, di] = integrator.maxwell((n, q.shape[-2], 3), g, q.device, q.dtype)
        n_events[r] = n
        deaths[r], births[r] = di, src
    return q, v, f, anc, n_events, deaths, births


# --------------------------------------------------------------------------- leakage guard
FR_METHODS = ("fr_estimated", "fr_uniform", "fr_oracle")
ALL_METHODS = ("abf",) + FR_METHODS
ESTIMATED_TARGET_METHODS = ("fr_estimated",)


def assert_no_reference_leakage(method, reference_free_energy):
    """Structural gate: only ``fr_oracle`` may ever hold the reference."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "fr_oracle":
        if reference_free_energy is None:
            raise ValueError("fr_oracle requires the reference free energy.")
        return
    if reference_free_energy is not None:
        raise AssertionError(
            f"NO-REFERENCE-LEAKAGE VIOLATION: method={method!r} received a reference free "
            "energy; only fr_oracle may.")
