"""Persistent Fisher--Rao probability mass, held strictly apart from ABF.

Frozen protocol: ``docs/V4A_PREREGISTRATION.md``.

v3 conflated three objects every 500 steps by resampling at every FR
opportunity.  v4-A separates them::

    FR weights           ->  probability mass          (this module)
    physical propagation ->  statistical information   (the ABF accumulators)
    resampling           ->  particle representation   (a separate concern)

**This object holds no reference to any ABF accumulator, estimator or bias.**
That is deliberate architecture, not convention: :class:`PersistentMass` cannot
corrupt the mean-force estimate because it has no way to reach it.  The
mathematical rule it enforces is that changing ``w_i`` alone can never change
``F̂'``; only physical propagation creates an ABF observation.

Why the weights may not enter the estimator: after repeated updates,
``w_i(t)`` is a *path* functional -- each multiplier depends only on ``xi`` at
the instant it was applied, but two replicas at the same *current* ``xi`` can
carry wildly different weights (measured on this benchmark: a factor of 8e6
inside one narrow bin).  ABF's justification needs the biasing factor to be
constant on the current fibre, which a path weight is not.

Everything is stored and updated as ``log w``.  Weights that span hundreds of
nats are ordinary here, so a normalized-weight representation would underflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

LOG_KDE_CONST = 0.5 * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))


def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


@dataclass
class MassDiagnostics:
    ess: float                  # particle ESS = 1 / sum_i w_i^2
    ess_frac: float             # ess / K
    w_max: float
    ess_anc_mass: Optional[float] = None
    m_max: Optional[float] = None


class PersistentMass:
    """The FR mass sidecar for one run: owns ``{log w_i}`` and nothing else."""

    def __init__(self, n_particles: int, device=None, dtype=torch.float64):
        self.K = int(n_particles)
        self.dtype = dtype
        self.log_w = torch.full((self.K,), -float(torch.log(torch.tensor(
            float(self.K), dtype=dtype))), device=device, dtype=dtype)
        self._normalize()

    # -- state ------------------------------------------------------------
    def _normalize(self) -> None:
        """Renormalize so ``LSE(log w) == 0``; every ``log w <= 0`` after this."""
        if not torch.isfinite(self.log_w).all():
            raise FloatingPointError(
                "non-finite log weight; failing closed rather than clipping it "
                "to a plausible value")
        self.log_w = self.log_w - logsumexp(self.log_w)

    def reset_uniform(self) -> None:
        """Called only by the representation layer, after a resampling."""
        self.log_w = torch.full_like(self.log_w, 0.0)
        self._normalize()

    @property
    def weights(self) -> torch.Tensor:
        """Normalized weights.  For diagnostics and resampling only."""
        return torch.exp(self.log_w)

    # -- the weighted marginal, in log space -------------------------------
    def log_density_at(self, z: torch.Tensor, eta: float) -> torch.Tensor:
        """``log p_w(z_i) = LSE_j [ log w_j + log K_eta(z_i - z_j) ]``.

        Returned **up to an additive constant**, which is all the FR update
        needs: the normalization that follows absorbs any constant.  That is
        also what makes the equal-weight case reduce exactly to the unweighted
        KDE rather than approximately.
        """
        d = (z.unsqueeze(1) - z.unsqueeze(0)) / eta
        log_k = -0.5 * d * d                      # constant -log(eta*sqrt(2pi)) dropped
        return logsumexp(self.log_w.unsqueeze(0) + log_k, dim=1)

    # -- the Fisher--Rao mass update ---------------------------------------
    def fr_update(self, log_q_at: torch.Tensor, log_p_at: torch.Tensor,
                  theta: float = 1.0) -> None:
        """``log w_i <- log w_i + theta (log q(z_i) - log p_w(z_i))``, renormalized.

        Moves probability mass only.  Positions are untouched, no replica is
        created or destroyed, and nothing here can reach an ABF accumulator.
        """
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if theta == 0.0:
            return
        self.log_w = self.log_w + theta * (log_q_at - log_p_at)
        self._normalize()

    # -- diagnostics -------------------------------------------------------
    def ess(self) -> float:
        """``exp(-LSE(2 log w))`` -- safe to exponentiate since ESS lies in [1, K]."""
        return float(torch.exp(-logsumexp(2.0 * self.log_w)))

    def w_max(self) -> float:
        return float(torch.exp(self.log_w.max()))

    def mass_ancestry(self, ancestors: torch.Tensor) -> Tuple[float, float]:
        """``(ESS_anc^mass, m_max)`` with ``m_a = sum_{i: a_i = a} w_i``.

        Distinct from count ancestry, and the distinction is the point: an arm
        that never resamples has ESS_anc^count = K by construction while its
        mass may sit almost entirely on one ancestor.
        """
        K = self.K
        log_m = torch.full((K,), -float("inf"), dtype=self.log_w.dtype,
                           device=self.log_w.device)
        for a in torch.unique(ancestors):
            sel = ancestors == a
            log_m[int(a)] = logsumexp(self.log_w[sel])
        alive = torch.isfinite(log_m)
        ess_mass = float(torch.exp(-logsumexp(2.0 * log_m[alive])))
        return ess_mass, float(torch.exp(log_m[alive].max()))

    def diagnostics(self, ancestors: Optional[torch.Tensor] = None) -> MassDiagnostics:
        d = MassDiagnostics(ess=self.ess(), ess_frac=self.ess() / self.K,
                            w_max=self.w_max())
        if ancestors is not None:
            d.ess_anc_mass, d.m_max = self.mass_ancestry(ancestors)
        return d

    # -- within-fibre localization (diagnostic only) ------------------------
    def log_fibre_ess(self, z: torch.Tensor, z_eval: torch.Tensor,
                      h: float) -> torch.Tensor:
        """``log ESS_fibre(z*)`` for ``u_i = w_i K_h(z* - xi_i)``, per evaluation point.

        ``ESS_fibre = (sum_i u_i)^2 / sum_i u_i^2``, computed as
        ``2 LSE(log u) - LSE(2 log u)`` so it survives the weight spreads that
        make this diagnostic worth having.

        Why it exists: keeping ``w`` out of the ABF estimator removes the direct
        bias, but a *resampling* converts a path-dependent mass distribution back
        into an actual physical population.  Two replicas at the same ``xi`` with
        different fibre coordinates and very different weights can be selected
        against each other, perturbing the realized conditional law even though
        no weight ever entered a force accumulator.  This measures how few
        replicas actually carry the mass near each ``z``.
        """
        d = (z_eval.unsqueeze(1) - z.unsqueeze(0)) / h
        log_u = self.log_w.unsqueeze(0) - 0.5 * d * d
        return 2.0 * logsumexp(log_u, dim=1) - logsumexp(2.0 * log_u, dim=1)

    def fibre_ess(self, z: torch.Tensor, z_eval: torch.Tensor,
                  h: float) -> torch.Tensor:
        return torch.exp(self.log_fibre_ess(z, z_eval, h))

    # -- the representation trigger (decision only; it does not resample) ---
    def needs_resample(self, rho_resample: float) -> bool:
        """Degeneracy-triggered, never called "conditional" in this project:
        "conditional" already means the fibre law pi(dq | xi = z)."""
        return self.ess() < float(rho_resample) * self.K

    def take_indices(self, src: torch.Tensor) -> None:
        """Reorder the masses to follow a reallocation of positions."""
        self.log_w = self.log_w[src]
        self._normalize()
