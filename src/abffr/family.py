"""The v3 bias/target family: one carrier, one shape function, no duplication.

Frozen protocol: ``docs/V3_PREREGISTRATION.md`` (v3.1) with Amendments 1-2.

A single estimated free energy carries both the applied bias and the FR target::

    A'_t(z) = Fhat'_t(z)          (running ABF mean-force estimate)
    A_t(z)  = integral of A'_t

Given a shape function ``g``::

    bias potential      B_t = g(A_t) - A_t
    applied bias force  -dB_t/dz = A'_t * [1 - g'(A_t)]
    FR target           q_t \\propto exp[-beta * g(A_t)]

Every quantity below is derived from the *same* ``g`` object.  The v3.0 draft
coded the force and the target from two different smoothings of F, which made
"consistency" false; deriving both here is what prevents that returning.

Consistency, stated correctly (Amendment 2): ``q_t`` is the stationary marginal
of ``B_t`` **in the estimated model**.  The true stationary marginal is
``p* \\propto q_t * exp[-beta (F - A_t)]``, equal to ``q_t`` only as ``A_t -> F``.

Track P is the deliberate violation and is expressed by giving the scheme two
*different* families -- a flat force family and a physical target family -- so
the mismatch is explicit in the code rather than implied by a comment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import torch_utils as tu

EPS = 1e-12


@dataclass(frozen=True)
class Family:
    """A shape function ``g`` and its derivative, in dimensionless form.

    ``kind`` is one of:

    ``flat``      g = 0                      -> full flattening (standard ABF)
    ``capped``    g = softplus_a(u)/(a beta) -> flat core, physical tail
    ``tempered``  g = A / gamma_wt           -> well-tempered
    ``physical``  g = A                      -> zero applied bias

    with the dimensionless depth ``u = beta (A - min A) - c_cut`` and sharpness
    ``a``.  ``c_cut`` and ``a`` are dimensionless (kT), so the family transfers
    across temperatures unchanged.
    """

    kind: str
    c_cut: Optional[float] = None      # dimensionless (kT), ``capped`` only
    sharpness: float = 2.0             # dimensionless, ``capped`` only
    gamma_wt: Optional[float] = None   # dimensionless, ``tempered`` only

    def __post_init__(self):
        if self.kind not in ("flat", "capped", "tempered", "physical"):
            raise ValueError(f"unknown family kind: {self.kind!r}")
        if self.kind == "capped":
            if self.c_cut is None:
                raise ValueError("capped family requires c_cut (dimensionless kT)")
            if self.sharpness <= 0:
                raise ValueError("capped family requires sharpness > 0")
        if self.kind == "tempered":
            if self.gamma_wt is None or self.gamma_wt <= 0:
                raise ValueError("tempered family requires gamma_wt > 0")

    # -- primitives -------------------------------------------------------

    def _u(self, A: torch.Tensor, beta: float) -> torch.Tensor:
        """Dimensionless depth beta (A - min A) - c_cut, per row."""
        A_min = A.min(dim=1, keepdim=True).values
        return beta * (A - A_min) - float(self.c_cut)

    def beta_g(self, A: torch.Tensor, beta: float) -> torch.Tensor:
        """``beta * g(A)`` -- dimensionless, and the only form the target needs.

        Working in ``beta g`` rather than ``g`` keeps the target exponent in
        natural units and avoids a multiply-then-divide by beta.
        """
        if self.kind == "flat":
            return torch.zeros_like(A)
        if self.kind == "physical":
            return beta * A
        if self.kind == "tempered":
            return beta * A / float(self.gamma_wt)
        a = float(self.sharpness)
        # softplus(a u)/a, computed stably for large |u|
        return torch.nn.functional.softplus(a * self._u(A, beta)) / a

    def g_prime(self, A: torch.Tensor, beta: float) -> torch.Tensor:
        """``dg/dA`` -- dimensionless; the bias-force multiplier is ``1 - g'``."""
        if self.kind == "flat":
            return torch.zeros_like(A)
        if self.kind == "physical":
            return torch.ones_like(A)
        if self.kind == "tempered":
            return torch.full_like(A, 1.0 / float(self.gamma_wt))
        return torch.sigmoid(float(self.sharpness) * self._u(A, beta))

    # -- derived quantities (never write these formulas anywhere else) ------

    def bias_force_multiplier(self, A: torch.Tensor, beta: float) -> torch.Tensor:
        """Multiplier m(z) such that the applied bias force is ``A' * m``."""
        return 1.0 - self.g_prime(A, beta)

    def beta_bias_potential(self, A: torch.Tensor, beta: float) -> torch.Tensor:
        """``beta * B = beta * (g(A) - A)``."""
        return self.beta_g(A, beta) - beta * A

    def target(self, A: torch.Tensor, beta: float, dx: float) -> torch.Tensor:
        """Normalized FR target ``q \\propto exp[-beta g(A)]`` on the grid."""
        return _normalize_log_density(-self.beta_g(A, beta), dx)


def oracle_target(F_ref: torch.Tensor, A: torch.Tensor, family: Family,
                  beta: float, dx: float) -> torch.Tensor:
    """The true stationary marginal of the *applied* bias (Amendment 2).

    ``q_oracle \\propto exp[-beta (F_ref + B_t)]`` with ``B_t`` built from the
    candidate's own estimated carrier.  Only the target is oracle-ized; the bias
    is left exactly as the deployable arm applies it, so this isolates
    target-estimation error alone rather than changing bias and target together.
    """
    log_q = -(beta * F_ref + family.beta_bias_potential(A, beta))
    return _normalize_log_density(log_q, dx)


def stationary_marginal(F_ref: torch.Tensor, beta_bias: torch.Tensor,
                        beta: float, dx: float) -> torch.Tensor:
    """``p* \\propto exp[-beta(F_ref + B)]`` -- the engineering-gate reference.

    Gate 1A compares a long no-FR run against this for an arbitrary frozen
    carrier; Gate 1B is the special case ``A = F_ref``, where it must equal the
    family's own target.
    """
    return _normalize_log_density(-(beta * F_ref + beta_bias), dx)


def _normalize_log_density(log_p: torch.Tensor, dx: float) -> torch.Tensor:
    """Exponentiate and normalize a log-density row-wise, max-shifted.

    The shift is what lets a target spanning tens of nats be represented at all;
    the v2 operator clipped the score instead, which changed the flow.
    """
    shifted = log_p - log_p.max(dim=1, keepdim=True).values
    p = torch.exp(shifted)
    return p / tu.trapezoid(p, dx).clamp_min(EPS).unsqueeze(1)


@dataclass(frozen=True)
class Scheme:
    """A named arm: which family sets the bias, and which sets the FR target.

    For every consistent arm the two are the same object.  Track P sets them
    deliberately different (flat force, physical target); ``consistent`` reports
    which is which so the analysis cannot silently mix them up.
    """

    name: str
    force_family: Family
    target_family: Optional[Family]   # None for the no-FR arms

    @property
    def consistent(self) -> bool:
        if self.target_family is None:
            return True                      # no FR: nothing to disagree with
        return self.force_family == self.target_family


# The frozen v3.1 members.  c_cut / gamma_wt values live in the campaign config,
# not here; these are the shapes.
def plain_abf() -> Scheme:
    return Scheme("plain_abf", Family("flat"), None)


def track_p(physical_target: bool = True) -> Scheme:
    """Full flattening bias with a physical target: the deliberate mismatch."""
    return Scheme("track_p_physical", Family("flat"),
                  Family("physical") if physical_target else None)


def capped(c_cut: float, sharpness: float = 2.0, with_fr: bool = True) -> Scheme:
    fam = Family("capped", c_cut=c_cut, sharpness=sharpness)
    return Scheme(f"capped_c{c_cut:g}", fam, fam if with_fr else None)


def tempered(gamma_wt: float, with_fr: bool = True) -> Scheme:
    fam = Family("tempered", gamma_wt=gamma_wt)
    return Scheme(f"tempered_g{gamma_wt:g}", fam, fam if with_fr else None)


def consistent_physical() -> Scheme:
    fam = Family("physical")
    return Scheme("consistent_physical", fam, fam)


def family_from_config(spec: Optional[dict]) -> Optional[Family]:
    """Build a :class:`Family` from a config block, or ``None``."""
    if not spec:
        return None
    spec = dict(spec)
    kind = spec.pop("kind")
    return Family(kind=kind, **spec)


def scheme_from_config(v3: Optional[dict]) -> Optional[Scheme]:
    """Build the arm's :class:`Scheme` from the campaign config.

    ``target_family`` defaults to the force family, which is what makes an arm
    consistent; Track P sets it explicitly to a different family, and that is
    the only place in the campaign where the two differ.  The engine asks this
    object for every bias/target quantity so no family formula is ever written
    outside :mod:`abffr.family`.
    """
    if not v3 or not v3.get("enabled", False):
        return None
    force = family_from_config(v3.get("family"))
    if force is None:
        raise ValueError("v3.family is required when v3.enabled is true")
    target = family_from_config(v3.get("target_family"))
    if target is None and v3.get("operator", "none") != "none":
        target = force                       # consistent arm
    return Scheme(v3.get("name", "v3"), force, target)
