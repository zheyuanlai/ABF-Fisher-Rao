"""Bi-channel torus: an ABP biasing xi = phi with a genuinely hidden coordinate psi.

Purpose (docs/PREREGISTRATION_APPLICATION_MAP.md, Phase F).  Every system this
campaign has run was limited by something an adaptive-biasing POTENTIAL repairs by
itself: discovery (Type D), adaptation rate (Type B), or an oscillatory establishment
transient (Type A).  In all three the bias acts along the coordinate that limits the
estimator, so population reallocation is redundant with the base method -- which is
exactly what gateway, WCA at every K, both torus cells and vacuum alanine reported.

The one deficit an ABP cannot repair is Type C: the bias flattens p(xi) and says
nothing about p(z | xi) for a coordinate z it does not bias.  This system is built to
exhibit Type C on purpose, with the mechanism the ABF literature calls a hidden or
orthogonal barrier (Lelievre-Minoukadeh bi-channel):

    s(psi) = (1 + cos psi) / 2                                (channel indicator)
    V(phi, psi) = Hperp (1 - cos 2 psi)/2                      (barrier between channels)
                + Delta (1 - cos psi)/2                        (channel B lies Delta higher)
                + s(psi)       Ha (1 - cos 2 phi)/2            (channel A phi-profile)
                + (1 - s(psi)) Hb (1 + cos 2 phi)/2            (channel B: wells at +-pi/2)

Two channels run from -pi to pi in phi.  Channel A (psi ~ 0) has its phi-wells at
0, pi; channel B (psi ~ pi) has its phi-wells exactly where A has its barriers, so
F(phi) is a genuine mixture and an under-populated channel B corrupts the free
energy the ABP is estimating -- a Type-C error that is visible in e_F, not only in a
diagnostic.  Ha = Hb makes the two channel partition functions equal by the
translation phi -> phi + pi/2, so the exact channel ratio is p_B/p_A = e^{-beta Delta}
and the target population is known analytically.

Why this is the decisive cell for the Fisher-Rao question:

* the adaptation gain g_SHUS -- the arm that matched or beat marginal FR on every
  system so far -- rescales the phi-bias and by construction cannot lower a barrier
  in psi.  For the first time it is not a competing explanation.
* marginal FR is blind here by construction: two walkers at the same phi in
  different channels get the SAME score, so the marginal step cannot prefer the rare
  channel.  It is carried as a control, and a null from it is a prediction, not a
  disappointment.
* the fiber-wise step of fisher_rao_cond.py is the only arm that can see the deficit,
  and it leaves the phi-marginal invariant, so it cannot buy its result by perturbing
  the occupancy signal SHUS deposits from.

Cost knobs and what they control (Kramers, D = 1):
    tau_{A->B} ~ (pi/Hperp) e^{beta (Hperp + Delta/2)},   tau_{B->A} ~ ... e^{beta (Hperp - Delta/2)}
    equilibrium channel-B fraction = 1 / (1 + e^{beta Delta}).
The Type-C window needs tau_{B->A} >> T (the channel populations cannot equilibrate
within the run) while channel B is still REACHED (else the cell is Type D/discovery
and no reallocation can help by construction).  Both are screen outputs, gated on
plain-SHUS rows before any reallocation arm exists.

Reference: the CV is a coordinate of the configuration space, so F(phi, psi) = V
exactly and F(phi) = -beta^{-1} log int e^{-beta V} dpsi by quadrature on the
production grid -- no reference simulation, no MBAR, nothing to confound the
accuracy claim.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..events1p import fr_event1p
from ..events_cond import fr_event_cond
from ..fisher_rao_cond import weight_ess
from ..grid import DEVICE, DTYPE, EPS
from ..grid1p import (Grid1P, ShusAccumulator1P, binned_density1p, integral1p,
                      interp1p, kl_to_uniform1p, tv_to_uniform1p)
from ..grid2d import (GridT2, binned_density2, integral2, periodic_gaussian_kernel,
                      smooth2, torus_distance, wrap_periodic)
from ..resampling import ancestor_stats, surviving_ancestors
from .gateway import Method, _fires_at_block, _schedule_source  # shared arm logic

REFERENCE_ID = "bichannel-analytic-v1"
PI = math.pi
NG = 96
GRID2 = GridT2(x1min=-PI, L1=2 * PI, n1=NG, x2min=-PI, L2=2 * PI, n2=NG)
GRID1 = Grid1P(xmin=-PI, L=2 * PI, n=NG)
REGIONS = ("chanA", "chanB")


# -----------------------------------------------------------------------------
# physics
# -----------------------------------------------------------------------------
def V_of(phi, psi, Hperp, Delta, Ha, Hb):
    s = 0.5 * (1.0 + torch.cos(psi))
    return (0.5 * Hperp * (1.0 - torch.cos(2.0 * psi))
            + 0.5 * Delta * (1.0 - torch.cos(psi))
            + s * 0.5 * Ha * (1.0 - torch.cos(2.0 * phi))
            + (1.0 - s) * 0.5 * Hb * (1.0 + torch.cos(2.0 * phi)))


def gradV_of(phi, psi, Hperp, Delta, Ha, Hb):
    s = 0.5 * (1.0 + torch.cos(psi))
    ds = -0.5 * torch.sin(psi)
    A = 0.5 * Ha * (1.0 - torch.cos(2.0 * phi))
    B = 0.5 * Hb * (1.0 + torch.cos(2.0 * phi))
    dphi = torch.sin(2.0 * phi) * (s * Ha - (1.0 - s) * Hb)
    dpsi = (Hperp * torch.sin(2.0 * psi) + 0.5 * Delta * torch.sin(psi)
            + ds * (A - B))
    return dphi, dpsi


def reference_surface(Hperp, Delta, Ha, Hb, device=DEVICE, dtype=DTYPE):
    """Exact F(phi, psi) = V on the production grid, zero-mean.  -> (n1, n2)."""
    P1, P2 = GRID2.mesh(device, dtype)
    F = V_of(P1, P2, Hperp, Delta, Ha, Hb)
    return F - F.mean()


def reference_objects(beta, Hperp, Delta, Ha, Hb, device=DEVICE, dtype=DTYPE):
    """F1(phi), p_cond(psi | phi), rho(phi) and the exact channel-B fraction.

    F1 is the marginal free energy the ABP is estimating; p_cond is the reference
    the Type-C diagnostic E_cond is scored against; p_B_ref is the equilibrium
    population of the rare channel, which the reallocation arms must approach from
    below and must not overshoot.
    """
    P1, P2 = GRID2.mesh(device, dtype)
    F2 = V_of(P1, P2, Hperp, Delta, Ha, Hb)
    rho2 = torch.exp(-beta * (F2 - F2.min()))
    Z1 = rho2.sum(dim=1) * GRID2.dx2                       # (n1,)
    F1 = -torch.log(torch.clamp(Z1, min=EPS)) / beta
    F1 = F1 - F1.mean()
    p_cond = rho2 / torch.clamp(rho2.sum(dim=1, keepdim=True), min=EPS) / GRID2.dx2
    rho1 = torch.exp(-beta * F1)
    rho1 = rho1 / (rho1.sum() * GRID1.dx)
    inB = channel_weight_B(P2).to(dtype)
    p_B = float((rho2 * inB).sum() / torch.clamp(rho2.sum(), min=EPS))
    pB_phi = ((rho2 * inB).sum(dim=1)
              / torch.clamp(rho2.sum(dim=1), min=EPS))          # (n1,)
    return {"F2": F2 - F2.mean(), "F1": F1, "p_cond": p_cond, "rho1": rho1,
            "p_B_ref": p_B, "pB_phi_ref": pB_phi,
            # Under a CONVERGED bias the stationary law is p_ref(psi|phi) x uniform
            # in phi, so the channel fraction a converged run should show is the
            # UNIFORM-phi average of pB_phi_ref, not its Boltzmann average.  The two
            # coincide on a symmetric cell and differ on an asymmetric one -- the
            # pilot showed a plain-SHUS run walking past p_B_ref with no reallocation
            # anywhere, which is this effect and not an overshoot.
            "p_B_ref_biased": float(pB_phi.mean())}


def type_c_amplitude(cfg, device="cpu", dtype=DTYPE):
    """e_F the run converges to if channel B is NEVER populated -- the size of the
    Type-C error in the quantity the ABP is actually estimating.  Pre-run computable,
    and the number that decides whether a cell is worth running at all: it must stand
    far above the mollifier floor e* of analytic_floors()."""
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    P1, P2 = GRID2.mesh(device, dtype)
    F2 = V_of(P1, P2, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb)
    rho = torch.exp(-cfg.beta * (F2 - F2.min())) * (1.0 - channel_weight_B(P2))
    F1A = -torch.log(torch.clamp(rho.sum(dim=1) * GRID2.dx2, min=EPS)) / cfg.beta
    d = ref["F1"] - F1A
    d = d - d.mean()
    return float(torch.sqrt((d * d).mean()))


def channel_of(psi):
    """0 = channel A (psi near 0), 1 = channel B (psi near pi)."""
    return (torch.cos(psi) < 0).long()


def channel_weight_B(psi):
    """Channel-B membership as a weight: 1 in B, 0 in A, 1/2 exactly on the divide.

    The production grid puts nodes exactly on psi = +-pi/2, and a strict inequality
    would then assign both dividing lines to the same channel -- enough to move
    p_B_ref off 1/2 on a symmetric cell and quietly bias every conditional score
    against it.  Walkers land on the divide with probability zero, so this reduces to
    the indicator for the population and only fixes the grid reference.
    """
    c = torch.cos(psi)
    tol = 100.0 * torch.finfo(c.dtype).eps
    return torch.where(c.abs() < tol, torch.full_like(c, 0.5), (c < 0).to(c.dtype))


# -----------------------------------------------------------------------------
# configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BiChannelConfig:
    beta: float = 4.0
    Hperp: float = 2.0           # orthogonal barrier between the channels
    Delta: float = 0.5           # channel-B offset: p_B/p_A = e^{-beta Delta}
    Ha: float = 1.0              # channel-A phi-barrier
    Hb: float = 1.0              # channel-B phi-barrier (= Ha keeps Z_A = Z_B)
    K: int = 1024
    dt: float = 1e-3
    n_steps: int = 400_000
    block: int = 20
    eps_bw: float = 0.06         # mollifier bandwidth (deposits)
    eta_bw: float = 0.25         # KDE bandwidth (marginal, conditional, FR score)
    n_saves: int = 400
    profile_every: int = 8       # cadence for the 1D (CV) profile series
    joint_every: int = 40        # cadence for the FULL (phi, psi) KDE: 96x96 frames
                                 # are ~50x a 1D frame, so the joint is stored coarsely
                                 # while the channel-resolved profile P_B(phi) -- all
                                 # the conditional scoring actually needs -- is stored
                                 # at profile cadence
    ess_window_steps: int = 4000
    n_strata: int = 32           # xi-strata for fiber-wise reallocation; the width
                                 # 2 pi / 32 = 0.196 is the resolution at which the
                                 # phi-marginal is held invariant
    cv: str = "phi"              # "phi" = the 1D CV this phase is about; "phipsi" =
                                 # the AUGMENTED CV, i.e. just bias the hidden
                                 # coordinate too.  That is the baseline an
                                 # application scientist asks for first, and it is
                                 # scored on the SAME reduced quantity F(phi) so the
                                 # two approaches are directly comparable.
    init: str = "chanA"          # "chanA": every walker starts in channel A -- the
                                 # rare channel must be REACHED, which makes the
                                 # deficit an unrelaxed conditional (Phase F, Type C
                                 # = a BIAS).  "stationary": walkers drawn from the
                                 # exact stationary law of the CONVERGED bias
                                 # (uniform in phi x p_ref(psi | phi)), so the
                                 # represented conditional is correct in expectation
                                 # and what remains is finite-K noise (Phase J = a
                                 # VARIANCE).  "boltzmann": the unbiased equilibrium.
                                 # The last two consult the reference and are
                                 # experimental CONDITIONS shared by every arm.
    warm_start: bool = False     # start the accumulator at its analytic fixed point
                                 # R* = K_eps e^{-beta F1} instead of at R = 1, i.e.
                                 # begin the run already converged.  With
                                 # init="stationary" this removes the establishment
                                 # transient entirely and the run measures only the
                                 # estimator's variance about its fixed point.

    @property
    def T_total(self) -> float:
        return self.n_steps * self.dt

    def channel_times(self):
        """Kramers estimates (D = 1): (tau_A->B, tau_B->A, equilibrium p_B)."""
        pre = PI / max(self.Hperp, 1e-9)
        up = self.beta * (self.Hperp + 0.5 * self.Delta)
        dn = self.beta * (self.Hperp - 0.5 * self.Delta)
        return (pre * math.exp(up), pre * math.exp(dn),
                1.0 / (1.0 + math.exp(self.beta * self.Delta)))


def analytic_floors(cfg: BiChannelConfig, device="cpu", dtype=DTYPE):
    """(e*, KL*) of the mollified SHUS fixed point on F1 -- computable pre-run."""
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    from ..grid1p import smooth1p
    k, r = periodic_gaussian_kernel(cfg.eps_bw, GRID1.dx, GRID1.n, device, dtype)
    F1 = ref["F1"].unsqueeze(0)
    rho = torch.exp(-cfg.beta * F1)
    rho_m = smooth1p(rho, k, r)
    F_star = -torch.log(torch.clamp(rho_m, min=EPS)) / cfg.beta
    F_star = F_star - F_star.mean(dim=1, keepdim=True)
    d = F_star - (F1 - F1.mean(dim=1, keepdim=True))
    p_star = rho / torch.clamp(rho_m, min=EPS)
    p_star = p_star / integral1p(p_star, GRID1).unsqueeze(1)
    return {"e_star": float(torch.sqrt((d * d).mean())),
            "kl_star": float(kl_to_uniform1p(p_star, GRID1))}


def conditional_floors(cfg: BiChannelConfig, K: int, n_rep=64, seed=777,
                       device="cpu", dtype=DTYPE, biased=True):
    """Finite-K floors of the conditional metrics: the SAME estimators applied to
    EXACT Boltzmann samples of this cell.

    None of these vanish under perfect sampling, so their floors are preregistered
    before any outcome is read -- exactly as the ALA-1 conditional metric was.
    Returns sorted replicate arrays for E_cond, E_chan and |P_B - p_B_ref|.
    """
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    rho2 = torch.exp(-cfg.beta * ref["F2"])
    if biased:
        # the null a CONVERGED run is measured against: phi uniform (the bias has
        # flattened it), psi drawn from the exact conditional p_ref(psi | phi)
        rho2 = rho2 / torch.clamp(rho2.sum(dim=1, keepdim=True), min=EPS)
    w = (rho2 / rho2.sum()).reshape(-1)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    cdf = torch.cumsum(w, dim=0)
    u = torch.rand((n_rep, K), device=device, dtype=dtype, generator=gen)
    idx = torch.searchsorted(cdf, u.reshape(-1).contiguous()).clamp(max=w.numel() - 1)
    i1 = (idx // GRID2.n2).reshape(n_rep, K)
    i2 = (idx % GRID2.n2).reshape(n_rep, K)
    j1 = torch.rand((n_rep, K), device=device, dtype=dtype, generator=gen) - 0.5
    j2 = torch.rand((n_rep, K), device=device, dtype=dtype, generator=gen) - 0.5
    z1 = GRID2.x1min + (i1.to(dtype) + j1) * GRID2.dx1
    z2 = GRID2.x2min + (i2.to(dtype) + j2) * GRID2.dx2
    z1 = wrap_periodic(z1, GRID2.x1min, GRID2.L1)
    z2 = wrap_periodic(z2, GRID2.x2min, GRID2.L2)
    k1, r1 = periodic_gaussian_kernel(cfg.eta_bw, GRID2.dx1, GRID2.n1, device, dtype)
    k2, r2 = periodic_gaussian_kernel(cfg.eta_bw, GRID2.dx2, GRID2.n2, device, dtype)
    p2 = binned_density2(z1, z2, k1, r1, k2, r2, GRID2)
    ec = _e_cond(p2, ref["p_cond"].unsqueeze(0))
    ech = _e_chan(p2, ref["pB_phi_ref"].unsqueeze(0))
    target = ref["p_B_ref_biased"] if biased else ref["p_B_ref"]
    pB = ((torch.cos(z2) < 0).to(dtype).mean(dim=1) - target).abs()
    return {"e_cond": np.sort(ec.detach().cpu().numpy()),
            "e_chan": np.sort(ech.detach().cpu().numpy()),
            "p_B_err": np.sort(pB.detach().cpu().numpy())}


def stationary_init(cfg: BiChannelConfig, K: int, seed: int, biased=True,
                    device=DEVICE, dtype=DTYPE):
    """Exact draw from a stationary law of this cell.  -> (phi, psi), each (K,).

    biased=True is the law a CONVERGED phi-bias samples: uniform in phi, with psi from
    the exact conditional p_ref(psi | phi).  It is stationary for the frozen-bias
    dynamics, so an ensemble started there stays correct in expectation for the whole
    run whatever the channel-exchange time is -- which is what turns the Type-C
    deficit from a bias into a variance.  biased=False is the unbiased Boltzmann law.

    Grid inverse-CDF with uniform jitter inside the cell (the sampler already used by
    conditional_floors, so the run's initial condition and the metric floors it is
    scored against are drawn from the same construction).
    """
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    rho2 = torch.exp(-cfg.beta * (ref["F2"] - ref["F2"].min()))
    if biased:
        rho2 = rho2 / torch.clamp(rho2.sum(dim=1, keepdim=True), min=EPS)
    w = (rho2 / rho2.sum()).reshape(-1)
    rng = np.random.default_rng(1000 + int(seed))
    u = torch.as_tensor(rng.random(K), device=device, dtype=dtype)
    idx = torch.searchsorted(torch.cumsum(w, dim=0), u).clamp(max=w.numel() - 1)
    j = torch.as_tensor(rng.random((2, K)) - 0.5, device=device, dtype=dtype)
    z1 = GRID2.x1min + ((idx // GRID2.n2).to(dtype) + j[0]) * GRID2.dx1
    z2 = GRID2.x2min + ((idx % GRID2.n2).to(dtype) + j[1]) * GRID2.dx2
    return (wrap_periodic(z1, GRID2.x1min, GRID2.L1),
            wrap_periodic(z2, GRID2.x2min, GRID2.L2))


def warm_start_R(cfg: BiChannelConfig, device=DEVICE, dtype=DTYPE):
    """The accumulator's analytic fixed point on this cell: R* = K_eps e^{-beta F1}."""
    from ..grid1p import smooth1p
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    k, r = periodic_gaussian_kernel(cfg.eps_bw, GRID1.dx, GRID1.n, device, dtype)
    rho = torch.exp(-cfg.beta * (ref["F1"] - ref["F1"].min())).unsqueeze(0)
    return torch.clamp(smooth1p(rho, k, r), min=EPS)


def target_log_q(methods, refs, M, device, dtype):
    """Per-row log q(psi | phi) for a batch, or None if every arm is uniform.

    "reparam" with amplitude a targets a UNIFORM distribution in
    psi' = psi + a sin psi (monotone for |a| < 1), whose density in psi is exactly
    (1 + a cos psi) / L2 -- normalized on the circle with no quadrature.  a = 0.8 is
    a 9:1 skew, i.e. a large and entirely arbitrary mis-specification of the target,
    which is what choosing a descriptor over a nonlinear function of it would do.
    """
    if all(m.cond_target == "uniform" for m in methods):
        return None
    B = len(refs)
    R = B * M
    out = torch.empty((R, GRID2.n1, GRID2.n2), device=device, dtype=dtype)
    psi = GRID2.x2(device, dtype).reshape(1, -1)
    for b in range(B):
        for j, m in enumerate(methods):
            r = b * M + j
            if m.cond_target == "uniform":
                out[r] = -math.log(GRID2.L2)
            elif m.cond_target == "reparam":
                a = float(m.cond_target_a)
                assert abs(a) < 1.0, "reparametrization needs |a| < 1 to stay monotone"
                out[r] = (torch.log1p(a * torch.cos(psi)) - math.log(GRID2.L2)
                          ).expand(GRID2.n1, GRID2.n2)
            elif m.cond_target == "oracle":
                out[r] = torch.log(torch.clamp(refs[b]["p_cond"], min=EPS))
            else:
                raise AssertionError(f"unknown cond_target {m.cond_target!r}")
    return out


def reduce_to_phi(F2_hat, beta):
    """F(phi) = -beta^{-1} log int e^{-beta F2(phi,psi)} dpsi, zero-mean.

    The deliverable is F(phi) in both CV choices, so an augmented-CV run is scored on
    its reduction, not on the 2D surface it happens to have learned.  F2_hat:
    (R, n1, n2), beta: (R, 1) -> (R, n1).
    """
    b = beta.reshape(-1, 1, 1)
    e = -b * F2_hat
    e = e - e.amax(dim=(1, 2), keepdim=True)
    Z = torch.clamp(torch.exp(e).sum(dim=2) * GRID2.dx2, min=EPS)
    F1 = -torch.log(Z) / beta.reshape(-1, 1)
    return F1 - F1.mean(dim=1, keepdim=True)


def analytic_floors_2d(cfg, device="cpu", dtype=DTYPE):
    """Mollifier floor of an AUGMENTED-CV run, scored on the reduced F(phi).

    The 2D accumulator's fixed point is mollified on a 96x96 grid and then reduced,
    so its floor on F(phi) is not the 1D floor and must be quoted separately or the
    head-to-head is unfair to one side.
    """
    from ..shus2d import mollified_fixed_point2
    ref = reference_objects(cfg.beta, cfg.Hperp, cfg.Delta, cfg.Ha, cfg.Hb,
                            device, dtype)
    fp = mollified_fixed_point2(ref["F2"], cfg.beta, cfg.eps_bw, GRID2, device, dtype)
    beta = torch.full((1, 1), cfg.beta, device=device, dtype=dtype)
    F1_star = reduce_to_phi(fp["F_star"].unsqueeze(0), beta)[0]
    d = F1_star - ref["F1"]
    d = d - d.mean()
    return {"e_star": float(torch.sqrt((d * d).mean())),
            "e_star_2d": fp["e_star"], "kl_star_2d": fp["kl_star"]}


def _e_chan(p2, pB_phi_ref):
    """E_chan = int p(phi) |P_B(phi) - P_B_ref(phi)| dphi.

    The channel-resolved conditional error: the TV of the two-outcome conditional
    "which channel", resolved in xi.  This is the PRIMARY Type-C readout on this
    system -- E_cond is a KDE-vs-KDE total variation over the full psi axis whose
    finite-K floor (~0.15 at K = 1024) is as large as the entire deficit it is meant
    to measure, an instrument limitation recorded rather than hidden.  E_chan's floor
    is a local binomial fluctuation, roughly an order of magnitude below the signal.
    """
    mB = channel_weight_B(GRID2.x2(p2.device, p2.dtype)).to(p2.dtype)
    col = torch.clamp(p2.sum(dim=2), min=EPS)
    pB_phi = (p2 * mB).sum(dim=2) / col
    p_phi = col * GRID2.dx2
    return (p_phi * (pB_phi - pB_phi_ref).abs()).sum(dim=1) * GRID2.dx1


def _e_cond(p2, p_cond_ref):
    """E_cond = int p(phi) TV(p(psi|phi), p_ref(psi|phi)) dphi.  p2: (R, n1, n2)."""
    p_phi = p2.sum(dim=2) * GRID2.dx2
    p_c = p2 / torch.clamp(p2.sum(dim=2, keepdim=True), min=EPS) / GRID2.dx2
    tv_col = 0.5 * ((p_c - p_cond_ref).abs().sum(dim=2) * GRID2.dx2)
    return (p_phi * tv_col).sum(dim=1) * GRID2.dx1


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
        for a in ("K", "dt", "n_steps", "block", "eps_bw", "eta_bw", "n_saves",
                  "profile_every", "joint_every", "ess_window_steps",
                  "n_strata", "cv", "init", "warm_start"):
            assert getattr(c, a) == getattr(c0, a), f"non-uniform {a} across configs"
    is2d = c0.cv == "phipsi"
    assert c0.cv in ("phi", "phipsi"), f"unknown cv {c0.cv!r}"
    assert not (is2d and any(m.use_fr or m.sham for m in methods)), (
        "the augmented-CV arm is a plain-ABP baseline; reallocation on an already "
        "biased coordinate is a different experiment and is not mixed into it")
    K, dt, n_steps, block = c0.K, c0.dt, c0.n_steps, c0.block
    assert n_steps % block == 0
    n_blocks = n_steps // block
    n_strata = c0.n_strata

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

    def rowmask(fn):
        return torch.tensor([fn(m) for m in methods], device=device).repeat(B)
    is_fr_row = rowmask(lambda m: m.use_fr and not m.sham)
    is_sham_row = rowmask(lambda m: m.sham)
    is_cond_row = rowmask(lambda m: m.cond_fr)
    # a sham inherits its partner's geometry: it must be the null for THAT step
    partner_cond = torch.tensor([methods[partner_col[j]].cond_fr
                                 for b in range(B) for j in range(M)], device=device)
    is_wt_row = rowmask(lambda m: m.cond_weighted)
    is_state_row = rowmask(lambda m: m.cond_state)
    assert all(m.cond_fr or not m.cond_state for m in methods), (
        "the discrete-state score is a CONDITIONAL (fiber-wise) allocation rule")
    assert all(m.cond_fr or m.sham or not m.cond_weighted for m in methods), (
        "weighted selection is defined for the FIBER-WISE step only: the marginal "
        "step moves the very marginal the accumulator learns from, and compensating "
        "weights there would cancel the arm outright")
    for j, m in enumerate(methods):
        if m.sham:
            assert not (m.cond_weighted and not methods[partner_col[j]].cond_fr), (
                f"sham {m.name!r} carries weights but shadows a MARGINAL arm; "
                f"weighted selection is fiber-wise only")
            assert m.cond_weighted == methods[partner_col[j]].cond_weighted, (
                f"sham {m.name!r} must carry the same weighting as the arm it "
                f"shadows ({m.shadows!r}); otherwise it is not that arm's null")
    is_coarse_row = rowmask(lambda m: m.coarse_bins > 0)
    coarse_nb = torch.tensor([m.coarse_bins for m in methods], device=device,
                             dtype=torch.long).repeat(B)
    cond_nb1 = torch.tensor([m.cond_bins1 for m in methods], device=device,
                            dtype=torch.long).repeat(B)
    cond_nb2 = torch.tensor([m.cond_bins2 for m in methods], device=device,
                            dtype=torch.long).repeat(B)
    theta0 = torch.tensor([m.theta if (m.use_fr and not m.sham) else 0.0
                           for m in methods], device=device, dtype=dtype).repeat(B)
    alpha_ess = torch.tensor([m.alpha_ess for m in methods], device=device,
                             dtype=dtype).repeat(B)
    assert all(m.g_shus > 0 for m in methods), "g_shus must be positive"
    gain = torch.tensor([m.g_shus for m in methods], device=device,
                        dtype=dtype).repeat(B)

    k1e, r1e = periodic_gaussian_kernel(c0.eta_bw, GRID2.dx1, GRID2.n1, device, dtype)
    k2e, r2e = periodic_gaussian_kernel(c0.eta_bw, GRID2.dx2, GRID2.n2, device, dtype)

    def cfg_b(fn):
        return torch.tensor([fn(c) for c in cfgs], device=device, dtype=dtype)

    def to_run(t_b):
        return t_b.repeat_interleave(M).reshape(R, 1)
    beta = to_run(cfg_b(lambda c: c.beta))
    Hperp = to_run(cfg_b(lambda c: c.Hperp))
    Delta = to_run(cfg_b(lambda c: c.Delta))
    Ha = to_run(cfg_b(lambda c: c.Ha))
    Hb = to_run(cfg_b(lambda c: c.Hb))
    noise_amp = torch.sqrt(2.0 * dt / beta)

    refs = [reference_objects(float(c.beta), float(c.Hperp), float(c.Delta),
                              float(c.Ha), float(c.Hb), device, dtype) for c in cfgs]
    cond_log_q = target_log_q(methods, refs, M, device, dtype)
    F1_ref = torch.stack([r["F1"] for r in refs]).repeat_interleave(M, dim=0)
    p_cond_ref = torch.stack([r["p_cond"] for r in refs]).repeat_interleave(M, dim=0)
    pB_phi_ref = torch.stack([r["pB_phi_ref"] for r in refs]).repeat_interleave(M, dim=0)
    rho_ref = torch.stack([r["rho1"] for r in refs]).repeat_interleave(M, dim=0)
    p_B_ref = np.array([r["p_B_ref"] for r in refs])

    # initial conditions: every walker in the channel-A well at (0, 0); paired arms
    X1 = torch.empty((B, K), device=device, dtype=dtype)
    X2 = torch.empty((B, K), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        init = cfgs[b].init
        assert init in ("chanA", "stationary", "boltzmann"), f"unknown init {init!r}"
        if init == "chanA":
            rng = np.random.default_rng(1000 + int(sd))
            X1[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device,
                                    dtype=dtype)
            X2[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device,
                                    dtype=dtype)
        else:
            X1[b], X2[b] = stationary_init(cfgs[b], K, sd, init == "stationary",
                                           device, dtype)
    X1 = wrap_periodic(X1.repeat_interleave(M, dim=0), GRID2.x1min, GRID2.L1)
    X2 = wrap_periodic(X2.repeat_interleave(M, dim=0), GRID2.x2min, GRID2.L2)
    anc = torch.arange(K, device=device).unsqueeze(0).expand(R, K).clone()
    anc_g = anc.clone()
    # statistical weights, mean 1 (Phase I).  Equal-weight arms keep them at exactly
    # 1.0 forever, so every weighted code path below is the identity for them.
    W = torch.ones((R, K), device=device, dtype=dtype)
    if is2d:
        from ..shus2d import ShusAccumulator2
        shus = ShusAccumulator2(R, GRID2, beta, c0.eps_bw, device, dtype, gain=gain)
        rho_ref2 = torch.exp(-beta.reshape(R, 1, 1)
                             * torch.stack([r["F2"] for r in refs]
                                           ).repeat_interleave(M, dim=0))
        rho_ref2 = rho_ref2 / integral2(rho_ref2, GRID2).reshape(R, 1, 1)
    else:
        R0 = None
        if c0.warm_start:
            R0 = torch.cat([warm_start_R(c, device, dtype).expand(M, -1)
                            for c in cfgs], dim=0)
        shus = ShusAccumulator1P(R, GRID1, beta, c0.eps_bw, device, dtype, gain=gain,
                                 R_init=R0)
    dep_ref_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)
    dep_self_cur = torch.full((R,), float("nan"), device=device, dtype=dtype)

    gen_n = torch.Generator(device=device)
    gen_n.manual_seed(2000 + batch_seed)
    gen_f = torch.Generator(device=device)
    gen_f.manual_seed(3000 + batch_seed)

    save_steps = sorted({*range(0, n_steps, max(1, n_steps // c0.n_saves)),
                         n_steps - 1})
    n_saves = len(save_steps)
    save_set = set(save_steps)
    prof_steps = save_steps[:: c0.profile_every]
    if save_steps[-1] not in prof_steps:
        prof_steps = prof_steps + [save_steps[-1]]
    n_prof = len(prof_steps)
    prof_set = set(prof_steps)
    joint_steps = save_steps[:: c0.joint_every]
    if save_steps[-1] not in joint_steps:
        joint_steps = joint_steps + [save_steps[-1]]
    n_joint = len(joint_steps)
    joint_set = set(joint_steps)

    ts = {k: torch.zeros((R, n_saves), device=device, dtype=dtype) for k in
          ("l2_f", "kl_u", "tv_u", "e_cond", "e_chan", "ess_anc", "wmax",
           "ess_anc_glob", "wmax_glob", "n_anc", "dep_ref", "dep_self",
           # Phase I: the price and the proof of weighted selection.  ess_w is the
           # weight ESS (1 for every equal-weight arm); wmax_w the largest single
           # share of the represented mass; w_sum the running total weight, which
           # the stratum-wise renormalization holds at K.
           "ess_w", "wmax_w", "w_sum")}
    # P is the channel split of the REPRESENTED law (weights); P_n is the channel
    # split of the PARTICLES.  They are the same series for every equal-weight arm
    # and the direct measurement of the decoupling for a weighted one.
    ts["P"] = torch.zeros((R, n_saves, 2), device=device, dtype=dtype)
    ts["P_n"] = torch.zeros((R, n_saves, 2), device=device, dtype=dtype)
    prof = {"pmf": torch.zeros((R, n_prof, NG), device=device, dtype=dtype),
            "marg": torch.zeros((R, n_prof, NG), device=device, dtype=dtype),
            "pB_phi": torch.zeros((R, n_prof, NG), device=device, dtype=dtype),
            "joint": torch.zeros((R, n_joint, NG, NG), device=device, dtype=dtype)}
    if is2d:
        prof["pmf2"] = torch.zeros((R, n_joint, NG, NG), device=device, dtype=dtype)
    ev = {k: torch.zeros((R, max(n_events, 1)), device=device, dtype=dtype)
          for k in ("theta", "ess_fr", "turnover")}
    tot_turn = torch.zeros(R, device=device, dtype=dtype)
    save_ptr, prof_ptr, joint_ptr, event_ptr = 0, 0, 0, 0
    ar = torch.arange(K, device=device).unsqueeze(0).expand(R, K)

    for step in range(n_steps):
        if c0.ess_window_steps > 0 and step % c0.ess_window_steps == 0:
            anc = ar.clone()

        # ---- physical propagation: overdamped EM on the torus, bias on phi ONLY
        g1, g2 = gradV_of(X1, X2, Hperp, Delta, Ha, Hb)
        if is2d:
            b1, b2 = shus.bias_force_at(X1, X2)
        else:
            b1, b2 = shus.bias_force_at(X1), 0.0
        z1 = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        z2 = torch.randn((B, K), device=device, dtype=dtype,
                         generator=gen_n).repeat_interleave(M, dim=0)
        X1 = wrap_periodic(X1 + (-g1 + b1) * dt + noise_amp * z1,
                           GRID2.x1min, GRID2.L1)
        X2 = wrap_periodic(X2 + (-g2 + b2) * dt + noise_amp * z2,
                           GRID2.x2min, GRID2.L2)

        shus.deposit(X1, X2) if is2d else shus.deposit(X1, W)

        if (step + 1) % block == 0:
            if is2d:
                r_n = shus.R / integral2(shus.R, GRID2).reshape(R, 1, 1)
                inc = shus.update(dt, K)
                d_n = inc / torch.clamp(integral2(inc, GRID2),
                                        min=EPS).reshape(R, 1, 1)
                dep_ref_cur = torch.sqrt(((d_n - rho_ref2) ** 2).mean(dim=(1, 2)))
                dep_self_cur = torch.sqrt(((d_n - r_n) ** 2).mean(dim=(1, 2)))
            else:
                r_n = shus.R / integral1p(shus.R, GRID1).unsqueeze(1)
                inc = shus.update(dt, K)
                d_n = inc / torch.clamp(integral1p(inc, GRID1), min=EPS).unsqueeze(1)
                dep_ref_cur = torch.sqrt(((d_n - rho_ref) ** 2).mean(dim=1))
                dep_self_cur = torch.sqrt(((d_n - r_n) ** 2).mean(dim=1))
            blk = (step + 1) // block
            if event_ptr < n_events and event_blocks[event_ptr] == blk:
                active = fires[event_ptr]
                sel = ar.clone()
                W_cond = W                       # weights after the fiber-wise step
                turn = torch.zeros(R, device=device, dtype=torch.long)
                th_u = torch.zeros(R, device=device, dtype=dtype)
                ef = torch.full((R,), float("nan"), device=device, dtype=dtype)
                marg_rows = (active & is_fr_row & ~is_cond_row)
                marg_sham = (active & is_sham_row & ~partner_cond)
                if bool((marg_rows | marg_sham).any()):
                    s_m, t_m, th_m, e_m = fr_event1p(
                        X1, marg_rows, marg_sham, is_coarse_row, coarse_nb, partner,
                        theta0, alpha_ess, k1e, r1e, GRID1, gen_f)
                    touched = (marg_rows | marg_sham).unsqueeze(1)
                    sel = torch.where(touched, s_m, sel)
                    turn = torch.where(touched[:, 0], t_m, turn)
                    th_u = torch.where(touched[:, 0], th_m, th_u)
                    ef = torch.where(touched[:, 0], e_m, ef)
                cond_rows = (active & is_fr_row & is_cond_row)
                cond_sham = (active & is_sham_row & partner_cond)
                if bool((cond_rows | cond_sham).any()):
                    s_c, t_c, th_c, e_c, W_c = fr_event_cond(
                        X1, X2, cond_rows, cond_sham, cond_nb1, cond_nb2, n_strata,
                        partner, theta0, alpha_ess, k1e, r1e, k2e, r2e, GRID2,
                        gen_f, cond_log_q, W, is_wt_row, channel_of(X2),
                        is_state_row, len(REGIONS))
                    touched = (cond_rows | cond_sham).unsqueeze(1)
                    W_cond = torch.where(touched, W_c, W_cond)
                    sel = torch.where(touched, s_c, sel)
                    turn = torch.where(touched[:, 0], t_c, turn)
                    th_u = torch.where(touched[:, 0], th_c, th_u)
                    ef = torch.where(touched[:, 0], e_c, ef)
                ev["theta"][:, event_ptr] = th_u
                ev["ess_fr"][:, event_ptr] = ef
                ev["turnover"][:, event_ptr] = turn.to(dtype)
                tot_turn += turn.to(dtype)
                # ESTIMATOR PROTECTION: walker arrays only
                X1 = torch.gather(X1, 1, sel)
                X2 = torch.gather(X2, 1, sel)
                # weighted rows take the compensating weights the fiber-wise step
                # computed; every other row (marginal FR, sham, plain SHUS) carries
                # its weights through the same gather as its walkers -- which for
                # the equal-weight arms is exactly 1 in, exactly 1 out
                W = torch.where(is_wt_row.unsqueeze(1), W_cond,
                                torch.gather(W, 1, sel))
                anc = torch.gather(anc, 1, sel)
                anc_g = torch.gather(anc_g, 1, sel)
                event_ptr += 1

        if step in save_set:
            F_hat2 = shus.f_estimate() if is2d else None
            F_hat = reduce_to_phi(F_hat2, beta) if is2d else shus.f_estimate()
            d = F_hat - F1_ref
            d = d - d.mean(dim=1, keepdim=True)
            ts["l2_f"][:, save_ptr] = torch.sqrt((d * d).mean(dim=1))
            p_hat = binned_density1p(X1, k1e, r1e, GRID1, W)
            ts["kl_u"][:, save_ptr] = kl_to_uniform1p(p_hat, GRID1)
            ts["tv_u"][:, save_ptr] = tv_to_uniform1p(p_hat, GRID1)
            p2 = binned_density2(X1, X2, k1e, r1e, k2e, r2e, GRID2, W)
            ts["e_cond"][:, save_ptr] = _e_cond(p2, p_cond_ref)
            ts["e_chan"][:, save_ptr] = _e_chan(p2, pB_phi_ref)
            e_, w_ = ancestor_stats(anc, K)
            ts["ess_anc"][:, save_ptr] = e_
            ts["wmax"][:, save_ptr] = w_
            eg_, wg_ = ancestor_stats(anc_g, K)
            ts["ess_anc_glob"][:, save_ptr] = eg_
            ts["wmax_glob"][:, save_ptr] = wg_
            ts["n_anc"][:, save_ptr] = surviving_ancestors(anc_g, K)
            ts["dep_ref"][:, save_ptr] = dep_ref_cur
            ts["dep_self"][:, save_ptr] = dep_self_cur
            lab = channel_of(X2)
            wsum = torch.clamp(W.sum(dim=1), min=EPS)
            for j in (0, 1):
                inj = (lab == j).to(dtype)
                ts["P"][:, save_ptr, j] = (W * inj).sum(dim=1) / wsum
                ts["P_n"][:, save_ptr, j] = inj.mean(dim=1)
            ts["ess_w"][:, save_ptr] = weight_ess(W)
            ts["wmax_w"][:, save_ptr] = W.max(dim=1).values / wsum
            ts["w_sum"][:, save_ptr] = W.sum(dim=1) / float(K)
            if step in prof_set:
                prof["pmf"][:, prof_ptr] = F_hat
                prof["marg"][:, prof_ptr] = p_hat
                mB = channel_weight_B(GRID2.x2(device, dtype)).to(dtype)
                col = torch.clamp(p2.sum(dim=2), min=EPS)
                prof["pB_phi"][:, prof_ptr] = (p2 * mB).sum(dim=2) / col
                prof_ptr += 1
            if step in joint_set:
                prof["joint"][:, joint_ptr] = p2
                if is2d:
                    prof["pmf2"][:, joint_ptr] = F_hat2
                joint_ptr += 1
            save_ptr += 1
        if progress is not None and step % progress == 0:
            print(f"    step {step}/{n_steps}", flush=True)

    worstP = float((ts["P"].sum(dim=2) - 1.0).abs().max())
    assert worstP < 1e-9, f"channel fractions do not sum to 1 (worst {worstP:.3e})"

    t_axis = np.array([s * dt for s in save_steps])
    prof_t = np.array([s * dt for s in prof_steps])
    joint_t_axis = np.array([s * dt for s in joint_steps])
    ev_t = np.array([k * block * dt for k in event_blocks])

    def npy(t):
        return t.detach().cpu().numpy()

    recs = []
    for b in range(B):
        for m in range(M):
            r = b * M + m
            l2 = npy(ts["l2_f"][r])
            recs.append(dict(
                config=asdict(cfgs[b]), seed=int(seeds[b]),
                method=asdict(methods[m]), batch_seed=batch_seed,
                reference_id=REFERENCE_ID,
                eval_window=(GRID1.xmin, GRID1.xmin + GRID1.L),
                time=t_axis, profile_time=prof_t, joint_time=joint_t_axis,
                x_grid=npy(GRID1.x(device, dtype)),
                psi_grid=npy(GRID2.x2(device, dtype)),
                F_ref=npy(F1_ref[r]), F2_ref=npy(refs[b]["F2"]),
                p_cond_ref=npy(refs[b]["p_cond"]),
                pB_phi_ref=npy(refs[b]["pB_phi_ref"]), p_B_ref=float(p_B_ref[b]),
                p_B_ref_biased=float(refs[b]["p_B_ref_biased"]),
                pmf_t=npy(prof["pmf"][r]), marginal_t=npy(prof["marg"][r]),
                pB_phi_t=npy(prof["pB_phi"][r]), joint_t=npy(prof["joint"][r]),
                **({"pmf2_t": npy(prof["pmf2"][r])} if is2d else {}),
                l2_f_t=l2, kl_u_t=npy(ts["kl_u"][r]), tv_u_t=npy(ts["tv_u"][r]),
                e_cond_t=npy(ts["e_cond"][r]), e_chan_t=npy(ts["e_chan"][r]),
                ess_anc_t=npy(ts["ess_anc"][r]), wmax_t=npy(ts["wmax"][r]),
                ess_anc_glob_t=npy(ts["ess_anc_glob"][r]),
                wmax_glob_t=npy(ts["wmax_glob"][r]), n_anc_t=npy(ts["n_anc"][r]),
                dep_ref_l2_t=npy(ts["dep_ref"][r]),
                dep_self_l2_t=npy(ts["dep_self"][r]),
                P_regions=npy(ts["P"][r]), P_regions_n=npy(ts["P_n"][r]),
                ess_w_t=npy(ts["ess_w"][r]), wmax_w_t=npy(ts["wmax_w"][r]),
                w_sum_t=npy(ts["w_sum"][r]),
                event_time=ev_t, event_theta=npy(ev["theta"][r]),
                event_ess_fr=npy(ev["ess_fr"][r]),
                event_turnover=npy(ev["turnover"][r]),
                final_l2_f=float(l2[-1]),
                int_l2_f=float(np.trapezoid(l2, t_axis)),
                final_e_cond=float(npy(ts["e_cond"][r])[-1]),
                final_e_chan=float(npy(ts["e_chan"][r])[-1]),
                int_e_chan=float(np.trapezoid(npy(ts["e_chan"][r]), t_axis)),
                final_p_B=float(npy(ts["P"][r])[-1, 1]),
                final_p_B_n=float(npy(ts["P_n"][r])[-1, 1]),
                final_ess_w=float(npy(ts["ess_w"][r])[-1]),
                min_ess_w=float(npy(ts["ess_w"][r]).min()),
                total_turnover=float(tot_turn[r]),
            ))
    return recs


# -----------------------------------------------------------------------------
# F2a: clone decorrelation in the hidden coordinate
# -----------------------------------------------------------------------------
def clone_decorrelation(cfg: BiChannelConfig, seeds, t0=200.0, lag_max=400.0,
                        n_bins=8, n_par=8, batch_seed=777, device=DEVICE,
                        dtype=DTYPE, progress=None):
    """Q4a's instrument, adapted to a Type-C system (Phase F2a).

    Spin plain SHUS up to t0, freeze the learned bias, pick n_par parents in each of
    n_bins equal phi-strata, duplicate each into two children with independent noise,
    and watch how fast a clone forgets its parent.

    Two measures, because one number would hide the distinction a conditional
    correction turns on:

    * m_chan(tau): excess probability that siblings still share a CHANNEL over the
      same-bin independent baseline.  A population correction is carried by exactly
      this; if it decays inside an event stride the correction cannot persist.
    * m_psi / m_phi(tau): the frozen Q4a measure 1 - d_sib/d_ind on RMS torus pair
      distance.  Fast decay here means siblings are not redundant for conditional
      averages -- the good case for variance.

    Returns per-row arrays; tau_clone is the first lag with m <= 1/e.
    """
    B = len(seeds)
    K, dt, block = cfg.K, cfg.dt, cfg.block
    n_spin = int(round(t0 / dt))
    assert n_spin % block == 0, "t0 must be a whole number of adaptation blocks"

    beta = torch.full((B, 1), cfg.beta, device=device, dtype=dtype)
    Hp = torch.full((B, 1), cfg.Hperp, device=device, dtype=dtype)
    Dl = torch.full((B, 1), cfg.Delta, device=device, dtype=dtype)
    Ha = torch.full((B, 1), cfg.Ha, device=device, dtype=dtype)
    Hb = torch.full((B, 1), cfg.Hb, device=device, dtype=dtype)
    noise_amp = torch.sqrt(2.0 * dt / beta)

    X1 = torch.empty((B, K), device=device, dtype=dtype)
    X2 = torch.empty((B, K), device=device, dtype=dtype)
    for b, sd in enumerate(seeds):
        rng = np.random.default_rng(1000 + int(sd))
        X1[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device, dtype=dtype)
        X2[b] = torch.as_tensor(rng.normal(0.0, 0.15, K), device=device, dtype=dtype)
    X1 = wrap_periodic(X1, GRID2.x1min, GRID2.L1)
    X2 = wrap_periodic(X2, GRID2.x2min, GRID2.L2)
    shus = ShusAccumulator1P(B, GRID1, beta, cfg.eps_bw, device, dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(2000 + batch_seed)

    for step in range(n_spin):                                  # spin-up
        g1, g2 = gradV_of(X1, X2, Hp, Dl, Ha, Hb)
        b1 = shus.bias_force_at(X1)
        X1 = wrap_periodic(X1 + (-g1 + b1) * dt + noise_amp * torch.randn(
            (B, K), device=device, dtype=dtype, generator=gen), GRID2.x1min, GRID2.L1)
        X2 = wrap_periodic(X2 - g2 * dt + noise_amp * torch.randn(
            (B, K), device=device, dtype=dtype, generator=gen), GRID2.x2min, GRID2.L2)
        shus.deposit(X1)
        if (step + 1) % block == 0:
            shus.update(dt, K)
    Fp_frozen = shus.Fp.clone()                                 # the bias is now fixed

    # stratified parents: n_par per phi-bin, drawn without replacement
    bw = GRID1.L / n_bins
    bidx = torch.remainder(((X1 - GRID1.xmin) / bw).long(), n_bins)
    cnt = torch.zeros((B, n_bins), device=device, dtype=dtype)
    cnt.scatter_add_(1, bidx, torch.ones_like(X1))
    assert float(cnt.min()) >= n_par, (
        f"a phi-stratum holds {float(cnt.min()):.0f} < {n_par} walkers at t0; "
        "the parent draw would sample with replacement")
    key = bidx.to(dtype) + torch.rand((B, K), device=device, dtype=dtype, generator=gen)
    order = key.argsort(dim=1)
    off = (torch.cumsum(cnt, dim=1) - cnt).long()
    take = (off.unsqueeze(2) + torch.arange(n_par, device=device).view(1, 1, -1)
            ).reshape(B, n_bins * n_par)
    parents = torch.gather(order, 1, take)                      # (B, n_bins*n_par)
    P = parents.shape[1]

    C1 = torch.gather(X1, 1, parents).repeat_interleave(2, dim=1)   # (B, 2P)
    C2 = torch.gather(X2, 1, parents).repeat_interleave(2, dim=1)
    N = 2 * P
    Fp_c = Fp_frozen
    gen_c = torch.Generator(device=device)
    gen_c.manual_seed(3000 + batch_seed)

    # sibling pairs (2i, 2i+1); independent baseline pairs child0 of parent j with
    # child0 of parent (j+1) inside the SAME phi-bin
    sib_a = torch.arange(0, N, 2, device=device)
    sib_b = sib_a + 1
    j = torch.arange(P, device=device)
    nxt = (j % n_par + 1) % n_par + (j // n_par) * n_par
    ind_a, ind_b = 2 * j, 2 * nxt

    n_fine = int(round(20.0 / dt))
    fine_every = max(1, int(round(0.2 / dt)))
    coarse_every = max(1, int(round(10.0 / dt)))
    n_lag = int(round(lag_max / dt))
    rec = sorted({0, *range(fine_every, min(n_fine, n_lag) + 1, fine_every),
                  *range(coarse_every, n_lag + 1, coarse_every)})
    out = {k: torch.zeros((B, len(rec)), device=device, dtype=dtype)
           for k in ("d_sib_psi", "d_ind_psi", "d_sib_phi", "d_ind_phi",
                     "same_sib", "same_ind")}

    def record(ptr):
        for nm, a, b in (("sib", sib_a, sib_b), ("ind", ind_a, ind_b)):
            d2 = torus_distance(C2[:, a], C2[:, b], GRID2.L2)
            d1 = torus_distance(C1[:, a], C1[:, b], GRID2.L1)
            out[f"d_{nm}_psi"][:, ptr] = torch.sqrt((d2 * d2).mean(dim=1))
            out[f"d_{nm}_phi"][:, ptr] = torch.sqrt((d1 * d1).mean(dim=1))
            out[f"same_{nm}"][:, ptr] = (channel_of(C2[:, a])
                                         == channel_of(C2[:, b])).to(dtype).mean(dim=1)

    rec_set = {s: i for i, s in enumerate(rec)}
    if 0 in rec_set:
        record(rec_set[0])
    for step in range(1, n_lag + 1):
        g1, g2 = gradV_of(C1, C2, Hp, Dl, Ha, Hb)
        b1 = interp1p(C1, Fp_c, GRID1)
        C1 = wrap_periodic(C1 + (-g1 + b1) * dt + noise_amp * torch.randn(
            (B, N), device=device, dtype=dtype, generator=gen_c), GRID2.x1min, GRID2.L1)
        C2 = wrap_periodic(C2 - g2 * dt + noise_amp * torch.randn(
            (B, N), device=device, dtype=dtype, generator=gen_c), GRID2.x2min, GRID2.L2)
        if step in rec_set:
            record(rec_set[step])
        if progress is not None and step % progress == 0:
            print(f"    lag step {step}/{n_lag}", flush=True)

    npy = lambda t: t.detach().cpu().numpy()
    lag_t = np.array([s * dt for s in rec])
    m_psi = 1.0 - npy(out["d_sib_psi"]) / np.clip(npy(out["d_ind_psi"]), 1e-30, None)
    m_phi = 1.0 - npy(out["d_sib_phi"]) / np.clip(npy(out["d_ind_phi"]), 1e-30, None)
    same_ind = npy(out["same_ind"])
    m_chan = (npy(out["same_sib"]) - same_ind) / np.clip(1.0 - same_ind, 1e-12, None)

    def tau_of(m):
        out_ = []
        for row in m:
            hit = np.where(row <= 1.0 / math.e)[0]
            out_.append(float(lag_t[hit[0]]) if len(hit) else float("nan"))
        return np.array(out_)

    return {"lag": lag_t, "m_psi": m_psi, "m_phi": m_phi, "m_chan": m_chan,
            "tau_psi": tau_of(m_psi), "tau_phi": tau_of(m_phi),
            "tau_chan": tau_of(m_chan), "d_ind_psi": npy(out["d_ind_psi"]),
            "same_ind": same_ind}
