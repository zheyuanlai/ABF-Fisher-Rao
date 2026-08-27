"""The allocation arms, as one object per run.

Frozen protocol: ``docs/QR_DECOUPLING_PREREGISTRATION.md``.

This is the orchestration layer that the four Stage-0 modules were built for.
It owns, per run: the Fisher--Rao cell mass, the online difficulty estimator,
the replica-count target, and the decision of whether a count change is worth
its genealogy.  It emits *indices and hold times only* -- it never touches an
ABF accumulator, and the engine is what applies them.

The arms, and what differs between them::

    A0    qr disabled -- plain ABF, no cells, no mass, no estimator
    A2    mass only; r never changes            identity gate: must equal A0
    A3    r = uniform                           count balancing, the incumbent
    A4a   r ∝ sqrt(a)                           leverage only, no estimation
    A4b   r ∝ sqrt(a Gamma_hat)                 leverage x measured difficulty
    A5    r ∝ sqrt(a Gamma_hat + lam q^2)       ESS-constrained, rho frozen
    A6a   the same r as A4a, held by the BIAS instead of by birth--death
    A6b   the same r as A4b, held by the BIAS instead of by birth--death

Why A6 exists
-------------
Measured, not assumed.  Across a run the allocation target is *stable* --
opportunity-to-opportunity drift in ``r*`` is 0.000 total variation for A4a and
6% of the target-occupancy gap for A4b -- while the gap between ``r*`` and the
actual occupancy stays at 0.18-0.29 TV.  The arm is not chasing noise in its own
estimate: the dynamics are pulling the population away from ``r*`` continuously,
and birth--death is buying back a fraction of that every opportunity at the cost
of genealogy it never recovers.

But a replica density is exactly what a bias potential controls.  Sampling under
``A(z)`` gives ``p(z) ∝ exp(-beta(F - A))``, so

    A(z) = Fhat(z) + log r*(z) / beta   =>   p(z) ∝ r*(z)

makes the target allocation the *stationary* occupancy: no cloning, no genealogy,
no leak between opportunities.  The mean force stays estimable because the extra
term is a function of ``z`` alone, which is the same fibre-conditional
invariance that licenses ABF itself.

That makes A6 the honest comparator for A4, and sharpens the campaign's
question.  If a birth--death arm cannot beat the bias-held arm carrying the
*identical* ``r*``, then reallocation buys nothing that a bias cannot, and
whatever is left for it to do lives in establishment -- moving mass into a newly
discovered region faster than diffusion can -- rather than in allocation.

A3, A4a, A4b and A5 share one opportunity schedule, one benefit gate, one floor,
one resampler and one rejuvenation rule.  **Only ``r`` differs.**  That is the
whole design: an arm-specific anything else would make a margin unattributable,
which is how the previous four campaigns lost their positives.

A2 consumes no randomness at all -- the mass update is deterministic -- so it is
trajectory-identical to A0 by construction rather than by tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import allocation as al
from . import balanced_representation as br
from . import cell_mass as cm
from . import information as inf

#: Arms this module implements.  A0 is "not enabled", which is why it is absent.
ARMS = ("A2", "A3", "A4a", "A4b", "A5", "A6a", "A6b")

#: Arms that never change replica counts.
MASS_ONLY = ("A2",)

#: Arms that hold their allocation with the bias rather than with birth--death.
#: They never resample, so they pay no genealogy at all.
BIAS_HELD = ("A6a", "A6b")

#: Arms that read the online difficulty estimator.
USES_GAMMA = ("A4b", "A5", "A6b")


@dataclass
class QRConfig:
    arm: str
    n_cells: int = 32
    rho: float = 0.5                      # A5 only
    floor_fraction: float = al.FLOOR_FRACTION
    eps_gene: float = 0.1
    benefit_threshold: float = 0.10
    #: Occupancy chi-square per live cell required before a count move is even
    #: considered.  Its expectation under pure multinomial noise is 1, so 2.0 is
    #: "twice what fluctuation explains" rather than a tuned number.  Without it
    #: the benefit statistic fires on noise 84% of the time -- see
    #: ``balanced_representation.occupancy_chi2``.
    chi2_threshold: float = 2.0
    #: Width of the no-move band, in standard deviations of the target count.
    deadband_z: float = 1.0
    opportunity_every: int = 500
    burnin_fraction: float = 0.20
    stop_fraction: float = 0.80
    history_capacity: int = 400
    theta: float = 1.0

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown qr arm {self.arm!r}; have {list(ARMS)}")
        if not 0.0 <= self.burnin_fraction < self.stop_fraction <= 1.0:
            raise ValueError("qr needs 0 <= burnin < stop <= 1")
        if self.stop_fraction >= 1.0:
            raise ValueError(
                "the allocation window must close strictly before the end of "
                "the run: the long-time limit is ABF's by construction, and a "
                "window that silently ran to the end is what cost v3 every arm")


@dataclass
class QRDecision:
    """What the engine should apply.  ``src is None`` means "change nothing"."""

    src: Optional[np.ndarray] = None
    hold: Optional[np.ndarray] = None      # per-output-slot rejuvenation steps
    bias_increment: Optional[np.ndarray] = None   # grid-level, A6 only
    row: Dict = field(default_factory=dict)


def config_from_dict(cfg: Dict) -> Optional[QRConfig]:
    block = (cfg.get("qr", {}) or {})
    if not bool(block.get("enabled", False)):
        return None
    known = set(QRConfig.__dataclass_fields__) | {"enabled"}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(
            f"unknown qr keys {unknown}; a knob the protocol does not define is "
            f"a knob nobody preregistered")
    return QRConfig(**{k: v for k, v in block.items() if k != "enabled"})


class QRArm:
    """Per-run allocation state for one batch row."""

    #: Set after a resampling to the step at which the last clone finishes its
    #: rejuvenation hold.  Not a new knob: it is the same bound that sets the
    #: hold, used for the thing the hold implies.
    _cooldown_until: int = 0

    def __init__(self, qcfg: QRConfig, n_particles: int, x_grid: np.ndarray,
                 eval_mask: np.ndarray, beta: float, dt: float,
                 obs_interval: int):
        self.cfg = qcfg
        self.K = int(n_particles)
        self.beta = float(beta)
        self.dt = float(dt)
        self.obs_interval = int(obs_interval)

        x_grid = np.asarray(x_grid, dtype=float)
        self.x0, self.x1 = float(x_grid[0]), float(x_grid[-1])
        J = int(qcfg.n_cells)
        self.J = J
        self.edges = np.linspace(self.x0, self.x1, J + 1)
        self.cell_of_grid = np.clip(
            np.digitize(x_grid, self.edges) - 1, 0, J - 1)

        # Static leverage: pure grid geometry and the evaluation mask.  No free
        # energy enters here, which is why the mask has to be geometric.
        self.a_cell = al.cell_reduce(
            al.leverage(x_grid, eval_mask), self.cell_of_grid, J)

        self.mass = cm.CellMass(n_cells=J, theta=float(qcfg.theta))
        self.hist = inf.MeanForceHistory(n_cells=J,
                                         capacity=int(qcfg.history_capacity))
        self._s2_sum = np.zeros(J)
        self._s2_n = 0
        self._cooldown_until = 0
        self.rows = []

    # -- geometry ---------------------------------------------------------
    def cell_of(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(np.asarray(x, dtype=float), self.edges) - 1,
                       0, self.J - 1)

    # -- the online difficulty stream -------------------------------------
    def observe(self, x: np.ndarray, force: np.ndarray,
                mean_force_at: np.ndarray,
                eligible: Optional[np.ndarray] = None) -> None:
        """One ABF observation batch.

        Fed from the *same eligible stream the accumulator sees*, which is what
        keeps the clone -> higher measured difficulty -> more clones loop from
        closing: held-out replicas contribute to neither.
        """
        if self.cfg.arm not in USES_GAMMA:
            return
        cell = self.cell_of(x)
        self.hist.push(cell, force, eligible)
        self._s2_sum += inf.conditional_force_variance(
            cell, force, mean_force_at, self.J, eligible)
        self._s2_n += 1

    def gamma_hat(self) -> np.ndarray:
        if self._s2_n == 0:
            return np.ones(self.J)
        tau = inf.tau_from_lag1(
            self.hist, obs_interval=float(self.obs_interval) * self.dt)
        return inf.gamma_hat_decomposed(self._s2_sum / self._s2_n, tau)

    # -- the allocation target --------------------------------------------
    def r_star(self, q_cell: np.ndarray) -> tuple:
        arm = self.cfg.arm
        if arm in ("A4a", "A6a"):
            r, lam, ess = al.r_neyman(self.a_cell), 0.0, float("nan")
            return al.apply_floor(r, self.cfg.floor_fraction), lam, ess
        if arm == "A6b":
            g = self.a_cell * self.gamma_hat()
            return (al.apply_floor(al.r_neyman(g), self.cfg.floor_fraction),
                    0.0, float("nan"))
        if arm == "A3":
            r, lam, ess = al.r_uniform(self.J), 0.0, float("nan")
        elif arm == "A4a":
            r, lam, ess = al.r_neyman(self.a_cell), 0.0, float("nan")
        else:
            g = self.a_cell * self.gamma_hat()
            if arm == "A4b":
                r, lam, ess = al.r_neyman(g), 0.0, float("nan")
            else:
                out = al.r_ess_constrained(g, q_cell, rho=float(self.cfg.rho))
                r, lam, ess = out.r, out.lam, out.ess_fraction
        return al.apply_floor(r, self.cfg.floor_fraction), lam, ess

    def bias_increment(self, r_target: np.ndarray) -> np.ndarray:
        """``d/dz [ log r*(z) / beta ]`` on the profile grid, as a force.

        The engine adds this to the ABF bias force, which makes ``r*`` the
        stationary occupancy instead of something birth--death has to keep
        re-imposing.  ``log r*`` is carried to the grid by linear interpolation
        between cell centres rather than as a piecewise constant: differentiating
        a step function would put the entire increment on the cell boundaries as
        spikes, which is a discretisation artefact and not the intended force.
        """
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        log_r = np.log(np.maximum(r_target, 1e-300))
        grid_x = np.linspace(self.x0, self.x1, self.cell_of_grid.size)
        log_r_grid = np.interp(grid_x, centres, log_r)
        return np.gradient(log_r_grid, grid_x) / self.beta

    # -- one opportunity ---------------------------------------------------
    def opportunity(self, step: int, x: np.ndarray, A_grid: np.ndarray,
                    rng: np.random.Generator) -> QRDecision:
        """Update the mass, then decide whether to move any counts."""
        cell = self.cell_of(x)
        counts = np.bincount(cell, minlength=self.J).astype(int)
        occupied = counts > 0

        # 1. Fisher--Rao on the mass.  Every arm does this; it moves nothing
        #    physical and cannot reach the estimator.
        A_cell = np.array([A_grid[self.cell_of_grid == j].mean()
                           for j in range(self.J)])
        self.mass.fr_step(cm.log_target_from_free_energy(A_cell, self.beta))
        q_cell = self.mass.mass

        row = dict(step=int(step), arm=self.cfg.arm,
                   mass_ess=al.mass_ess_fraction(q_cell, counts / self.K),
                   n_occupied=int(occupied.sum()), resampled=False,
                   benefit=0.0, lam=0.0, n_replacements=0, duplicate_pairs=0.0,
                   cooling=bool(step < self._cooldown_until))

        if self.cfg.arm in MASS_ONLY:
            self.rows.append(row)
            return QRDecision(row=row)

        if self.cfg.arm in BIAS_HELD:
            # The allocation is held by the bias, so an "opportunity" here only
            # refreshes the increment.  Nothing is cloned and nothing is killed.
            r_target, _, _ = self.r_star(q_cell)
            row["r_star"] = r_target.tolist()
            row["occupancy"] = (counts / self.K).tolist()
            self.rows.append(row)
            return QRDecision(row=row, bias_increment=self.bias_increment(r_target))

        # 1b. Never resample on top of clones that have not yet rejuvenated.
        #     Without this the arm chases noise in its own Gamma_hat: r* jitters
        #     between opportunities, the occupancy test fires every time, and
        #     genealogy compounds.  Measured before this rule, A4b fired at 22
        #     of 24 opportunities and drove ancestor ESS to 8 of 256.  The
        #     cooldown is not a new parameter -- resampling before the previous
        #     clones are independent is precisely what the rejuvenation bound
        #     says costs more than it buys.
        if step < self._cooldown_until:
            self.rows.append(row)
            return QRDecision(row=row)

        # 2. Where should the replicas be?
        r_target, lam, ess_pred = self.r_star(q_cell)
        g = self.a_cell * (self.gamma_hat() if self.cfg.arm in USES_GAMMA
                           else np.ones(self.J))
        want = al.desired_counts(r_target, self.K, occupied=occupied)
        benefit = br.resample_benefit(g, counts / self.K, want / self.K)
        chi2 = br.occupancy_chi2(counts, want)
        row["r_star"] = r_target.tolist()
        row["occupancy"] = (counts / self.K).tolist()
        row.update(benefit=float(benefit), lam=float(lam),
                   ess_predicted=float(ess_pred), chi2=float(chi2))

        # 3. Two questions, in order, because they are different questions.
        #    Is the occupancy wrong beyond sampling noise?  And if so, is
        #    correcting it worth the genealogy?  The benefit statistic cannot
        #    answer the first: sum g/r is convex, so an exactly-on-target
        #    population still shows an apparent gain from being equalised.
        if chi2 < float(self.cfg.chi2_threshold):
            self.rows.append(row)
            return QRDecision(row=row)
        if benefit < float(self.cfg.benefit_threshold):
            self.rows.append(row)
            return QRDecision(row=row)

        # Move only the part sampling noise cannot explain.  Whether to
        # reallocate and how far are different questions; snapping to the
        # target answers the second one with "all the way", and most of that
        # distance is fluctuation.
        want = al.deadband_counts(counts, want, z=float(self.cfg.deadband_z))
        if int(np.abs(want - counts).sum()) == 0:
            row["deadband_absorbed"] = True
            self.rows.append(row)
            return QRDecision(row=row)
        res = br.resample_cells(cell, want, rng)
        tau = inf.tau_from_lag1(
            self.hist, obs_interval=float(self.obs_interval) * self.dt)
        tau_med = float(np.nanmedian(tau)) if np.isfinite(tau).any() else 1.0
        tau = np.where(np.isfinite(tau) & (tau > 0), tau, tau_med)

        hold = np.zeros(self.K, dtype=np.int64)
        out_cell = cell[res.src]
        for j in np.flatnonzero(want > 0):
            sel = out_cell == j
            n_clone = int(res.is_clone[sel].sum())
            if not n_clone:
                continue
            D = n_clone * 2.0            # pairs contributed within this cell
            hold[sel & res.is_clone] = br.rejuvenation_steps(
                D=D, n_children=int(want[j]), tau=float(tau[j]), dt=self.dt,
                eps_gene=float(self.cfg.eps_gene))

        self.mass.log_M = self.mass.log_M      # mass is on cells; counts moved
        self._cooldown_until = int(step) + int(hold.max())
        row.update(resampled=True, n_replacements=int(res.n_replacements),
                   duplicate_pairs=float(res.duplicate_pairs),
                   hold_max=int(hold.max()), n_clones=int(res.is_clone.sum()),
                   cooldown_until=int(self._cooldown_until))
        self.rows.append(row)
        return QRDecision(src=res.src, hold=hold, row=row)


def firing_steps(n_steps: int, qcfg: QRConfig) -> np.ndarray:
    """Opportunity steps.  Same three-phase shape clean-v2 froze."""
    burn = int(round(qcfg.burnin_fraction * n_steps))
    stop = int(round(qcfg.stop_fraction * n_steps))
    every = int(qcfg.opportunity_every)
    return np.array([s for s in range(1, int(n_steps) + 1)
                     if burn <= s < stop and (s - burn) % every == 0])
