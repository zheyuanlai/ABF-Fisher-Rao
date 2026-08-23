# Can RC-WFR's speed be kept without its bias?

The molecular phase left one thing unfixed.  RC-WFR's Wasserstein step moves `z`
without a Metropolis correction -- correcting it would need `F`, which is what is
being computed -- so the method carries a residual bias that the conditional move
cannot touch.  Measured on pentane: the corrected arm flattens at
`e_F = 0.0206` (slope -0.044) while ABF keeps converging (-0.231) and overtakes
somewhere past 4e8 force evaluations.  Over that same run the fiber conditional
fell to 0.0014 nats without moving `e_F`, so the residual is provably the
marginal half.

## The question

    Is that residual carried in the DEPOSITS, or in the SAMPLER?

If in the deposits, then switching transport off and re-estimating from
post-switch samples alone should recover the statistical convergence rate while
keeping the configurations RC-WFR spent its budget building.  If in the sampler,
nothing is recovered and the caveat is permanent.

## The design

    Stage A   0 < t < t_switch      RC-WFR: W transport + learned Metropolis
                                    conditional move.  Spreads replicas across z
                                    and builds correct fiber configurations.
    Stage B   t_switch < t < T      transport and Fisher-Rao OFF.  The replicas
                                    are now a stratified set of windows at
                                    RC-WFR-chosen z, and the run is ordinary
                                    constrained TI.

Every run carries **two** mean-force accumulators, so one job reports both
estimators:

| estimator | what it uses |
|---|---|
| `e_F` | every deposit, both stages |
| `e_F_prod` | only post-switch deposits |

The Metropolis conditional move stays on in stage B.  It is exact, so it cannot
introduce bias; it only accelerates fiber relaxation at fixed `z`.

`t_switch` is absolute (in steps), which means one long run reads as several
switch FRACTIONS: a run of 1.6e6 steps with `t_switch = 1e5` is a 6% switch at
full budget and a 25% switch when read at 4e5.

## Preregistered readings

| outcome | reading |
|---|---|
| `e_F_prod` keeps falling at ~-0.4 and passes the persistent arm | the residual was in the deposits; the caveat is removable |
| `e_F_prod` flattens at the same level | the residual is in the sampler; the caveat is permanent |
| `e_F_prod` falls but plateaus higher | RC-WFR's z-allocation is itself suboptimal as a stratification |

The cost accounting is unchanged: `fe` counts every gradient evaluation from
`t = 0`, so the discarded stage-A deposits are paid for in full.  Comparisons are
against the SAME 16 seeds used for the persistent, ABF, cold-TI and naive-lift
long runs.

## Also in this campaign

* **alanine `z = (phi, psi)`** -- the complete-coordinate control.  With the
  hidden torsion promoted into the reaction coordinate, any surviving advantage
  cannot be hidden-mode repair.  Needs 2-D machinery: the mean force is a vector
  field the estimator does not force to be a gradient, so `F` is its
  least-squares potential (`laplacian F = div f_hat`, solved spectrally), and the
  residual curl is reported as a free convergence diagnostic.
* **heptane** -- three candidate hidden torsions at increasing distance from `z`,
  so the `S_k tau_k^2` selection rule is tested against three contrasts rather
  than hexane's one.
* **OPES / ABP** -- the adaptive-biasing-POTENTIAL family, so the baseline set
  covers both major adaptive approaches and not only adaptive-biasing-force.

## First outcome: freezing the replicas IN PLACE fails, and says why

`t_switch = 2.5e4` (1.6% of budget), 16 seeds, read to 4.31e8 force evaluations:

| | e_F | late slope |
|---|---|---|
| persistent RC-WFR | **0.0207** | -0.044 |
| WFR -> TI, frozen in place, all deposits | 0.0411 | +0.050 |
| WFR -> TI, frozen in place, post-switch only | 0.0412 | +0.030 |

The switched arm does not recover the statistical rate -- it is DEAD FLAT at
twice the persistent arm's error, and the two estimators agree, so discarding the
stage-A deposits changes nothing.  The residual is not in the deposits.

But the run also says what went wrong, and neither cause is intrinsic:

1. **Frozen-in-place replicas are an uneven stratification.**  256 walkers left
   wherever transport dropped them have Poisson-distributed local density, and
   because the SAME set is reused for the entire production stage the unevenness
   never averages away -- it becomes a bias through the low-count ramp.
2. **The learned proposal degenerates once `z` stops moving.**  Each window then
   sees only its own handful of walkers, so `nu_hat(y|z)` collapses toward a
   delta per window and the Metropolis move becomes a no-op.  `D_cond` is 0.11
   against the persistent arm's 0.0014.

Both have direct fixes, and both are one line:

* `snap_at_switch` -- assign the replicas to a UNIFORM grid of windows by sorted
  rank at the switch.  The displacement is small when transport has already
  equidistributed them, it is a one-time move followed by the whole production
  stage of relaxation, and it turns stage B into a genuine stratified TI whose
  windows carry RC-WFR-built fibers.
* `freeze_lift_at_switch` -- stop UPDATING the learned conditional at the switch
  and keep using what it learned while `z` was still moving.

## The selection rule, tested on five mode-contrasts

Heptane gives three candidate hidden torsions at increasing distance from `z`,
and -- unlike hexane -- they all relax at essentially the SAME rate
(`tau` = 9.7e4, 9.4e4, 8.5e4 steps), so only the coupling differs.  The
diagnostic `S_k tau_k^2` was computed from the reference before any arm ran:

| promoted | `S_k` | `tau_k` | `S_k tau_k^2` | e_F | D_cond(phi2) | vs naive lift |
|---|---|---|---|---|---|---|
| none | - | - | - | 0.0446 | 0.181 | - |
| `phi2` | 0.0764 | 9.7e4 | **7.1e8** | **0.0167** | 0.0062 | **-58.1%** [-70.6, -45.9] |
| `phi3` | 0.0029 | 9.4e4 | 2.6e7 | 0.0348 | 0.188 | +9.7% [-44.6, +40.9] |
| `phi4` | 0.0024 | 8.5e4 | 1.7e7 | 0.0445 | 0.188 | +0.2% [-20.7, +57.9] |
| stratified TI, cold | - | - | - | 0.0482 | 0.165 | +35.8% [-24.6, +80.9] |

Only the strongly coupled mode is worth promoting, and the two that are not
coupled are worth nothing at all -- their confidence intervals span zero and
their conditional error in `phi2` is untouched (0.188 vs 0.181), which is the
signature of a move that is working but moving the wrong coordinate (its
Metropolis acceptance is the highest of the three).

Across hexane and heptane the rule now has **five** contrasts, and they separate
cleanly by about an order of magnitude in the predicted damage:

| mode | `S_k tau_k^2` | measured reduction in `e_F` | CI excludes zero |
|---|---|---|---|
| HEP `phi2` | 7.1e8 | **58.1%** | yes |
| HEX `phi2` | 2.6e8 | **44.3%** | yes |
| HEP `phi3` | 2.6e7 | -9.7% | no |
| HEP `phi4` | 1.7e7 | -0.2% | no |
| HEX `phi3` | 6.8e6 | -29.1% | no |

So the practical rule is usable as stated: compute `S_k` from the (z, y_k)
histogram a thermodynamic-integration run already accumulates, `tau_k` from
watching one window relax, and promote the modes whose product is an order of
magnitude above the rest.  On these systems that is exactly one mode, and
promoting a second buys nothing.

## The switch works once the replicas are snapped

Pentane, 16 seeds, same block as the persistent / ABF / cold-TI long runs:

| arm | estimator | e_F @1.07e8 | e_F @4.3e8 | late slope |
|---|---|---|---|---|
| persistent RC-WFR | all | **0.0228** | 0.0208 | **-0.044** |
| WFR -> TI, frozen in place @2.5e4 | all | 0.0380 | 0.0411 | +0.050 |
| WFR -> TI, frozen in place @2.5e4 | post-switch | 0.0387 | 0.0412 | +0.030 |
| WFR -> TI, **snapped** + frozen proposal @1e5 | all | 0.0282 | 0.0221 | **-0.220** |
| WFR -> TI, **snapped** + frozen proposal @1e5 | post-switch | 0.0339 | **0.0220** | **-0.318** |

**That is the result the campaign was after.**  The persistent arm is parked at
a bias floor (slope -0.044).  The snapped switch reaches the same level and is
still converging at -0.32, close to the statistical rate, so it does not have
that floor.  The two estimators converge onto each other, which says the
stage-A deposits are not the problem once the stratification is fixed.

The trade is explicit rather than free: at 1.07e8 the switched arm is worse
(0.0282 vs 0.0228) because it paid 2.7e7 force evaluations for transport it then
stopped using, and at 4.3e8 they are level.  Extrapolating the fitted slopes one
more factor of four puts the switched arm 28% below the persistent one and still
falling.  So:

    RC-WFR persistent   -> fastest at small budgets, parked at a floor
    RC-WFR -> TI (snap) -> slightly behind early, keeps converging, wins later

and `t_switch` is the dial between them.  What was a permanent caveat is now a
scheduling choice.

## Alanine with `z = (phi, psi)`: the advantage IS hidden-mode repair

The complete-coordinate control.  Reference is a stratified 2-CV constrained TI
run (4096 windows x 4 rows, curl fraction 0.027, between-row s.e. 0.056 kJ/mol,
`F_rms` 15.7, estimator floor 0.100).  It recovers C7eq at IUPAC
`(-77, +56)` and the beta/C5 basin at `(-150, +158)`.

| arm | e_F (kJ/mol) | I_F | curl | coverage | vs stratified TI |
|---|---|---|---|---|---|
| stratified constrained TI, cold | **0.254** | 0.385 | 0.052 | 1.000 | - |
| RC-WFR (W + Fisher-Rao) | 0.337 | 0.582 | 0.046 | 0.786 | **+31.5%** [+5.2, +61.6] |
| RC-WFR, W only (no Fisher-Rao) | 0.561 | 3.878 | 0.082 | 0.727 | **+114.5%** [+85.0, +147.6] |
| ABF, multiple walkers | 43.6 | 43.7 | 0.508 | **0.083** | did not converge |

ABF is not given a percentage because it did not do the job at this budget: it
covered 8% of the surface and its mean-force field is 51% non-gradient, so its
`F` is undefined over most of the domain and the number is not a comparison.
Filling a 91 kJ/mol two-dimensional surface with an adaptive bias from one basin
takes much longer than the stratified methods need, which is a real property of
the setting and not a defect of the implementation -- but it means the honest
2-D comparison here is RC-WFR against stratified TI.

**With the hidden torsion promoted into the reaction coordinate, RC-WFR loses to
plain stratified TI** -- by 31.5%, with the interval excluding zero.  That is the
cleanest possible statement of what the method is for:

    the advantage comes from repairing an INCOMPLETE reaction coordinate.
    Once the coordinate is complete, reaction-coordinate transport is a pure
    cost and stratification wins.

Two supporting details.

* **Fisher-Rao earns its place here.**  Removing it more than doubles the error
  (+114.5% vs TI, against +31.5% with it), where on a one-dimensional torsion
  removing it changed nothing (-6.5%, noise).  Discovery on a single period of a
  torsion is free; discovery on a two-torus with a cold start is not, and that
  is exactly the regime where a birth-death term has something to amplify.  The
  honest statement is that the relative roles of W and FR are problem-dependent,
  and this campaign now has one molecular example of each.
* **Part of the gap is coverage, not bias.**  RC-WFR reaches 0.786 of the coarse
  cells against stratified TI's 1.000 by construction; diffusive transport
  equidistributes a two-torus much less efficiently than a grid does.
