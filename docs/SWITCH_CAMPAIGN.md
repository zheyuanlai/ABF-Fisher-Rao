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
  and keep using what it learned while `z` was still moving.  (This one turned
  out to make no difference; see the ablation below.  It is kept as a flag
  because the reasoning that motivated it is still the right thing to check.)

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

## The switch: snapping fixes the failure, but does not buy an asymptotic win

(Read this section together with the final one.  The convergence claim made here
was tested to 9e8 force evaluations afterwards and did not hold; what survives is
the comparison between snapping and freezing in place.)

Pentane, 16 seeds, same block as the persistent / ABF / cold-TI long runs:

| arm | estimator | e_F @1.1e8 | e_F @4.3e8 | late slope |
|---|---|---|---|---|
| persistent RC-WFR | all | **0.0228** | **0.0208** | **-0.044** |
| ABF | all | 0.0322 | 0.0227 | -0.231 |
| stratified constrained TI, cold | all | 0.0363 | 0.0284 | -0.253 |
| RC-WFR, naive lift | all | 0.0448 | 0.0437 | -0.088 |
| WFR -> TI, frozen in place @2.5e4 | all | 0.0380 | 0.0411 | +0.050 |
| WFR -> TI, frozen in place @2.5e4 | post-switch | 0.0387 | 0.0412 | +0.030 |
| WFR -> TI, **snapped** @1e5 | all | 0.0282 | 0.0221 | -0.220 |
| WFR -> TI, **snapped** @1e5 | post-switch | 0.0339 | 0.0220 | **-0.318** |
| WFR -> TI, **snapped** @4e5 | all | **0.0228** | 0.0212 | -0.081 |
| WFR -> TI, **snapped** @4e5 | post-switch | **0.0228** | 0.0232 | -0.484 (vs total fe; -0.283 vs PRODUCTION fe, which is the honest axis) |

**The persistent arm is parked** -- slope -0.044, flatter than every baseline and
flat over a factor four in force evaluations -- and the switched arms are not.
But the slope has to be read against the right axis, and reading it against
TOTAL force evaluations overstates the effect.  An arm that switched earlier has
a more mature production estimate at the same total budget, so its measured slope
is naturally shallower for reasons that have nothing to do with bias.  Against
**production** force evaluations (`fe - fe_switch`), which is the axis the
post-switch estimator actually lives on:

| arm | switch at | production fe | e_F at end | slope vs production fe |
|---|---|---|---|---|
| persistent RC-WFR | - | - | 0.0207 | **-0.044** |
| switch @1e5 | 2.9e7 | 4.0e8 | 0.0220 | **-0.281** |
| switch @4e5 | 1.2e8 | 3.2e8 | 0.0228 | **-0.283** |
| automatic | 1.2e7 | 4.2e8 | 0.0232 | -0.051 |

and at a MATCHED production budget of 2.9e8 the three switched arms give 0.0219,
0.0252 and 0.0239 -- they are the same arm at different points on one curve.

So the honest statement is narrower than the raw numbers first suggested:

* persistent RC-WFR **is** parked (-0.044 over a factor four);
* the switched arms are **still converging** (-0.28), which the persistent arm is
  not;
* they are currently **level** with it (0.022 against 0.0207), not below it.

A crossing is imminent at that rate but has not been demonstrated, so a longer
run was done to measure it rather than extrapolate -- see the final section: the
crossing does NOT happen.  The `-0.484` figure
quoted earlier was measured over a window immediately after the switch, where the
estimator was still shedding its start-up transient, and it overstates the
asymptotic rate; it is withdrawn in favour of `-0.28`.

**One** implementation detail is load-bearing, and an ablation says which.  Two
fixes were applied together -- snapping the replicas onto a uniform window grid,
and freezing the learned conditional at the switch -- and running the snap
WITHOUT the freeze gives numbers that agree to 3e-12:

| @1e5 switch | e_F @1.1e8 | e_F @4.3e8 | late slope |
|---|---|---|---|
| snapped + frozen proposal | 0.0339 | 0.0220 | -0.318 |
| snapped only | 0.0339 | 0.0220 | -0.318 |

So the proposal-collapse diagnosis was wrong, or rather it was a SYMPTOM: once
the replicas are spread uniformly, every window keeps seeing a well-spread
ensemble and the learned conditional does not degenerate whether it is frozen or
not.  **The entire fix is the snapping.**  Freezing the replicas where transport
happened to leave them is an uneven stratification, and that unevenness -- not
the deposits, not the proposal -- is what turned the switch from a gain into a
loss.

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

## The adaptive baselines, and a bug in my own implementation of them

The comparison had only adaptive-biasing-FORCE.  Two adaptive-biasing-POTENTIAL
arms were added: an ABP/SHUS one (bias built from raw visit counts) and **OPES
proper** -- reweighted density `w_k = exp(beta V_{n-1}(s_k))`, the explicit
barrier floor `epsilon = exp(-beta DeltaE/(1-1/gamma))`, and normalisation by
`Z_n` over the explored region, with `gamma -> infinity` targeting the uniform
marginal every other arm targets.  Kernel compression is replaced by the same
fixed grid and Gaussian kernel the rest of the package uses; that changes the
representation of `P_hat`, not the algorithm, and it is stated rather than
hidden.

**Both arms were initially broken, and the symptom was diagnostic.**  They
reported `e_F ~ 0.20` on pentane against ABF's 0.048, with an apparent bias floor
(slope -0.056 out to 4.1e8) -- an unbiased method should not have one.  The cause:
they deposited the mean force with the `(det G)^{-1/2}` Fixman weight.  That
weight converts the RIGID measure a constrained sampler produces into `nu^xi`.
These arms sample UNCONSTRAINED, and their bias depends on `z` alone, so
conditioning on `z` already gives `nu^xi` -- the weight is pure error.  It is
exactly the wrong-weight control Gate I measured at 0.151 kcal/mol on butane, and
0.15 is precisely the floor they were sitting on.

After removing it, at 1e5 steps on pentane:

| arm | before | after |
|---|---|---|
| OPES | 0.1997 | **0.0650** |
| ABP / SHUS | 0.1915 | **0.0577** |
| ABF (unaffected -- it never had the weight) | 0.0505 | 0.0505 |

and at the confirmation budget (4e5 steps, 32 seeds) OPES reaches **0.0340**
against ABF's 0.0314 and RC-WFR's 0.0215.  So the corrected baseline set has all
three adaptive methods within 10% of each other, which is what one should expect,
and the earlier "OPES is 4x worse than ABF" statement was an artefact of my
implementation and is withdrawn.

Screened on their own knobs: OPES over kernel width `sigma` in {0.05, 0.10, 0.20}
and barrier limit in {8, 20, 40} k_B T; ABP over its gain across two decades.

## Alanine: the same picture, one step further back on the same curve

(Superseded in the same way as the pentane section: read with the crossing test
below.  The comparison against a matched-length persistent run stands; the
inference that the switch would eventually win does not.)

The first read of this was against the shorter confirmation run and was wrong.
Compared against a persistent run of the SAME length and seeds (8e5 steps,
16 seeds, seed block 90000):

| arm | estimator | e_F @1.07e8 | e_F @2.3e8 | late slope | vs persistent @2.3e8 |
|---|---|---|---|---|---|
| persistent RC-WFR | all | 0.5307 | **0.5349** | **+0.007** | - |
| switch @5e4 | post-switch | 0.6187 | 0.5719 | -0.078 | **+5.9%** [+2.3, +8.5] |
| switch @2e5 | post-switch | 0.7058 | 0.5791 | **-0.220** | **+5.7%** [+2.3, +10.7] |

Persistent RC-WFR on alanine is parked, exactly as on pentane -- slope +0.007
over the last factor four -- and the later switch is still converging at -0.220.
The switched arm is 5.7% behind at 2.3e8 because it discarded 5.5e7 force
evaluations of stage-A deposits; at -0.220 against +0.007 it repays that within
about 1.4x more budget.

So the qualitative result transfers.  What differs from pentane is how far along
it is: pentane's late switch reaches -0.283 against production budget, while
alanine's reaches -0.11.  Two reasons, and the second is a design lesson:

* alanine's persistent error is 3.4 estimator floors with a conditional error
  still at 0.15 nats, so a larger share of what remains is fiber sampling rather
  than transport bias, and switching does not touch that;
* after the switch each window holds **one** replica whose entire 60-coordinate
  fiber must be explored by time-averaging alone.  Pentane's fiber is a single
  torsion plus fast bonds and angles; alanine's is not.  The natural fix --
  several replicas per window after the switch, at the same total cost -- is a
  scheduling change this campaign did not test, and is the obvious next thing.

## Final baseline picture (corrected)

Pentane, 16 seeds, every family screened on its own knobs, after removing the
Fixman weight that never belonged in the unconstrained arms:

| arm | e_F @1.0e8 | e_F @4.1e8 | late slope |
|---|---|---|---|
| RC-WFR + learned Metropolis lift | **0.0215** | **0.0206** | -0.044, parked |
| ABF, multiple walkers | 0.0322 | 0.0227 | -0.231 |
| OPES (reweighted density, barrier floor) | 0.0341 | 0.0235 | -0.252 |
| ABP / SHUS (visit counts) | 0.0361 | 0.0258 | -0.211 |
| stratified constrained TI, cold | 0.0375 | 0.0285 | -0.253 |
| RC-WFR, naive rotation lift | 0.0450 | 0.0426 | -0.088 |

Paired change of RC-WFR against each adaptive baseline:

| vs | at 1.0e8 | at 4.1e8 |
|---|---|---|
| ABF | **-33.1%** [-38.2, -25.7] | **-8.1%** [-17.5, -5.9] |
| OPES | **-37.3%** [-39.5, -27.0] | **-10.8%** [-21.4, -0.8] |
| ABP / SHUS | **-37.2%** [-46.4, -29.2] | **-19.8%** [-24.8, -14.7] |

The three adaptive methods now land within 14% of each other, which is what one
should expect of properly implemented members of the same family, and the shape
of the RC-WFR result is unchanged by the correction: **a large advantage at
intermediate budgets that decays as the unbiased methods converge past it.**  At
1e8 it is a third better than any of them; at 4e8 it is 8-20% better and its
slope says it will not stay ahead.

Alanine is a different regime and the numbers there are dominated by the setting
rather than by the methods: the restricted `phi` arc means an unconstrained
sampler must be held inside it by walls and diffuse 160 degrees from one basin,
and at 1e8 both ABF (8.71) and OPES (8.24) are still at coverage ~0.5, against
stratified TI's 2.98 and RC-WFR's 0.53.

## The crossing was measured, and it does not happen

Both arms were run to ~9e8 force evaluations -- twice the previous largest
budget -- specifically to see whether the switched arm passes the persistent one.

| force evaluations | persistent RC-WFR | switched @4e5, post-switch |
|---|---|---|
| 1.0e8 | 0.0215 | 0.0215 (has not switched yet) |
| 2.0e8 | 0.0208 | 0.0328 (restarted estimator) |
| 4.0e8 | 0.0208 | 0.0243 |
| 6.0e8 | 0.0207 | 0.0227 |
| **8.5e8** | **0.0205** | **0.0211** |

Paired at 8.5e8: **+2.7% [-4.2, +9.8]** -- the interval spans zero.  The switched
arm catches up and stops there.  Its own rate decays too, from -0.28 measured at
the shorter budget to **-0.162** against production budget here.

**So the central claim of this campaign does not survive its own decisive test,
and is withdrawn.**  Switching transport off does not take RC-WFR below the level
persistent RC-WFR reaches; both approach ~0.020 kcal/mol and stop.

### What ~0.020 actually is

It is very unlikely to be RC-WFR's transport bias, because the arm with **no
transport at all after the switch** lands on the same number.  Three facts point
the same way:

* the switched arm's conditional error is 0.089 nats against the persistent
  arm's 0.0008 -- a hundredfold difference in fiber quality with **no**
  difference in `e_F`, so neither arm's error is fiber-limited either;
* every constrained-dynamics arm converges toward the same neighbourhood, and
  the naive-lift arm (0.0437, slope -0.088) is the only one clearly above it;
* Gate I already measured this on butane: stratified constrained TI settled at
  0.0531 against a 0.0488 estimator smoothing floor, an excess of 0.021 in
  quadrature, attributable to the projected-Euler constrained integrator, which
  **every** constrained arm here shares.

The estimator smoothing floor at `bw_mf = 0.05` is 0.0127, and 0.0205 is 1.6x
that.  The most likely reading is that the shared floor of the constrained
integrator plus the estimator sits at ~0.020, and that RC-WFR's transport bias is
somewhere BELOW it -- which would mean the bias this campaign set out to remove
was never the binding constraint at these budgets, and the earlier "persistent
RC-WFR is parked at a transport-bias floor" reading confused a shared numerical
floor for a method-specific one.

That is testable and was not tested: re-run persistent and switched at a smaller
`bw_mf` and a smaller `h`.  If the plateau moves with `bw_mf`, it is the
estimator; if with `h`, the integrator; if with neither, it really is transport.
**That experiment is the first thing to do next**, and until it is done, no claim
should be made about removing RC-WFR's asymptotic bias.

### What still stands

Nothing above touches the results that do not depend on the asymptotic story:

* the Metropolis-corrected learned conditional lift, -53.6% against the naive
  lift on pentane and -83.4% on alanine, indistinguishable from an oracle
  proposal;
* the minimum-norm lift being actively harmful, and its 12.6x degradation with
  transport rate;
* RC-WFR's -33% against ABF and -37% against OPES at 1e8 force evaluations, both
  with intervals excluding zero;
* the complete-coordinate control: with `z = (phi, psi)` RC-WFR loses to plain
  stratified TI by +31.5%, so the advantage is hidden-mode repair;
* the `S_k tau_k^2` selection rule over five mode-contrasts;
* freezing replicas in place being much worse than snapping them.

What is withdrawn is only the claim that the two-stage schedule removes an
asymptotic bias.  At these budgets it does not, and the reason may be that there
was no method-specific asymptotic bias there to remove.


## Where this leaves the project

| | before this campaign | after |
|---|---|---|
| the fiber conditional | fixed exactly by a Metropolis-corrected learned move | unchanged, and now shown irrelevant to the plateau (0.089 vs 0.0008 nats, same `e_F`) |
| the marginal `z`-transport | assumed to be the residual bias | **not established**; the ~0.020 plateau is reached by an arm with no transport, so it is probably a shared numerical floor |
| what the advantage IS | unclear | **hidden-mode repair**; with a complete CV the method loses to plain TI |
| which mode to promote | one contrast | a rule with five contrasts, separating by an order of magnitude |
| baselines | ABF, stratified TI | + OPES proper and ABP, both corrected after a bug of mine |
| stage B | untried | snapping to a uniform grid is essential; replicas per window are not |

Next, in the order they now matter:

1. **Identify the ~0.020 plateau.**  Vary `bw_mf` and vary `h` on persistent
   RC-WFR at a long budget.  If it moves with `bw_mf` it is the estimator, with
   `h` the integrator, with neither it really is transport.  Every asymptotic
   claim in this project is downstream of this one measurement, and it is cheap.
2. **A deployable fiber-equilibration diagnostic**, without which the automatic
   switch cannot be selective.  Three versions were tried here and all were
   sampling-noise dominated.
3. **Solvated alanine**, keeping `z = phi`, `y = psi` so the exact torsional
   proposal still applies while the fiber gains hundreds of solvent coordinates.
4. **NaCl** only after that; a hydration coordinate needs a genuinely new
   non-torsional conditional move, which is a separate algorithmic problem.
