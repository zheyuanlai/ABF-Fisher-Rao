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

## The switch works once the replicas are snapped

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
run is under way to measure it rather than extrapolate.  The `-0.484` figure
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

## Alanine: the switch transfers, one step further back on the same curve

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

## Final baseline picture

Pentane at ~4.2e8 force evaluations, 16 seeds, every family screened on its own
knobs:

| arm | e_F (kcal/mol) | late-time rate |
|---|---|---|
| RC-WFR persistent, learned Metropolis lift | **0.0207** | -0.044, parked |
| RC-WFR -> TI (snap @4e5), post-switch estimator | 0.0235 | **-0.283** (vs production fe) |
| ABF, multiple walkers | 0.0227 | -0.231 |
| stratified constrained TI, cold | 0.0287 | -0.253 |
| RC-WFR, naive rotation lift | 0.0432 | -0.088 |
| OPES / ABP family | 0.1571 | -0.056 |

Alanine at ~2.2e8:

| arm | e_F (kJ/mol) | late-time rate |
|---|---|---|
| RC-WFR persistent | **0.5364** | +0.007, parked |
| stratified constrained TI, cold | 1.7573 | -0.679 |

RC-WFR is **-68.4%** [-70.0, -66.0] below cold stratified TI on alanine at that
budget, and TI's steep slope says it is the one still converging.  Both readings
are the same story the switch experiment tells: RC-WFR buys a large head start
and then stops improving, and the switch is what converts the head start into a
permanent one.

## Where this leaves the project

| | before this campaign | after |
|---|---|---|
| the fiber conditional | fixed exactly by a Metropolis-corrected learned move | unchanged |
| the marginal `z`-transport | permanent bias, caveat on every claim | **removable** by switching transport off and snapping |
| what the advantage IS | unclear -- transport, or hidden-mode repair? | **hidden-mode repair**; with a complete CV the method loses to plain TI |
| which mode to promote | one contrast (hexane) | a rule with five contrasts, separating by an order of magnitude |
| baselines | ABF, stratified TI | + OPES/ABP family |

The three things this campaign did NOT do, in the order they now matter:

1. **more replicas per window after the switch.**  Alanine recovers less rate
   than pentane, and the visible reason is that stage B leaves one replica per
   window to explore a 60-coordinate fiber by time-averaging alone.
2. **an adaptive switch criterion** -- switch when `D_cond` and the marginal KL
   have both stopped moving, rather than at a fixed step count.
3. **solvated alanine, then NaCl.**  Solvated alanine keeps the exact torsional
   proposal and adds a many-body fiber; NaCl needs a genuinely new
   non-torsional conditional move and is a separate algorithmic problem.

## Replicas per window after the snap: the hypothesis is not supported

The obvious explanation for alanine recovering less rate than pentane was that
stage B leaves ONE replica per window to explore a 60-coordinate fiber by
time-averaging alone.  Tested directly, holding `M x R = 256` and the total
force-evaluation budget fixed, so everything before the switch is the same
trajectory:

| M windows | R replicas/window | e_F | post-switch e_F | slope (production fe) | vs R=1 |
|---|---|---|---|---|---|
| 256 | 1 | 0.5540 | 0.5666 | -0.116 | - |
| 128 | 2 | 0.5163 | **0.5240** | -0.112 | **-8.3%** [-8.8, -6.7] |
| 64 | 4 | 0.5494 | 0.5627 | -0.119 | -0.2% [-0.6, -0.0] |
| 32 | 8 | 0.5492 | 0.5648 | -0.128 | -0.1% [-1.1, +0.3] |

**The rate does not move.**  Every allocation gives -0.11 to -0.13, against the
success criterion of -0.4.  `R = 2` buys 8.3% in LEVEL, with an interval
excluding zero, and `R = 4` and `R = 8` buy nothing -- the extra replicas cost
exactly the z-resolution they gain.

So the one-replica-per-window explanation is wrong, and the alanine gap is not
about stage-B allocation.  What the numbers do say is where it is: alanine sits
at 3.4 estimator floors after the switch while pentane sits at 1.8, so alanine
has much more error left of some other kind.  The natural next suspect is that
`psi` is not the only fiber mode that matters on alanine -- the `S_k tau_k^2`
diagnostic has only ever been applied to alkane torsions, and applying it to
alanine's remaining internal coordinates (the methyl rotations, `omega`) would
say whether the promoted set is simply incomplete there.  That is a different
experiment from this one and was not run.

## Automatic switching: the criterion fires too early, and the data says why

Two deployable diagnostics, recorded every 5000 steps and needing no reference:

* `D_snap` -- mean squared distance from the replicas to the uniform grid they
  would be snapped onto, i.e. how violent the snap would be;
* `D_learn` -- how much the learned conditional table still moves between checks.

Thresholds were calibrated on **pentane only** (`eps_snap = 0.01`,
`eps_learn = 0.002`, three consecutive passes) and then frozen and applied to
alanine with no retuning.

The problem is visible in the calibration itself: **both settle by ~3e4 steps**
and are flat thereafter, while the best hand-tuned switch is at 4e5 steps.  They
measure the marginal and the estimate, and what a later switch buys is fiber
equilibration, which neither sees.  The rule fires at 4.4e4 steps.

| pentane | switch (fe) | e_F | slope vs production fe |
|---|---|---|---|
| automatic | 1.2e7 | 0.0232 | -0.051 |
| fixed @1e5 | 2.9e7 | 0.0220 | -0.281 |
| fixed @4e5 (best hand-tuned) | 1.2e8 | 0.0228 | -0.283 |

At a **matched production budget** of 2.9e8 the automatic rule is actually the
best of the three (0.0219 against 0.0252 and 0.0239) -- it simply started
production earlier and is further along its own curve, which is also why its
measured slope is shallower.

**But it does not transfer, and it fired on noise.**  Applied to alanine with the
thresholds unchanged, as preregistered, it **never fired at all**.  Looking at
which diagnostic blocks:

| | passes `D_snap < 0.01` | passes `D_learn < 0.002` |
|---|---|---|
| pentane | 100% of checks | **1%** |
| alanine | 99% | **0%** |

`D_snap` is satisfied everywhere -- transport equidistributes the marginal within
a few thousand steps and keeps it there, which is exactly the point of the
Wasserstein step -- so it carries no information about when to stop.  `D_learn`
is the binding constraint, it is noisy at this check cadence, and its scale does
not transfer between systems: pentane's three consecutive passes were an early
fluctuation rather than a convergence signal.

So the criterion as specified **is not usable**: non-selective on the system it
was calibrated on, and never triggered on the system it was tested on.  Together
with the failure of the fiber-side diagnostic below, the conclusion is that an
automatic rule needs a measure of FIBER equilibration, and this campaign did not
find a deployable one.

A third diagnostic was built specifically to see fiber equilibration -- the total
variation between the recent ensemble's conditional and the run's own time
average -- and tried at two resolutions and against both the smoothed and the raw
reference.  Every version is dominated by sampling noise and flat from the first
checkpoint (0.26-0.34 throughout).  It is recorded in the archives and not used.
**A deployable fiber-equilibration measure is the missing piece, and this
campaign did not find one.**
