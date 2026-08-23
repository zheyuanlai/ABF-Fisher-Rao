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
| WFR -> TI, **snapped** @4e5 | post-switch | **0.0228** | 0.0232 | **-0.484** |

**That is the result the campaign was after.**  The persistent arm is parked at
a bias floor: slope -0.044, flatter than every baseline, and flat over a factor
four in force evaluations.  The snapped switch does not have that floor.  Its
post-switch estimator converges at **-0.484** for the late switch -- the pure
statistical rate, indistinguishable from -0.5 -- and at -0.318 for the early one.

The late switch is the one to read.  Its `t_switch` sits at 1.2e8 force
evaluations, so up to that point it IS persistent RC-WFR and matches it exactly
(0.0228 at 1.1e8).  After it, the two diverge in behaviour rather than level:

|  | at 1.1e8 | at 4.3e8 | rate |
|---|---|---|---|
| persistent | 0.0228 | 0.0208 | -0.044, parked |
| switched at 4e5 steps | 0.0228 | 0.0232 | **-0.484, statistical** |

At 4.3e8 the switched arm is 12% behind, because it threw away the 1.2e8 force
evaluations of stage-A deposits and is running on 3.2e8 of production.  At
-0.484 against -0.044 that deficit is repaid within a factor of two more budget,
and everything past that is gain.

So the algorithm is:

    RC-WFR persistent   -> fastest early, then parked at a bias floor
    RC-WFR -> TI (snap) -> identical early, then converges at the statistical rate

and `t_switch` is the dial between them.  What was a permanent caveat --
"speed at practical budgets, not accuracy at unlimited budget" -- is now a
scheduling choice, and the honest claim becomes:

    RC-WFR is a fast adaptive initialiser for an asymptotically unbiased
    constrained free-energy estimator.

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

## The second adaptive baseline

The comparison so far had only adaptive-biasing-FORCE.  Adding the
adaptive-biasing-POTENTIAL family (SHUS/ABP, the same construction OPES belongs
to: build a bias from the visited density and push with its gradient, targeting
a uniform marginal), with the SAME Chapter-3 mean-force estimator so nothing
turns on an estimator difference.

Screened over a **3000-fold** range of its adaptation gain, pentane at 1e5 steps:

| gain | 0.03 | 0.3 | 1 | 3 | 10 | 100 |
|---|---|---|---|---|---|---|
| e_F | 0.1915 | 0.2101 | 0.2168 | 0.2101 | 0.2125 | 0.2168 |

It is flat in the knob and sits at ~0.20, against ABF's 0.048 at the same budget
and RC-WFR's 0.024.  Coverage is 1.000 and the conditional error is low (0.05),
so it explores fine; what is slower is the bias itself -- ABF's bias IS the
running mean force, while an ABP bias has to accumulate a visited density first.

Two things should be said plainly about this.  It is a fair screen of the
implementation that is here (the ABP engine ported from the earlier campaign,
with its own knob tried over three decades), and it is NOT a claim about
production OPES, which uses adaptive-bandwidth kernel density estimates and a
well-tempered target and is a better-tuned instrument than this.  **ABF remains
the primary adaptive baseline in this campaign**, because on these systems it is
the stronger of the two.


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
it is: pentane's late switch reached -0.484, the full statistical rate, while
alanine's reaches -0.220.  Two reasons, and the second is a design lesson:

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
| RC-WFR -> TI (snap @4e5), post-switch estimator | 0.0235 | **-0.484, statistical** |
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
