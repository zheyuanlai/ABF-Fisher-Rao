# Does reaction-coordinate WFR survive the move to molecules?

**Short answer: the mechanism transfers, sharpens, and acquires a fix the toy
phase could not have found. The apparent accuracy ceiling turned out to be the
measurement convention rather than the method, and once it was removed the
method proved unbiased.**

## The claim, frozen

> **Reaction-coordinate WFR is a fast population-level equilibration mechanism
> for free-energy calculation with an INCOMPLETE reaction coordinate.** Its
> practical molecular success depends on pairing reaction-coordinate transport
> with an *exact* Metropolis-corrected conditional move for the slow fiber
> modes; learned proposals are fine, uncorrected ones are worse than useless.
> In that regime it reaches the statistical-error floor substantially earlier
> than cold stratified TI (**-56.2%** [-64.4, -49.5] at 8.6e8 force evaluations)
> and than an oracle-warm-started TI ceiling (**-28.5%** [-39.0, -10.1]), while
> becoming statistically level with that ceiling at long budget. Once the
> relevant slow modes are promoted into the reaction coordinate itself, ordinary
> stratification is preferable (**+31.5%** [+5.2, +61.6] against it on the
> two-torsion control).

Three limits on that claim, stated up front rather than at the end.

* **The adaptive-biasing margins are from the old convention.** ABF, OPES and
  ABP were run at `h` = 2e-3, `b_mf` = 0.05, `n` = 129, where the shared 0.020
  numerical floor was comparable to the differences being measured: -33.1% vs
  ABF at 1e8 force evaluations, -8.1% at 4.1e8. They were not rerun at the lower
  floor. Read them as measured-at-that-floor, not as settled.
* **Everything here is one reaction coordinate on a small molecule in vacuum**,
  plus one two-dimensional control. Solvated and ionic systems were planned and
  deliberately not run.
* **A residual 0.003 kcal/mol in the floor is unexplained**, after four
  candidates were tested and three found real but insufficient (section 6c).

The toy phase (`MANIFOLD_FORMULATION.md`) concluded that RC-WFR's error lives in
the *lift*: how a configuration correctly distributed on `Sigma(z)` is carried to
`Sigma(z')`. This phase asks whether that survives on real molecules, with a
torsion as the reaction coordinate, references from ~2e11 force evaluations of
unbiased dynamics, and every baseline screened at least as hard as the new arms.

## Systems

| tag | system | coords | z | hidden slow mode | reference |
|---|---|---|---|---|---|
| BUT | united-atom butane, TraPPE | 12 | central torsion | - | unbiased MD |
| PEN | united-atom pentane, TraPPE | 15 | `phi1` | `phi2` (`tau_y` = 1.3e5 steps) | unbiased MD |
| HEX | united-atom hexane, TraPPE | 18 | `phi1` | `phi2` and `phi3` | unbiased MD |
| ALA | alanine dipeptide, AMBER ff14SB, vacuum | 66 | `phi` | `psi` (`tau_psi` = 2.3e4 steps) | stratified constrained TI |

## What the campaign found

### 1. The geometrically motivated lift is the worst one

Chapter 3's minimum-norm horizontal lift — move along `M^-1 grad xi` and SHAKE —
is not merely no better than a naive alternative, as the toy phase found. On a
molecule it is **actively worse than an internal-coordinate rotation**, by
**-53.7% [-56.6, -49.0]** on pentane. Its conditional error in the slow mode is
the *same* (0.170 vs 0.194): the extra damage is not in the slow mode at all. It
is in the fast modes the projection bends in order to buy the constraint. (On
alanine the two are separated at 1x, -38.8% for the rotation, but the CI spans
zero by 4x; pentane is the cleaner measurement.)

That also explains the toy phase's transport-rate pathology. Sweeping `kappa_W`
over a factor **64** on 16 fresh seeds:

| `kappa_W` | 0.0375 | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 | 2.4 |
|---|---|---|---|---|---|---|---|
| min-norm SHAKE | 0.098 | 0.115 | 0.159 | 0.249 | 0.414 | 0.722 | **1.231** |
| internal-coordinate rotation | 0.065 | 0.069 | 0.077 | 0.074 | 0.065 | 0.060 | 0.049 |
| + oracle conditional map | 0.028 | 0.028 | 0.025 | 0.031 | 0.027 | 0.025 | 0.029 |
| + Metropolis move (learned) | 0.027 | 0.025 | 0.022 | 0.025 | 0.024 | 0.023 | 0.029 |

The pathology belongs to the SHAKE lift specifically -- a factor **12.6** across
the sweep -- because its damage is a per-step *distortion* proportional to the
displacement it has to buy. A rotation distorts nothing and is flat. And for the
corrected arm both `e_F` and `D_cond` (0.0050-0.0055) are flat to seed noise over
the whole 64-fold range: **the transport rate stops being a hyper-parameter,
because the thing the tradeoff was made of has been removed.**

### 2. A lift learned from the run's own samples, applied uncorrected, is worse than no lift

This is the campaign's clearest negative result and it has two flavours.

* **learned CDF map**: +52.8% [+36.9, +69.8] worse than the naive lift. It can
  only rearrange, so it cannot create the spread a cold start lacks, and it
  rearranges according to a conditional inferred from the mis-transported
  ensemble.
* **learned refresh**: +506.9% [+473.8, +577.4] worse, with a conditional error
  of 1.49 nats against the naive lift's 0.19. From a cold start `nu_hat(y|z)` is
  a delta, so redrawing from it deletes the relaxation the dynamics keeps making.
  It is a fixed point, and forgetting does not escape it.

And an ORACLE refresh — the toy phase's best arm — is a **disaster on alanine**
(+180% over naive), using the exact conditional. Pentane's fiber is a
united-atom chain whose only structure is `phi2`; alanine's carries the C7eq
internal hydrogen bond and sixty coordinates correlated with `psi`. Drawing
`psi` from the marginal `nu(psi|phi)` is right about `psi` and wrong about
everything `psi` is coupled to.

### 3. The fix: make the conditional move Metropolis-correct

A rigid rotation of the distal fragment about a torsion axis is an isometry of
`R^{3A}`; it changes that torsion and nothing else; the internal-coordinate
Jacobian never depends on a torsion; and it does not move any of the four atoms
that define `xi`, so `det G` is invariant. Therefore

    accept with  min(1, exp(-beta[V(q') - V(q)]) * nu_hat(y|z') / nu_hat(y'|z'))

leaves the constrained ensemble on `Sigma(z')` **exactly** invariant, whatever
`nu_hat` is. The learned conditional sets the acceptance rate and nothing else.

`tests/test_mol.py` checks this directly: from a delta start, with a proposal
deliberately chosen to know nothing about the potential, the move drives the
ensemble to the unbiased-MD reference conditional (TV < 0.05) in 4e3 steps —
where natural relaxation needs `tau_y` = 1.3e5.

Confirmed on 32 fresh seeds at 1.07e8 matched force evaluations:

| | pentane `e_F` (kcal/mol) | alanine `e_F` (kJ/mol) |
|---|---|---|
| RC-WFR + Metropolis y-move, **learned** proposal | **0.0215** | see `MOLECULAR_RESULTS.md` |
| ...vs the same conditional used as an uncorrected refresh | 0.2917 | 15.28 |
| ...vs oracle proposal | 0.0196 (**+1.2% [-2.6, +15.3]**) | |
| ...vs naive rotation lift | 0.0475 (**-53.6%**) | |

**The learned proposal is statistically indistinguishable from the oracle.**
That is the practical result: no reference conditional is needed anywhere.

### 4. Against the baselines

| vs | pentane, 1x | pentane, 4x |
|---|---|---|
| ABF, multiple walkers | **-55.0%** | **-31.4%** |
| stratified constrained TI, cold | **-62.3%** | **-47.1%** |
| stratified constrained TI, warm (oracle initial conditional) | **-43.5%** | **-25.1%** |

### 5. The caveat, stated plainly -- and then measured

The advantage shrinks with budget and the reason is structural. The baselines are
unbiased; RC-WFR is not, because its Wasserstein step moves `z` without a
Metropolis correction and correcting it would need `F`. Late-time
`d log e_F / d log fe` at the confirmation budget is -0.42 for ABF and cold TI
(statistical) against -0.13 for the corrected RC-WFR arm. Extrapolating those
fits put the crossover with ABF at ~4e8 force evaluations.

**Rather than leave that as an extrapolation, it was run.** At ~4.3e8, 16 seeds:

| arm | at 1.07e8 | at ~4.3e8 | late slope |
|---|---|---|---|
| RC-WFR + Metropolis (learned) | 0.0215 | **0.0206** | -0.044 |
| ABF | 0.0314 | 0.0227 | -0.231 |
| stratified constrained TI, cold | 0.0378 | 0.0284 | - |
| RC-WFR, naive rotation lift | 0.0475 | 0.0437 | - |

The prediction was right to better than a factor two. RC-WFR is still ahead of
ABF at 4.1e8 (**-8.1%** [-17.5, -5.9]) but only just, and it is flat while ABF
is not.

**And the residual is provably not the fiber.** Over that same run the corrected
arm's conditional error fell from 0.0051 to **0.0014 nats** while `e_F` did not
move. The fiber conditional is essentially exact and `e_F` did not follow it
down.

**Nor is the residual a transport bias -- it is the numerics.** The 0.021 that
the corrected arm stopped at was measured directly (section 6 below, and
`MOLECULAR_RESULTS.md` section 20) and reconstructed from two terms with no
fitted parameters: the estimator's kernel smoothing at `b_mf = 0.05` contributes
0.01243, and the constrained integrator's time-step bias at `h = 2e-3`
contributes 0.01326, for 0.01818 in quadrature before any statistical error.
Lowering `h` to 1e-3 and `b_mf` to 0.02 puts butane's warm constrained TI at
**0.0056** with no detectable bias left.

So: **the conditional move removes the fiber half of RC-WFR's error exactly, and
is worth a factor two -- 0.021 against the naive lift's 0.044.** Everything below
0.021 was invisible to the convention the campaign ran at, which is why the
budget at which adaptive biasing overtakes RC-WFR was measurable but the reason
for the plateau was not.

### 6. ...and the marginal half is removable too, by switching transport off

Run RC-WFR as an initialiser rather than as the whole algorithm: at `t_switch`,
stop transporting, **snap the replicas onto a uniform window grid**, freeze the
learned proposal, and continue as ordinary stratified constrained TI, estimating
from post-switch samples only.

| | at 1.1e8 | at 4.3e8 | late-time rate |
|---|---|---|---|
| persistent RC-WFR | 0.0228 | 0.0208 | -0.044, parked |
| switched at 4e5 steps | 0.0228 | 0.0232 | **-0.283** (vs production fe) |
| switched at 1e5 steps | 0.0339 | 0.0220 | -0.318 |
| switched, frozen IN PLACE at 2.5e4 | 0.0387 | 0.0412 | +0.030, worse floor |

The late switch is identical to persistent RC-WFR up to its switch point and
then keeps converging (-0.283 against production budget) instead of parking.  It
is currently LEVEL with the persistent arm rather than below it; a crossing is
implied by the rates but was not demonstrated at the budgets run here. The **snapping
is the whole fix**: an ablation that snaps without freezing the proposal
reproduces the snapped result to 3e-12. Freezing the replicas where transport
happened to leave them is an uneven stratification, and because the same set is
reused for the entire production stage that unevenness never averages away.

**This did not survive its decisive test.**  Run to 9e8 force evaluations the
switched arm converges up to the persistent one and stops: +2.7% [-4.2, +9.8],
interval spanning zero.  Both approach ~0.020 kcal/mol, which the arm with no
transport also reaches.  The claim is withdrawn; see `docs/SWITCH_CAMPAIGN.md`.

**And the ~0.020 is now measured rather than suspected.**  On butane, with warm
stratified constrained TI and nothing else -- no transport, no exploration, no
hidden mode -- sweeping `h` at fixed physical time and accumulating three
bandwidths from each trajectory:

| | kcal/mol |
|---|---|
| estimator smoothing at `b_mf` = 0.05 | 0.01243 |
| constrained integrator at `h` = 2e-3 | 0.01326 |
| quadrature sum | **0.01818** |
| observed plateau | ~0.020 |

At `b_mf = 0.08` the measured `e_F` IS the analytic smoothing floor to three
digits and the residual is exactly zero -- the check this could have failed.
The reference's own time-step bias was measured too, by rerunning it four times
finer: 0.0018 kcal/mol, barely over its own block error, and eight times smaller
than the constrained arm's at the same step.  So the excess belongs to the
projection, not to the dynamics, and **the number every constrained arm
converged to was never any of their own.**

### 6b. Rerun at a convention where the floor is not doing the work

With the plateau explained, the comparison that mattered was worth running
again at `h` = 1e-3, `b_mf` = 0.02 and a 257-node grid, where the same
accounting puts the floor near 0.005 instead of 0.020.  Pentane, 16 seeds,
1024 windows, 3.2e6 steps, three arms:

| arm | 8.6e8 fe | 1.7e9 fe | 3.4e9 fe | late slope |
|---|---|---|---|---|
| stratified TI, warm (ceiling) | 0.0114 | 0.0093 | **0.0084** | +0.024 |
| RC-WFR + Metropolis y-move | 0.0088 | 0.0088 | **0.0087** | -0.037 |
| stratified TI, cold | 0.0210 | 0.0124 | **0.0102** | -0.302 |

**Everything moved down by a factor 2.4 and the ordering held.**  Against cold
TI, RC-WFR is **-56.2%** [-64.4, -49.5] at 8.6e8 and still **-11.6%**
[-20.9, -4.3] at 3.4e9.  Against the warm ceiling it is **-28.5%**
[-39.0, -10.1] at 8.6e8 and level at the end (+5.6% [-11.9, +24.8]).

**This is the test the withdrawn bias claim never had.**  Roughly 0.015 kcal/mol
of room was opened by lowering the floor; an RC-WFR-specific transport bias of
even half that would have separated it from the ceiling arm.  None is visible.
RC-WFR is instead flat from its first save -- 0.0088 to 0.0087 across a factor 4
in budget -- while warm TI takes the whole run to arrive at the same place and
cold TI is still descending when it stops.  The advantage is real and it lives
at the front of the budget axis, which is where a sampler that removes an
equilibration cost should live.

Two further facts from the same run.  The conditional error is 0.0003 nats for
RC-WFR against 0.0268 for both TI arms -- a factor 90, and the mechanism of
section 3 intact.  And RC-WFR's seed-to-seed scatter is **3.5x smaller** than
either TI arm's (0.0017 against 0.0058): its error is almost entirely
common-mode, so different seeds at the same budget return nearly the same
profile.

Of the residual ~0.008 that all three arms share, 0.0029 is accounted for by
three independently measured terms (kernel, the reference's 180-bin resolution,
the reference's own noise).  The rest is arm-DEPENDENT -- the arms differ from
each other by 0.004-0.006 against a row mean's 0.0014 -- so it is not a common
reference offset.  The likely cause is named in `MOLECULAR_RESULTS.md` section
21.1 and deliberately left unmeasured.

### 6c. The floor that replaced it, itemised — and one candidate refuted

The new floor is 0.0093 on pentane, and it is not a bias. Halving `h` again from
1e-3 changes nothing (self-difference 0.0027 against a 0.0042 noise floor) and
halving `b_mf` from 0.02 to 0.01 moves `e_F` by 0.00008. Itemised:

| term | kcal/mol | how it was obtained |
|---|---|---|
| statistics | 0.00772 | row scatter, de-biased |
| estimator kernel, at the measured sampling density | 0.00263 | section 23 |
| reference 180-bin resolution | 0.00196 | resample-and-return on a smooth profile |
| reference's own time step | 0.00238 | the reference rerun 4x finer |
| reference block noise | 0.00085 | its 8 independent blocks |
| **quadrature of known terms** | **0.00876** | |
| observed | 0.00928 | |
| unaccounted | 0.00306 | |

Known terms carry **89% of the squared error**, dominated by ordinary statistics.

The leading suspect for the rest was that the kernel's floor had been computed
for a *uniform* sampling density, while the Fixman weight makes even
grid-placed windows uneven and RC-WFR's windows are equidistributed by transport
instead. **It was measured and refuted.** With each arm's actual density the
extra term is 0.0017 for the TI arms and **0.0003** for RC-WFR — far too small
to be the 0.003 residual or the 0.004–0.006 by which the arms differ.

It also came back the opposite way round from the assumption: **RC-WFR's
effective sampling density is the more uniform one**, by a factor seven.
Placing windows on a perfect grid does not give a flat sampling density once the
Fixman weight is applied; transport-equidistributed windows happen to
compensate.

### 7. What the advantage actually is

With `z = (phi, psi)` -- alanine's hidden torsion promoted into the reaction
coordinate -- RC-WFR **loses** to plain stratified constrained TI, by
**+31.5%** [+5.2, +61.6]. The advantage measured everywhere else is
hidden-mode repair, not reaction-coordinate transport for its own sake.

That also relocates Fisher-Rao. On a one-dimensional torsion removing it changes
nothing (-6.5%, noise): one period of a torsion has no discovery problem for
birth-death to solve. On the two-torus with a cold start, removing it more than
doubles the error (+114.5% vs stratified TI, against +31.5% with it). The
relative roles of W and FR are problem-dependent, and this campaign now has one
molecular example of each.

### 8. Which mode to promote: a rule, tested five times

`S_k tau_k^2` -- conditional sensitivity to `z` times the square of the
relaxation time, both measurable from a run's own output before any arm is run:

| mode | `S_k tau_k^2` | measured reduction in `e_F` | significant |
|---|---|---|---|
| heptane `phi2` | 7.1e8 | **58.1%** | yes |
| hexane `phi2` | 2.6e8 | **44.3%** | yes |
| heptane `phi3` | 2.6e7 | -9.7% | no |
| heptane `phi4` | 1.7e7 | -0.2% | no |
| hexane `phi3` | 6.8e6 | -29.1% | no |

Heptane's three candidate torsions all relax at the SAME rate (9.7e4, 9.4e4,
8.5e4 steps), so only coupling varies -- and only the coupled one is worth
promoting. The rule separates the two groups by an order of magnitude.

## Where the numbers come from

* `MOLECULAR_PLAN.md` — systems, gates, arms, preregistered predictions
* `MOLECULAR_METHOD.md` — the construction and the exactness argument
* `MOLECULAR_RESULTS.md` — every measurement in the order it was taken, including
  the wrong ones and why
* `figures/figMOL*.png` — regenerated by `figures/make_mol_figures.py`
* `tests/test_mol.py` — 12 engineering tests, ~7 min on one GPU
