# Does reaction-coordinate WFR survive the move to molecules?

**Short answer: the mechanism transfers, sharpens, and acquires a fix the toy
phase could not have found — but the honest claim is speed at practical budgets,
not accuracy at unlimited budget.**

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
move. The fiber conditional is essentially exact and `e_F` still sits at 1.6
estimator floors.

So: **the conditional move removes the fiber half of RC-WFR's error exactly, and
is worth a factor two -- the corrected floor is 0.021 against the naive lift's
0.044. The marginal half remains, and it sets a budget, measured at a few times
1e8 force evaluations here, beyond which plain adaptive biasing wins.**

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
transport also reaches, so it is most likely the shared constrained-integrator
plus estimator floor rather than a transport bias.  The claim is withdrawn; see
`docs/SWITCH_CAMPAIGN.md`.

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
