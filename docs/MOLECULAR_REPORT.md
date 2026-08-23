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
**-53.7% [-56.6, -49.0]** on pentane and +26% on alanine. Its conditional error
in the slow mode is the *same* (0.170 vs 0.194): the extra damage is not in the
slow mode at all. It is in the fast modes the projection bends in order to buy
the constraint.

That also explains the toy phase's transport-rate pathology. Sweeping `kappa_W`
over a factor 16:

| `kappa_W` | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 |
|---|---|---|---|---|---|
| min-norm SHAKE | 0.118 | 0.167 | 0.250 | 0.423 | **0.725** |
| internal-coordinate rotation | 0.070 | 0.075 | 0.060 | 0.074 | 0.047 |
| + oracle conditional map | 0.026 | 0.028 | 0.037 | 0.030 | 0.032 |

The pathology belongs to the SHAKE lift specifically, because its damage is a
per-step *distortion* proportional to the displacement it has to buy. A rotation
distorts nothing and is flat in `kappa_W`. With the conditional transported, the
conditional error *improves* with faster transport (0.050 -> 0.017), because
faster transport visits more fibers and the map re-equilibrates at each one.

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

### 5. The caveat, stated plainly

The advantage shrinks with budget and the reason is structural. The baselines are
unbiased; RC-WFR is not, because its Wasserstein step moves `z` without a
Metropolis correction and correcting it would need `F`. Late-time
`d log e_F / d log fe` is -0.42 for ABF and cold TI (statistical) against -0.13
for the corrected RC-WFR arm (partly bias-limited). Extrapolating the fits one
decade puts the crossover at ~3.4x this campaign's largest budget for ABF and
~7.6x for cold TI.

So: **the conditional move removes the fiber half of RC-WFR's error exactly, and
is worth a factor two. The marginal half remains, and it sets a budget beyond
which plain stratification wins.**

## Where the numbers come from

* `MOLECULAR_PLAN.md` — systems, gates, arms, preregistered predictions
* `MOLECULAR_METHOD.md` — the construction and the exactness argument
* `MOLECULAR_RESULTS.md` — every measurement in the order it was taken, including
  the wrong ones and why
* `figures/figMOL*.png` — regenerated by `figures/make_mol_figures.py`
* `tests/test_mol.py` — 12 engineering tests, ~7 min on one GPU
