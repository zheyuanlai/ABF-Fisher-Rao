# Molecular RC-WFR: results log

Every measurement in the order it was taken, including the ones that were wrong
and why.  Systems, engine and gates are defined in
[`MOLECULAR_PLAN.md`](MOLECULAR_PLAN.md).

## 0. Engine

United-atom TraPPE alkanes, flexible bonds and angles, Brownian dynamics with
mobility `M^-1` and step `h = dt/gamma = 0.002` (kcal/mol, Angstrom, amu, T = 300 K).
`xi` is a torsion; `Sigma(z)` is enforced by SHAKE along `M^-1 grad xi`; the
mean force is the Chapter-3 / den Otter-Briels estimator

    f = (grad xi^T M^-1 grad V)/G  -  beta^-1 div(M^-1 grad xi / G),
    G = grad xi^T M^-1 grad xi,

and constrained dynamics on the BARE potential samples the rigid measure, so
every deposit carries the Fixman weight `(det G)^-1/2` and the accumulator's
self-normalisation turns it into `E_rgd[w f]/E_rgd[w] = F'(z)`.

Two implementation choices that turned out to matter more than expected:

* **`torch.func`, not `autograd.grad`.** The latter forces a dynamo graph break,
  and the compiled inner loop then costs 4 ms of kernel launches per step
  instead of 0.3 ms, for any batch size.  These loops are launch-bound; the
  batch axis is close to free and every seed and hyper-parameter configuration
  goes into it.
* **The lift must be an internal-coordinate rotation, not a SHAKE projection.**
  Rotating the distal fragment about the torsion axis changes that dihedral
  exactly and preserves every bond length, bond angle and other torsion.  A
  SHAKE projection buys the same constraint by bending bonds; for a small step
  that is a perturbation, for a large one it is a catastrophe (the first
  refresh-lift arm returned `e_F ~ 7e10` because a pi-sized y-jump projected
  along `M^-1 grad xi` left the molecule hundreds of kcal/mol uphill).

## 1. Reference free energies

Unbiased Brownian dynamics, 131072 walkers x 2e6 steps = 2.2e11 force
evaluations per system, statistics in 8 independent blocks.
`F_ref(z) = -beta^-1 log p(z)` uses none of the constrained machinery.

| system | F span | min bin count | block s.d. of F_ref |
|---|---|---|---|
| butane, `xi = phi` | 4.548 kcal/mol | 1.4e5 | 0.0010 kcal/mol |
| pentane, `xi = phi1` | 4.720 kcal/mol | 1.4e5 | - |

Butane reproduces the expected TraPPE profile: trans at 0, gauche at +-115 deg
0.84 kcal/mol up, the trans-gauche barrier 3.28 and the cis barrier 4.55; the
profile is symmetric under `phi -> -phi` to 4 decimals.

**The first force field was wrong and the reference caught it.**  Including the
1-4 LJ pair alongside the TraPPE torsion series double counts the same sterics
and put the cis barrier at 10.6 kcal/mol instead of 4.55.  TraPPE-UA computes
intramolecular LJ only for sites four or more bonds apart; butane then has no
intramolecular pair at all and pentane exactly one, the 1-5 CH3...CH3 contact.
Both references were rerun.

## 2. Gate I -- does the Chapter-3 engine reproduce unbiased MD?

**A. The mean-force formula, with the sampler taken out of the question.**
`F` from the unbiased histogram vs `F` from thermodynamic integration of the
mean force deposited along the SAME unbiased trajectories:

| system | RMS \|F_TI - F_hist\| | with the WRONG (rigid) weight | span |
|---|---|---|---|
| butane | **0.0131** | 0.1511 | 4.55 |
| pentane | **0.0121** | - | 4.72 |

The wrong-weight control is the point: applying `(det G)^-1/2` to samples that
are already drawn from `nu^xi` (unbiased MD conditioned on z) is exactly the
error a constrained sampler makes in reverse, and it is 11x larger than the
residual.  The geometry is doing real work and doing it in the right direction.

**B. The constrained sampler.**  Stratified constrained TI, 1024 windows,
7.3e7 force evaluations, cold-started:

| quantity | value | requirement |
|---|---|---|
| `e_F` vs the unbiased reference | 0.0531 kcal/mol | <= 2 estimator floors |
| estimator smoothing floor at the same `bw_mf = 0.10` | 0.0488 | - |
| Fixman reweighting ESS | 0.980 | > 0.9 |
| SHAKE residual | 2.7e-15 rad | < 1e-9 |

`e_F` plateaus (log-log slope -0.07, not -0.5), so what is left is bias, and
essentially all of it is the estimator's own kernel smoothing, which every arm
shares.  **Gate I passes.**

The estimator floor is the RMS error an exact `F'` incurs by passing through the
same kernel-smoothed binned accumulator:

| `bw_mf` | 0.04 | 0.05 | 0.06 | 0.08 | 0.10 |
|---|---|---|---|---|---|
| butane | 0.0140 | 0.0164 | 0.0203 | 0.0322 | 0.0488 |
| pentane | 0.0107 | 0.0127 | 0.0165 | 0.0287 | 0.0455 |

The campaign uses `bw_mf = 0.05`, floor 0.0127 kcal/mol against `F_rms = 1.262`.

## 3. The hidden slow mode is real, and slow

Pentane's conditional `p(phi2 | phi1)` from the reference:

| `phi1` | P(phi2 trans) | P(g+) | P(g-) |
|---|---|---|---|
| 0 (trans) | 0.636 | 0.182 | 0.181 |
| +115 (g+) | 0.817 | 0.177 | **0.006** |
| -115 (g-) | 0.816 | **0.006** | 0.178 |

The pentane effect suppresses the g-g+ combination 28-fold, so the conditional
changes drastically across the fiber -- exactly the structure a naive lift
mishandles.

Fiber relaxation at fixed `phi1 = 0`, starting every replica in the trans basin
of `phi2`: P(gauche) reaches 0.0429 at 1e4 steps, 0.2552 at 1e5, 0.3562 at 3e5
against an equilibrium 0.3635.

    tau_y  ~  1.3e5 steps.

Every budget in this campaign is quoted against that number.  Below `tau_y` the
fiber is frozen and the lift is the only thing that can place the conditional;
far above it, no lift is needed.

## 4. The lift, on a molecule

Two lift families are available once `xi` is a torsion.

* **minimum-norm horizontal** (`shake`) -- move along `M^-1 grad xi` and SHAKE.
  This is the Chapter-3 geometric answer.
* **internal-coordinate rotation** (`rot`) -- rotate the distal fragment about
  the torsion's central bond.  Changes that dihedral exactly and preserves every
  bond length, bond angle and other dihedral, because the two planes defining
  any other dihedral are both carried by the same rotation.

Neither knows anything about `nu^xi(. | z)`.  On top of `rot` the fiber torsion
can additionally be transported:

| lift | what it does to the slow mode y |
|---|---|
| `ymap` | `y' = F^-1_{z'}(F_z(y))`, the exact 1-D continuity-equation solution |
| `yref` | `y' ~ nu(. | z')`, an independent draw: transports AND spreads |
| `ymh` | propose `y' ~ nu_hat(. | z')`, accept by Metropolis |

The Metropolis variant is the one that matters practically.  A rigid rotation
about the torsion axis is an isometry of `R^{3A}`; the internal-coordinate
Jacobian depends on bonds and angles but never on a torsion; and the rotation
does not move any of the four atoms that define `xi`, so `det G` is invariant.
Therefore

    accept with  min(1, exp(-beta [V(q') - V(q)]) nu_hat(y|z') / nu_hat(y'|z'))

leaves the constrained ensemble on `Sigma(z')` exactly invariant, whatever
`nu_hat` is.  The learned conditional then sets the acceptance rate and nothing
else: a degenerate `nu_hat` makes the move a no-op instead of a catastrophe.
`nu_hat` carries a 2% uniform background so the proposal is bounded away from
zero and the same density serves the draw and the acceptance ratio.

Cost: two energy evaluations per replica per lift event, charged to `fe`
(10% at `n_cond = 20`).

## 5. Pilot, pentane, 40k steps (0.3 tau_y), 8 seeds

| arm | e_F | D_cond |
|---|---|---|
| RC-WFR, min-norm SHAKE lift | 0.2622 | 0.198 |
| RC-WFR, rotation lift | 0.1009 | 0.199 |
| RC-WFR + oracle y-map | 0.0493 | 0.029 |
| RC-WFR + oracle y-refresh | 0.0424 | 0.010 |
| stratified constrained TI (cold) | 0.0876 | 0.232 |

Three things were already visible in the pilot and survived into the screen.

1. **The geometrically motivated lift is the worst one.**  Minimum-norm
   horizontal transport is 2.6x worse than a plain rotation, at IDENTICAL
   conditional error in `phi2` (0.198 vs 0.199).  Its extra damage is not in the
   slow mode at all -- it is in the fast modes it bends to buy the constraint.
   The toy campaign found min-norm merely useless (-4.5%); on a molecule it is
   actively harmful.
2. **Transporting the slow mode is worth about half the error**, and 85-95% of
   the conditional error.
3. **A CDF map cannot fix a cold start.**  It is a rearrangement: a delta stays a
   delta.  The refresh spreads as well as transports, and beats the map on
   `D_cond` by 3x.

Two implementation facts, both found the expensive way:

* the mean force must be deposited at the END of a relaxation window, not the
  start.  Depositing immediately after a lift reads the strain the lift
  introduced, and inflates every arm's error;
* a learned REFRESH is self-reinforcing.  From a cold start the ensemble is a
  delta in y, so `nu_hat` is a delta, so the refresh redraws the delta, and it
  actively destroys the relaxation the dynamics would otherwise have made:
  `D_cond` went to 1.22 against the naive lift's 0.21.  This is what motivated
  the Metropolis correction.

Metropolis-corrected arms, 40k steps, N = 128, `bw_mf = 0.05`:

| arm | e_F | D_cond | acceptance |
|---|---|---|---|
| `ymh` (oracle proposal) | 0.0588 | 0.0096 | 0.96 |
| `lmh` (learned proposal) | 0.0604 | 0.0108 | 0.67 |

The learned proposal matches the oracle.  `D_cond` for the learned arm falls
from 0.031 to 0.010 inside the first 20k steps -- about a tenth of `tau_y`.

## 6. Screen, pentane, 1e5 steps (0.77 tau_y), 8 seeds, 15 (kappa, theta) configurations

Each arm at its own best configuration, selected on median `I_F`:

| arm | best kappa | I_F | e_F final | D_cond | worst config's I_F |
|---|---|---|---|---|---|
| RC-WFR, min-norm SHAKE | 0.075 | 0.1219 | 0.1014 | 0.192 | 0.7325 |
| RC-WFR, rotation | 1.2 | 0.0579 | 0.0466 | 0.211 | 0.0886 |
| RC-WFR + oracle y-map | 0.15 | 0.0346 | 0.0247 | 0.041 | 0.0466 |
| RC-WFR + oracle y-refresh | 0.075 | 0.0322 | 0.0273 | 0.0093 | 0.0448 |
| RC-WFR + learned y-map (2e5) | 0.3 | 0.0905 | 0.0840 | 0.234 | 0.1119 |

Estimator floor 0.0127; `F_rms` 1.262.

### The transport-rate stress test came out of the screen for free

`e_F` against `kappa_W` at `theta = 0.3`:

| kappa | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 |
|---|---|---|---|---|---|
| min-norm SHAKE | 0.118 | 0.167 | 0.250 | 0.423 | **0.725** |
| rotation | 0.070 | 0.075 | 0.060 | 0.074 | 0.047 |
| + oracle y-map | 0.026 | 0.028 | 0.037 | 0.030 | 0.032 |
| + oracle y-refresh | 0.027 | 0.033 | 0.027 | 0.027 | 0.031 |

and the conditional error `D_cond` for the same sweep:

| kappa | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 |
|---|---|---|---|---|---|
| rotation | 0.195 | 0.208 | 0.210 | 0.208 | 0.211 |
| + oracle y-map | 0.050 | 0.038 | 0.029 | 0.022 | **0.017** |
| + oracle y-refresh | 0.009 | 0.010 | 0.010 | 0.009 | 0.010 |

The toy campaign's P4 said: with a wrong lift, faster reaction-coordinate
transport makes the free energy worse; with a conditionally correct one it does
not.  On a molecule that is true, and the mechanism is now visible:

* the pathology belongs to the **minimum-norm SHAKE lift specifically**.  Its
  damage is a per-step DISTORTION, proportional to the displacement it has to
  buy, so it scales with `kappa` -- a factor 6.2 over the range swept, and its
  worst configuration is 57x the estimator floor;
* the internal-coordinate rotation has no such pathology -- it distorts nothing
  -- and its `e_F` is flat in `kappa`.  Its damage is the conditional LAG, which
  saturates once transport is faster than fiber relaxation, and `D_cond` is
  accordingly flat at 0.21;
* with the conditional transported, `D_cond` *improves* monotonically with
  `kappa` (0.050 -> 0.017 for the map).  Faster transport visits more fibers per
  unit time and the map re-equilibrates `y` at each one.

So the toy's tradeoff was two different effects superposed.  Separated on a
molecule: **distortion scales with transport rate; conditional lag does not; and
transporting the conditional turns transport rate from a liability into an
asset.**

### Baselines at the same budget

| arm | best knob | I_F | e_F final | D_cond |
|---|---|---|---|---|
| stratified constrained TI, cold | 128 windows | 0.0796 | 0.0613 | 0.200 |
| stratified constrained TI, warm (oracle init) | 64 windows | 0.0474 | 0.0343 | 0.176 |
| ABF, multiple walkers | ramp 200 | 0.0668 | 0.0476 | 0.049 |

ABF has the LOW conditional error of the three baselines and still the worse
free energy: its dynamics is unconstrained, so `phi2` relaxes by itself while
the bias flattens `phi1`, but that costs it the stratification that makes the
mean-force estimator efficient.  RC-WFR with a transported conditional
(0.025-0.027) beats all three at matched force evaluations, including the
oracle-initialised warm TI (0.034).

Screening summary at 1e5 steps, 8 seeds, each family at its own best knob:

| arm | I_F | e_F | D_cond | vs floor |
|---|---|---|---|---|
| RC-WFR min-norm SHAKE | 0.1219 | 0.1014 | 0.192 | 8.0x |
| RC-WFR rotation (naive) | 0.0579 | 0.0466 | 0.211 | 3.7x |
| stratified TI cold | 0.0796 | 0.0613 | 0.200 | 4.8x |
| ABF | 0.0668 | 0.0476 | 0.049 | 3.7x |
| stratified TI warm (oracle) | 0.0474 | 0.0343 | 0.176 | 2.7x |
| RC-WFR + oracle y-map | 0.0346 | 0.0247 | 0.041 | 1.9x |
| RC-WFR + oracle y-refresh | 0.0322 | 0.0273 | 0.0093 | 2.1x |
| RC-WFR + learned y-map (2e5) | 0.0905 | 0.0840 | 0.234 | 6.6x |
| RC-WFR + learned y-refresh (2e5) | 0.0881 | 0.1284 | **1.175** | 10x |

The two learned uncorrected lifts are the campaign's clearest negative result.
The refresh does not merely fail to help: its conditional error is 5.6x the
naive lift's, because from a cold start `nu_hat(y|z)` is a delta and redrawing
from it every 20 steps deletes the relaxation the dynamics keeps making.  The
map is safer -- it can only rearrange -- but it still ends up worse than doing
nothing, because it rearranges according to a conditional inferred from the
mis-transported ensemble.

## 7. Hexane: which fiber mode has to be promoted?

Pentane has one hidden torsion, so "promote the slow mode" has only one reading.
Hexane has two, and they differ in the right way:

| | relaxation timescale | coupling to `z = phi1` |
|---|---|---|
| `phi2` (adjacent) | same torsional barrier | P(g-) falls 0.159 -> 0.005 as `phi1: 0 -> +115` |
| `phi3` (one bond further) | same torsional barrier | P(g+) moves 0.155 -> 0.187 only |

Both are equally SLOW; only `phi2` is strongly COUPLED (through the 1-5 contact
that also makes pentane work).  The toy campaign's stress test varied timescale
at fixed coupling; hexane varies coupling at fixed timescale, which is the case
a practitioner actually faces when choosing what to promote.

Reference quality: 2.2e11 force evaluations, block populations stable to 5e-4
and g+/g- symmetric to 1e-3, `RMS|F_TI - F_hist| = 0.0132` on a span of 4.66.

## 8. The Metropolis-corrected lift, screened (pentane, 1e5 steps, 8 seeds)

| arm | I_F | e_F | D_cond | vs floor | needs an oracle? |
|---|---|---|---|---|---|
| RC-WFR + Metropolis y-move, **oracle** proposal | 0.0312 | 0.0230 | 0.0103 | 1.8x | yes |
| RC-WFR + Metropolis y-move, **learned** proposal, bw_z 0.15 | 0.0335 | 0.0250 | 0.0108 | 2.0x | **no** |
| RC-WFR + Metropolis y-move, **learned** proposal, bw_z 0.25 | 0.0336 | 0.0247 | 0.0106 | 1.9x | **no** |

Against the same-budget alternatives: 0.0466 (naive rotation lift), 0.0613
(cold stratified constrained TI), 0.0476 (ABF), 0.0343 (warm stratified TI with
an oracle initial conditional).

Three things are worth separating out.

1. **The learned proposal costs 8% relative to the oracle one, not a factor.**
   That is the Metropolis correction doing exactly what it is supposed to: the
   proposal determines the acceptance rate (0.67 learned vs 0.96 oracle) and
   nothing else.  Compare the same learned conditional used WITHOUT correction:
   as a refresh it produced `D_cond = 1.18`, 110x worse.
2. **It is insensitive to the density-estimation knobs** -- 0.0250 vs 0.0247
   across a factor 1.7 in bandwidth -- which is the practical difference between
   a method and a tuning exercise.
3. **It beats the oracle-initialised warm TI baseline** (0.0247 vs 0.0343) while
   using no oracle at all.

## 9. Alanine dipeptide, and why its reference cannot be unbiased MD

Ace-Ala-Nme, AMBER ff14SB, vacuum, 22 atoms.  The torch inner loop reproduces
OpenMM to 1.19e-9 relative on energy and 1.67e-10 on forces over 24 thermally
displaced configurations (`scripts/mol_ala_gate.py`).  Masses are set uniform at
12 amu: masses do not enter `e^{-beta V}`, so `F(z)` is exactly unchanged and
the mean force becomes mass-free, while the X-H bonds stop forcing `h ~ 4e-7`.
With that choice alanine's torsional diffusion per step is 1.88e-4 rad^2 against
pentane's 1.69e-4 -- the two systems share a budget scale.

**The first reference attempt failed, and the failure is instructive.**  Unbiased
Brownian dynamics from a uniform start reported `P(C7ax) = 0.3548` in ALL EIGHT
independent blocks, identical to four decimals: that basin sits behind a ~14 k_B T
barrier in `phi`, nothing entered or left it in 7e5 steps, and the "measurement"
was of the initial condition.  Re-seeding cannot fix that -- it only freezes the
same error.  A second, milder failure sat on top: the C7eq/C5 balance was still
drifting monotonically across all eight blocks.

The reference is therefore a **stratified constrained-TI** one (128 windows,
1024 replicas, fiber torsions started uniformly, 2e6 steps, 8 independent
replicate rows).  Each window is pinned at its own `phi`, so the barrier in
`phi` never has to be crossed; only the fiber has to relax, and it starts spread
rather than at a point.  This is the engine Gate I validated against unbiased MD
on butane and pentane, so it is a checked instrument.

| | value |
|---|---|
| between-row s.e. of `F_ref` | 0.087 kJ/mol |
| drift over the last save interval | 0.021 kJ/mol |
| Fixman reweighting ESS | 0.965 |
| `F` span over the full circle | 63.7 kJ/mol |

**It recovers physics the unbiased run destroyed.**  Constrained TI puts the
C7ax basin **+7.50 kJ/mol** above the C7eq minimum -- the accepted vacuum value
for alanine dipeptide (~1.5-2 kcal/mol).  The unbiased run said **-0.00**, i.e.
exactly degenerate, because that was its initial ratio.

Cross-checked on the sub-arc where the unbiased run IS ergodic, the two
references agree to `RMS = 0.38 kJ/mol` on a 28.3 kJ/mol span, and the
disagreement is a smooth monotone tilt that crosses zero exactly at the
C7eq/C5 barrier -- the signature of the unbiased run's unconverged basin
balance, whose block series was still moving in that direction when it stopped.

### The campaign domain

`phi` carries a ~96-degree arc with no Boltzmann weight at all (`F > 18 k_B T`)
and, behind it, C7ax.  A cumulative TI integral around the periodic circle would
have to cross both, so the CV is rotated by -105 degrees and the domain
restricted to the ergodic arc `[-80, +80]` around C7eq, the beta/C5 region and
the barrier between them -- the standard negative-`phi` half of the Ramachandran
map.  The grid is reflecting there rather than periodic.

### Alanine: campaign constants

| quantity | value |
|---|---|
| `F_rms` over the arc | 6.71 kJ/mol |
| reference s.e. | 0.062 kJ/mol |
| estimator smoothing floor at `bw_mf = 0.05` | 0.156 kJ/mol |
| fiber relaxation `tau_psi` at the C7eq window | 2.3e4 steps |
| pentane's `tau_y` for comparison | 1.3e5 steps |

Alanine's hidden mode is roughly **six times faster** than pentane's relative to
the same step budget, which makes it a deliberately different regime: at 1e5
steps `psi` gets four relaxation times where pentane's `phi2` gets less than one.
P5 says the lift should matter LESS here, and that is a falsifiable prediction
rather than a hope.

Two bugs that the alanine setup exposed, both fixed and both worth recording
because they are the kind that produce plausible wrong numbers:

* `TorsionCV` carries ONE angular offset for every dihedral it holds, so
  rotating the CV to move the inaccessible arc to the domain edge also rotated
  `psi`.  Reading the reference conditional at the unrotated `psi` compared two
  different coordinates and reported `D_cond ~ 3.5` nats.
* The conditional itself cannot come from the unbiased run.  A uniform target in
  `phi` sends replicas to the arc edges, where `F` reaches 20-28 kJ/mol and the
  unbiased reference has almost no samples -- 10.6% of its conditional cells were
  empty.  The reference conditional is therefore taken from a STRATIFIED
  constrained run, which visits every window equally: 0.08% empty cells, and
  every `z` slice carries 2.6-3.8e5 samples.

## 10. Alanine screen (1e5 steps = 4.3 tau_psi, 8 seeds, 3 kappa values)

| arm | e_F (kJ/mol) | I_F | D_cond | vs naive |
|---|---|---|---|---|
| RC-WFR min-norm SHAKE | 6.86 | 7.15 | 0.86 | +26% |
| RC-WFR rotation (naive) | 5.46 | 4.85 | 0.94 | - |
| RC-WFR + oracle y-map | 1.74 | 2.95 | 0.43 | **-68%** |
| RC-WFR + oracle y-**refresh** | **15.28** | 14.93 | 4.10 | **+180%** |
| RC-WFR + Metropolis y-move, oracle proposal | **0.82** | 1.90 | 0.36 | **-85%** |
| RC-WFR + Metropolis y-move, **learned** proposal | **1.02** | 2.70 | 0.27 | **-81%** |

Estimator floor 0.156 kJ/mol; `F_rms` 6.71.

Two things are much sharper here than on pentane.

**The uncorrected refresh is a disaster on a real peptide** -- 2.8x worse than
doing nothing at all, using the EXACT conditional.  On pentane the same move was
the best oracle arm.  The difference is what else is in the fiber: pentane's is
a united-atom chain whose only structure is `phi2`, while alanine's carries the
C7eq internal hydrogen bond and sixty other coordinates correlated with `psi`.
Drawing `psi` from the marginal `nu(psi | phi)` is right for `psi` and wrong for
everything `psi` is correlated with, and the configuration it lands on is
strained.  This is the sharpest available demonstration that "transport the
conditional of the slow mode" is not by itself a safe instruction.

**The Metropolis correction fixes exactly that**, and by a factor 6.7 over the
naive lift.  It rejects the proposals that break the rest of the fiber, using
the full potential, so the move is right about `psi` AND about everything `psi`
is coupled to.  And the learned proposal again costs little relative to the
oracle one (1.02 vs 0.82) while needing no reference at all.

## 11. Pentane confirmation: 32 fresh seeds, three budgets, frozen hyper-parameters

Force evaluations at 4x: 1.07e8.  Estimator floor 0.0127 kcal/mol, `F_rms` 1.262.
Every arm ran at the configuration frozen on the 8 screening seeds, which no
confirmation seed ever saw.

| arm | e_F at 0.5x | at 1x | at 4x | D_cond | ESS_Fix |
|---|---|---|---|---|---|
| RC-WFR + Metropolis y-move, oracle proposal | 0.0296 | 0.0256 | **0.0196** | 0.0049 | 0.980 |
| RC-WFR + Metropolis y-move, **learned** proposal | 0.0264 | 0.0259 | **0.0215** | 0.0051 | 0.980 |
| RC-WFR + oracle y CDF-map | 0.0334 | 0.0290 | 0.0223 | 0.0270 | 0.979 |
| stratified constrained TI (cold) | 0.0818 | 0.0633 | 0.0378 | 0.1352 | 0.980 |
| RC-WFR + oracle y refresh | 0.0471 | 0.0410 | 0.0419 | 0.0297 | 0.979 |
| RC-WFR, naive rotation lift | 0.0714 | 0.0596 | 0.0475 | 0.1940 | 0.980 |
| RC-WFR + learned y CDF-map | 0.0794 | 0.0758 | 0.0710 | 0.1917 | 0.980 |
| RC-WFR + full conditional refresh | 0.0915 | 0.0898 | 0.0902 | 0.0073 | 0.979 |
| RC-WFR, min-norm SHAKE lift | 0.1286 | 0.1181 | 0.1003 | 0.1704 | 0.979 |
| RC-WFR + learned y refresh | 0.0713 | 0.0516 | 0.2917 | 1.4860 | 0.974 |

Paired relative change at 4x, median with a 95% bootstrap CI (bold = CI excludes zero):

| comparison | change in e_F | change in I_F |
|---|---|---|
| rotation lift vs min-norm SHAKE | **-53.7%** [-56.6, -49.0] | -47.9% [-53.1, -44.7] |
| + Metropolis y-move (learned) vs naive rotation | **-53.6%** [-57.7, -50.1] | -57.1% [-59.4, -52.6] |
| + Metropolis y-move (oracle) vs naive rotation | **-54.4%** [-60.2, -49.4] | -57.2% [-60.7, -50.1] |
| **learned vs oracle proposal** | **+1.2%** [-2.6, +15.3] | +3.5% [-6.6, +15.6] |
| + oracle y CDF-map vs naive rotation | **-52.0%** [-57.3, -47.9] | -51.0% [-56.0, -46.4] |
| + oracle y refresh vs naive rotation | **-10.8%** [-18.5, -0.2] | -23.4% [-31.2, -12.7] |
| + LEARNED y CDF-map vs naive rotation | **+52.8%** [+36.9, +69.8] | +34.6% [+23.7, +53.7] |
| + LEARNED y refresh vs naive rotation | **+506.9%** [+473.8, +577.4] | +140.1% [+132.3, +165.9] |
| + Metropolis y-move (learned) vs cold stratified TI | **-47.1%** [-52.7, -35.6] | -58.2% [-61.6, -51.0] |

The learned proposal is **statistically indistinguishable from the oracle one**:
+1.2% with a CI spanning zero.  That is the campaign's central practical claim,
and it is the direct consequence of the correction -- an uncorrected lift built
from the same estimate is 53% (map) or 507% (refresh) WORSE than doing nothing.

### The "ceiling" arm is not a ceiling, and the reason is worth keeping

`wfr_qref` replaces the whole configuration with a draw from a library of exact
conditional samples.  It has the **best conditional error of any arm** (0.0049,
tied with the Metropolis arms) and one of the worst free energies.  Three
separate things are wrong with it, and only two are fixable:

| version | e_F | D_cond |
|---|---|---|
| as first written | 0.0902 | 0.0073 |
| + SHAKE re-projection onto the exact `z'` after the draw | 0.0835 | 0.0049 |
| + deposits unweighted | 0.0567 | 0.0049 |
| for comparison: `wfr_lmh` | **0.0215** | 0.0051 |

1. The library is bucketed in `z`, so a drawn configuration sits somewhere in the
   bucket rather than at `z'`; re-project.
2. The library is drawn from `nu^xi` (unbiased MD), but the shared estimator
   applies the `(det G)^{-1/2}` weight that converts `nu_rgd` to `nu^xi`.  For
   this arm alone that double-counts, and removing it is worth 32%.
3. **What is left is not fixable.**  The arm redraws every 20 steps and then runs
   constrained dynamics, which pulls the configuration back toward `nu_rgd`, so
   at deposit time the ensemble is a MIXTURE of the two measures and no single
   weight is right for it.  The perfect conditional error alongside a
   three-times-worse free energy is that mixture, not a lift error.

So there is no clean "conditional-oracle ceiling" arm in this design.  The
practical ceiling is `ti_warm` -- stratified TI started from the oracle
conditional, 0.0299 -- and the fully practical `wfr_lmh` at 0.0215 is below it.

### Against the baselines, and the honest caveat

At 4x (1.07e8 force evaluations), the fully practical arm -- RC-WFR with a
Metropolis-corrected LEARNED conditional move, no oracle anywhere:

| vs | change in e_F at 1x | at 4x |
|---|---|---|
| ABF, multiple walkers | **-55.0%** [-61.3, -48.0] | **-31.4%** [-38.1, -25.3] |
| stratified constrained TI, cold | **-62.3%** [-64.9, -54.6] | **-47.1%** [-52.7, -35.6] |
| stratified constrained TI, warm (ORACLE initial conditional) | **-43.5%** [-50.5, -26.1] | **-25.1%** [-38.6, -19.2] |

**The advantage shrinks with budget, and that is not noise.**  Late-time
`d log e_F / d log fe`:

| arm | slope | reading |
|---|---|---|
| ABF | -0.416 | still converging, near the statistical rate |
| stratified TI, cold | -0.412 | still converging |
| stratified TI, warm | -0.249 | partly bias-limited |
| RC-WFR + Metropolis (learned) | -0.133 | partly bias-limited |
| RC-WFR + oracle y CDF-map | -0.146 | partly bias-limited |
| RC-WFR, naive rotation | -0.183 | partly bias-limited |
| RC-WFR, min-norm SHAKE | -0.081 | bias floor |

The baselines are unbiased and RC-WFR is not: its Wasserstein step moves `z`
without a Metropolis correction, and correcting it would need `F`, which is what
is being computed.  The conditional move fixes the FIBER half of that, not the
marginal half.  Extrapolating the fitted power laws -- an extrapolation of about
one decade, so a guide and not a measurement -- the baselines catch up at

| vs | crossover |
|---|---|
| ABF | ~4e8 force evaluations (3.4x this campaign's largest budget) |
| stratified TI, cold | ~9e8 (7.6x) |
| stratified TI, warm (oracle) | ~3e9 (23x) |

So the correct claim is a **speed** claim at practical budgets, not an accuracy
claim at unlimited budget.  That is the same structural obstruction the toy phase
identified, unchanged: an unconditional move in `xi` cannot be Metropolis-
corrected without knowing `F`.  What the molecular phase adds is that the OTHER
half of the error -- the fiber conditional -- can be removed exactly, and is
worth a factor two.

## 12. Transport-rate stress test on fresh seeds (16 seeds, 1e5 steps, kappa over 64x)

`e_F` (kcal/mol):

| `kappa_W` | 0.0375 | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 | 2.4 |
|---|---|---|---|---|---|---|---|
| min-norm SHAKE | 0.0980 | 0.1153 | 0.1592 | 0.2492 | 0.4140 | 0.7216 | **1.2313** |
| naive rotation | 0.0651 | 0.0688 | 0.0766 | 0.0739 | 0.0654 | 0.0603 | 0.0485 |
| + oracle y-refresh | 0.0432 | 0.0453 | 0.0447 | 0.0462 | 0.0440 | 0.0397 | 0.0387 |
| + oracle y CDF-map | 0.0277 | 0.0282 | 0.0253 | 0.0311 | 0.0269 | 0.0249 | 0.0289 |
| + Metropolis (oracle) | 0.0258 | 0.0230 | 0.0271 | 0.0237 | 0.0258 | 0.0266 | 0.0269 |
| + Metropolis (learned) | 0.0269 | 0.0245 | 0.0219 | 0.0251 | 0.0244 | 0.0229 | 0.0286 |

`D_cond`:

| `kappa_W` | 0.0375 | 0.075 | 0.15 | 0.3 | 0.6 | 1.2 | 2.4 |
|---|---|---|---|---|---|---|---|
| naive rotation | 0.147 | 0.151 | 0.206 | 0.183 | 0.210 | 0.213 | 0.199 |
| + oracle y-refresh | 0.0295 | 0.0299 | 0.0293 | 0.0310 | 0.0301 | 0.0311 | 0.0291 |
| + oracle y CDF-map | 0.0480 | 0.0424 | 0.0233 | 0.0182 | 0.0137 | 0.0112 | **0.0111** |
| + Metropolis (learned) | 0.0055 | 0.0054 | 0.0053 | 0.0052 | 0.0050 | 0.0053 | 0.0054 |

Three separate behaviours, cleanly separated:

* **min-norm SHAKE**: a factor **12.6** across the sweep.  Its damage is a
  per-step distortion, so it scales with the displacement it has to buy.  At the
  fastest transport it is 97 estimator floors.
* **naive rotation**: flat, and slightly DECREASING.  It distorts nothing; its
  damage is the conditional lag, which saturates once transport is faster than
  the fiber relaxes, so pushing `kappa` further only buys coverage.
* **the corrected arm**: flat to within the seed noise in both `e_F` (0.022-0.029)
  and `D_cond` (0.0050-0.0055) across a **64-fold** range of transport rate.

The last line is the strongest form of the toy phase's P4.  For the corrected
method the transport rate stops being a hyper-parameter: there is no tradeoff
left to tune, because the thing the tradeoff was made of has been removed.  For
the oracle CDF-map the conditional error actually IMPROVES 4.3x with faster
transport.

## 13. Mechanism ablations (pentane, 16 seeds, same 4x budget)

| arm | e_F | D_cond | coverage |
|---|---|---|---|
| full RC-WFR, naive rotation lift | 0.0475 | 0.194 | 1.00 |
| W only (Fisher-Rao removed) | 0.0444 | 0.183 | 1.00 |
| W only + oracle y-refresh | 0.0415 | 0.029 | 1.00 |
| ...vs full W+FR with the same lift | 0.0419 | 0.030 | 1.00 |
| Fisher-Rao only (transport removed) | 1.2622 | - | **0.02** |
| probability-flow W (`w_mode='flow'`) | 1.2622 | - | **0.02** |

Two of these are worth stating plainly because they cut against the toy phase.

**On a torsional CV the Fisher-Rao half buys nothing.**  Removing it changes
`e_F` by -6.5% (0.0475 -> 0.0444), i.e. nothing outside seed noise, where the toy
campaign found removing it turned a 50% win over replica exchange into a 27%
loss.  The difference is the domain: a torsion is one period long and plain
diffusion covers it in 4e4 steps, so there is no discovery problem for birth-death
to solve.  The toy campaign's factor 2.4 came from a CV domain of length 24, where
diffusion WAS the bottleneck.  Fisher-Rao is worth what discovery is worth, and on
a single torsion discovery is free.

**Fisher-Rao alone cannot explore, exactly as designed.**  Selection only
reallocates walkers among the `z` values that already exist; with the transport
step removed, coverage stays at 2%.

**The probability-flow variant is degenerate from a point start**, which is a
reproduction of a known toy-phase failure, not a new result: the deterministic
step's velocity is `-kappa grad log p_hat`, which vanishes for a delta initial
ensemble, so nothing ever moves.  The toy phase fixed it with an initial jitter;
the molecular arms start from a single basin without one.

## 14. Which fiber mode has to be promoted -- predicted before it is measured

The manifold phase's spectral estimate says a mode's damage scales as
`S_k tau_k^2`, with `S_k` the conditional's sensitivity to `z` and `tau_k` its
relaxation time.  Both are measurable from a run's own output.  Hexane:

| mode | `S_k` (sensitivity) | `tau_k` (steps) | predicted damage | rank |
|---|---|---|---|---|
| `phi2` (adjacent to `z`) | 0.0658 | 6.2e4 | 2.55e8 | **1** |
| `phi3` (one bond further) | 0.00068 | 1.0e5 | 6.8e6 | 2 |

`phi3` is the SLOWER of the two by a factor 1.6, and the diagnostic still ranks
it second by a factor 38, because coupling beats timescale here (96x against
2.6x).  The toy phase could only vary timescale; this is the first test of the
other axis, and the hexane arms measure the actual benefit of promoting each.

## 15. Alanine dipeptide confirmation: 16 fresh seeds, 1.07e8 matched force evaluations

Estimator floor 0.156 kJ/mol; `F_rms` 6.71; reference s.e. 0.062.

| arm | e_F at 0.5x | at 1x | at 4x | D_cond | ESS_Fix |
|---|---|---|---|---|---|
| RC-WFR + Metropolis y-move, **learned** proposal | 1.056 | 0.659 | **0.526** | 0.161 | 0.975 |
| RC-WFR + Metropolis y-move, oracle proposal | 0.696 | 0.609 | 0.579 | 0.247 | 0.975 |
| stratified constrained TI, warm (ORACLE initial conditional) | 0.817 | 0.781 | 0.653 | 0.398 | 0.976 |
| RC-WFR + oracle y CDF-map | 1.646 | 1.272 | 0.681 | 0.260 | 0.975 |
| stratified constrained TI, cold | 2.773 | 3.836 | 2.980 | 0.429 | 0.974 |
| RC-WFR, naive rotation lift | 3.512 | 4.142 | 3.325 | 0.323 | 0.976 |
| RC-WFR, min-norm SHAKE lift | 5.326 | 7.210 | 3.357 | 0.455 | 0.975 |
| RC-WFR + full conditional refresh | 3.955 | 1.796 | 7.372 | 0.182 | 0.975 |
| ABF, multiple walkers | 16.558 | 14.341 | 8.707 | 1.086 | - |
| RC-WFR + oracle y **refresh** | 16.200 | 15.163 | **15.204** | 4.095 | 0.978 |

Paired relative change at 4x (bold = 95% bootstrap CI excludes zero):

| comparison | change in e_F |
|---|---|
| + Metropolis (learned) vs naive rotation | **-83.4%** [-84.9, -81.5] |
| + Metropolis (oracle) vs naive rotation | **-82.4%** [-84.1, -79.2] |
| + oracle y CDF-map vs naive rotation | **-78.3%** [-80.4, -75.9] |
| + oracle y **refresh** vs naive rotation | **+361.4%** [+336.3, +421.1] |
| + Metropolis (learned) vs cold stratified TI | **-82.4%** [-83.3, -81.6] |
| + Metropolis (learned) vs ABF | **-94.0%** [-94.4, -93.1] |
| + Metropolis (learned) vs warm stratified TI (oracle) | **-17.5%** [-22.4, -14.0] |
| **learned vs oracle proposal** | **-6.5%** [-10.5, -0.7] |

The learned proposal is not merely as good as the oracle one here, it is very
slightly BETTER.  That is not a claim about learning beating a reference: both
are only proposals and the Metropolis step makes either exact, so the difference
is acceptance efficiency.  The "oracle" is a marginal conditional
`nu(psi | phi)` estimated from a stratified run; the learned one adapts to the
ensemble that is actually there.

Two caveats stated rather than buried.

* **ABF is at its worst here** and part of that is the setup, not the method.
  The restricted arc means an unconstrained sampler must be held inside it by
  walls, and it has to diffuse 160 degrees over a 7 kJ/mol internal barrier from
  a single starting basin; its coverage is 0.40 at 1e5 steps.  A stratified
  method is placed across the whole domain by construction.  ABF's late-time
  slope is -0.373, the steepest of any arm, so it is still converging fast.
* **The naive-rotation and min-norm arms are statistically indistinguishable at
  4x** (-5.3% [-21.3, +33.4]).  On alanine the min-norm lift's extra damage is
  visible at 1x (-38.8% for the rotation) but washes out by 4x; the pentane
  separation (-53.7%) is the cleaner measurement.

## 16. Hexane: the ranking was predicted, then measured

16 seeds, 2e5 steps, matched force evaluations, all arms at the same
`(kappa, theta)` (taken from pentane's frozen naive arm, so no hexane arm is
tuned against another).

| promoted | e_F (median [IQR]) | I_F | D_cond(phi2) | MH accept | vs naive lift |
|---|---|---|---|---|---|
| none (naive rotation) | 0.0414 [0.0283,0.0526] | 0.0486 | 0.1765 | - | - |
| `phi2` -- adjacent, S=0.066, tau=6.2e4 | **0.0200** | 0.0267 | **0.0042** | 0.863 | **-44.3%** [-56.4, -25.9] |
| `phi3` -- distal, S=0.00068, tau=1.0e5 (**the slower mode**) | 0.0485 | 0.0546 | 0.1862 | 0.891 | +29.1% [-14.2, +37.9] |
| both | 0.0214 | 0.0244 | 0.0036 | 0.840 | **-44.5%** [-51.8, -25.5] |
| ABF | 0.0383 | 0.0750 | 0.0265 | - | +8.5% [-30.1, +46.7] |

Read it in this order.

1. Promoting `phi2` is worth **-44.3%**, with a CI excluding zero.
2. Promoting `phi3` -- **the slower of the two modes, by a factor 1.6** -- is
   worth nothing at all: +29.1% with a CI spanning zero, and its conditional
   error is unchanged (0.186 vs 0.177).  Its Metropolis acceptance is the
   highest of the three (0.891), so the move is working; it is simply moving
   the wrong coordinate.
3. Promoting both is **not better than promoting `phi2` alone** (-44.5% vs
   -44.3%).  There is no second thing to fix.

The diagnostic ranked `phi2` first by a factor 38, computed from statistics a
thermodynamic-integration run already produces, before any of these arms ran.
`S_k tau_k^2` is therefore usable as a selection rule, and the axis it selects
on is **coupling, not timescale** -- which is the axis the toy phase could not
vary.

## 17. The predicted crossover, measured

The confirmation's fitted convergence rates put the crossover with ABF at ~4e8
force evaluations, 3.4x the largest budget run there.  Rather than leave that as
an extrapolation, the same arms were run to 1.6e6 steps (4.1-4.7e8 force
evaluations), 16 seeds:

| | at 1.07e8 (confirmation) | at ~4.3e8 | late-time slope |
|---|---|---|---|
| RC-WFR + Metropolis (learned) | 0.0215 | **0.0206** | -0.044 |
| ABF, multiple walkers | 0.0314 | 0.0227 | -0.231 |
| stratified constrained TI, cold | 0.0378 | 0.0284 | - |
| paired change vs ABF | **-31.4%** [-38.1, -25.3] | **-8.1%** [-17.5, -5.9] | |

The prediction was right to within less than a factor two, and the picture it
describes is exactly what happened: RC-WFR is flat (slope -0.044, a bias floor)
while ABF keeps converging (-0.231).  RC-WFR is still ahead at 4.1e8 with a CI
excluding zero, but only by 8%, and it will be overtaken shortly after.

**And the residual is provably not the fiber.**  Over the same long run the
corrected arm's conditional error fell from 0.0051 to **0.0014 nats** while its
free-energy error did not move (0.0215 -> 0.0207).  The fiber conditional is
essentially exact and `e_F` still sits at 1.6 estimator floors.  What is left is
the uncorrected Wasserstein step in `z` -- the structural obstruction, unchanged
from the toy phase, and the one thing this campaign did not fix.

## 18. Alanine transport-rate sweep (16 fresh seeds, 1e5 steps)

`e_F` (kJ/mol):

| `kappa_W` | 0.0375 | 0.15 | 0.6 | 2.4 | max/min |
|---|---|---|---|---|---|
| min-norm SHAKE | 6.025 | 7.095 | 6.741 | 6.137 | 1.2x |
| naive rotation | 5.106 | 6.210 | 5.276 | 3.799 | 1.6x |
| + Metropolis (oracle) | 3.588 | 0.705 | 0.644 | **0.635** | 5.6x |
| + Metropolis (learned) | 3.615 | 0.938 | 0.799 | **0.660** | 5.5x |

`D_cond`:

| `kappa_W` | 0.0375 | 0.15 | 0.6 | 2.4 |
|---|---|---|---|---|
| naive rotation | 0.989 | 0.946 | 0.864 | 0.826 |
| + Metropolis (learned) | 0.147 | 0.168 | 0.159 | 0.150 |

The alanine version of the stress test says the same thing in the opposite
voice.  On pentane the corrected arm was FLAT in `kappa_W` over 64x; here it
**improves 5.6x** with faster transport, because alanine's domain is a
160-degree arc with 22-28 kJ/mol walls and the slowest transport cannot cover
it inside 1e5 steps.  Either way the conclusion is the same: once the fiber
conditional is handled correctly, faster reaction-coordinate transport is never
a liability, and here it is the single largest available gain.

The min-norm lift does NOT show its pentane pathology here (1.2x, not 12.6x).
Its per-step distortion scales with the displacement it has to buy, and on a
264-degree domain at the same `kappa` those displacements are smaller relative
to everything else that is going wrong -- alanine's naive-lift error is 5 kJ/mol
against a 0.156 floor, so the distortion is not the binding constraint.

## 19. What the long runs settle

At ~4.3e8 force evaluations, 16 seeds:

| arm | at 1.07e8 | at ~4.3e8 |
|---|---|---|
| RC-WFR + Metropolis (learned) | 0.0215 | **0.0206** |
| ABF | 0.0314 | 0.0227 |
| stratified constrained TI, cold | 0.0378 | 0.0284 |
| RC-WFR, naive rotation lift | 0.0475 | 0.0437 |

Both RC-WFR arms are on their bias floors; both baselines are still converging.
The corrected arm's floor (0.021) is **half** the naive lift's (0.044), and it
is 1.6 estimator floors -- while its conditional error over the same run fell to
0.0014 nats.  The fiber is solved; the marginal is not.

## 20. What the 0.020 plateau is made of

Every constrained arm -- persistent RC-WFR, the switched arm, the arm with no
transport at all -- converged to about `e_F = 0.020` kcal/mol and stopped.  Two
arms differing by a factor **100** in conditional error (0.089 against 0.0008
nats) reached the same number.  That is not what a method-dependent bias looks
like, so the plateau was measured directly instead of being attributed.

The test system is butane, because it has no hidden slow mode: its fiber is
bonds and angles, which relax in ~1e3 steps.  The arm is warm stratified
constrained TI -- no transport, no exploration, nothing that could confound a
numerical measurement.  `h` is swept over `{2e-3, 1e-3, 5e-4, 2.5e-4}` at a
**fixed physical time** of 400 (so `n_steps` and the deposit interval both scale
as `1/h`, holding the statistical term still while the discretisation term
moves), and three bandwidths are accumulated from the SAME trajectory at each
`h`, since the bandwidth changes only the estimator.  `N = 1024` windows, 8 rows,
a 257-node grid.

### 20.1 The measurement

`e_F` against the unbiased-MD reference (median over rows, kcal/mol):

| `h` | `b_mf` = 0.08 | 0.04 | 0.02 |
|---|---|---|---|
| 2.0e-3 | 0.03313 | 0.01579 | 0.01474 |
| 1.0e-3 | 0.03109 | 0.00864 | 0.00556 |
| 5.0e-4 | 0.03069 | 0.00886 | 0.00525 |
| 2.5e-4 | 0.03077 | 0.00878 | 0.00607 |

### 20.2 The decomposition, with no fitted parameters

The **smoothing** term is not fitted.  It is computed by differentiating the
reference and putting it back through the estimator's own pipeline -- the same
Nadaraya-Watson kernel on the same grid, then the same trapezoid integration --
and comparing the smoothed reconstruction against the unsmoothed one.  Taking
the difference of two reconstructions rather than comparing to `F_ref` cancels
both the reference's noise and the differentiate/re-integrate mismatch, leaving
only what the kernel did:

| grid | `b_mf`=0.08 | 0.05 | 0.04 | 0.02 |
|---|---|---|---|---|
| n=129 | 0.03155 | **0.01243** | 0.00797 | 0.00108 |
| n=257 | 0.03158 | 0.01244 | 0.00798 | 0.00200 |
| n=513 | 0.03156 | 0.01244 | 0.00798 | 0.00200 |

It is grid-independent from n=129 up: at these bandwidths the kernel, not the
grid, sets the resolution.

The **statistical** term is the row scatter, which is the statistical part of a
single row's `e_F` (not of the row mean): 0.0047-0.0055 at this budget.

Removing both leaves the **discretisation** term:

| `h` | `b_mf`=0.08 | 0.04 | 0.02 |
|---|---|---|---|
| 2.0e-3 | 0.00846 | 0.01250 | 0.01353 |
| 1.0e-3 | 0.0 | 0.0 | 0.00230 |
| 5.0e-4 | 0.0 | 0.0 | 0.00053 |
| 2.5e-4 | 0.0 | 0.0 | 0.00297 |

**The check the decomposition could have failed.**  At `b_mf = 0.08` the
measured `e_F` (0.0331) IS the analytic smoothing floor (0.0316) to three
digits at every `h`, and the residual after removing it is exactly zero from
`h = 1e-3` down.  Nothing was tuned to make that happen.

### 20.3 The same answer without any reference

`e_F` is measured against a reference that is itself unbiased MD at `h = 2e-3`,
so it carries an O(h) error of its own.  Shrinking `h` in the constrained arm
alone would then make it converge AWAY from the reference.  Rather than assume
that away, the arm is also compared to itself:

| `h` | `‖F(h) - F(h_min)‖` |
|---|---|
| 2.0e-3 | 0.01326 |
| 1.0e-3 | 0.00315 |
| 5.0e-4 | 0.00273 |
| 2.5e-4 | 0 (by construction) |

No reference appears anywhere in that column, and it agrees with the
reference-based discretisation term (0.01353) to 0.0003.  Only the largest `h`
clears the 0.0048 noise floor, so the order is bounded rather than fitted:
**p > 1.5**, consistent with the drop by a factor 4.2 across a factor 2 in `h`.

### 20.4 The reference's own time-step bias, measured

Butane's reference was rerun as unbiased MD at `h = 5e-4`, four times finer:

    ‖F_ref(h=2e-3) - F_ref(h=5e-4)‖ = 0.00181 kcal/mol
    block standard errors:            0.00090 and 0.00081

So the reference's discretisation bias is barely above its own statistical
error, and **eight times smaller** than the constrained arm's bias at the same
step.  Every existing reference in this campaign stands.  The asymmetry is the
informative part: unconstrained overdamped Langevin at `h = 2e-3` is essentially
converged, while the same step through the **projection** is not.  The excess
belongs to the constrained integrator specifically.

### 20.5 The plateau, reconstructed

The campaign ran at `h = 2e-3`, `b_mf = 0.05`, `n = 129`.  Its two measured
numerical terms are

    smoothing               0.01243
    constrained integrator  0.01326
    quadrature sum          0.01818     before any statistical error

against an observed plateau of ~0.020.  **The number every constrained arm
converged to was the estimator and the integrator.**  That is why arms differing
by a factor 100 in conditional error shared it, and it is the last piece of the
withdrawn asymptotic-bias claim (section 19, `SWITCH_CAMPAIGN.md`).

### 20.6 Gate A

At `h = 1e-3`, `b_mf = 0.02`, `n = 257`, butane reaches **0.0056**, of which
0.0047 is statistical and 0.0020 smoothing -- leaving no detectable bias, a
factor 3.6 under the plateau.  The floor is not a property of the method, and
lowering it costs a factor 2 in force evaluations per unit physical time and
nothing else.

Reproduced by `scripts/mol_floor_study.py` and `scripts/mol_floor_fit.py`;
figure `figures/figMOL11_floor.png`.
