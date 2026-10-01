# Alanine dipeptide with an incomplete collective variable: ABF (+ uniform Fisher--Rao) on phi alone or psi alone

**Date:** 2026-09-30. **Branch:** `main` (uncommitted, on top of `ebc123b` + the histogram study).
**GPU:** 3 only. **Status:** design, rules and predictions written BEFORE any run; results appended below.

## 1. Question

The alanine studies so far used the complete 2-D CV (phi, psi) and found ABF sufficient (kernel
2026-08, FR-start timing 2026-09-04, histogram estimator 2026-09-30, `docs/ALANINE_HISTOGRAM_ABF.md`).
The user asks what happens with an INCOMPLETE CV: bias phi alone, or psi alone, and see whether the
results are still good, for ABF and for ABF + uniform-target FR.

## 2. What is being tested

With a 1-D CV the omitted angle is a hidden coordinate. Two facts fix the prior:

* The slow angle is phi: the C7eq -> C7ax crossing (15.8 kT along phi at 300 K) is the process the
  2-D ABF flattens; the psi barriers (C7eq <-> C5, ~1-2 kT) are fast. So biasing phi alone should
  still discover and flatten the slow transition (the hidden psi relaxes within picoseconds), while
  biasing psi alone leaves the phi barrier unflattened: the spontaneous crossing rate is of order
  one per microsecond and the aggregate simulated time is 0.2 microseconds per seed
  (2048 walkers x 100 ps), so C7ax should stay undiscovered in most seeds.
* The project's theorem (`docs/V2_PREREGISTRATION.md`, [[v2-campaign-status]]): a birth-death step that
  reads only the CV marginal leaves the conditional distribution of the hidden coordinate unchanged in
  expectation, d/dt p(y|xi)|_FR = 0. FR can reallocate walkers along psi but cannot make a walker cross
  the phi barrier, nor preferentially clone one that did (at fixed psi the clone/kill decision is blind to phi).

Reference for a 1-D CV: the exact marginal of the accepted 2-D reference over the hidden angle,
`F1(xi) = -kT log sum_hidden exp(-F2/kT)` on the same 97 cells (`core1d_ala.marginal_reference_1d`;
+inf cells contribute nothing). Its 8 kT mask and equilibrium weights are built exactly as in 2-D
(`metrics_ala.build_masks` on `F1`). For psi alone a second, diagnostic reference is the
VISITED-SIDE conditional marginal (the same sum restricted to phi < 0): the prediction is that
psi-only ABF converges to that one, not to the full marginal.

## 3. Engine

`src/alanine/core1d_ala.py` (new file; the 2-D engine is untouched). CV = `alkanes.cv.DihedralCV`
with IUPAC values (den Otter's 1-D instantaneous force for ONE dihedral; not the phi-component of the
2-D vector force, because the two dihedrals share three atoms). Everything else is the 2-D engine's
code or convention: force field, BAOAB, dt 1 fs, gamma 1/ps, 300 K, float64, initial ensemble
(the cached C7eq ensemble of every 2026-09 alanine arm), fixed-consumption RNG, the full-state
kill-and-clone step imported from `core2d_ala`, 6 ps age-aware genealogy window, non-finite containment.
Estimator: the adaptive-box histogram (1-D boxes of 1..9 bins, c_min 800, 5 ps ramp; the walker feels
its OWN bin's value, the PMF is the exact integral of the piecewise-constant force with the circular
mean removed), plus a kernel-ABF check arm per CV (h 0.08, min_count 200, alkanes convention).
Recorded every save: the 1-D PMF and mean force, counts, the 1-D walker marginal and KL to uniform,
the JOINT (phi, psi) walker histogram on the 97 x 97 grid (the hidden coordinate), basin fractions
and first hits from the accepted 2-D watershed labels, genealogy, temperature, raw accumulators.
Tests: `tests/test_alanine_1d.py` (values == the 2-D engine's angles, geometry == DihedralCV, box
sums, exact PMF, own-bin bias, marginal reference, arm equality at rate 0, FR fires, both CVs finite).

## 4. Arms (all N 2048, 100 ps, seeds 0-15, `rng_seed` 20260903, init cache, one at a time on GPU 3)

| stage | CV | method | notes |
|---|---|---|---|
| `phi_abf` | phi | abf | histogram c800 / 5 ps ramp |
| `phi_u02_t0` | phi | fr_uniform | FR from 0.5 ps, rate 0.02 |
| `phi_u15_t0` | phi | fr_uniform | FR from 0.5 ps, rate 0.15 |
| `phi_abf_kernel` | phi | abf | kernel estimator check |
| `psi_*` | psi | the same four | |

Launch order: phi_abf, psi_abf, phi_u02_t0, psi_u02_t0, phi_u15_t0, psi_u15_t0, phi_abf_kernel, psi_abf_kernel.

## 5. Endpoints and rules (frozen)

* Primary per CV: the 1-D FES error e(t) = equilibrium-weighted aligned L2 of the arm's PMF against
  the FULL marginal reference on its 8 kT mask; I_F on W1 = [5, 100] ps and on the own window
  [1, 100] ps; final error. FR vs ABF: paired relative change, median + BCa CI, win rate, with the
  FR-timing verdict rules (ACCELERATION_POSITIVE at median <= -10 % with CI < 0; NEUTRAL inside
  +-10 % with |final| <= 5 %; HARMFUL when the CI lower bound > 0 or final > +5 % with CI > 0).
* Hidden-coordinate endpoints: C7ax first-hit time (censored = never), C7ax occupancy at 20 / 100 ps
  vs the reference 3.1 %, and the conditional distance
  D_cond(t) = mean over CV bins with >= 50 walkers of TV( p_walkers(hidden | xi), p_ref(hidden | xi) )
  (equal weight per bin, NOT weighted by the walker marginal -- the mixture trap of
  `docs/ADVERSARIAL_AUDIT` / [[adversarial-audit-before-the-long-run]]).
* psi-only: e(t) against the visited-side reference is reported alongside; "converges to the
  conditional" means final error vs the visited-side reference < 0.5 x final error vs the full one.
* Cross-CV context: the 2-D histogram ABF arm's PMF marginalised over the hidden angle is scored on
  the same 1-D reference (a 2-D method's implied 1-D result), so the 1-D arms can be placed against it.

## 6. Predictions (recorded before any run)

- P1 phi-only ABF is GOOD: C7ax discovered in 16/16 seeds by 10 ps; final e(F(phi)) within 2x of the
  2-D arm's marginalised error; D_cond(psi | phi) falls below 0.05 by 20 ps.
- P2 psi-only ABF is NOT good against the full reference: C7ax first hit censored in >= 12/16 seeds,
  C7ax occupancy < 0.5 % at 100 ps, final error vs the full reference > 2x the error vs the
  visited-side reference (it converges to the conditional marginal).
- P3 FR is NEUTRAL on phi-only at both rates (the 2-D result carried over).
- P4 FR does NOT repair psi-only: C7ax discovery and D_cond unchanged (the theorem); vs its own ABF
  it is NEUTRAL at rate 0.02; rate 0.15 NEUTRAL or HARMFUL, not positive.
- P5 kernel-ABF check arms agree with the histogram arms in sign on every endpoint (estimator-independence).
If P4 fails (FR reaches C7ax or lowers D_cond), that is the first positive of marginal FR on a hidden
coordinate and must be replicated on fresh seeds before it is reported.

## 7. Results and verdict (all eight arms; read 2026-09-30T10:00Z; 16 paired seeds; 11.4-11.9 ms/step)

Tables: `results/alanine_1d/analysis/scoreboard.md`, `arms.csv`, `fr.csv`, `summary.json`; figures
`figures/phi_panels.*`, `figures/psi_panels.*`. Errors are the equilibrium-weighted aligned L2 of the
1-D PMF against the FULL marginal reference (8 kT mask); "visited" is against the phi < 0 conditional
marginal (psi only). D_cond by window = equal-weight TV of p(hidden | xi) vs the reference conditional
on 15-deg pooled bins (multinomial floor in brackets; the reference's own cell noise adds ~0.03).

### ABF alone: the incomplete CV works for phi, fails for psi

| arm | I_F W1 | final | C7ax first hit | C7ax @100 ps | D_cond [5,20] / [50,100] | hidden mass phi > 0 [50,100] (target) |
|---|---|---|---|---|---|---|
| phi_abf (histogram) | 76.3 | **0.374** | 3.07 ps, 0/16 censored | 0.13 | 0.095 (0.041) / 0.074 (0.017) | - |
| phi_abf_kernel | 68.1 | 0.417 | 3.12 ps, 0/16 | 0.16 | 0.091 / 0.070 | - |
| psi_abf (histogram) | 205.2 | **2.070** (visited 0.514, ratio 4.0) | 4.41 ps, 0/16 censored | 0.14 | 0.321 (0.021) / 0.249 (0.012) | 0.129 (0.321) |
| psi_abf_kernel | 203.7 | 2.051 (visited 0.635, ratio 3.2) | 4.51 ps, 0/16 | 0.15 | 0.319 / 0.240 | 0.134 (0.321) |
| 2-D (phi, psi) histogram ABF, marginalised (context) | - | 0.210 (phi) / 0.242 (psi) | 16/16 | - | - | - |

**phi alone (P1 PASS).** C7ax is discovered in every seed by 3 ps, the phi marginal is fully
flattened (KL to uniform 0.03 at 100 ps: on the circle the uniform target is reachable, unlike the
torus), the hidden psi conditional stays within 0.03-0.06 of the reference above the sampling floor,
and F(phi) ends at 0.37 kJ/mol, 1.8x the 2-D method's marginalised 0.21 (the residual is the hidden
psi lag plus the coarser 1-D discretisation floor). Good enough to use.

**psi alone (P2 half PASS, half FAIL).** The prediction that C7ax stays undiscovered was WRONG: once
psi is flattened the walkers spend time at psi ~ -50 .. -130 deg where the phi crossing is cheap, so
C7ax is hit by 4.4 ps in every seed. What holds is the substance: the hidden phi equilibrates on a
timescale longer than the run. The walkers' mass on the phi > 0 side climbs linearly, 0.00 -> 0.02 ->
0.07 -> 0.13 over the four windows, against a target of 0.32 (the psi-uniform average of the
reference conditional); D_cond sits at 0.25-0.32 (floor 0.01-0.02); and the final F(psi) lies ON the
phi < 0 conditional profile (error 0.51 vs it, 2.07 vs the full marginal, ratio 4.0 > the
preregistered 2), i.e. it reports a barrier 9 kJ/mol too high between psi -130 and -50 deg where the
phi > 0 side dominates the true marginal. psi alone is NOT good: a hidden slow coordinate, converging
on a >100 ps timescale.

### FR on the incomplete CV: neutral at best, and it slows the hidden coordinate

FR vs the matched histogram ABF arm (own window [1, 100] ps, full reference; frozen rules):

| arm | CV | rate | dI_F own | final | D_cond [50,100] vs ABF | hidden mass phi > 0 [50,100] arm / ABF | C7ax @100 arm / ABF | ESS min | verdict |
|---|---|---|---|---|---|---|---|---|---|
| phi_u02_t0 | phi | 0.02 | -1.34 % [-3.30, -0.51] 12/16 | +5.70 % [-0.27, +7.42] 4/16 | +1.3 % [-2.2, +3.6] | - | 0.134 / 0.135 | 0.91 | INCONCLUSIVE (~neutral: 1 % integrated gain, 6 % endpoint loss) |
| phi_u15_t0 | phi | 0.15 | **+18.9 % [+12.9, +21.1] 0/16** | +67.1 % [+55.3, +78.8] 0/16 | +11.6 % [+9.7, +13.5] 0/16 | - | 0.151 / 0.135 | 0.47 | **HARMFUL** |
| psi_u02_t0 | psi | 0.02 | +0.24 % [+0.03, +0.34] 2/16 | +0.78 % [+0.43, +0.88] 1/16 | +3.4 % [+2.4, +4.1] 1/16 | 0.105 / 0.129 | 0.113 / 0.144 | 0.96 | NEUTRAL_SIG |
| psi_u15_t0 | psi | 0.15 | +1.31 % [+1.03, +1.53] 0/16 | +3.30 % [+2.79, +3.51] 0/16 | **+14.5 % [+12.3, +16.1] 0/16** | **0.064 / 0.129** | **0.064 / 0.144** | 0.77 | NEUTRAL_SIG |

**P3 (FR neutral on phi) FAILS at the high dose:** in 1-D the uniform target is reachable and the
marginal is flat by 20 ps, so a rate-0.15 FR keeps killing and cloning a population that is already
where the target wants it (18.6 events per opportunity, ESS 0.47): +19 % integrated, +67 % at the
end, and the hidden psi conditional 12 % further from the reference. The gentle dose is a 1 %
integrated gain against a 6 % endpoint loss: inconclusive by the rules, neutral in substance.

**P4 (FR does not repair psi) PASSES, with a sharper finding.** FR never censors C7ax and the PMF
error is unchanged (+0.2 % / +1.3 %), but the hidden coordinate gets WORSE monotonically in dose:
the walkers' mass on the phi > 0 side at 50-100 ps is 0.129 (ABF) -> 0.105 (rate 0.02) -> 0.064
(rate 0.15), C7ax occupancy 0.144 -> 0.113 -> 0.064, D_cond +3.4 % -> +14.5 %. Mechanism, read
from the score: the psi region where the hidden phi is still equilibrating (psi ~ -50 .. -130) is
exactly where the ABF bias, learned from the phi < 0 population, has not caught up, so that region is
transiently OVER-populated relative to the uniform target and FR kills there -- preferentially
removing the very walkers that carry the new phi > 0 information, and cloning phi < 0 walkers
elsewhere. This is the theorem's practical corollary on an incomplete CV: marginal reallocation
cannot improve the hidden conditional in expectation, and with a lagging bias it degrades it.
(The pentane 2-D study recorded the same sign: "harms hidden conditional when active".)

**P5 PASS.** The kernel check arms agree in sign on every endpoint (phi: kernel ABF 9.7 % better
integrated, endpoint n.s.; psi: within 1 %); the conclusions are estimator-independent.

**Verdict.** With an incomplete CV, ABF on the slow angle (phi) still works (F(phi) to 0.37 kJ/mol,
C7ax found in 3 ps); ABF on the fast angle (psi) does not (F(psi) converges to the phi < 0 conditional,
9 kJ/mol wrong at the barrier, on a >100 ps hidden timescale). Uniform-target Fisher-Rao does not
rescue the incomplete CV: it is neutral-to-harmful on the PMF and, on psi, slows the hidden
coordinate's equilibration in proportion to its dose. Nothing here licenses FR as a remedy for a
missing slow coordinate; it argues the opposite.
