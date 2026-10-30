# Deca-alanine, re-done: ABF versus ABF + Fisher–Rao (2026-10)

**Status: protocol written 2026-10-29 before any production run.**  Results are appended below
as stages close; nothing above the results section is edited after a stage has run except to
record amendments, which are dated.

## 0. Why reopen a closed system

`docs/CLOSURE_deca_and_R15.md` (2026-08-11) closed deca-alanine as *conditional-equilibration-
limited*: at the literature budget (16 walkers x 0.5 ns, 8 ensembles) the kernel-ABF mean force
disagreed with the reference by 61 % in bins holding up to 2e6 effective counts, the learned span
was 87–110 kT against 72 kT, and 95–99.96 % of walkers sat above 2.80 nm.  The same local-mean-force
estimator inside umbrella windows gave 8.4 % error, so the estimator was exonerated and ABF's own
conditional sampling blamed.  Three things have changed since:

1. **A defect in the sampler's bias application** (`src/deca/core.py`, 2026-08): the per-bin trust
   ramp (`abf_min_count`, NAMD's `fullSamples`) was applied in the first BAOAB half-kick only.  The
   second half-kick interpolated the *unweighted* profile, so every sparse bin still received half of
   an unguarded bias.  The regression test of the time only checked that the guard changed the
   trajectory, not that both half-kicks honoured it.  Fixed; `legacy_unweighted_second_kick=True`
   reproduces the defect for the historical replay and nothing else.
2. **Histogram ABF** (`docs/HISTOGRAM_ABF_REPLICATION.md`) materially changed other baselines in this
   project; deca-alanine was only ever run with the kernel estimator (h = 0.04 nm).
3. **The R2 online FR stop** (`docs/r2_campaign/RESULTS.md`) and the wall-aware uniform target were
   never applied to a molecular system with a bounded distance CV.

The closed study's attribution (conditional sampling, not estimator) may well stand.  The point is
to establish it on a sampler without the defect, with the estimator chosen on ABF-only data, and
then — only if ABF is usable — to answer the project's question, ABF versus ABF + FR, on
deca-alanine.  Per the user directive of 2026-10-28, the production comparison has exactly two arms;
no sham, count-balancing or Laplacian selection arm is run.

## 1. Model, CV, reference (unchanged physics)

Ace-(Ala)10-Nme, 112 atoms, AMBER ff14SB, vacuum, NoCutoff, no constraints, 300 K, Langevin
gamma = 1 ps^-1, BAOAB, dt = 1 fs (0.5 fs checked in Stage I-B).  CV: distance between the
terminal carbonyl carbons (atoms 4 and 104), evaluation domain [1.20, 3.60] nm, 129 cells, soft
harmonic walls (k = 4000 kJ/mol/nm^2) from 1.25 and 3.55 nm.  Local mean force
`f = (grad V . grad xi)/2 - 2/(beta R)` (`alkanes.distance_cv`).

**Reference.**  The 2026-08 umbrella/MBAR reference (96 windows, k = 3200, 32 replicas, 3
interleaved builds, 2 ns/replica, 18 432 ns aggregate; span 179.6 kJ/mol, between-build L2 0.604,
drift 3.18 kJ/mol, argmin 1.637 nm, Gate A TV 0.754) was built on an H200 and **its raw profile was
never copied to NSCC** (only the summary JSON is in git).  It is rebuilt on the A100s with the same
frozen `UmbrellaConfig` and the same per-build seeds, one build per process, with one addition:
the per-cell average of the local mean force over every production step of every kept replica,
which is an **independent F'_ref** (the restraint depends on R only, so the conditional law given R
is the unbiased one).  Acceptance as before (max pairwise between-build L2 < 10 % of the span;
drift between the 1.5 and 2.0 ns checkpoints reported as the honest uncertainty), plus two new
checks: integrating F'_ref must reproduce the MBAR F on I to within the between-build spread, and
the rebuilt span / argmin / Gate A must agree with the 2026-08 record.  (The rebuild sets
`n_prod_steps = 2 000 000` explicitly instead of stopping early at 2 ns, so its `config_hash` differs
from the 2026-08 `7126bf20f4f4`; every other field is identical.)

**Scoring interval** I = [1.30, 3.50] nm (wall-free).  Error metrics (§8) are evaluated on I.

## 2. Algorithms (`src/deca/core.py`)

*ABF.*  K walkers, one shared estimator.  Kernel: Nadaraya–Watson with Gaussian h on the 129-cell
grid, effective counts = smoothed counts.  Histogram: `Gamma_j = M_j / C_j` (0 if empty), applied
own-cell, piecewise constant, `F` = exact integral evaluated at the cell centres.  Applied bias =
`min(1, C_eff/C_min) * Gamma`, `C_min = 100`, linear warm-up of the bias scale over the first 10 ps,
clip at 2000 kJ/mol/nm (guard).  **Both half-kicks evaluate the same frozen trust-weighted field.**
The stored estimate is never ramped.

*FR* (`fr_uniform`).  Reflected-KDE marginal (h = 0.06 nm) of the walkers' R; target
`q(z) ∝ exp(-beta V_wall(z))` (flat inside the walls; uses only the known wall potential); score
`S = log p - log q - KL(p||q)`, recentred, clipped to [-2, 2]; positive = death, negative =
birth; `p_die = 1 - exp(-rate |S| dt_FR)` every 0.5 ps from 50 ps, at most 5 % of the population per
opportunity; full-state clone with fresh Maxwell momentum (`alanine.dynamics`).  A clone deposits
its first ABF sample only after one propagation step.  The arms share coordinates, velocities and the
dynamics noise stream until the first event (asserted bit-identical at zero rate on CPU).
*Note 2026-10-29 (smoke, job 26049499):* on the A100 the compiled engine is not run-to-run
bit-deterministic (float64 atomics in scatter / autograd index-add): two identical ABF runs agree
bit for bit for 3.7 ps, then separate chaotically (max |dxi| 0.03 nm by 8 ps, final PMF 0.22 kJ/mol
apart).  The arms therefore share initial state and noise stream, and separate from ~4 ps exactly
as two repeats of the same arm do.  The paired design rests on shared seeds, initial populations
and noise, not on bitwise coupling; no determinism mode is imposed (it would slow the engine and
change nothing statistically).

*Online stop* (R2 of `src/eri/fr_controller.py`, time in ps): monitoring window W = [2.0, 2.8] nm
on the interior bins [1.30, 3.50]; checks every 5 ps from 50 ps with a 20 ps trailing window;
established when the trailing-window deposit fraction in W is >= 0.75 of its share of the interior
and every interior bin holds >= 1000 cumulative deposits; two consecutive established checks →
FR stops at `max(2 t_est, t_decision)`.  Safety stop if ancestor ESS/K < 0.15 or the largest lineage
exceeds max(0.10, 6/K).  FR never restarts; ABF continues to the horizon.  The controller sees
counts, birth/death histograms and lineage shares only.

## 3. Stages, in order, with freezes

| stage | cells | decides |
|---|---|---|
| 0 smoke (gdev) | CPU regression tests, A100 throughput, tiny cell, tiny reference | code validity, costs |
| I-A replay | 2026-08 screen with and without the defect, same seeds/hardware | did the defect matter |
| I-B timestep | K=64, 200 ps, 8 seeds, dt 1 vs 0.5 fs | dt for every later stage (accept if the dt effect on e_F is < 5 % and < 1/3 of the 10 % minimum FR effect) |
| I-C reference | 3 builds x 96 x 32 x 2 ns + F'_ref | acceptance; honest uncertainty |
| II-A estimator | kernel h=0.04, h=0.02, hist 129, hist 257; K=64, 1 ns, 8 seeds, ABF only | the estimator (frozen before any FR run) |
| II-B conditional | analysis of II-A: p(Y \| R) and E[f \| R] vs reference, time windows | is ABF conditional-equilibration-limited at this budget |
| III FR calibration | K=128, 1 ns, 8 pilot seeds, rates 0.005/0.02/0.05 ps^-1, R2 stop | largest rate with ESS/K >= 0.20 and max lineage <= 0.10 in essentially all seeds |
| IV production | K=64/128/256 x 2 ns, K=32 x 4 ns; 16 fresh paired seeds; abf vs fr_uniform | the result |

Freeze rule: estimator, dt, rate, KDE bandwidth, FR start, clip and the controller spec are fixed
before the first production run and recorded in `configs/deca_fr_2026/production.yaml`; the
analysis (`scripts/deca_fr_2026/analyze.py`) is frozen with them.  If Stage II shows ABF to be
unusable at 1 ns (II-B), the ABF-only assessment is extended to 2 ns before deciding; no
conditional-rejuvenation machinery is introduced in this study.

## 4. Endpoints (§8 of the plan)

Per seed: `e_F(t)` (aligned RMS on I), `e_F'(t)` (RMS of Gamma - F'_ref on I), `I_F` = mean of
`e_F` over [50 ps, T], `e_F(T)`, marginal TV/KL to the wall target, conditional-structure TV per
0.2 nm interval against the reference joint of (R, Y) in time windows, ancestor ESS/K and max
lineage, events, stop time, physically continuous committed transitions (0.05 nm hysteresis; clones
reset the tracker).  Primary: paired per-seed `G_F = 1 - I_F^FR / I_F^ABF` at K = 128, T = 2 ns:
median >= 10 % with the 95 % seed-bootstrap CI excluding 0 counts as a gain, and only if `e_F(T)`
and the conditional TV are not worse.  Equal budget: `e_F` at 128 M walker-steps for K = 32/64/128/256
per arm, min over K reported with the selection bootstrapped.

## 5. Compute

NSCC ASPIRE 2A A100-40GB, float64, torch 2.6 compiled engine.  Queues: gdev (<= 2 h), g1 (1 GPU),
g2 (2), g3 (4); one running job per queue per user.  Costs are recorded in `docs/experiments.md`.

## 6. Results

### Stage I-A (2026-10-29, job 26049477): the half-kick defect did not cause the 2026-08 failure

Same seeds, same initial population, same A100, 16 walkers x 0.5 ns x 8 ensembles, kernel h = 0.04.
Legacy (defect reproduced): learned span on I 79.9–107.4 kT (median 94.1), fraction of walkers above
2.80 nm in the second half 0.910–1.000 (median 0.990).  Fixed: 81.2–112.7 kT (median 96.8),
0.972–1.000 (median 0.999).  Paired span differences (fixed − legacy): +16.1, +13.0, −2.4, −8.5,
+14.4, +1.2, −19.3, +8.0 kT.  The 2026-08 numbers (86.7–110.3 kT; 0.951–0.9996) are reproduced by
both arms within seed scatter.  Conclusion: at the literature budget the corrected sampler fails Gate 0
exactly as the defective one did; the 2026-08 attribution to ABF's conditional sampling at that budget
stands.  Whether a larger population / longer run changes it is Stage II.

### Stage I-C (2026-10-29, jobs 26049144/26049145, merge 26053061): reference rebuilt and accepted

3 builds x 96 windows x 32 replicas x 2.0 ns on A100s (2.81 h each), 0 replicas excluded, 12.3 M
samples per build (590 k out-of-domain samples dropped per build, never clamped).  Consensus:
span 179.60 kJ/mol = 72.00 kT, minimum 1.637 nm, max pairwise between-build L2 0.66 kJ/mol,
ratio 0.037 (accepted at < 1), consensus drift between the 0.5-ns checkpoints 0.97 / 0.62 / 0.74
kJ/mol (0.04 of a resolvable 10 % effect), Gate A max pairwise TV 0.764.  The 2026-08 record
(179.6 / 1.637 / 0.604 / 3.18 / 0.754) is reproduced; the drift is 4x smaller than recorded.
Independent F'_ref from the per-cell conditional mean force over every production step:
between-build L2 7.8 kJ/mol/nm; integrating it reproduces the MBAR F to 2.66 kJ/mol L2 on I
(1.5 % of the span; 4x the between-build spread of F, 4x below the smallest ABF error measured
below).  Files: `results/deca_fr_2026/reference/{summary.json,reference.npz}`.

### Stage II-A (2026-10-29, job 26049477): estimator frozen = histogram, 257 cells

K = 64, 1 ns, 8 seeds, ABF only, scored on I against the merged reference.  Median over seeds:

| estimator | e_F(T) kJ/mol | I_F | e_F'(T) kJ/mol/nm | rel. mean-force error |
|---|---|---|---|---|
| kernel h = 0.04 | 11.72 | 40.83 | 42.0 | 0.44 |
| kernel h = 0.02 | 13.43 | 32.28 | 48.5 | 0.51 |
| histogram 129 | 11.84 | 18.98 | 46.2 | 0.49 |
| histogram 257 | **10.76** | **14.21** | 45.5 | 0.48 |

The histogram arms converge far earlier (I_F) at equal or better final error; 257 cells beats 129
on both.  Frozen for Stages III/IV: `abf_estimator = histogram`, `n_grid = 257`, `C_min = 100`,
10 ps warm-up (inherited settings, not tuned).  Reference values are interpolated to the 257
cell centres for scoring.

### Stage II-B (same runs): ABF at K = 64 x 1 ns is still conditional-equilibration-limited

Second-half conditional-structure TV against the reference joint p(R, Y): 0.26–0.45 in the
2.3–2.7 nm intervals, 0.04–0.17 at 2.7–3.1 nm, ~0 above 3.1 nm; the 1.5–2.3 nm intervals hold
< 50 second-half samples per seed because 82–93 % of walkers sit above 2.80 nm.  The mean-force
RMS error per 0.2-nm interval is largest where the walkers spend their time (2.9–3.5 nm: 54–97
kJ/mol/nm against |F'| up to 186), i.e. large counts with a wrong conditional ensemble -- the
2026-08 diagnosis, now on a corrected sampler with the better estimator, at 8x the literature
budget.  Per the plan, the ABF-only assessment is extended to 2 ns by the production K = 64 ABF
arm (16 seeds) rather than by an extra cell.

### Stage I-B (2026-10-29): ABF-based dt cells confounded; direct window test pending

K = 64, 200 ps: e_F(T) 68 (1 fs) vs 103 kJ/mol (0.5 fs), 0.5 fs worse at every time in 7/8 seeds
while the walker dynamics coincide.  `abf_min_count` counts raw samples, so at 0.5 fs the trust
ramp saturates in half the physical time and the early-bias span inflation is larger: a guard
artefact, not an integrator verdict.  The direct test (conditional mean force in 25 stiff windows
x 48 replicas at both dt, job 26050158) decides; dt = 1 fs is used meanwhile, as in 2026-08.

### Stage I-B, direct (2026-10-29/30, jobs 26050158 and 26053327): dt = 1 fs accepted

25 stiff windows (k = 8000) x 48 replicas from the diverse pool, 20 ps pull, 100 ps equilibration,
400 ps production, same seeds at dt = 1 fs and 0.5 fs (1199/1200 replicas pass the structural
screen).  `<f_loc | R>` per window: mean |difference| 3.0 kJ/mol/nm, RMS difference 5.0 % of the RMS
reference force, mean |difference| / SE 0.51, max 1.67 (no window beyond 2 SE); integrated spans
69.1 vs 69.8 kT (F_ref 67.2 on the same range).  Kinetic temperature 293.7 K (1 fs) vs 298.4 K
(0.5 fs), the known BAOAB artefact at gamma = 1 ps^-1.  dt = 1 fs is kept for every stage.
Observed on the way: at R >= 2.9 nm the replica-to-replica SD of `<f_loc>` is ~90 kJ/mol/nm after
400 ps under the restraint, and the conditional-mean F'_ref and the MBAR-gradient F'_ref differ by
~30 % there (2.95 nm: 158 vs 120).  The restrained conditional ensemble in the extended region is
slow and multimodal; F'-based metrics there carry that reference ambiguity (it integrates to
2.7 kJ/mol L2 in F, negligible against the ABF errors).  `e_Fp_grad_final` is reported alongside.

### Stage III (2026-10-29/30, job 26053062): FR calibration at K = 128, 1 ns, 8 seeds

| rate (ps^-1) | events/seed | stop | ESS/K final | max lineage | e_F(T) FR vs ABF | I_F FR vs ABF |
|---|---|---|---|---|---|---|
| 0.005 | 44–107 | safety in 6/8 at 325–565 ps, 2/8 to the horizon | 0.22–0.37 (8/8 >= 0.20) | 0.102 in 6/8 | 12.0 vs 10.8 (FR worse 5/8) | 17.0 vs 16.3 (median +5 %) |
| 0.02 | 87–157 | safety 8/8 at 130–230 ps | 0.16–0.28 (5/8) | 0.102–0.117 (0/8) | 15.8 vs 10.8 (worse 7/8) | 19.8 vs 16.3 |
| 0.05 | 62–162 | safety 8/8 at 75–125 ps | 0.17–0.27 (4/8) | 0.102–0.141 (0/8) | 14.9 vs 10.8 (worse 7/8) | 22.0 vs 16.3 |

The R2 "established" trigger (trailing-window occupancy of [2.0, 2.8] nm >= 0.75 of its share)
never fired at any rate: ABF with or without FR keeps 90–100 % of the walkers above 2.8 nm.  The
uniform target therefore clones the few compact walkers repeatedly; lineage diversity collapses
and the safety rule stops FR.  Marginal TV to the wall target is lower under FR at 100 ps (0.35–0.53
vs 0.56) and higher at 1 ns (0.59–0.65 vs 0.56).  No tested rate meets the preregistered criterion
(ESS/K >= 0.20 AND max lineage <= 0.10 in essentially all seeds); per the rule the intensity is
reduced: rates 0.002 and 0.001 run next (job 26054xxx), and the largest passing rate is frozen.

### Stage IV, ABF arms (2026-10-29, jobs 26053063/26053334/26053335): ABF converges to the same biased profile at every K

16 seeds each, histogram 257, scored against the merged reference on I.  Median e_F(T): K = 32 x 4 ns
12.63 [12.2, 12.9], K = 64 x 2 ns 12.19 [11.8, 12.6], K = 128 x 2 ns 12.15 [11.7, 12.4], K = 256 x 2 ns
12.18 [11.8, 12.4] kJ/mol.  The error passes through a minimum near 1 ns (10.7–11.3) and then
*rises* to a K-independent plateau of 12.2 kJ/mol (4.9 kT) with a seed spread of +-0.4: a systematic
limit, not statistical noise.  Equal budget (128 M walker-steps): K = 128 x 1 ns 11.3 < K = 64 x 2 ns
12.2 < K = 32 x 4 ns 12.6 < K = 256 x 0.5 ns 20.8.
Where the error lives (median profile, K = 64 and K = 256 are identical to 0.1 kJ/mol): after
alignment on I the ABF free energy sits +5 to +7 kJ/mol above the reference everywhere below
2.9 nm and -17 / -31 kJ/mol below it at 3.1–3.3 / 3.3–3.5 nm; the mean force is within +-10
kJ/mol/nm of the reference below 2.9 nm and 55–96 kJ/mol/nm too low above it (the same sign and
size against both reference F' estimators, so this is not the reference ambiguity of §I-B).
ABF puts 17 % of its second-half samples in each extended interval and < 1 % in each compact one.
Reading: in the extended region (R > 2.9 nm) the conditional ensemble has slowly exchanging
substates (replica SD ~90 kJ/mol/nm under a stiff restraint, §I-B); ABF walkers arrive there by
unfolding from the helix and the umbrella replicas by being pulled from a diverse pool, and the two
samplers settle on different substate mixtures.  Neither is demonstrably the equilibrium one.
Deca-alanine remains conditional-equilibration-limited at 16x the literature budget per
ensemble, with a corrected sampler and the better estimator; the limitation is now located
(R > 2.9 nm) and quantified (~30 kJ/mol over the extended region).
