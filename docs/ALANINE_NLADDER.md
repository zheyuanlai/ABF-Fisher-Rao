# Alanine dipeptide: the walker-count ladder (the last lever for an establishment-limited marginal)

**Date:** 2026-10-01. **Branch:** `main` (uncommitted). **GPU:** 3 only. **Status:** design, predictor,
rules and predictions written BEFORE any run.

## 1. Question

Every alanine knob is closed (estimator, ABF/FR burn-in, dose, start, incomplete CV, temperature,
target support, short budget; `docs/ALANINE_HISTOGRAM_ABF.md`, `ALANINE_1D_CV.md`, `ALANINE_LOWT.md`).
The 150 K study measured WHY: ABF establishment on this system is count-limited -- the time to
accumulate c_min samples per cell of a ~T-independent kJ/mol landscape -- and with N = 2048 walkers
that takes ~7 ps of a 100 ps run. The one variable that mechanism leaves is samples per cell per
unit time, N x 1000 / n_cells. This study lowers N to make the marginal establishment-limited within
the budget and asks whether uniform-target FR then accelerates ABF. It is a regime test with a stated
chance of a negative, not a rescue: starving N lengthens the estimator's trust time, not a physical
bottleneck, the FR score from a 64-walker KDE on a 9409-cell torus is noisy, and the ZIF-8 lesson
(inadequate counts equalised = harm) applies.

## 2. Design (300 K, the accepted reference; everything else = the selected 300 K configuration)

Ladder N in {2048 (existing: `results/alanine_histogram/hA_c800_w5`, `hu02_t0`, `hu15_t0`), 256, 64}.
Per new N: `abf` (the predictor arm), `u02_t0`, `u15_t0` (torus-uniform FR from 0.5 ps, rates 0.02 /
0.15). Histogram estimator, adaptive boxes, c_min 800, 5 ps ramp; kde 0.15, score_clip 2,
max_event_fraction 0.05 (N = 256: 12 events per opportunity cap; N = 64: 3), fr_every 500, 6 ps
age window; 100 ps; seeds 0-15; `rng_seed` 20260903; init C7eq thermalised 20 ps and cached PER N
(`results/alanine_nladder/init_c7eq_seeds0-15_N<N>.npz`). Stage names `N256_abf`, `N256_u02_t0`, ...
Launch order: N256_abf, N64_abf, N256_u02_t0, N256_u15_t0, N64_u02_t0, N64_u15_t0.
Analysis `scripts/analyze_alanine_lowT.py --root results/alanine_nladder --ref results/alanine/reference/reference.npz`
(matched ABF = same N, estimator, c_min, warm-up).

## 3. Frozen rules

Predictor on each `abf` arm (read before its FR arms): t_est = first save with >= 95 % of the 8 kT
mask cells at raw count >= 800 and C7ax visited; t_F = first save with the error within 2x its final
value; ESTABLISHMENT-LIMITED iff t_est >= 30 ps and t_est > t_F (t_est censored at 100 ps counts as
>= 30 ps). FR verdicts: `docs/FR_START_TIMING.md` rules on the raw read-out (300 K floor 0.559),
own window [1, 100] ps; genealogy floors ESS_age >= 0.30, max share <= 0.05, events < 5 % of N per
opportunity (at N = 64 the ESS floor means >= 19 independent lineages).

**Reading rule.** A positive counts only if (i) the regime call on that N is ESTABLISHMENT-LIMITED,
(ii) the own-window median <= -10 % with CI < 0, (iii) the endpoint clause and floors hold, and
(iv) the effect is larger at the more starved N (monotone in t_est). A positive at N = 64 with floor
violations is reported as such, not as a rescue. A null or harm at both N closes the alanine example
as the paper's atomistic neutrality control, with the count-limited mechanism as the explanation.

## 4. Predictions (recorded before any run)

- P1 t_est: N = 256 -> 40-70 ps (ESTABLISHMENT-LIMITED); N = 64 -> censored at 100 ps.
- P2 (the hypothesis) at N = 256, rate 0.02: ACCELERATION_POSITIVE. Stated prior: at most even odds.
- P3 at N = 64: the FR score is noise-dominated; NEUTRAL or HARMFUL, floors likely violated at rate 0.15.
- P4 if P2 holds, the gain is larger at N = 64 than at N = 256 only if the floors hold there; otherwise
  the ladder shows a maximum at intermediate starvation, which is the ZIF-8 pattern.
- P5 ABF alone gets worse monotonically with smaller N on every endpoint (the regime is real).

## 5. Amendment A1 (2026-10-01T04:40Z; the N = 256 predictor read, NO FR arm read -- u02/u15 still running)

**Predictor on `N256_abf`:** t_est = 37 ps (N = 2048: 7 ps), t_F = 12 ps, C7ax first hit 4.9 ps, 0/16
censored; ABF raw error 5.73 at 5 ps -> 0.725 at 20 -> 0.554 at 100 (floor 0.559), I_F W1 77.2
(N = 2048: 75.3). P1 holds (t_est ~ 5x longer at 8x fewer walkers) and the rule as written calls it
ESTABLISHMENT-LIMITED.

**The rule's inequality is in the wrong direction, and this is caught before any FR arm is read.**
The project's own ZIF-8 result (closed 2026-08-31, HARMFUL +3.7 %, 14/16) had exactly this
signature: F converged (54 ps) BEFORE the marginal established (77 ps), and the lesson recorded then
was that ABF needs ADEQUATE per-cell counts, not equal ones, and that the predictor must test whether
the ERROR is allocation-limited. "t_est > t_F" means the error has converged before the counts are
adequate, i.e. the count threshold is NOT what limits the error, so reallocating counts cannot help
it. The favourable regime is the opposite ordering: the error keeps falling until establishment
completes, t_F >= t_est. At N = 256 the error is within 2x of its final value at 12 ps while the
counts reach 800 per cell only at 37 ps (the adaptive boxes pool coarse cells meanwhile), and the
integrated error is within 3 % of the N = 2048 arm's: the error is NOT allocation-limited.

Corrected reading (applies to this study and to the 150 K study, where t_est = 7 ps makes the call
FAST under either rule): ESTABLISHMENT-LIMITED (favourable) iff t_est >= 30 ps AND t_F >= t_est;
"COUNT-STARVED, ERROR-CONVERGED" (the ZIF-8 pattern) iff t_est >= 30 ps and t_F < t_est. N = 256 is
the latter. **Revised prediction for the N = 256 FR arms, recorded before they are read: NEUTRAL at
rate 0.02, NEUTRAL or HARMFUL at 0.15.** The analyzer now prints both calls. The verdict rules and the
reading rule of section 3 are unchanged; only the regime label is corrected.

## 6. Results and verdict (all six arms; read 2026-10-01T05:15Z; 3.6-4.9 ms/step, 6-8 min per arm)

Tables: `results/alanine_nladder/analysis/scoreboard.md`, `fr.csv`, `summary.json`; figure
`figures/lowT_curves.*`. N = 2048 rung from the 300 K histogram study.

**Predictor (corrected reading, amendment A1):**

| N | t_est | t_F | C7ax first hit | ABF I_F W1 | ABF final | KL(p||U) at 100 ps (finite-N floor log(9409/N)) | call |
|---|---|---|---|---|---|---|---|
| 2048 | 7 ps | 14 ps | 4.4 ps | 75.3 | 0.544 | 1.67 (1.52) | FAST |
| 256 | 37 ps | 12 ps | 4.9 ps | 77.2 | 0.554 | 3.62 (3.60) | count-starved, error-converged |
| 64 | censored (> 100 ps) | 15 ps | 7.1 ps | 90.6 | 0.571 | 4.99 (4.99) | count-starved, error-converged |

Starving N lengthens the count-establishment time as designed (P1 PASS) and degrades ABF only
mildly (P5 PASS: +3 % and +20 % integrated, endpoints at the reference floor), but the error
converges by 12-15 ps at every N: the error is never limited by per-cell adequacy, because the
adaptive boxes pool coarse cells until the fine ones fill. The walker marginal's KL to uniform sits
exactly on its finite-N floor at every N, i.e. the instantaneous marginal of N walkers on 9409 cells
cannot be uniform, and that floor, not any landscape feature, is what the FR score reads.

**FR vs matched ABF (own window [1, 100] ps, raw read-out):**

| arm | rate | dI_F own | final | events/opp (cap) | ESS min | verdict |
|---|---|---|---|---|---|---|
| N256_u02_t0 | 0.02 | -0.75 % [-1.42, +0.09] 10/16 | -0.59 % | 0.4 (12) | 0.90 | NEUTRAL |
| N256_u15_t0 | 0.15 | +0.47 % [-0.66, +1.80] 5/16 | -0.15 % | 3.0 (12) | 0.62 | NEUTRAL (max-share floor violated) |
| N64_u02_t0 | 0.02 | +5.61 % [-1.80, +7.59] 6/16 | +0.21 % | 0.1 (3) | 0.86 | NEUTRAL (floor violated) |
| N64_u15_t0 | 0.15 | +2.15 % [-2.51, +5.32] 7/16 | -0.07 % | 0.5 (3) | 0.67 | NEUTRAL (floor violated) |

No arm approaches ACCELERATION_POSITIVE; time-to-accuracy speed-ups are 0.94-1.00; the error
ratio never drops below 0.96. At N = 64 the frozen dose fires 0.1-0.5 events per opportunity (the
score is noise on a 64-walker KDE) and a single clone already breaches the 5 % lineage-share floor.
P2 (the hypothesis) FAILS; P3 PASS (noise-dominated, floors violated); P4 moot.

**Verdict.** The walker-count ladder does not rescue the example. Lowering N creates count
starvation but not an allocation-limited error, exactly the ZIF-8 pattern the corrected predictor
named before the FR arms were read, and uniform-target Fisher-Rao is neutral at every N. With the
estimator, burn-in, dose, start, CV completeness, temperature, target support, budget window and
walker count all tried, vacuum alanine dipeptide on (phi, psi) is CLOSED as the project's atomistic
neutrality control: ABF is sufficient, the mechanism is count-limited establishment on a
~T-independent kJ/mol landscape whose error converges before the counts do, and marginal
reallocation has nothing to act on. The reusable product of the five alanine studies is the
corrected predictor: FR can only help where the error keeps falling until establishment completes
(t_F >= t_est), which is the gateway/WCA/LTA signature and not alanine's.
