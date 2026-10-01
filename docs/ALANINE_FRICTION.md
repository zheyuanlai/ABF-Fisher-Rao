# Alanine dipeptide: the friction ladder (making marginal establishment dynamics-limited)

**Date:** 2026-10-01. **Branch:** `main` (uncommitted). **GPU:** 3 only. **Status:** design, predictor,
kill rule and predictions written BEFORE any run.

## 1. Question and premise

The alanine (phi, psi) example is closed as ABF-sufficient on every knob tried (estimator, burn-in,
dose, start, CV completeness, temperature, target support, budget window, walker count; see
`docs/ALANINE_NLADDER.md` section 6). The mechanism measured along the way: ABF's establishment of
the marginal is COUNT-limited (c_min samples per cell of a ~T-independent kJ/mol landscape, ~7 ps
at N = 2048) and the error converges before the counts do (t_F < t_est whenever establishment is
stretched by starving N). The project's positives (gateway, WCA, LTA) have the opposite ordering:
the error keeps falling until the walkers have physically reached and populated every region, i.e.
establishment is DYNAMICS-limited. The LTA 80-300 K sweep turned a null positive because, in
overdamped Brownian dynamics, mobility sets the spreading rate; the alanine temperature move failed
because at gamma = 1 ps^-1 the torsions move inertially and spreading (~6 ps) barely depends on T.

Friction is the correct transfer of that mechanism: with gamma = 10-50 ps^-1 the backbone torsions
become diffusive, the spreading time across the flattened torus grows ~ gamma, the count rate per
cell is unchanged, and the free energy does not depend on gamma (the 300 K reference stands). The
stiff orthogonal modes still relax fast relative to a diffusive torsion, so no hidden slow
coordinate is created (the ZIF-8 / 1-D failure mode). Physically gamma ~ 50 ps^-1 is an implicit
solvent viscosity. Stated prior: roughly even odds; this is the last alanine experiment.

## 2. Design (300 K, N 2048, accepted reference, selected configuration)

Ladder gamma in {1 (existing: `results/alanine_histogram/hA_c800_w5`, `hu02_t0`, `hu15_t0`), 10, 50}
ps^-1. Stage 1 (now): `g10_abf`, `g50_abf` -- the predictor arms. Stage 2 (only where the predictor
is favourable): `g<γ>_u02_t0`, `g<γ>_u15_t0` (torus-uniform FR from 0.5 ps, rates 0.02 / 0.15).
Histogram adaptive boxes, c_min 800, 5 ps ramp; kde 0.15, score_clip 2, cap 0.05, fr_every 500,
6 ps age window; dt 1 fs (BAOAB is stable at gamma dt = 0.05); 100 ps; seeds 0-15; `rng_seed`
20260903; init C7eq thermalised 20 ps AT the run's gamma and cached per gamma
(`results/alanine_friction/init_c7eq_seeds0-15_N2048_g<γ>.npz`). Analysis
`scripts/analyze_alanine_lowT.py --root results/alanine_friction --ref results/alanine/reference/reference.npz`
(matched ABF = same gamma, estimator, c_min, warm-up).

## 3. Frozen rules

**Predictor** (corrected form, `docs/ALANINE_NLADDER.md` A1), read on each `abf` arm BEFORE its FR
arms are launched: t_est = first save with >= 95 % of the 8 kT mask cells at raw count >= 800 and
C7ax visited; t_cov = first save with >= 95 % of the mask visited at all; t_F = first save with the
error within 2x its final value. FAVOURABLE iff t_est >= 30 ps AND t_F >= t_est. FR arms are run
only for a gamma whose call is FAVOURABLE.
**Kill rule:** if neither gamma = 10 nor 50 is FAVOURABLE, no FR arm is run and alanine stays closed.
**Verdicts** for any FR arm: `docs/FR_START_TIMING.md` rules on the raw read-out (floor 0.559), own
window [1, 100] ps, floors ESS_age >= 0.30, max share <= 0.05, events < 5 % of N per opportunity.
A positive must also be monotone in gamma (larger at 50 than at 10) to be reported as a regime effect.

## 4. Predictions (recorded before any run)

- P1 spreading slows with friction: t_cov 6 ps (gamma 1) -> >= 20 ps (10) -> >= 60 ps (50); C7ax first
  hit 4 ps -> >= 10 ps -> >= 30 ps.
- P2 the error ordering flips: at gamma = 50, t_F >= t_est (FAVOURABLE); at gamma = 10 borderline.
- P3 (the hypothesis) at gamma = 50, rate 0.02: ACCELERATION_POSITIVE; rate 0.15 positive integrated
  with a larger endpoint cost; both larger than at gamma = 10.
- P4 the conditional stays fast: no heating, the hidden-coordinate signature of the 1-D study absent.
- P5 ABF alone converges more slowly with friction (the regime is real), endpoints still at the floor.

## 5. Results: predictor arms (read 2026-10-01T13:20Z; 17.4 ms/step; the kill rule applies)

| gamma (ps^-1) | t_cov (mask visited) | t_est (counts >= 800) | C7ax first hit | t_F | ABF error @5 / 20 / 50 / 100 ps | ABF I_F W1 | call |
|---|---|---|---|---|---|---|---|
| 1 (existing) | 5 ps | 7 ps | 4.4 ps | 14 ps | 5.7 / 0.73 / 0.59 / 0.54 | 75.3 | FAST |
| 10 | 5 ps | 7 ps | 3.5 ps | 16 ps | 7.4 / 0.90 / 0.62 / 0.55 | 83.2 | FAST |
| 50 | 5 ps | 8 ps | 4.1 ps | (0 ps; see below) | 9.7 / 6.09 / 3.49 / 1.89 | 377.2 | FAST |

**P1 FAILS: friction does not slow the spreading.** The mask is visited by 5 ps and C7ax is hit by
3.5-4.1 ps at every gamma. The reason is that under ABF the walkers are DRIVEN, not diffusing: the
bias gradient on a torsion is tens of kJ/mol per radian, and even the overdamped drift it produces at
gamma = 50 crosses a cell in well under a picosecond. The LTA ethane spread by thermal diffusion
through zeolite windows; alanine's walkers spread by the bias itself, so neither temperature nor
friction touches the establishment time. Three predictor arms (gamma 1, 10, 50) agree to within 1 ps.

**What gamma = 50 does change is the estimator's variance, not the marginal.** ABF's error is 5x worse
integrated (377 vs 75) and 3.4x the floor at 100 ps (1.89, still falling: 3.49 at 50 ps). The marginal
is flat (KL to uniform 1.68 at 100 ps = the gamma-1 value), every cell is trusted by 8 ps, and the
ensemble is not heated, so this is correlation-limited sampling: at high friction the orthogonal
degrees of freedom decorrelate slowly, the instantaneous forces deposited in a cell are strongly
autocorrelated, and the effective sample count per cell falls by ~gamma. (The t_F = 0 entry is the
rule's artefact when the final error is still within 2x of the initial one.) Counts are already equal
across the accessible region, so there is nothing for count reallocation to fix; the lever in this
regime would be decorrelation of the orthogonal modes (the project's fibre-relaxation line), not FR.
P2 FAILS; the clause t_est >= 30 ps fails at every gamma.

**Kill rule applied.** Neither gamma is FAVOURABLE, so no FR arm is launched and alanine stays closed.
This was the last lever: the five-study mechanism now reads in full -- on vacuum alanine (phi, psi)
the bias itself spreads the walkers in ~5 ps regardless of N, T or gamma; establishment is complete
long before the error converges only when sampling is starved (small N) or correlated (large gamma),
and in both of those cases the marginal is already as flat as the walker count allows, so marginal
reallocation has no target. The example is the project's atomistic neutrality control.
