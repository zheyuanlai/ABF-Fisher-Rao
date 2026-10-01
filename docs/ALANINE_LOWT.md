# Alanine dipeptide at low temperature: making the (phi, psi) marginal establishment-limited

**Date:** 2026-09-30. **Branch:** `main` (uncommitted, on top of the histogram and 1-D studies).
**GPU:** 3 only. **Status:** design, rules and predictions written BEFORE the 150 K reference or any arm ran.

## 1. Question and premise

Every alanine test so far was at 300 K on the complete (phi, psi) CV, and every one was a null for
uniform-target Fisher-Rao: ABF alone establishes the whole 8 kT mask by ~15 ps of a 100 ps run
(C7ax by 4 ps), so there is nothing left to redistribute. The project's positives (gateway, WCA,
LTA) share one signature: a marginal-establishment phase that occupies a large share of the budget
while the conditional stays fast. The LTA precedent is direct: a 300 K null became positive at every
temperature of an 80-300 K sweep, with the gain growing as T fell, diagnosed as marginal
establishment starvation. The user asks whether lowering T can put alanine in that regime.

Lowering T scales every barrier in kT (C7eq -> C7ax: 15.8 kT at 300 K, 31.6 kT at 150 K), so ABF has
to build a much taller bias before walkers cross and the establishment phase stretches. Vacuum
Ace-Ala-Nme's orthogonal degrees of freedom are stiff vibrations and methyl rotors, so the
conditional should stay fast down to ~150 K (the ZIF-8 failure mode, a slow conditional, is not
expected). This is a regime search, not a tuning: one temperature, a preregistered predictor read
on the ABF-only arm before any FR arm is read, and an honest closure either way.

## 2. Design (first temperature: 150 K)

**Reference.** `scripts/run_alanine_reference.py --temperature 150 --kappa 100` (umbrella kappa
halved so the restrained width sqrt(kT/kappa) equals the 300 K reference's 6.4 deg and the 24 x 24
window overlap geometry is unchanged; all other settings the accepted ones: 16 copies, 100 ps equil,
1000 ps prod, MBAR, n_grid 97), then `analyze_alanine_reference.py` (acceptance gates) and
`bootstrap_alanine_reference.py` (F_se = the read-out floor). Output `results/alanine_T150/reference/`.
The reference must pass the same acceptance gates as the 300 K one before any arm is scored.

**Arms** (N 2048, 100 ps, seeds 0-15, `rng_seed` 20260903, init C7eq thermalised 20 ps AT 150 K and
cached; histogram estimator, adaptive boxes, c_min 800, 5 ps ramp = the configuration selected by
the histogram study; FR knobs unchanged: kde 0.15, score_clip 2, cap 0.05, fr_every 500, 6 ps window):

| stage | method | FR start | rate | target |
|---|---|---|---|---|
| `abf` | abf | - | - | - (the PREDICTOR arm; read first) |
| `u02_t0` | fr_uniform | 0.5 ps | 0.02 | uniform on the torus (the closed target) |
| `u15_t0` | fr_uniform | 0.5 ps | 0.15 | uniform on the torus |
| `s02_t0` | fr_support | 0.5 ps | 0.02 | uniform on the VISITED support |
| `s15_t0` | fr_support | 0.5 ps | 0.15 | uniform on the visited support |

**The support target (new method `fr_support`).** The uniform torus target is unreachable on
alanine (sterically excluded area; KL floor 1.67 at 300 K, larger at 150 K where the accessible
region shrinks), so part of every FR event pushes walkers toward cells that cannot be populated.
`fr_support` uses q proportional to the indicator of cells whose cumulative ABF count is >= 1
(visited at least once), renormalised on the torus. It consults only the ABF count accumulator
(no reference, no bias, no EMA), is a constant of the algorithm once the accessible region is
visited, and equals `fr_uniform` exactly when every cell has been visited. It is NOT the
physical/EMA target ruled out in 2026-08. The engine records KL(p || q_support) alongside
KL(p || uniform). Existing methods stay bit-identical (fixture tests).

## 3. Frozen rules

**Read-outs and statistics** as in `docs/ALANINE_HISTOGRAM_ABF.md`: histogram arms scored on the raw
150 K reference (equilibrium weight, 8 kT mask), floor = bootstrap F_se; secondary km / sm read-outs;
own window [1, 100] ps, W1 [5, 100] ps; paired median + BCa CI, win rate; verdict rules of
`docs/FR_START_TIMING.md` (ACCELERATION_POSITIVE at median <= -10 % with CI < 0; NEUTRAL inside
+-10 % with |final| <= 5 %; HARMFUL when the CI lower bound > 0 or final > +5 % with CI > 0);
genealogy floors ESS_age >= 0.30, max share <= 0.05, events < 5 % of N per opportunity.

**Predictor (read on `abf` BEFORE any FR arm).** t_est = first time at which >= 95 % of the 8 kT mask
cells are trusted (count-based, from the saved trust fraction and the final counts) and C7ax has been
visited in every seed; t_F = first time the ABF error falls within 2x its final value. The regime is
called ESTABLISHMENT-LIMITED if t_est >= 0.3 x 100 ps (30 ps) and t_est > t_F; FAST otherwise.
Expectation, recorded now: FR can only be positive in the establishment-limited call; a positive in
the fast call would be unexplained and must be replicated on fresh seeds before it is reported.

**Ladder rule.** If 150 K is FAST and FR is null, go colder (100 K) with the same design; if 150 K is
establishment-limited and FR is null or harmful, the example is closed honestly (no further T).
If positive, replicate at 200 K (the milder side) to show the effect scales with the regime.

## 4. Predictions (recorded before any run)

- P1 the 150 K reference passes the acceptance gates with kappa 100 (overlap geometry unchanged).
- P2 ABF alone at 150 K is establishment-limited: C7ax first hit > 10 ps (300 K: 4 ps), t_est >= 30 ps.
- P3 the torus-uniform arms: rate 0.02 NEUTRAL; rate 0.15 NEUTRAL or HARMFUL (unreachable target).
- P4 the support-uniform arms: rate 0.02 ACCELERATION_POSITIVE (median <= -10 % on the own window,
  CI < 0) with the endpoint within +5 %; rate 0.15 positive integrated but with a larger endpoint
  cost. P4 is the study's hypothesis; its failure closes the example.
- P5 the FR gain, if any, scales with t_est: it must be larger at 150 K than the 300 K null (0 %) and,
  if 200 K is run, intermediate there.

## 5. Results at 150 K (read 2026-10-01T03:00Z; reference 2.55 h, five arms 17.0-17.5 ms/step)

**Reference (P1 PASS).** All five acceptance gates pass with kappa 100: 576/576 seeds, MBAR converged
(resid 9.8e-9, 133 iterations), minimum nearest-neighbour overlap in the 8 kT region 0.068 with 0 of
69 pairs below 0.03, median overlap 0.115, global minimum at C7eq (-74, +48 deg). kT = 1.247 kJ/mol;
8 kT mask = 805 cells (300 K: 2239); C7ax-C7eq = 6.20 kJ/mol = 5.0 kT (300 K: 8.53 kJ/mol = 3.4 kT);
raw read-out floor (bootstrap F_se, equilibrium-weighted) 0.216 kJ/mol (300 K: 0.559).
`results/alanine_T150/reference/`, `results/alanine_T150/analysis/scoreboard.md`, `figures/lowT_curves.*`.

**Predictor on the ABF arm: FAST. P2 FAILS.** t_est (95 % of mask cells at count >= 800 and C7ax
visited) = 7 ps, identical to 300 K (7 ps); coverage of the mask 6 ps; C7ax first hit 4.86 ps
(300 K 4.35), 0/16 censored; t_F = 30 ps. By the frozen rule the regime is FAST, so no FR effect was
expected, and none appeared.

| arm | target | rate | dI_F own | final | ratio min (t) | events/opp | ESS min | verdict |
|---|---|---|---|---|---|---|---|---|
| u02_t0 | torus | 0.02 | +1.41 % [-0.08, +2.23] 5/16 | +1.19 % | 0.995 (3 ps) | 3.5 | 0.90 | NEUTRAL |
| s02_t0 | visited support | 0.02 | +1.07 % [-0.45, +1.69] 6/16 | +0.26 % | 0.993 (3 ps) | 3.6 | 0.90 | NEUTRAL |
| u15_t0 | torus | 0.15 | +2.87 % [+1.42, +4.87] 2/16 | +3.28 % | 0.905 (3 ps) | 29.1 | 0.47 | NEUTRAL_SIG |
| s15_t0 | visited support | 0.15 | +1.91 % [+1.30, +3.15] 2/16 | +3.63 % | 0.907 (3 ps) | 29.5 | 0.46 | NEUTRAL_SIG |

Support vs torus target, paired at equal rate: -0.03 % [-0.90, +0.72] (0.02), -0.51 % [-1.51, +0.25]
(0.15): the two targets are the same arm. **Reason (a design flaw, recorded):** with
`fr_support_min_count = 1` every one of the 9409 cells, including the 2054 the reference never
visited, has been entered by at least one of 2048 walkers by 20 ps (49.6 % of the torus by 5 ps,
100 % by 20 ps, and every cell reaches 800 counts by 100 ps), so the "visited support" IS the torus
after 20 ps and KL(p || support) = KL(p || uniform) = 1.67 at the end. A support target needs a
threshold relative to the uniform expectation (e.g. 1 % of the mean count per cell), not 1 sample.
P3 PASS, P4 FAIL (the hypothesis), P5 moot.

**Why lowering T did not create the regime (the mechanism, learned here).** In vacuum the free
energy in kJ/mol is nearly temperature-independent (range 92.6 kJ/mol at 300 K, 79.9 at 150 K;
C7ax-C7eq 8.5 -> 6.2), and ABF's establishment is the time to accumulate c_min samples per cell of a
landscape measured in kJ/mol: with N = 2048 walkers depositing 2048 samples per step over a mask of
800-2200 cells, every cell reaches 800 counts within ~1 ps of being reached, and the walkers reach
the whole mask by ~6 ps at either temperature. ABF never waits for a thermal barrier crossing, so
the kT-scaling of barriers that made the LTA ethane (overdamped BD, mobility proportional to T)
establishment-limited at low T does not act here. Halving T halves kT and shrinks the mask but
leaves t_est at 7 ps; it also halves the reference floor (0.56 -> 0.22), which is why the ABF
endpoint looks better in absolute terms.

**Amendment A1 (2026-10-01T03:05Z, deviation from the ladder rule, labelled).** The frozen ladder
says FAST + null -> 100 K. That step is dropped: t_est is temperature-invariant by the mechanism
above (7 ps at both 300 K and 150 K), so 100 K would reproduce the same FAST call at the cost of a
5 h reference + arms. The quantity that controls establishment on this system is samples per cell
per unit time, N x 1000 / n_cells: the lever is the walker count N (or the budget), not T. Predicted
t_est ~ 7 ps x (2048 / N) for the count-limited part plus the ~6 ps spreading time: N = 256 -> ~60 ps,
N = 64 -> beyond a 100 ps run. This is offered as the next experiment, NOT launched: the user's
standing direction is to avoid spending GPU time on questions that are settled.

**Verdict at 150 K.** The alanine (phi, psi) example is FAST at 150 K exactly as at 300 K; uniform
Fisher-Rao is NEUTRAL at the gentle dose and NEUTRAL_SIG (worse by 2-3 %, ESS 0.47) at the high dose
with either target. Temperature is not the knob that saves this example.

**Note (2026-10-01T04:40Z, from `docs/ALANINE_NLADDER.md` amendment A1):** the predictor's inequality
"t_est > t_F" labels the ZIF-8 (harmful) ordering as favourable; the favourable regime is t_F >= t_est.
The 150 K call is FAST under either reading (t_est = 7 ps), so nothing here changes.
