# Gateway replica ladder at equal force-evaluation budget

*2026-10-04. Design frozen in `configs/gateway_replica_ladder/design.json` before any ladder run.
Engine `src/gateway_numba.py` (CPU port), tests `tests/test_gateway_numba.py`, runner
`scripts/run_gateway_replica_ladder.py`, analyzer `scripts/analyze_gateway_replica_ladder.py`, data and
figures under `results/gateway_replica_ladder/`.*

## 1. Question

The user's point: an N-replica run to time T spends N x T force evaluations, so the fair comparator
for N replicas to time T is one replica to time N T (and, more generally, any (N', T') with N' T' = N T).
Errors are plotted against the **normalised budget** u = b / B, where b = N x (steps so far).
The question is whether ABF + uniform Fisher-Rao (mFR) helps **more with more replicas or with
fewer**. The answer must be honest about whether a different (N, T) split of the same budget beats
both arms.

## 2. Design (frozen before data)

* Cell: the frozen gateway anchor (beta 16, beta H = 8 kT, s 0.1, r 32, dt 4e-4).
* Budget: B = 2048 x 100000 = 2.048e8 walker-steps **per arm**. Rungs are N = 2048, 1024, ..., 2, 1,
  with n_steps = B / N, so T runs from 40 (the accepted confirmatory shape) to 81920.
* Arms: histogram (P0) ABF and histogram ABF + uniform FR at 180 bins (primary) and 45 bins
  (continuity with the accepted confirmation). The four arms of one (N, seed) share initial
  conditions and Langevin noise slot by slot. FR is not defined at N = 1.
* FR knobs are frozen in physical time: gamma 1.5, eta 0.1, fr_every 10, clip 3, ramp 10000 steps
  (= the accepted int(0.1 n_steps) at the anchor), cap = max(1, floor(0.08 N)). The cap is identical
  to the accepted law for N >= 13; the accepted law would switch FR off at N <= 12.
* Init: all walkers start in the left basin (`one_right` is undefined at N = 1). 32 seeds per N
  (7100-7131).
* Metrics:
  * Primary: the normalised integrated error Ibar_F = (1/B) int_0^B e_F(b) db, taken as the mean over
    200 budget fractions.
  * Secondary: the final e_F(B), and u_eps = the first budget fraction after which e_F stays at or
    below the median ABF e_F(B) at N = 2048.
  * Effects are per-seed paired (FR - ABF)/ABF: median, bootstrap 95 % CI, and wins.
* Why 180 bins is primary: its e_F floor is 2e-8, against 0.0030 at 45 bins. Long, thin rungs could
  otherwise reach a floor that would hide FR-vs-ABF differences.

## 3. Engine validation (gate passed before reading the ladder)

* `tests/test_gateway_numba.py` (12 tests pass):
  * Init is identical to the torch engine.
  * ABF dynamics and the P0 accumulators are **bitwise** equal to `gateway_core.simulate_batch` on CPU
    when both are driven by the same noise stream (45 and 180 bins, 3000 steps).
  * The FR KDE and uniform score equal the torch code to round-off at N = 1, 3, 37, 2048.
  * The resampling law (events, cap, who dies or is copied, pool law) matches `resample_indices` in
    distribution for N = 8 to 64, including cap 0 and cap 1.
* Gate (a): the analyzer's scorer reproduces the accepted `l2_f_t` from the accepted accumulators
  (worst difference 8e-16).
* Gate (b): numba at N = 2048 on the accepted confirmation seeds (5300-5315, left init) vs the
  accepted GPU rows:

| | ABF I_F | ABF e_F(T) | FR effect, I_F (16/16 wins both) | FR effect, e_F(T) | FR events |
|---|---|---|---|---|---|
| GPU (accepted) | 0.943 | 0.00633 | -30.0 % [-31.0, -21.6] | -48.5 % | 6344 |
| CPU numba | 0.915 | 0.00617 | -25.8 % [-30.1, -20.1] | -43.7 % | 6326 |

  All four clauses pass.
* Cost: 30-40 s of one core per (N, seed) job with 4 arms, for every N. The whole ladder (384 jobs) took
  192 s wall on 100 processes. The GPU engine is dispatch-bound at ~0.9 ms per step, so N = 1 would
  take about 50 h on it.

## 4. A reference defect found on the way (post-hoc, class 9)

Every rung with N <= 256 ended at the same e_F(B) of about 0.0015 against the analytic F. That is a
deterministic floor, and the cause is the integrator. The Euler-Maruyama y-update at fixed x has
stationary variance 1 / (beta omega^2 (1 - omega^2 dt / 2)), so the sampler converges to

    <f | x> = 4 H x (x^2 - 1) + omega' / (beta omega (1 - omega^2 dt / 2)).

At the gate, omega_in^2 dt / 2 = 0.205 (a 26 % variance inflation). Integrated over x this gives an
RMS F error of **0.001383** against the continuum F. The 32-seed pooled accumulators measure
0.00137-0.00139 at N = 1, 4, 16 and 64 (a 1 % match).

Against this dt-consistent reference (`--reference em`), the pooled serial ABF error drops to 0.00011.
That is 1/sqrt(32) of the single-seed 0.00054, so this reference is the sampler's true large-sample
limit. Under the analytic reference the floor compresses every endpoint near it. That includes the
accepted confirmation, where the FR final effect at N = 2048 is -59 % (analytic) vs -70 % (EM), and
it hid the small-N FR harm reported below.

Both scorings are reported. The analytic one is preregistered; the EM one is post-hoc, labelled
POST-HOC in every figure, and is the right yardstick for a method comparison: both arms share the
integrator, and only the reference changes.

## 5. Results (180 bins, 32 paired seeds per N)

Effects are ABF + FR vs ABF, median paired change [bootstrap 95 % CI] (wins out of 32).

| N | T | ABF Ibar_F (EM / analytic) | dIbar_F, EM ref | dIbar_F, analytic ref | ABF e_F(B), EM | FR e_F(B), EM | d e_F(B), EM ref |
|---|---|---|---|---|---|---|---|
| 2048 | 40 | 0.0229 / 0.0230 | **-30.6 %** [-33.7, -28.0] 32/32 | -29.5 % 32/32 | 0.00531 | 0.00164 | **-69.7 %** 32/32 |
| 1024 | 80 | 0.0143 / 0.0145 | **-40.9 %** [-45.4, -36.7] 31/32 | -36.7 % 31/32 | 0.00310 | 0.00130 | -57.9 % 31/32 |
| 512 | 160 | 0.0080 / 0.0085 | **-36.9 %** [-43.8, -29.5] 31/32 | -31.1 % 31/32 | 0.00136 | 0.00090 | -37.6 % 23/32 |
| 256 | 320 | 0.0048 / 0.0054 | **-42.4 %** [-49.4, -32.5] 28/32 | -35.2 % 28/32 | 0.00096 | 0.00062 | -16.0 % [-46.9, +3.1] |
| 128 | 640 | 0.0023 / 0.0033 | **-34.0 %** [-42.7, -11.3] 26/32 | -26.2 % 28/32 | 0.00050 | 0.00057 | -18.6 % [-37, +41] |
| 64 | 1280 | 0.0017 / 0.0026 | -20.9 % [-37.1, +5.8] 19/32 | -12.6 % 21/32 | 0.00050 | 0.00055 | +18.6 % [-15, +46] |
| 32 | 2560 | 0.0013 / 0.0021 | -0.7 % [-26, +33] 16/32 | +4.6 % 14/32 | 0.00049 | 0.00052 | +0.7 % |
| 16 | 5120 | 0.0010 / 0.0017 | +21.3 % [-15, +59] 13/32 | +8.5 % 11/32 | 0.00034 | 0.00057 | **+49.6 %** [+18.8, +96.4] 7/32 |
| 8 | 10240 | 0.0011 / 0.0019 | -7.8 % [-23, +8] 19/32 | -8.5 % 21/32 | 0.00044 | 0.00046 | -1.5 % |
| 4 | 20480 | 0.0008 / 0.0018 | **+18.1 %** [+3.2, +34.0] 9/32 | -8.6 % 22/32 | 0.00037 | 0.00052 | **+56.7 %** [+5.4, +90.2] 10/32 |
| 2 | 40960 | 0.0009 / 0.0017 | **+21.0 %** [+9.9, +55.5] 9/32 | +6.7 % 10/32 | 0.00053 | 0.00049 | -5.6 % |
| 1 (serial) | 81920 | 0.0010 / 0.0018 | -- | -- | **0.00054** | -- | -- |

The full tables, including the 45-bin arm, u_eps, event counts and ESS, are in
`results/gateway_replica_ladder/scoreboard{,_emref}.md`. Figures are
`figures/fig_ladder_headline{,_emref}.png` (curves vs b/B, effect vs N, error vs N) and
`figures/fig_ladder_curves_h{180,45}{,_emref}.png` (one panel per N).

**Answer to the question: FR's benefit grows with the number of replicas.** On the integrated error it
is -31 to -42 % (26-32 of 32 seeds) for every N >= 128. It fades at N = 64 (CI spans 0), is neutral at
N = 8-32, and is harmful at N = 2, 4 and 16 under the dt-consistent reference (+18 to +21 % integrated,
up to +57 % final). The final-error benefit is confined to N >= 512 and grows to -70 % at N = 2048.

The mechanism at small N is visible in the event counts. With N <= 16, 80-100 % of FR events are
deaths: a KDE of a handful of walkers is dominated by each walker's own kernel, so S > 0 for almost
everyone. FR then degenerates into random resampling, which costs diversity and repairs nothing. The
seed-pooled FR endpoint (0.00019-0.00029 at N = 4-16, vs ABF's 0.00006-0.00009) shows the harm is
partly a **bias**, not only variance.

**The fairness question, answered.** At a fixed number of force evaluations, ABF itself improves
steeply as N falls. Ibar_F is 0.023 at N = 2048, 0.0023 at N = 128 and about 0.001 at N <= 16 (10-20x).
The final e_F is 0.0053 at N = 2048 and 0.0003-0.0005 at N <= 128.

So the 2048-replica ABF + FR run (Ibar_F 0.0154, e_F(B) 0.00164) is beaten by serial one-walker ABF
at the same budget: 15x on the integrated error and 3x at the endpoint. Serial ABF reaches the
2048-replica ABF endpoint after 1.3 % of the budget; 2048-replica ABF + FR needs 30 % of it. **In this
system mFR saves physical (parallel, wall-clock) time, not force evaluations.** It accelerates exactly
the short, many-replica runs that are an inefficient use of a fixed force-evaluation budget. Where the
budget is spent efficiently (few replicas, long T), establishment is no longer the bottleneck and FR
has nothing left to repair.

**The other axis (`figures/fig_ladder_physical_time{,_emref}.png`, `scripts/plot_gateway_replica_ladder_walltime.py`).**
The same runs plotted against physical time per replica t, which is wall-clock when the replicas run in
parallel:

* ABF's time to reach eps (= the median ABF e_F(B) at N = 2048) stalls at about 45 time units for every
  N >= 64. This is the establishment wall, and more replicas do not move it.
* ABF + FR cuts that time to 12-18 for N >= 128, i.e. 3-4x faster.
* For N <= 32 the two arms coincide.

So FR is a wall-clock accelerator for wide parallel runs, and it is that regime only. At a fixed
force-evaluation budget the same wide runs are dominated by narrow ABF.

Caveat for the wall-clock reading: with N replicas on parallel hardware the wall time is about T,
not N T. At N = 2048 the run needs 1e5 sequential steps; N = 1 needs 2e8.

## 6. Predictions (recorded in design.json before any ladder run)

| | prediction | verdict |
|---|---|---|
| P1 | FR's relative Ibar_F benefit grows with N | **PASS** (both references) |
| P2 | at N <= 8, FR is neutral or harmful | **PASS** (analytic: neutral; EM: harmful at N = 2 and 4, neutral at 8) |
| P3 | ABF alone improves as N falls from 2048 to ~128 | **PASS**, and it keeps improving down to N ~ 4-16 |
| P4 | serial N = 1 ABF ends more accurate than N = 2048 ABF + FR, **and** has the worst early-budget error | first half **PASS** (0.00054 vs 0.00164); second half **FAIL**: the serial walker has the *best* early-budget error, because every force evaluation advances the one walker's physical time and establishment is physical-time-limited |
| P5 | FR saves physical time, not force evaluations | supported |

## 7. What this does and does not say

* It is one 1-D-CV toy whose ABF is establishment-limited in physical time. A system where the
  per-walker exploration itself is slow (several transition paths unresolved by the CV, which is the
  regime where many walkers help ABF) could change the ABF-vs-N curve. The WCA dimer ladder is the next
  check (port and review in progress, 2026-10-04).
* FR's knobs are the anchor-tuned ones (gamma 1.5, eta 0.1, cap 0.08 N). A small-N-tuned FR (a wider
  KDE, or a leave-one-out score that removes the self-kernel bias) was not tried. The leave-one-out
  score is the obvious fix for the death-dominated regime.

## 8. Exploratory: is the small-N harm a score artefact? (post-hoc, `configs/gateway_replica_ladder/exploratory_score_fix.json`)

Two textbook fixes for the death-dominated small-N score were tested, each as an extra arm paired with
ABF on shared noise (32 seeds, N = 1024, 256, 64, 16, 4, dt-consistent reference):

* **empirical centring**: S = log p(x_i) - mean_j log p(x_j);
* **leave-one-out + empirical centring**: each walker's own kernel is removed from its density.

The data are in `results/gateway_replica_ladder/score_fix/` (`scoreboard.md`, `summary.json`) and the
figure is `figures/fig_score_fix_emref.png`. Tests `test_v4_loo_self_term` pass (15/15 in total).

Integrated-error effect vs ABF (wins out of 32). Death fraction = the share of FR events that are deaths.

| N | accepted score | empirical centring | LOO + empirical | death fraction (acc / emp / LOO) |
|---|---|---|---|---|
| 1024 | -32.3 % (30) | -33.5 % (31) | -32.7 % (32) | 0.53 / 0.50 / 0.50 |
| 256 | -41.5 % (28) | -38.0 % (26) | -36.0 % (29) | 0.56 / 0.50 / 0.50 |
| 64 | -27.7 % (22) | -13.4 % (20) | +6.7 % (15) | 0.63 / 0.50 / 0.50 |
| 16 | +23.9 % [+7.5, +36.9] (10) | +1.7 % (15) | +41.0 % [+3.4, +110] (10) | 0.80 / 0.50 / 0.70 |
| 4 | +7.0 % (13) | +26.8 % (13) | +22.7 % (13) | 0.96 / 0.50 / 0.60 |

* E1 (deaths fall to ~50 %): **PASS** for empirical centring (0.50 everywhere); partial for LOO (0.60-0.70).
* E2 (the small-N harm disappears): **FAIL**. No variant gives a benefit at N <= 16, and LOO is the worst
  at N = 16, with 3.4x the events and ESS/N 0.12.
* E3 (no fix creates a small-N benefit; the large-N benefit is unchanged): **PASS**.

So the small-N failure is not a centring artefact. With a long physical run the population is already
established, and resampling only costs diversity. Replication note: the accepted arm's integrated effect
at N = 1024 is -32.3 % here vs -40.9 % in the ladder (same seeds and init, different noise stream).
That gap is a reminder that a 32-seed median effect carries a few points of realisation noise beyond its
bootstrap CI.
