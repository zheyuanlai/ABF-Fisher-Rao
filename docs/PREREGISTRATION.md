# Preregistration - RC-WFR-TI campaign

Frozen before the confirmation runs.  Outcomes are appended, never edited.

## Object

`RC-WFR-TI`: no adaptive bias.  Replica labels Z_i live in reaction-coordinate
space and evolve by a Wasserstein-Fisher-Rao flow toward the UNIFORM target;
each replica carries a physical fiber configuration relaxed by constrained MD;
F is reconstructed by thermodynamic integration of the conditional mean force.

## Endpoints (co-primary)

* `I_F`  = (1/B) * integral_0^B e_F(fe) d(fe), the budget-normalized integrated
  gauge-optimal L2 free-energy error over the eval window.  ACCURACY.
* `tau_eps` = force evaluations to first reach e_F <= eps and stay below it for
  the trailing 20% of the budget; `S_eps = tau_base / tau_arm`.  SPEED.

Secondary: `e_F_final`, KL(p_t||u), CV coverage, hidden-channel L1 error
`chan`, ancestry ESS and surviving-ancestor fraction, wall clock.

## Cost currency

FORCE EVALUATIONS.  All arms use the same N and n_steps.  Replica-exchange
energy evaluations are charged.  W steps / FR resampling / KDE / TI quadrature
are free in this currency and reported separately in wall clock.

## Estimator floor

Shared binned mean-force estimator, bw_mf = 0.02 on G = 361 => floor e_F = 0.0040.
No claim is made about differences at or below 2x the floor.

## Calibration protocol

Stage 1 screens each arm's own knobs on the CALIBRATION system (EB) with
calibration seeds (base seed 1000, 4 rows).  Every arm gets a screen; the
baselines (ABF ramp, SHUS gain, RE-TI exchange period) are screened at least as
hard as RC-WFR.  Winners are FROZEN, then Stage 2 re-runs them on FRESH
confirmation seeds (base seed 2000+) with no further tuning.

## Hypotheses and decision rules

Declared "supported" only if the paired median relative change in `I_F` is
<= -10% AND the 95% bootstrap CI upper endpoint is < 0, on >= 16 fresh seeds.

* **H0 (mechanism).**  `wfr` beats `w_only` and `fr_only`.
  Prediction from Phase 0: FR alone cannot expand support and must fail
  outright; W alone equilibrates diffusively (O(L^2)); WFR fronts (O(L)).
* **H1 (vs adaptive biasing).**  `wfr` beats tuned `abf` and tuned `shus` on
  >= 2 systems.
* **H2 (vs classical stratification).**  `wfr` beats `ti_warm`, `ti_cold`,
  `reti_warm`, `reti_cold`.  THIS IS THE DECISIVE TEST: stratified TI already
  achieves uniform CV coverage by construction, so any RC-WFR advantage over
  ABF that TI also shows is an advantage of stratification, not of WFR.
* **H3 (validity).**  The identity lift's hysteresis bias stays below the
  estimator floor at the budget where H1/H2 are claimed; `wfr` must track
  `wfr_oracle` (exact conditional refresh).
* **H4 (geometry).**  `wfr` beats `w_count` (count balancing) and `w_sham`.
  PRIOR: H4 is expected to FAIL for the count control - a uniform-target FR
  step with histogram density IS count balancing.  A tie is the predicted,
  publishable outcome; only `w_sham` must lose.

## Pre-registered mechanism predictions (falsifiable)

P1. RC-WFR's advantage over ABF grows with CV domain size L, because ABF's CV
    equilibration is diffusive (O(L^2)) while W+FR is a front (O(L)).
P2. RC-WFR's advantage over RE-TI grows with the number of spectator dofs
    m_spec, because Hamiltonian-exchange acceptance decays with system size
    while the RC-WFR lift cost does not.
P3. RC-WFR degrades when tau_fiber is long relative to the conditional budget
    (`SLOWFIB`), via lift hysteresis; the degradation is removed by `lift=oracle`.

## Kill criteria

If H2 fails on every system - i.e. plain stratified TI or RE-TI matches or
beats RC-WFR everywhere - the method has no practical case over classical
stratification and the campaign reports that as its result.

---

# OUTCOMES (appended after the confirmation runs; the design above is not edited)

* **H0 (mechanism) — SUPPORTED.** `wfr` beats `w_only` (0.0361 vs 0.0765) and
  `fr_only` (1.037, coverage 0.067) on EB. In the calibrated probability-flow variant,
  removing Fisher-Rao costs a factor 2.4 (EB) and 2.5 (CHANNEL). Both halves are
  necessary, and the marginal-level prediction (W scales O(L^2), W+FR scales O(L)) was
  confirmed quantitatively in Phase 0.

* **H1 (vs adaptive biasing) — SUPPORTED, with a stated condition.** RC-WFR beats
  tuned ABF by 62.6% [-66, -58] on EB (probability-flow variant) and by 72-83% on the
  long-CV torsion systems; it beats SHUS by 1-2 orders of magnitude everywhere. It
  LOSES to ABF by 191% on the shortest CV domain tested. The margin is monotone in CV
  domain length, as predicted by P1.

* **H2 (vs classical stratification) — SPLIT.** RC-WFR beats cold-start RE-TI on the
  hidden-channel system and beats cold-start stratified TI on the easy system ONLY when
  given an exact analytic lift. Model-free, on an easy fiber, plain fixed-window TI
  still wins by ~40%. The preregistered kill criterion ("TI or RE-TI matches or beats
  RC-WFR everywhere") is NOT met.

* **H3 (validity) — REJECTED as stated, and the failure is the campaign's main
  scientific content.** The identity lift has a systematic bias floor 3-28x the
  estimator floor that grows with kappa, is independent of `n_cond`, and does not
  decrease with more compute. The deterministic probability-flow step reduces it to
  ~2x the floor by self-annihilating, but the bias remains extensive in fiber modes.

* **H4 (geometry) — REJECTED for the count control, UPHELD for the sham control**,
  as predicted. Count balancing ties smooth Fisher-Rao three separate times; the
  matched-turnover sham is 2.3x worse.

* **P1 (advantage over ABF grows with L) — CONFIRMED.** +191% at L=3, tie at L=6,
  -72% at L=12, -82% at L=24.

* **P2 (advantage over RE-TI grows with system size) — FALSIFIED.** The gap widens in
  the wrong direction (+33% -> +270% as m_spec goes 0 -> 128) because exchange
  acceptance decays slowly (0.975 -> 0.814) while the lift bias is extensive.

* **P3 (degradation when tau_fiber is long, removed by lift=oracle) — CONFIRMED.**
  `lift=oracle` reaches 1.0-1.1x the estimator floor at every kappa on both systems.

Two findings were NOT anticipated by this preregistration and are recorded as such:
the probability-flow form of the W step, and its incompatibility with FR resampling
without a resample-move jitter.
