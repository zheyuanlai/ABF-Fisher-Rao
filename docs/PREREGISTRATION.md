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
