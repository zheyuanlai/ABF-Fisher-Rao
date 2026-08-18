# PREREGISTRATION — WCA dimer stage (frozen 2026-08-18, before any WCA SHUS run)

Continuation of the gateway campaign (`PREREGISTRATION_GATEWAY.md`, outcome recorded
there). The WCA stage asks whether the mechanism established on the gateway —
*temporary directed marginal population reallocation damps an establishment
transient of an accumulating ABP* — survives in a many-body system where a clone
copies its entire solvent environment.

## References (hard rules)

* `e_F` may be scored ONLY against the corrected high-precision reference
  (hp_v3, label "HP reference v2..."), which exists for cell **b1h2** alone.
  `wca.load_reference` refuses any other cell, grid, or label. No silent fallback.
* The superseded phase-tier TI profiles for b2h6 / b4h1 are permitted ONLY as
  gate-tolerance proxies (the KL* term of D_tol). They never score `e_F`.
  If a non-b1h2 cell becomes FR-eligible, an hp-grade reference is built for it
  BEFORE any confirmatory scoring; stored `pmf_t` makes rescoring free.

## Stage 4: plain-SHUS screen (NO FR)

* Cells (selected by the old campaign for reasons unrelated to SHUS+FR):
  **b1h2** (beta=1, h=2; previously "starved"), **b2h6** (beta=2, h=6),
  **b4h1** (beta=4, h=1; previously "easy"). Old ABF regime labels are ignored.
* 8 seeds per cell (0..7), K=1024 replicas, dt=2e-3, T=500, block=20 steps,
  `eps_bw = 0.025`, `eta_bw = 0.07` (floors from `mollified_marginal_floor`
  printed and recorded before the run is interpreted).
* Gates, identical in form to the gateway (frozen there before any FR data):
  * `T_hit`: first persistent time the stretched region (xi > 0.75) holds >= 1
    replica (hold 0.05);
  * `T_est`: trailing-window MEDIAN of `D_t = KL(p_hat||u) <= D_tol`, hold 0.10,
    `D_tol = 1.5 x (KL*(cell, eps_bw) + noise95(K, eta_bw))`;
  * **eligible for FR** iff median `T_hit/T <= 0.2` AND median establishment gap
    `(T_est - T_hit)/T >= 0.25`.
* If all three cells are SHUS-sufficient or discovery-limited: record the null,
  do NOT tune WCA until FR wins somewhere else. That outcome would mean the
  gateway mechanism is conditional, which is itself a result.

## Stage 5 (only if a cell passes): transfer discipline

* **The FR strength is TRANSFERRED from the gateway, not re-tuned:**
  `theta = 0.01`, stride = 10 adaptation blocks — frozen here, before any WCA FR
  run. Only the window is chosen from WCA SHUS-only data, by the same rule:
  `t_on >= Q90(T_hit)`, `t_off` inside the establishment interval, `< Q50(T_est)`.
  (A strength selected on a toy that transfers to a many-body system with only
  timing re-derived is the strong version of the claim. If it fails while the
  cell clearly has an establishment transient, a small WCA pilot MAY then be run,
  and the transfer failure is reported alongside it.)
* Five arms as on the gateway: shus / fr_temp / fr_persistent / sham / count
  (coarse 9 bins). >= 16 fresh matched seeds. Endpoints and thresholds identical
  to the gateway confirmatory — no WCA-specific definitions of "faster".
* Mandatory frozen-bias validation, and the clone-decorrelation diagnostic:
  track parent/child decorrelation in xi and in one solvent descriptor after
  cloning (the practical face of the Cov(S, K_perp) defect).

## STAGE-4 OUTCOME (2026-08-18, seeds 0-7 — recorded, not to be edited)

**All three cells are SHUS-sufficient; the preregistered null fires and the WCA
FR branch is closed without any WCA FR run.**

* Discovery is immediate everywhere (median T_hit/T = 0.003-0.005, even for
  b2h6's ~12 kT dimer barrier: SHUS floods it within a few time units).
* The marginal reaches its tolerance within t ~ 25-50 of T = 500 (T_est/T
  0.00-0.01 under the trailing-median rule; establishment gap ~ 0). The
  stretched-occupancy overshoot exists but resolves by t ~ 50 — the same
  transient shape as the gateway, ~50x shorter relative to budget.
* Interpretation (the outcome the prereg anticipated): the gateway mechanism is
  CONDITIONAL. Temporary directed reallocation helps an accumulating ABP where
  establishment is slow relative to the budget; on these WCA cells SHUS
  establishes almost immediately, so there is nothing for population damping to
  damp. This asymmetry vs the old ABF campaign (where WCA b1h2 was starved) is
  itself informative: SHUS's occupancy-driven flooding removes the population
  deficit that mFR-ABF exploited.
* Anomaly, documented (not affecting gates): b1h2's e_F(T) plateaus at ~0.16 kT
  (13x the mollifier floor) as a smooth monotone TILT of F_hat vs the hp_v3
  reference (+0.36 compact to -0.31 stretched), while the sampled marginal sits
  at the noise floor. SHUS flattens the marginal it actually samples, so
  F_hat -> F_dyn of the discretized clipped dynamics; the tilt measures
  F_dyn - F_TI, common-mode across any arms. Attribution (closed as far as it
  can be without a new reference build):
  - NOT our dt: halving left error (0.186 -> 0.177) and tilt slope
    (-0.63 -> -0.79) unchanged (dt_check.json);
  - NOT our force clip: 4x the clip at dt = 5e-4 left the slope unchanged
    (-0.90 -> -0.85, clip_check.json);
  - the remaining, consistent explanation is a PROTOCOL-FAMILY disagreement:
    hp_v3 integrates the smoothed constrained mean force sampled by naive
    per-step distance projection (no metric/Fixman weighting of the projected
    measure, its own dt fixed at 2e-3), while SHUS measures the occupancy free
    energy of the same engine. The old ABF campaign compared mean-force against
    mean-force, so this systematic canceled there; occupancy-vs-mean-force
    exposes it. Deciding which family is closer to the exact F would need an
    independent estimate (metric-corrected TI or umbrella-reweighted unbiased
    runs) — out of scope for the closed WCA branch, and irrelevant to paired
    arm comparisons and to the reference-free gates that produced the verdict.

## Stopping rule

Terminate the WCA branch if no cell passes the Stage-4 gate, or if transferred
temporary FR neither wins nor at least reproduces the gateway's sham/count
ordering on an eligible cell.
