# PREREGISTRATION — Gateway campaign (DRAFT until Stage-1 freeze)

Status: **DRAFT**. Structure and thresholds below are fixed now; items marked
`[FREEZE @ Stage 1]` get their numerical values from SHUS-only calibration data and
are frozen — with a dated commit touching this file — **before any confirmatory FR
run exists**. No value in this file may change after that commit.

## Stages and gates

0. **Engineering validation.** `tests/` must pass (SHUS sign/gauge/reweighting/
   protection, FR finite-step law, resampling unbiasedness, checkpoint/resume,
   determinism). No scientific output.
1. **Plain-SHUS calibration.** Calibration seeds `0..7`. Calibrate ONLY SHUS
   quantities: `eps_bw`, `eta_bw`, grid, `block`, `K`. `[FROZEN 2026-08-18]`:
   `eps_bw = 0.02`, grid `n = 361`, `eta_bw = 0.10`, `block = 20`, `K = 1024`,
   `dt = 2e-4`. Basis (SHUS-only data; no FR run existed): mollified SHUS has the
   analytic fixed point `F* = -b^-1 log(K_eps * e^{-bF})` with a non-uniform
   stationary marginal; at `eps = 0.07` this put a ~1 kT bias floor and a
   KL* ~ 0.3 marginal floor under every betaH = 8 kT cell (observed plateaus
   matched the analytic prediction to <3%). `eps = 0.02` reduces the worst floor
   to ~0.13 kT / KL* ~ 0.08 (`gw.mollified_fixed_point`).
   Then record, per seed: `e_F(t)`, `D_t = KL(p_hat_t || u)`, cosine modes of
   `F_t - F_ref`, `T_hit`, `T_est`, round trips.
   * `T_hit`: first persistent time every region (minus/gate/plus) holds >= 1 walker
     (hold fraction 0.05).
   * `T_est`: first time the TRAILING-WINDOW MEDIAN of `D_t` is `<= D_tol` (hold
     fraction 0.1). Median rule frozen 2026-08-18 after the SHUS-only screen showed
     an all-saves rule trips on single-save KL spikes even at the noise floor.
   * `D_tol = 1.5 x (KL*(cell, eps_bw) + noise95)`: the analytic marginal floor of
     the mollified fixed point plus the 95th-percentile finite-K KDE noise floor
     (`abpfr.diagnostics.kde_noise_floor`, identical kernel) — both computable
     before any run.
   * Estimator-consistent error `e_eps(t) = ||F_hat_t - F*||` is recorded alongside
     `e_F(t)`; the primary accuracy metric remains `e_F` against the TRUE `F_ref`.
   * **Gate 1 (FR has a job) — FINAL RULE, frozen 2026-08-18 from SHUS-only data
     (no FR run on any cell existed):** early discovery, median `T_hit/T <= 0.2`,
     AND a substantial post-discovery establishment transient, operationalized as
     median establishment gap `(T_est - T_hit)/T >= 0.25` (the closed ABF campaign's
     gap convention). The draft absolute cutoff `T_est/T >= 0.4` was replaced
     because it is denominator-sensitive to the arbitrary run length T; the gap
     criterion is the quantity the hypothesis is actually about. Supporting (not
     gating) evidence recorded per cell: sign changes of `a_1(t)` after `T_hit` and
     ring-out time (last `t` with `e_F > 2 e*`).
   * **Gate-1 outcome (screen of 2026-08-18, seeds 0-7):** easy_A SHUS-sufficient
     (gap 0.08); mid_B gap 0.27 but settles to its floor by 0.31 T (flooding-
     flavored); cold_C gap 0.31, anchor_D gap 0.34 (3 a_1 sign changes, ring-out
     0.87 T), hot_E gap 0.27. **Frozen Stage-2 cell: anchor_D** (beta=16, H=0.5,
     s=0.10, r=32; strongest underdamped establishment transient). Frozen anchor
     thresholds: `eps* = 0.0110` (median final plain-SHUS `e_F`, energy units);
     ladder `e0/2, e0/4, e0/8` with `e0 = 0.236`. If plain SHUS had converged
     smoothly after discovery everywhere, the gateway FR campaign would stop (a
     small WCA SHUS-only screen still runs before abandoning the branch).
2. **FR pilot (rate/window) — design frozen 2026-08-18 from SHUS-only anchor_D
   data, before any FR run on the cell.** Pilot seeds `8..15` (disjoint from
   calibration 0-7 and production 100+). Weak-late-short grid (the Stage-0 smoke
   showed early/strong FR suppresses SHUS learning):
   * `theta in {0.01, 0.025, 0.05}`; FR stride in {5, 10} adaptation blocks;
   * window: `t_on = 6.0` (>= Q90 of SHUS-only `T_hit` = 5.5, rounded up);
     `t_off in {14, 22}` (25% and 50% of the establishment interval
     `[t_on, Q50(T_est) = 38.1]`; both < Q50(T_est), so FR acts only while
     "discovered but not yet established");
   * arms per config: plain SHUS, SHUS+FR, SHUS+matched-turnover sham, all paired.
   Selection rule (frozen):
   * reject any config with min windowed `ESS_anc/K < 0.5` (tightened from the
     draft 0.30 after the smoke's dips to ~0.5 under an aggressive schedule) or
     median paired `e_F(T) > 1.05 x` plain SHUS;
   * a surviving config counts as a WIN only if median paired `Delta I_F <= -10%`
     vs plain SHUS AND it beats its own sham (median paired `I_F^FR < I_F^sham`);
   * among winners choose the **smallest intervention**, ordered by turnover
     budget `gamma_eff * (t_off - t_on)`, `gamma_eff = -ln(1-theta)/dt_FR`.
   Then `theta, stride, window` are frozen for Stage 3. Global-ancestry acceptance
   floors for Stage 3 (`n_anc`, `ESS_glob`) will be set from pilot data before any
   confirmatory run.
   * **Pilot outcome (2026-08-18, seeds 8-15): FROZEN Stage-3 configuration
     `theta = 0.01, stride = 10 blocks, window [6, 14]`** (gamma_eff = 0.25, the
     smallest budget among 4 winners; median paired dI_F = -11.6%, e_F(T) ratio
     0.856, min windowed ESS 0.88, final n_anc/K 0.72). Structure of the grid:
     every t_off = 14 config improved I_F (12-27%) with better final error;
     every t_off = 22 config worsened final error (ratio 1.04-1.13) — FR through
     the overshoot peak suppresses the correction phase, as the feedback mechanism
     predicts. theta = 0.05 stride 5 was rejected on ESS despite the largest gain
     (the safety rail binding as intended). The matched-turnover shams were inert
     (I_F within ~1% of baseline everywhere): the gain is carried by the
     Fisher-Rao DIRECTION, not by generic turnover.
   * Frozen Stage-3 ancestry floors (from pilot winner margins): min windowed
     `ESS_anc/K >= 0.5` and final `n_anc/K >= 0.5`.
3. **Confirmatory (5 arms, 32 fresh matched seeds each) — arms finalized
   2026-08-18, before any confirmatory run:** production seeds `100..131` on
   anchor_D; same `K`, steps, force budget, paired seeds and noise.
   * `shus` — plain baseline;
   * `fr_temp` — the frozen pilot winner: theta 0.01, stride 10 blocks, [6, 14];
   * `fr_persistent` — same theta/stride, window [6, T] (overdamping prediction);
   * `sham` — matched-turnover control shadowing `fr_temp`;
   * `count` — count balancing: same theta/stride/window as `fr_temp`, but weights
     from a piecewise-constant 9-bin histogram instead of the fine KDE (does the
     fine Fisher-Rao geometry matter beyond coarse balancing?).
   Mandatory endpoint: frozen-bias validation — each arm's final learned bias is
   scored by an independent fresh-population run (paired noise per seed), so the
   comparison cannot be an artifact of the online estimator's statistics.

## Co-primary endpoints (confirmatory)

With `e_F(t)` the gauge-optimal interior L2 error and `I_F = \int_0^T e_F dt`:

* **Accuracy:** median paired relative change `(I_F^FR - I_F^SHUS)/I_F^SHUS <= -10%`
  with paired bootstrap 95% CI entirely below 0.
* **Speed:** `S_eps* = tau^SHUS(eps*) / tau^FR(eps*)`, `tau` with a 0.2T persistence
  window, right-censored runs kept visible. Success: median `S >= 1.25`, CI above 1.
  * `eps* =` median FINAL plain-SHUS error over independent calibration seeds
    `[FREEZE @ Stage 1]`; secondary ladder `eps in {e0/2, e0/4, e0/8}`.
* **Claim levels** (fixed vocabulary): Level 1 "better finite-budget computation"
  (I_F only); Level 2 "transient acceleration" (S > 1 but final error worse);
  Level 3 "sustained convergence acceleration" (S > 1 AND
  `e_F^FR(T) <= 1.05 e_F^SHUS(T)` AND frozen-bias validation agrees in sign).

## Secondary metrics (all stored, none primary)

`e_F(T)`, `e_{F'}(t)`, `D_t`, TV, cosine modes `a_k(t)`, `T_hit`, `T_est`, FR event
fraction, `ESS_FR`, `ESS_anc`, `w_max`, turnover, wall-clock overhead of FR
(`(t_wall^FR - t_wall^SHUS)/t_wall^SHUS`), frozen-bias endpoint.

## Mechanistic prediction (falsifiable, stated in advance)

Temporary moderate FR damps the post-discovery establishment transient (visible as
faster decay / fewer sign changes of `a_1(t)` and earlier flattening of `D_t`) and
hands back a converging SHUS after `T_off` with `R_F(t) <~ 1`. Persistent strong FR
eventually loses the advantage or harms adaptation (the linearized
`lambda_slow ~ -kappa k^2/gamma` prediction). If the sham or count-balancing arm
matches temporary FR, we do NOT claim the Fisher-Rao geometry matters.

## Stopping rules

Terminate the application branch if (a) plain SHUS shows no persistent post-discovery
transient on gateway AND on the WCA screen, or (b) temporary FR cannot beat calibrated
plain SHUS on the gateway confirmatory study without ancestry collapse. We do not add
systems until something favorable appears.
