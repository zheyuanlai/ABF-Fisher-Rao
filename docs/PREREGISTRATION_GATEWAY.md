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
   * **Gate 1 (FR has a job):** `T_hit / T <= 0.2` AND `T_est / T >= 0.4`, or a clear
     slow post-discovery mode in `a_1(t)`. If plain SHUS converges smoothly after
     discovery on the gateway, the gateway FR campaign is stopped (a small WCA
     SHUS-only screen still runs before abandoning the application branch).
2. **FR pilot (rate/window).** Pilot seeds disjoint from calibration and production.
   Grid: `theta in {0.025, 0.05, 0.10, 0.20}`, windows ~ {10%, 25%, 50%} of the
   post-discovery budget, `T_on` driven by SHUS-only discovery information (never by
   error vs F_ref). Selection rule (fixed now):
   * reject any config with `min ESS_anc / K < 0.30` or `e_F(T) > 1.05 x` plain SHUS;
   * among survivors choose the **smallest intervention** achieving >= 10% pilot
     integrated-error gain. Then `theta, window` are frozen. `[FREEZE @ Stage 2]`.
3. **Confirmatory (5 arms, 32 fresh matched seeds each):** plain SHUS; temporary FR;
   persistent FR; matched-turnover sham; count balancing. Same `K`, same steps, same
   force-evaluation budget, paired seeds and noise.

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
