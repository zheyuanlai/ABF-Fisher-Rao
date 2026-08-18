# SPEC: mollified SHUS + temporary marginal Fisher-Rao

Frozen algorithmic conventions for the application branch. Anything listed here is
calibrated once (Stage 1) and then held fixed for every SHUS+FR comparison; changing an
item in this file after FR production runs exist invalidates the preregistration.

## 1. SHUS (the ABP)

State: accumulator `R_n(z) > 0` on the reaction-coordinate grid.

* **Bias sign.** Replicas evolve under `V_n(q) = V(q) - F_n(xi(q))`, with
  `F_n = -beta^{-1} log R_n`. A visited region grows `R`, lowers `F_n` there, raises
  `V_n` there, and pushes walkers away. At the fixed point `F_n -> F + const` and the
  biased marginal is uniform.
* **Deposit.** Every MD step, each replica deposits at its *post-step* position with
  weight `exp(-beta F_n(xi)) = R_n(xi)` (linear interpolation), where `F_n` is
  **block-frozen**: it only changes at block boundaries.
* **Update.** After each adaptation block of `L = block` steps:
  `R_{n+1} = R_n + (dt / K) * mollify(block deposits)`, where `mollify` is the Gaussian
  grid kernel `delta_eps` (bandwidth `eps_bw`, radius `4*eps/dx`, reflect-padded
  convolution — identical discretization to the validated ABF-FR engine).
* **Gauge.** After every update, `R <- R / max(R)` per row. This is a pure gauge
  transformation: forces depend only on `d/dz log R`, and all future deposit weights
  scale by the same constant, so the entire trajectory of profiles is scale-invariant
  (test: `test_gauge_invariance_of_forces`). It exists solely to prevent overflow —
  SHUS mass grows exponentially once the marginal is flat.
* **Consistency.** Samples from the biased equilibrium `p ~ exp(-beta(F - F_n))`
  deposited with weight `exp(-beta F_n)` increment `R` proportionally to
  `exp(-beta F)`: the bias cancels *exactly*, independent of the current `R_n` (test:
  `test_reweighting_consistency_increment_proportional_to_gibbs`). Hence
  `F_n -> F + C` regardless of the adaptation history.
* **Reporting gauge.** `F_hat` is centered on the eval window; L2 errors additionally
  subtract the optimal additive constant (mean of the difference over the window).

## 2. Fisher-Rao step (uniform target, finite theta)

* **Target.** `q_FR = u = 1/|M|` on the full reflected domain. Frozen project
  decision: no bias-aware or lagged targets (they carry no independent information).
* **Finite step.** At an event, estimate `p_hat` (KDE, bandwidth `eta_bw`, same
  binned-density discretization as everything else), assign
  `a_k = [u / p_hat(xi_k)]^theta`, normalize, and resample exactly `K` replicas by
  **systematic resampling**. This realizes `p^+ ~ p^{1-theta} u^theta`, the exact
  finite-time Fisher-Rao flow toward `u` with `theta = 1 - exp(-gamma tau_FR)`
  (tests: `test_finite_step_matches_power_interpolation`,
  `test_resampled_population_matches_target`).
* **Degeneracy control.** `theta` is halved per row until `ESS_FR >= alpha_ess * K`;
  a row that cannot meet the floor fires a no-op (`theta = 0` gives uniform weights,
  and systematic resampling with uniform weights is the identity).
* **Cloning.** Children copy `(x, y)` as-is (overdamped dynamics; nothing to
  re-thermalize on the gateway) and inherit the ancestor label. Decorrelation must
  come from subsequent *physical* propagation — this is the covariance-defect risk the
  theory branch quantifies, and we do not soften it by resampling hidden coordinates.
* **Schedule.** Events fire every `fr_every_blocks` adaptation blocks inside a hard
  window `[t_on_frac, t_off_frac] * T` ("temporary"); persistent FR is the same arm
  with `t_off_frac = 1`. No online gates inside the production loop — the window is
  fixed from pilot data (reviewer-proof, per plan).

## 3. Cycle ordering and estimator protection (the invariant)

```
physical propagation (block)  ->  SHUS update  ->  FR resampling  ->  next block
```

An FR event gathers walker arrays `(X, Y, ancestry)` **only**. It cannot touch the
SHUS accumulator or its deposit buffer (test:
`test_estimator_protection_fr_event_cannot_touch_accumulator`). Clones therefore
contribute to the estimator only after physical propagation in the next block.

## 4. Controls

* **Matched-turnover sham.** Copies its partner FR arm's realized turnover count at
  the partner's event times, killing/cloning uniformly random walkers. Fires on the
  partner's schedule by construction; separates "directed FR reallocation" from "any
  resampling of this magnitude" (test: `test_sham_copies_partner_turnover_and_timing`).
* **Count balancing.** Defined at Stage-3 preregistration time (not implemented yet;
  the old campaign found it tied FR on WCA, so it is a mandatory confirmatory arm).
* **Pairing.** All arms of one (config, seed) share initial conditions and Langevin
  noise; comparisons are paired per seed.

## 5. Batching / hardware

One batch flattens `B (config, seed) rows x M methods` to `R = B*M` rows of `K`
walkers, all resident on a single GPU (production target: one H200, float64). No
host<->device synchronization inside the step loop except the rare theta-backoff at FR
events; time series accumulate on-device and transfer once at the end. Reproducibility
is per (device, seed): device RNG streams differ between CPU and GPU, so cross-device
tests are component-level (`test_cpu_gpu_component_equivalence`), while
checkpoint/resume and re-run determinism are bitwise on a fixed device
(`test_checkpoint_resume_bitwise_equal`, `test_deterministic_given_seed`).

## 6. Storage schema

Every production run stores the FULL `pmf_t` and `marginal_t` time series plus
`time, x_grid, F_ref`, and metadata `reference_id, eval_window, config, method, seed`
(`abpfr.io.save_run` hard-asserts this). ~400 checkpoints per run so time-to-accuracy
is well resolved and any rescoring never needs a rerun.
