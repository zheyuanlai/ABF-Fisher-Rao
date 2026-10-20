# Figure provenance

## Conventions common to all seven figures

* Every CSV row is a checkpoint that the simulation saved. No time point was interpolated, smoothed or added.
* The CSV statistics are the median and 25th / 75th percentiles across seeds of per-seed errors (numpy's linear
  percentile).
* Paired statistics are per-seed relative changes (FR − ABF)/ABF. They are summarised by the median, a bootstrap 95 %
  CI of the median (10 000 resamples, unless stated otherwise) and FR wins / n.
* Scoring code: the repository's own scorers, imported from the main checkout at commit 13e77275 with no modified
  tracked files (recorded in `meta/<exp>.json → provenance`).
* Machine-readable copies of everything below: `meta/<exp>.json` and `manifest.json`.

Abbreviations:

* t.u.: time units;
* EM: overdamped Euler–Maruyama with unit mobility;
* P0 histogram: the textbook online bin-average ABF estimator;
* own-bin bias: each walker is biased by the bin mean force of the bin it occupies;
* IQR: interquartile range.

---

## 1. Entropic gateway — `gateway_*.csv`

1. **Purpose.** A constructed establishment-limited 2-D toy. The x-marginal needs walkers to pass through a narrow
   high-ω gate, so ABF's estimate is limited by how fast the barrier region is populated.
2. **Run directory.** `results/equal_budget_v2/gateway/N2048/s<seed>_{abf,fr}.npz` (64 files).
3. **Commit.** Runs at 55a4e78d (`git_dirty: false`); engine `src/gateway_ladder_numba.py` (gateway_ladder_numba/2,
   sha256 bdeaa40a…).
4. **Configuration.** `configs/equal_budget_v2/gateway_production.json` (frozen 2026-10-10). Timestep:
   `docs/equal_budget/GATEWAY_TIMESTEP_VALIDATION.md`.
5. **Seeds.** 32 paired seeds, 8100–8131 (all kept).
6. **Replicas.** N = 2048.
7. **Dynamics.** EM at h = 2.5 × 10⁻⁵ (validated). T = 40; 1.6 × 10⁶ steps.
8. **ABF.** P0 histogram, 180 bins on [−1.8, 1.8], own-bin bias Γ_j = M_j / (C_j + 1), no ramp.
9. **FR KDE and target.** Binned reflect-padded Gaussian KDE, η = 0.1. Uniform target on [−1.8, 1.8].
10. **FR schedule.**
    * Rate g(t) = 1.5 (1 − e^{−t/4}).
    * Every 0.004 t.u. (160 steps), from t = 0, never stopped.
    * Cap max(1, ⌊0.08 N⌋) = 163 per opportunity; score clip 3.
11. **Reference.**
    * Exact analytic F(x) = H(x²−1)² + β⁻¹ log ω(x), F′(x) = 4Hx(x²−1) + ω′/(βω).
    * β = 16, H = 0.5, ω_out = 1, ω_in = 32, s = 0.1.
    * Frozen file `results/equal_budget_v2/references/gateway_reference.npz`, verified by the scorer.
12. **Errors.** `scripts/equal_budget/eqb_metrics.GatewayScorer`, the accepted
    `analyze_gateway_replica_ladder.Scorer` with the analytic reference.
    * e_F: centred RMS over the 151 nodes of [−1.5, 1.5].
    * e_F′: L² over [−1.5, 1.5] of the bin force against F′ on a per-bin Gauss–Legendre sub-grid. It contains the
      hard piecewise-constant floor 0.0323.
    * e_ξ: ‖p_t − u‖ on [−1.8, 1.8], u = 1/3.6.
13. **Marginal.** **Instantaneous** replica histogram `hist_inst` (180 bins, raw).
14. **Data status.** Original data, post-processing only.
15. **Reproduced statistics.**
    * Ī_F (mean of e_F over the 200 uniform budget fractions): **−31.4 % [−34.7, −27.7], 31/32**.
    * e_F(T): **−61.4 % [−65.1, −55.2], 32/32**.
    * Both identical per seed to the frozen analysis `results/equal_budget_v2/gateway/analysis/summary.json`.
    * Audit figure "−29.7 %": the independent dt-refinement run set (seeds 7100–7131,
      `results/numerical_validation/gateway_dt/dt2.5e-05`), re-scored by the same Scorer: −29.67 %, 30/32.
      That set saved no instantaneous histogram, so it cannot supply panel (c).
16. **Caveats.**
    * The historical h = 4 × 10⁻⁴ figures are not used.
    * The first save is at t = 0.004, not t = 0.
    * FR acts from the first opportunity, so the arms differ from the first FR event.
    * The finite-N floor of panel (c) is 0.156. FR reaches it by t ≈ 6 and then sits slightly below it
      (sub-binomial scatter, a genuine effect of flattening by selection).
    * The gateway is a constructed toy.

---

## 2–3. Ethane in rigid all-silica LTA, 150 K and 300 K — `lta150_*.csv`, `lta300_*.csv`

1. **Purpose.** A molecular entropic-barrier system. Ethane crosses the 8-ring window between α-cages.
2. **Run directory.** `results/lta_histogram/production_T{150,300}/{abf,fr_uniform}.npz` (16 seeds per file). The
   matched-sham file `fr_sham.npz` in the same directory is not read.
3. **Commit.** Runs at 56ab774d (campaign base); engine `src/lta/core_lta.py`; config hash 8fae317868d3.
4. **Configuration.** `configs/lta_histogram/campaign.json` (frozen 2026-10-03), `docs/LTA_HISTOGRAM_REPLICATION.md`.
5. **Seeds.** 16 paired seeds: 1020–1035 at 150 K, 1060–1075 at 300 K.
6. **Replicas.** N = 1024.
7. **Dynamics.** EM at dt = 2 × 10⁻⁴ (engine units; validated in `docs/numerical_validation/LTA_VALIDATION.md`).
   T = 60; 300 000 steps; saves every 0.6 t.u.
8. **ABF.**
   * P0 histogram, 180 circular bins.
   * ABF bias ramp over 0 ≤ t ≤ 4.
   * Reported estimator: the production accumulators after the estimator burn-in at t = 4.
9. **FR KDE and target.** Wrapped Gaussian KDE, bandwidth 0.1 rad. Uniform target 1/(2π).
10. **FR schedule.**
    * Rate 0.2, every 5 steps, from t = 4, never stopped.
    * Cap ⌊0.02 N⌋ = 20; score clip 2.
11. **Reference.**
    * Independent exact Metropolis MC umbrella reference (`results/numerical_validation/lta/T{T}/MC`, 640 chain files).
    * F by the density route; F′ = pooled MC conditional mean force.
    * Loader: `scripts/convergence_fr/systems.py::load_lta`.
12. **Errors.**
    * e_F: aligned full-circle RMS over the 180 bins (kJ/mol).
    * e_F′: RMS of the bin mean force against the MC mean force (kJ/mol/rad).
    * e_ξ: ‖p̂_t − 1/(2π)‖ over the circle.
13. **Marginal.** **Instantaneous KDE-smoothed** replica marginal `p_hat`, as saved by the engine. It is FR's own
    marginal estimator, bandwidth 0.1 rad. The engine saves no raw histogram.
14. **Data status.** Original data, post-processing only.
15. **Reproduced statistics** (I_F = trapezoid over all saves on [0, 60]).

    | | I_F | e_F(T) |
    |---|---|---|
    | 150 K | **−28.5 % [−29.4, −26.6], 16/16** | **−54.4 % [−58.3, −46.1], 16/16** |
    | 300 K | **−13.7 % [−14.8, −11.1], 16/16** | −7.9 % [−14.6, +0.7], 11/16 (**not resolved**) |

    * Both rows are identical to `LTA_VALIDATION.md`.
    * The CSV curves are identical to `atlas_summary.json` (max deviation 0).
16. **Caveats.**
    * The e_F kink at t = 4 is the switch of the reported estimator to the production accumulators.
    * Panel (c) is smoothed by construction (KDE). Its floor differs from the raw-histogram figures.

---

## 4. Ethane in rigid Si-ITE (ITQ-3), 300 K, early-stopping FR (block E1) — `ite300_*.csv`

1. **Purpose.** Confirmatory test of FR used early only (on 1 ≤ t < 10), followed by plain ABF.
2. **Run directory.** `results/convergence_fr_campaign/E1_ite300_stop10/{abf,fr}_N1024_s<seed>.npz` (32 files).
3. **Commit.** Results committed at c29dc558; engine `src/eri/engine.py` (eri_engine/1, sha256 a165f0a9…). The
   runs record per-file sha256 provenance, not a git commit.
4. **Configuration.**
   * `scripts/convergence_fr/make_jobs.py` (block E1).
   * `docs/convergence_fr_campaign/PREREGISTRATION.md`.
   * Analysis `results/convergence_fr_campaign/analysis/E1.json`.
5. **Seeds.** 16 fresh paired seeds, 1100000–1100015.
6. **Replicas.** N = 1024.
7. **Dynamics.** EM at dt = 10⁻⁴. T = 50; 500 000 steps; saves every 0.125 t.u. (401 saves).
8. **ABF.**
   * P0 histogram, 180 circular bins.
   * Bias ramp(t) · min(1, C_j/200) · Γ_j; ramp over 0 ≤ t ≤ 1.
9. **FR KDE and target.** Unnormalised wrapped Gaussian KDE, bandwidth 0.1 rad. Uniform target.
10. **FR schedule.**
    * Rate 2.0, every 10 steps.
    * **On at t = 1, off at t = 10** (fr_stop_steps 100 000). No FR event after the stop (asserted).
    * Cap 20; score clip 2.
11. **Reference.** Exact MC reference `results/overnight/ite/reference` (variant ite_itq3), loader
    `systems.load_eri_cell`.
12. **Errors.**
    * e_F: aligned full-circle RMS.
    * e_F′: RMS of the bin mean force.
    * e_ξ: ‖p_t − 1/(2π)‖ over the circle.
13. **Marginal.** **Instantaneous** replica histogram `sv_h`: the current tagged-molecule bins at the save, raw,
    180 bins. Floor 0.167.
14. **Data status.** Original data, post-processing only.
15. **Reproduced statistics.** All identical to E1.json.
    * S_geo = **1.59 [1.31, 1.85]**, 16/16 seeds FR faster (sustained, kT ladder 0.080 … 0.032 kT).
    * Post-ramp Ī_F on [1, 50]: **−37.7 % [−42.2, −28.5], 16/16**.
    * e_F(T): **−32.3 % [−42.6, −24.1], 16/16**.
    * Full-window Ī_F: −24.5 %.
    * e_F′(T): −22.4 %.
16. **Caveats.**
    * Do not combine E1 with the continuous-FR block X1 (different seeds) as one three-arm experiment.
    * In panel (c) both arms sit at the finite-N floor from t ≈ 4 to 5.

---

## 5. Energetic metastability toy (NEW) — `meta_*.csv`

1. **Purpose.** An energetically metastable 2-D Gaussian–quartic multiwell with ξ = x. The archived "Case I"
   (`report/sections/04_case_metastability.tex`) is re-run with the final histogram-ABF methodology.
2. **Run directory.** `results/urops_2026/metastability/confirm/s<seed>_{abf,fr}.npz` (64 files, main checkout,
   git-ignored). The Stage 1–3 data are in the sibling directories `validation/`, `pilot_resolution/` and
   `pilot_rate/`.
3. **Commit.**
   * Engine and preregistration ef903fc2 (before any run).
   * Frozen configuration c06859d2 (before any confirmation run).
   * Engine `deliverables/UROPS_2026/scripts/meta_engine.py` (meta_engine/1).
4. **Configuration.** `metastability/PREREGISTRATION.md` (rules), `metastability/FROZEN.json` (selected values,
   UTC-stamped). Stage results: `metastability/RESULTS.md`.
5. **Seeds.** 32 fresh paired confirmation seeds, 910000–910031. The pilot seeds 900000–900007 were used only for
   the resolution and safety stages.
6. **Replicas.** N = 1000.
7. **Dynamics.** EM at h = 2 × 10⁻³ (largest candidate passing the Stage-1 gate; the archived value). T = 200;
   saves every 0.5 t.u. (401 saves).
8. **ABF.** P0 histogram, 120 bins on [−3, 3] (Stage-2 rule), own-bin bias M_j / (C_j + 1), no ramp.
9. **FR KDE and target.** Binned reflect-padded Gaussian KDE, η = 0.1. Uniform target 1/6 on [−3, 3].
10. **FR schedule.**
    * Rate γ (1 − e^{−t/20}) with γ = 0.2, chosen by the Stage-3 safety-only ladder (one rung below γ_safe = 0.5).
    * Every 0.01 t.u., from t = 0, never stopped.
    * Cap ⌊0.02 N⌋ = 20; score clip 2.
    * Death law: deaths ∝ S⁺, sources ∝ S⁻ (the WCA / ITE / LTA law).
11. **Reference.**
    * Independent y-quadrature (`src/abffr/reference.compute_reference`) on 6001 × 2001 nodes.
    * It equals the 4001-y-node quadrature and the archived 801-node profile to 1e-14.
12. **Errors.** Archived `src/abffr/metrics.py` conventions.
    * e_F: centred domain-RMS on [−2.5, 2.5] (2001-node trapezoid).
    * e_F′: domain-RMS of Γ(x) − F′(x). Piecewise-constant floor 0.139.
    * e_ξ: ‖p_t − 1/6‖ on [−3, 3].
13. **Marginal.** **Instantaneous** replica histogram (120 ABF bins, raw). Floor 0.141.
14. **Data status.** **NEW simulation**, ≈ 6 CPU core-hours in total, CPU only.
15. **Statistics** (I_F = trapezoid of e_F on [0, 200]).
    * I_F: **+51.9 % [+40.2, +60.7], 0/32** → **FR_HARMFUL** (frozen rule).
    * e_F(T): **+334 % [+291, +422], 0/32**.
    * I_F′: −1.0 % [−2.3, +2.4].
    * e_F′(T): +13.9 % [+11.4, +18.2].
16. **Caveats.**
    * Both arms start from the archived "mixed" initial condition.
    * The extra FR error is localised in the barrier bins, where the y-conditional is bimodal. This is a
      diagnostic, not a tested mechanism (`metastability/RESULTS.md`).
    * The prereg's expectation of "small or neutral" was not borne out.
    * The archived 5-seed kernel study (outcome-selected FR settings) is NOT used and is not reproduced.

---

## 6. Alanine dipeptide in vacuum, ξ = (φ, ψ) — `alanine_*.csv`

1. **Purpose.** Atomistic neutrality control. The 2-D torus CV has sterically excluded regions.
2. **Run directory.**
   * `results/alanine_histogram/hA_c800_w5/raw/hA_c800_w5__abf__c7eq__N2048__T100000__ns16__e852e5d318e3.npz` (ABF).
   * `results/alanine_histogram/hu02_t20/raw/hu02_t20__fr_uniform__c7eq__N2048__T100000__ns16__a1d3e25b4931.npz` (FR).
3. **Commit.** Runs at ebc123bb, recorded `dirty: true`. Engine `src/alanine/core2d_ala.py`; config hashes
   eb8a8a4d1f21 (ABF) and a8a3947d2a02 (FR).
4. **Configuration.**
   * `configs/alanine_histogram/alanine.yaml`.
   * Selection `configs/alanine_histogram/selected.json` (frozen rule: c_min 800, 5 ps ramp).
   * `docs/ALANINE_HISTOGRAM_ABF.md`.
5. **Seeds.** 16 paired seeds, 0–15.
6. **Replicas.** N = 2048.
7. **Dynamics.** Langevin, dt 1 fs, γ 1/ps, 300 K, AMBER ff14SB, float64 on CUDA. T = 100 ps; saves every 1 ps.
8. **ABF.**
   * Adaptive-box histogram on the 97 × 97 periodic grid: per cell, the smallest centred (2l+1)² box with
     ≥ 800 counts, l = 0 … 4.
   * 5 ps bias ramp.
   * Gradient projection every 50 steps.
9. **FR KDE and target.** Product wrapped Gaussian KDE, bandwidth 0.15 rad. Uniform target 1/(2π)² on the full
   torus.
10. **FR schedule.**
    * Rate 0.02, every 0.5 ps, **from t = 20 ps**, never stopped.
    * Cap ⌊0.05 N⌋ = 102; score clip 2.
11. **Reference.**
    * Umbrella sampling (24 × 24 windows × 16 copies, κ 200) + MBAR, `results/alanine/reference/reference.npz`.
    * Bootstrap noise floor of e_F: 0.559 kJ/mol.
12. **Errors.**
    * e_F: equilibrium-weighted (Boltzmann weight of F_ref on the 8 kT mask) aligned L² against the RAW reference
      (`metrics_ala.aligned_l2`, the study's own read-out). The kernel-matched read-out is not used.
    * e_F′: the same weight on the periodic central-difference gradient (`metrics_ala.grad_errors`).
    * e_ξ: ‖p_t − 1/(2π)²‖ on the full torus.
13. **Marginal.** **Instantaneous** replica histogram `marg_hist` (97² cells, raw, float32 storage). Supplementary:
    FR's own KDE marginal (0.15 rad) in `figs/data_supplementary/alanine_exi_kde.csv`.
14. **Data status.** Original data, post-processing only.
15. **Reproduced statistics.** Identical to the study's scoreboard.
    * dI_F on the own window [20, 100] ps: **+0.04 % [−1.36, +0.64]** (BCa CI, 8/16).
    * e_F(T): **−0.60 % [−0.91, +0.03]**.
    * dI_F on W1 = [5, 100] ps: −0.24 %.
    * Verdict **NEUTRAL** (equivalent: all six FR arms of the study have CIs inside ±4 %).
16. **Caveats.**
    * **Pairing is statistical, not bitwise.** CUDA scatter_add is nondeterministic; the arms are identical
      through 2 ps.
    * e_F reaches the reference-noise floor (0.559).
    * e_F′ is reference-noise limited (≈ 8.25 against an indicative 8.66).
    * Panel (c) sits at the finite-N floor 0.341 from ≈ 20 ps. The steric floor (0.025 for the reference's finite
      support) is not resolvable at N = 2048.
    * `hu02_t20` is one of six FR arms. It was not selected by outcome: it is the campaign configuration named in
      the task.

---

## 7. WCA dimer at the validated timestep — `wca_*.csv`

1. **Purpose.** Validated negative control. The earlier positive WCA result was an integrator artefact at
   dt = 0.002 and is NOT used.
2. **Run directory.** `results/numerical_validation/wca_confirmatory/dt0.000125_N256_T240/raw/`:
   * ABF: `s<seed>_abf.npz` (arm hist_abf);
   * FR: `s<seed>_frsham.npz` (arm hist_fr_uniform). The sham arm in the same file is not exported.
3. **Commit.** Runs at 92b33d7f; engine `src/wca_numba.py`.
4. **Configuration.** `configs/numerical_validation/wca_confirmatory_prereg.json`,
   `docs/numerical_validation/WCA_CONFIRMATORY.md`.
5. **Seeds.** 16 fresh paired seeds, 5100–5115.
6. **Replicas.** N = 256.
7. **Dynamics.** EM at dt* = 1.25 × 10⁻⁴ (validated in `WCA_LANGEVIN_VALIDATION.md`). T = 240; 1.92 × 10⁶ steps;
   saves every 5 t.u. (49 saves).
8. **ABF.**
   * P0 histogram, 160 bins on [−0.2, 1.2], edge-extrapolated.
   * Linear ramp over 0 ≤ t ≤ 20.
   * Reported estimator: production accumulators after the estimator burn-in at t = 20.
9. **FR KDE and target.** Reflected Gaussian KDE, bandwidth 0.07. Uniform target on [−0.2, 1.2].
10. **FR schedule.**
    * Rate 0.1 (no ramp), every 0.01 t.u., **from t = 40**, never stopped.
    * Cap ⌊0.02 N⌋ = 5; score clip 2.
11. **Reference.**
    * Exact Gibbs by Metropolis MC (`rescore_wca_history.mc_reference`): F by the density route, F′ = MC bin means.
    * Audit: `WCA_REFERENCE_AUDIT.md`.
12. **Errors.** `wca_numba.score_ladder_result`, the accepted read-out.
    * e_F = l2_f_t: centred on [−0.1, 1.1], in kT.
    * e_F′ = l2_fp_t: the histogram arms' own e_F′.
    * e_ξ: ‖p_t − 1/1.4‖ on [−0.2, 1.2].
13. **Marginal.** **Instantaneous** replica histogram `z_hist` (160 bins, raw). Supplementary interior-only
    version: `figs/data_supplementary/wca_exi_interior.csv`.
14. **Data status.** Original data, post-processing only.
15. **Reproduced statistics.** Identical to `summary.json` / `WCA_CONFIRMATORY.md`.
    * I_F (trapezoid on [0, 240]): **+9.9 % [+2.3, +13.2], FR wins 4/16** → **FR_HARMFUL**.
    * e_F(T): **+131 % [+93, +194], 1/16**.
    * e_F′(T): +10.5 % [+7.0, +16.5].
    * Pooled long-run FR tilt against ABF: 0.017 kT.
16. **Caveats.**
    * Only 49 checkpoints exist.
    * The e_F / e_F′ spike at t = 20 is the estimator burn-in reset (genuine).
    * Panel (c) is dominated by the ~16 % of replicas held in the soft-wall tails, which are clamped into the two
      edge bins. The interior-only error sits at the finite-N floor in both arms.
    * The historical matched sham (≈ neutral, −0.8 %) is part of the original analysis but not of this two-arm
      export.
