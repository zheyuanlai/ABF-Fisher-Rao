# Online histogram (P0) ABF mean-force estimator: replication of the entropic-bottleneck and WCA results

**Date:** 2026-09-27. **Branch:** `histogram-abf` off `main` @ `64567ea`. **GPU:** 3 only.
**Status:** implementation plan written BEFORE any code change (this section); calibration rule and
seed blocks frozen in `configs/histogram_abf/campaign.json` BEFORE any run; results appended below as
they close.

## 1. Implementation plan (from the code as it stands at 64567ea)

### What the two engines actually do today

**Entropic bottleneck** (`src/eb_abffr_core.py`, Case II of the report, `report/sections/05_case_entropic_bottleneck.tex`,
`report/tables/eb_design.tex`). Base regime verified in `PhysConfig` defaults and the table:
beta 8, H 2.5, omega_out 1, omega_in 25, s 0.25, N 256, dt 1e-3, 40 000 steps, save every 400; FR gamma 15,
eta 0.10, fr_every 10, score_clip 3, max_event_fraction 0.08, ramp_fraction 0.10 (gamma ramps as
1 - exp(-t/ramp) over the first 10 % of the run), fr_burnin 0, target EMA 0.005 (unused by the
uniform target). CV domain [-1.8, 1.8], grid of 181 nodes (dx 0.02), evaluation window [-1.5, 1.5], F gauged
at x = 0 then additive-constant aligned (mean over the window) for e_F. The report's Case II confirmatory
arm is `fr_estimated`; `fr_uniform` was run on 5 seeds in Stage 0 (median final e_F 0.104 vs ABF 0.210,
I_F 6.47 vs 10.67). The later gateway study (`src/gateway_core.py`) is a different cell and a different
engine; Case II is not superseded for this benchmark, so Case II is used.

*ABF estimator (object A, replaced here):* every step the walkers deposit `C[round((x-XMIN)/dx)] += 1`,
`Sf[...] += f` on the 181-node grid; the mean force is `smooth(Sf, h=0.07) / (smooth(C, h) + min_count 1.0)`
(reflect-padded Gaussian convolution, radius 4h), `B = cumtrapz(F')` gauged at x = 0, and the walker feels
`interp1d(x, F')` (linear interpolation of the grid profile). This runs every step for every arm.
*FR marginal (object B, untouched):* `binned_density(X, eta=0.10)` = count histogram on the same 181-node
grid, Gaussian-smoothed with bandwidth eta, normalised; score `log p - log q - KL`, clipped; birth-death
`fr_resample_indices`. Lives entirely in the FR block of `simulate_batch`, reads nothing from the ABF
estimator except `Bbias` for the estimated/oracle targets (not used by `fr_uniform`).

**WCA dimer** (`src/wca_abffr_core.py`, `src/wca_phase_jobs.py`, `scripts/run_uniform_wca.py`,
`scripts/run_wca_corrected_confirmation.py`). Accepted corrected Case IX cell verified in
`configs/wca_phase_diagram_production.yaml` + `configs/information_campaign/wca_corrected_confirmation_prereg.json`:
beta 1, h 2, w 2, n_dim 10 (M = 100), a 1.5, sigma 1, epsilon 1, N 1024, 120 000 steps, save every 2500,
dt 2e-3, grid 160 nodes on [-0.2, 1.2] (dz 0.008805), eval window [-0.1, 1.1]; reference
`cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz` (label "HP reference v2 ... unsmoothed", present,
160 nodes). Kernel ABF: `abf_bandwidth` 0.025, `abf_smooth_sigma` 0.5 grid points, `abf_force_clip` 40,
`abf_warmup_steps` 10 000 (bias scale ramps linearly 0 -> 1), `estimator_burn_in_steps` 10 000 (a second
"production" estimator starts then and is the one reported), `abf_edge_extrapolate` true,
`mean_force_sample_clip` 500. FR: `fr_rate` 0.10, `fr_every` 5, `fr_start_steps` 20 000, `target_ema_rate`
0.005, `max_event_fraction` 0.02, `score_clip` 2.0, `kde_bandwidth` 0.070 (the FR marginal KDE, reflected
Gaussian at the sample positions, `kde_1d_torch`). Accepted scoring convention: e_F = additive-constant
aligned RMS (trapezoid/width) over [-0.1, 1.1] of `pmf = cumtrapz(mean_force)`; primary read-out for the
kernel arm is the bank's h_read* = 0.0125 profile (`ReadoutBank`), legacy 0.025 alongside.

*ABF estimator (object A):* `TorchKernelABFEstimator.update` forms the full (160, N) Gaussian weight
matrix at every step (`num += w f`, `den += w`); `mean_force_profile` = num/den then a 0.5-point Gaussian
smoothing; `evaluate` re-forms that profile and linearly interpolates (edge-clamped); `pmf_profile`
re-forms it again and integrates (every step, for `A_hat`). *FR marginal (object B, untouched):*
`fr_score_torch` -> `kde_1d_torch` at `kde_bandwidth`, `fixed_population_birth_death_torch`.

### Design of the change

One estimator interface per engine, selected by configuration; the kernel path keeps its exact op sequence.

* EB: `PhysConfig.abf_estimator in {"kernel", "histogram"}`, `PhysConfig.abf_n_bins` (structural: uniform
  within a batch). `KernelABFEstimator` (the existing C/Sf/smooth/interp1d ops, unchanged order) and
  `HistogramABFEstimator` (batched, `(R, n_bins)` accumulators `C`, `M`). The loop calls
  `update`, `evaluate`, and only the histogram path skips the per-step grid profile / PMF / EMA when no
  method in the batch needs them (estimated or oracle target); at every save (and at the end) the
  profiles are rebuilt for the metrics.
* WCA: `SimConfig.abf_estimator`, `SimConfig.abf_n_bins`; `HistogramABFEstimator` with the same interface
  as `TorchKernelABFEstimator` (`update`, `evaluate`, `mean_force_profile`, `pmf_profile`,
  `effective_counts`) plus `bin_mean_force`, `counts`, `diagnostics`; both the all-steps bias estimator
  and the post-burn-in production estimator are built by one factory. The kernel branch of
  `run_sampler_gpu` is left as is (it keeps computing `A_hat` every step); the histogram branch computes
  `A_hat` only for the estimated/oracle targets or at a save. `PhaseRunSpec` gains the two fields;
  `spec_hash`/`config_hash` DROP them at their defaults so every legacy `run_id` is unchanged (otherwise
  every completed WCA run would be orphaned -- `src/wca_followup_jobs.py` records the same trap).
* Run-id for histogram runs carries the estimator and bin count so kernel/histogram files never collide.

**Histogram definition.** `n` equal bins on the CV domain (EB: [-1.8, 1.8]; WCA: [z_min, z_max] =
[-0.2, 1.2]); edges `e_j = lo + j*Delta`, bin `j` = `[e_j, e_{j+1})` for `j < n-1`, the last bin is closed
`[e_{n-1}, hi]`; index `clamp(floor((z - lo)/Delta), 0, n-1)`. Samples outside the domain (WCA soft wall;
EB reflects so none occur) are deposited in the boundary bin, matching the kernel's edge-extrapolate
reading. Online update per step: `idx = bin(z); C[idx] += 1; M[idx] += f` (scatter_add, no history).

**Trust rule (the engine's existing convention, not a new one).** EB: `Gamma_j = M_j / (C_j + min_count)`
with the engine's `min_count = 1.0` (the same soft ramp the kernel applies to its smoothed counts; an
empty bin gives exactly 0 bias). WCA: `Gamma_j = M_j / C_j` for `C_j > 0`, else 0 (the kernel's
`den > EPS` rule), with the existing linear warm-up ramp of the bias scale and the +-40 clip unchanged.
The applied force at a walker in bin `j` is `Gamma_j` (own bin, piecewise constant; no interpolation,
no smoothing). The reported "own" estimator is that same `Gamma_j` -- one object, so no read-out /
dynamics mismatch can arise.

**Own read-out.** `F_hat` = exact integral of the piecewise-constant force (piecewise linear, continuous),
evaluated at the existing grid nodes, gauged/aligned exactly as the kernel arm; `e_F` = RMS over the
existing evaluation window with the engine's own norm (EB: mean over nodes; WCA: trapezoid/width).
`e_F'` for a P0 profile: a step function has no well-defined value ON its jumps, and on the EB grid every
node sits on a bin edge for every ladder width (dx = 0.02 divides every Delta), which is exactly the
half-bin-shift trap recorded in `docs/GATEWAY_HISTOGRAM_ESTIMATOR.md`. The primary `e_F'` is therefore
the function-space RMS of `Gamma(x) - F'_ref(x)` over the window, computed by a composite Gauss-Legendre
rule with 8 nodes per bin (never on an edge; changed from a midpoint rule before any run when the unit
test showed the midpoint rule under-reads a locally linear residual by 1/64 in MSE; EB reference analytic at the sub-points, WCA reference
linearly interpolated between its 160 nodes, i.e. the same reading the trapezoid metric already uses).
The node-sampled variant ([left, right) rule) is stored alongside as a secondary. No KDE anywhere in
the histogram arms' primary numbers.

**Deterministic P0 floor** (no simulation): bin averages of `F'_ref` (64-point midpoint rule per bin),
P0 profile, exact integral, then the same `e_F` / `e_F'` conventions on the same window.

**Diagnostics recorded per histogram run:** counts per bin at every save, minimum count over the
evaluation window, fraction of window bins with `C_j < 10` (the trust-threshold diagnostic, fixed now,
affects no dynamics), maximum |applied bias force|, clipping fraction (WCA), wall time.

### Tests (before any production run)

`tests/test_histogram_abf.py`: (1) counts/force sums == brute-force numpy; (2) `Gamma_j * (C_j + min_count)
== M_j` exactly and `M_j / C_j` at `min_count = 0`; (3) `evaluate` returns the own-bin value, boundary
convention as documented; (4) exact PMF of a known piecewise-constant force; (5) empty bins finite and
zero-biased; (6) `abf_estimator="kernel"` bit-identical to the HEAD fixtures generated at 64567ea before
this code existed (`tests/fixtures/eb_pre_histogram_fixture.npz`, `wca_pre_relax_fixture.npz`);
(7) legacy `run_id`/`spec_hash`/`config_hash` unchanged (`tests/fixtures/histogram_abf_head_ids.json`);
(8) the FR marginal KDE / target / score / schedule / bandwidth source is untouched (source-level guard on
the FR block + numerical check that FR event counts are the same function of the population);
(9) short histogram smoke runs finite. Plus the existing WCA and gateway suites.

### Campaign (details, seeds and the frozen selection rule in `configs/histogram_abf/campaign.json`)

A1/B1 ABF-only bin-width ladders on fresh seeds (EB 5000-5007, 8 seeds; WCA 3000-3007), then the
predeclared rule, then A2/B2 four-arm confirmations on fresh seeds (EB 5100-5119, 20 seeds; WCA
3100-3115, 16 seeds). Efficiency from `scripts/profile_histogram_abf.py` (separate, synchronised
timing), never from instrumented production runs.

## 2. Results

### 2.1 Entropic bottleneck, A1: deterministic floors and ABF-only calibration (seeds 5000-5007, batch_seed 777)

Own read-out of the histogram ABF (P0 bins), one batch per width on identical Langevin noise. Floors from the analytic
reference. Rule (a): floor share <= 0.20; rule (b): paired median change to the next finer width <= 5 % (e_F(T), I_F), <= 10 % (e_F'(T)).

| n_bins | Delta | e_F(T) median | I_F median | e_F'(T) median | floor e_F | floor e_F' | share e_F | share e_F' | to finer: d e_F(T) / d I_F / d e_F'(T) | (a) | (b) | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45 | 0.080 | 0.0116 | 2.532 | 0.5761 | 0.0125 | 0.5753 | 1.08 | 1.00 | -23.0 % / -4.2 % / -27.5 % | FAIL | FAIL | 0.480 |
| 60 | 0.060 | 0.0085 | 2.477 | 0.4174 | 0.0071 | 0.4162 | 0.83 | 1.00 | -6.6 % / -1.3 % / -33.0 % | FAIL | FAIL | 0.477 |
| 90 | 0.040 | 0.0080 | 2.459 | 0.2797 | 0.0033 | 0.2778 | 0.41 | 0.99 | +2.3 % / +1.6 % / -49.0 % | FAIL | FAIL | 0.474 |
| 180 | 0.020 | 0.0083 | 2.476 | 0.1425 | 0.0000 | 0.1388 | 0.00 | 0.97 | +0.6 % / +3.2 % / -46.3 % | FAIL | FAIL | 0.476 |
| 360 | 0.010 | 0.0082 | 2.546 | 0.0765 | 0.0000 | 0.0694 | 0.00 | 0.91 | (finest) | FAIL | n/a | 0.469 |

Kernel ABF (h = 0.07, the accepted Case II estimator) on the same seeds and noise, context only: e_F(T) 0.2101, I_F 10.594, e_F'(T) 1.8423, 0.661 ms/step (the accepted Stage-0/1 medians are 0.210 / 10.67 / 1.83).

**Selection.** No width passed the frozen rule: rule (a) fails everywhere because the histogram ABF's own error IS the deterministic P0 floor (shares 0.9-1.1 for e_F', 0.4-1.1 for e_F; the Monte Carlo residue is negligible at 121 000+ samples per window bin), and rule (b) fails because e_F'(T) keeps falling with the floor (first order in Delta). The predeclared fallback (smallest normalised plateau violation) selected **n_bins = 60 (Delta = 0.060)**, frozen in `configs/histogram_abf/selected_bins.json` at 2026-09-27T10:19:39Z before any FR result was read.

**The finding that matters, visible in the ABF-only stage.** On identical seeds and noise the histogram ABF ends 25x more accurate than the kernel ABF (e_F(T) 0.0085 vs 0.210) and 4x better integrated (I_F 2.48 vs 10.6), and its e_F'(T) equals the discretisation floor. The kernel ABF's 0.21 is not sampling starvation of the barrier: its interior error (|x| <= 1.3) is 0.05-0.12 and the whole 0.21 comes from the outer 0.1 of the evaluation window, where its walker density is 0.004-0.009 and its mean force is 6-7 units short of the true wall force (F'_ref(1.5) = 18.75). The kernel smooths the accumulators across the frontier of the visited region, so at the steep outer walls (30 kT above the wells) it extrapolates the lower force of the visited side, the walkers cannot climb, the region is never visited and the bias never improves: a self-perpetuating frontier stall. The histogram bin at the frontier is locally unbiased, so its walkers climb bin by bin: the histogram ABF marginal is flat over the whole domain (40 % of samples at x > 0, minimum window-bin count 121 000, maximum applied bias 37.8 = F'_ref near the domain edge). The kernel arm reproduces the accepted Case II ABF numbers to three digits, so this is the accepted engine's behaviour, not a defect introduced here.

### 2.2 Entropic bottleneck, A2: four-arm confirmation (seeds 5100-5119, batch_seed 778, histogram Delta = 0.060)

Kernel arms scored by the accepted Case II convention (own h = 0.07 profile); histogram arms by their own bins. Medians (IQR) over 20 matched seeds; both estimators' batches share seeds, initial conditions and Langevin noise.

| arm | I_F | e_F(T) | e_F'(T) | interior e_F(T) (|x| <= 1.3, diagnostic) |
|---|---|---|---|---|
| kernel_abf | 10.620 (9.78-11.20) | 0.2062 (0.195-0.211) | 1.805 | 0.0422 |
| kernel_fr_uniform | 6.580 (6.37-6.71) | 0.1029 (0.100-0.106) | 1.195 | 0.0446 |
| hist_abf | 2.279 (2.12-2.47) | 0.0090 (0.008-0.010) | 0.417 | 0.0081 |
| hist_fr_uniform | 1.550 (1.47-1.65) | 0.0078 (0.007-0.008) | 0.417 | 0.0068 |

Fisher-Rao effect within each estimator (fr_uniform vs abf, paired by seed; median, bootstrap 95 % CI, wins):

| estimator | Delta I_F | Delta e_F(T) | Delta e_F'(T) | Delta interior e_F(T) (diagnostic) |
|---|---|---|---|---|
| kernel | -38.3 % [-40.5, -35.0], 20/20 | -49.6 % [-51.1, -47.7], 20/20 | -33.4 % [-35.0, -32.1], 20/20 | +6.5 % [+2.9, +8.6], 1/20 |
| hist | -31.2 % [-35.7, -28.3], 20/20 | -14.4 % [-19.4, -3.8], 15/20 | -0.0 % [-0.0, -0.0], 15/20 | -16.1 % [-21.4, -6.5], 15/20 |

Histogram vs kernel, same arm, paired on identical noise: ABF I_F -78.9 % [-80.1, -76.4], 20/20; ABF e_F(T) -95.6 % [-96.1, -95.3], 20/20; ABF+FR I_F -75.7 % [-77.6, -74.1], 20/20; ABF+FR e_F(T) -92.5 % [-92.8, -92.1], 20/20.

Safety: replacement fraction 0.0171 (kernel) vs 0.0124 (histogram); minimum windowed ancestor ESS/N over saves 0.016 vs 0.012 (the EB windowed ESS is read at the save before each 4000-step reset, so it is the fully decayed value in both arms; the accepted Case II value at the last save is ~0.09); conditional-variance absolute error (mean over the five probes) 0.0175 vs 0.0247 (ABF: 0.0150 vs 0.0214). Histogram support: minimum window-bin count 125266 (ABF) / 160227 (FR), no untrusted bins, maximum applied bias 37.8 (kernel 13.3).

Efficiency (end to end, R = 40 rows, 40 000 steps, GPU 3): kernel batch 33.3 s (0.832 ms/step), histogram batch 24.8 s (0.619 ms/step): **1.34x**.

**Verdict (frozen rules).** Same sign: True. CI rule (kernel CI upper < 0 implies histogram CI upper < 0): True. Magnitude, I_F (within 10 points or overlapping CIs): True. Magnitude, e_F(T) (within 15 points or overlapping CIs): False. **Replicates = False**: the integrated acceleration replicates (-38.3 % kernel vs -31.2 % histogram, 20/20 both), the endpoint gain does not (-49.6 % vs -14.4 %). Absolute accuracy acceptable: True (the histogram ABF is 4.7x better integrated and 23x better at the end, not worse).

**Reading.** The kernel's endpoint gain is the repair of the frontier stall described in 2.1: FR's uniform target pushes population into |x| > 1.4, where the kernel ABF alone never goes. In the interior the kernel FR arm is *worse* than kernel ABF (+6.5 % [+2.9, +8.6], 1/20), i.e. the accepted Case II endpoint gain lives entirely in the outer 0.1 of the window. Under the histogram there is no stall to repair: ABF alone reaches the discretisation floor (e_F'(T) = floor to three digits for both arms, hence Delta e_F'(T) ~ 0), and what remains of the FR effect is a genuine earlier convergence (I_F -31 %, the e_F(t) curves separate from t ~ 2 and stay separated) plus a small residual endpoint gain (-14 %) that now includes the interior (-16.1 % [-21.4, -6.5], 15/20). Diagnosis of the change: altered transition dynamics at the domain boundary (the kernel's frontier extrapolation vs the histogram's locally unbiased frontier bin), not discretisation bias, not sparse bins (no window bin below the trust threshold), not noisier piecewise-constant bias (the walker marginal is flat and the conditional diagnostics are unchanged), not genealogy (replacement fraction and ESS comparable).


### 2.3 Entropic bottleneck, post-hoc diagnostic: the stall is a kernel-bandwidth effect (ABF only, same 8 seeds and noise, CPU)

Added after A1 (not a preregistered endpoint; `scripts/run_histogram_abf_entropic.py --stage diag_kernel_h`, `scripts/analyze_histogram_abf_eb_diagnostics.py`). Kernel ABF at other online bandwidths against the histogram ladder, medians over 8 seeds:

| estimator | e_F(T) | e_F(T) interior |x| <= 1.3 | F' error at x = -1.5 | F' error at x = +1.5 | walker density at -1.5 | at +1.5 | max applied bias |
|---|---|---|---|---|---|---|---|
| kernel h=0.14 | 0.8896 | 0.4289 | +15.40 | -17.68 | 0.002 | 0.000 | 4.4 |
| kernel h=0.07 | 0.2101 | 0.0420 | +7.22 | -7.02 | 0.011 | 0.008 | 12.5 |
| kernel h=0.035 | 0.0135 | 0.0103 | -0.03 | +0.08 | 0.287 | 0.370 | 37.5 |
| kernel h=0.0175 | 0.0079 | 0.0078 | -0.01 | +0.02 | 0.241 | 0.339 | 39.0 |
| histogram Delta=0.080 | 0.0116 | 0.0095 | -1.19 | +1.19 | 0.351 | 0.340 | 37.0 |
| histogram Delta=0.060 | 0.0085 | 0.0078 | +1.67 | +1.78 | 0.292 | 0.317 | 37.8 |
| histogram Delta=0.040 | 0.0080 | 0.0080 | -0.01 | +0.00 | 0.290 | 0.324 | 38.6 |
| histogram Delta=0.020 | 0.0083 | 0.0084 | +0.57 | +0.58 | 0.286 | 0.294 | 39.5 |
| histogram Delta=0.010 | 0.0082 | 0.0083 | +0.29 | +0.29 | 0.289 | 0.291 | 39.9 |

Halving the accepted bandwidth once removes the stall (0.210 -> 0.0135) and halving it twice gives the histogram's number (0.0079); doubling it makes the stall catastrophic (0.89, F' 15-18 units short at the window edges, no walkers there). The frontier extrapolation of the smoothed accumulators is what holds the walkers off the walls, and its reach is the kernel's 4h support. The accepted Case II ABF baseline (h = 0.07) is therefore bandwidth-limited at the endpoint by a factor ~25, in the same way the ZIF-8 baseline was (docs/INFORMATION_CLOCK_AUDIT.md), and the histogram estimator removes the artefact at every width in the ladder. This diagnostic was not used for any selection; it explains the confirmation result of 2.2.


### 2.4 WCA dimer, B1: deterministic floors and histogram-ABF-only calibration (seeds 3000-3007, GPU 3)

Floors from the accepted TI reference (`cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz`, v2, never recomputed). Own read-out of the histogram ABF (post-burn-in production estimator, as the engine reports). Same frozen rule as A1.

| n_bins | Delta | e_F(T) median | I_F median | e_F'(T) median | floor e_F | floor e_F' | share e_F | share e_F' | to finer: d e_F(T) / d I_F / d e_F'(T) | (a) | (b) | min window count | clip fraction | sampler s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 0.03500 | 0.0896 | 40.82 | 0.6454 | 0.0055 | 0.5306 | 0.061 | 0.82 | -0.2 % / +0.6 % / -27.3 % | FAIL | FAIL | 1548494 | 0.055 | 143 |
| 80 | 0.01750 | 0.0888 | 39.24 | 0.4742 | 0.0014 | 0.2605 | 0.015 | 0.55 | -2.2 % / +8.1 % / -11.5 % | FAIL | FAIL | 791102 | 0.060 | 143 |
| 160 | 0.00875 | 0.0888 | 41.93 | 0.4175 | 0.0003 | 0.1271 | 0.003 | 0.30 | +4.6 % / +1.6 % / -1.2 % | FAIL | pass | 395532 | 0.049 | 142 |
| 320 | 0.00437 | 0.0917 | 42.30 | 0.4138 | 0.0001 | 0.0641 | 0.001 | 0.15 | (finest) | pass | n/a | 201854 | 0.043 | 143 |

**Selection.** No n_bins passed both clauses: 40/80/160 fail (a) on the e_F' share (0.82 / 0.55 / 0.30; the F' error plateaus at 0.41 from 160 bins on, so it is not discretisation-limited there but the floor is still 30 % of it), 320 passes (a) but has no finer neighbour for (b); 160 passes (b) (+4.6 % / +1.6 % / -1.2 %). The predeclared fallback selected **n_bins = 160 (Delta = 0.00875, one bin per reference node)**, frozen at 2026-09-27T11:41:18Z before any FR result was read. The histogram ABF's own e_F(T) is 0.089-0.092 at every width, i.e. the accepted kernel ABF number at h_read* (0.0889), so on this system the estimator swap does not move the ABF endpoint; the bias clip (|Gamma| > 40) fires on ~5 % of walker-steps, all of them at the compressed-dimer edge z < -0.1 outside the evaluation window where the physical mean force is -20 to -500 (both estimators clip there; the confirmation reports both).


## 3. Efficiency

Measured, not promised. `scripts/profile_histogram_abf.py` times each region with `torch.cuda.synchronize()` brackets (profiling mode only; the production loops carry no synchronisation) and times the unmodified engine loops end to end over a short horizon; `results/histogram_abf/profiling/profile_{cuda,cpu}.json`. GPU 3 (H200 NVL) and CPU (8 threads). EB: R = 4 rows (2 seeds x [abf, fr_uniform]), N = 256, float64, histogram Delta 0.06; WCA: N = 1024, float32, histogram n_bins 160. The kernel loop rebuilds its grid PMF every step (the accepted code path, kept as is); the histogram loop rebuilds it only at saves for abf / fr_uniform, so the per-step estimator cost of the kernel includes the PMF and the histogram's does not; the WCA sampler keeps two estimator instances (bias + post-burn-in), so its update is counted twice.

| device | system | physical force (us) | FR KDE + score (us) | kernel: update / evaluate / PMF (us) | histogram: update / evaluate / PMF (us) | estimator per step (us) | total per step, unmodified loop (us) | peak memory kernel / histogram |
|---|---|---|---|---|---|---|---|---|
| cuda | Entropic bottleneck | 163 | 435 | 42 / 185 / 45 | 44 / 76 / 105 | 272 -> 120 (**2.28x**) | 878 -> 617 (**1.42x**) | 0 / 1 MB |
| cuda | WCA dimer | 767 | 558 | 67 / 264 / 235 | 44 / 78 / 189 | 632 -> 167 (**3.78x**) | 1552 -> 1126 (**1.38x**) | 340 / 340 MB |
| cpu | Entropic bottleneck | 54 | 228 | 15 / 161 / 18 | 26 / 29 / 40 | 194 -> 55 (**3.51x**) | 506 -> 300 (**1.69x**) | n/a |
| cpu | WCA dimer | 236693 | 1092 | 617 / 123 / 90 | 35 / 32 / 45 | 1446 -> 101 (**14.26x**) | 200357 -> 167045 (**1.20x**) | n/a |

End to end on GPU 3, the actual campaign batches: EB confirmation, R = 40 rows x 40 000 steps, kernel 33.3 s vs histogram 24.8 s (**1.34x**); EB calibration, R = 8 ABF-only rows, kernel 26.4 s vs histogram 18.7-19.2 s (1.39x). The unmodified HEAD engines measured from a HEAD worktree before any edit (`profiling/head_baseline_timing_gpu3.json`): EB R = 4 batch 32.9 s; WCA sampler abf 195 s, fr_uniform 222 s per 120 000-step run, against which the histogram calibration runs took 142-156 s (abf; 1.3x). The WCA production wall times of all four confirmation arms are in 2.5.

Interpretation. Both engines are launch-bound on the GPU (the EB step is ~30 small kernels, the WCA physical force 0.77 ms), so removing the kernel's convolutions, interpolations and per-step PMF rebuild buys 1.4x in the total step on both systems even though the estimator itself is 2.3-3.8x cheaper; on the CPU the WCA physical force (237 ms per step at N = 1024, M = 100) dwarfs everything and the total moves 0.9-1.2x while the estimator is 14x cheaper. Memory is unchanged (the accumulators are a few kB either way). The online update never revisits a sample (two scatter-adds per step) and the free energy is reconstructed only at saves.


## 4. Code changes (branch `histogram-abf`)

Modified: `src/eb_abffr_core.py` (fields `abf_estimator`, `abf_n_bins`; `KernelABFEstimator` = the former inline
code op for op; `HistogramABFEstimator`; `reference_mean_force`; `p0_projection_floor`; the loop uses the
estimator interface; histogram records in `_finalize`), `src/wca_abffr_core.py` (same fields on `SimConfig`,
hash-neutral at defaults; `HistogramABFEstimator`; `make_abf_estimator`; numpy own-error / floor helpers;
`run_sampler_gpu` builds both estimators through the factory, rebuilds the PMF only when needed on the
histogram path, records bins per save and the applied-bias diagnostics), `src/wca_phase_jobs.py`
(`PhaseRunSpec` fields dropped from `spec_hash` at their defaults, `__hist<n>` tag in `run_id`, `build_sim`
pass-through, own-estimator scoring and records for histogram runs in `execute_run`), `.gitignore`
(un-ignore this doc), `tests/fixtures/README.md`.
Added: `tests/test_histogram_abf.py` (+ fixtures `eb_pre_histogram_fixture.npz`, `histogram_abf_head_ids.json`
generated at 64567ea), `configs/histogram_abf/{campaign.json, selected_bins.json}`,
`scripts/run_histogram_abf_entropic.py`, `scripts/run_histogram_abf_wca.py`,
`scripts/analyze_histogram_abf_entropic.py`, `scripts/analyze_histogram_abf_wca.py`,
`scripts/analyze_histogram_abf_eb_diagnostics.py` (post-hoc), `scripts/profile_histogram_abf.py`,
`scripts/plot_histogram_abf_summary.py`, `results/histogram_abf/` (summaries, tables, figures, provenance;
raw npz untracked as everywhere in this repository).
Untouched: every FR function (source-identical to 64567ea, asserted by the test), the reference files, every
frozen result directory, `src/gateway_core.py` (an unrelated uncommitted edit from an earlier session is left as
found).

### 2.5 WCA dimer, B2: four-arm confirmation (seeds 3100-3115, all four arms of a seed in one process, histogram n_bins = 160)

Kernel arms scored by the accepted corrected convention (read-out bank, h_read* = 0.0125; the legacy 0.025 read-out alongside); histogram arms by their own bins. Medians (IQR) over 16 seeds. Reference HP reference v2 (v2 campaign; unsmoothed, nonuniform acquisition).

| arm | I_F | e_F(T) | e_F'(T) | round trips | sampler s per run |
|---|---|---|---|---|---|
| kernel_abf | 41.16 (39.5-42.3) | 0.0912 (0.089-0.096) | 0.413 | 780678 | 237 |
| kernel_fr_uniform | 32.32 (30.8-35.2) | 0.0473 (0.044-0.050) | 0.362 | 790149 | 259 |
| hist_abf | 41.01 (39.1-42.7) | 0.0893 (0.086-0.093) | 0.418 | 704606 | 142 |
| hist_fr_uniform | 31.26 (29.8-34.3) | 0.0449 (0.043-0.048) | 0.394 | 730940 | 166 |

Kernel arms at the legacy read-out 0.025: ABF e_F(T) 0.09288, I_F 41.73 (the accepted confirmation on seeds 700-715 measured 0.09058 / 0.08888 at 0.025 / 0.0125).

Fisher-Rao effect within each estimator (fr_uniform vs abf, paired; median, bootstrap 95 % CI, wins):

| estimator | Delta I_F | Delta e_F(T) | Delta e_F'(T) |
|---|---|---|---|
| kernel @ h_read* 0.0125 | -19.0 % [-25.4, -14.8], 16/16 | -47.2 % [-51.5, -40.2], 16/16 | -11.3 % [-13.9, -8.7], 16/16 |
| kernel @ legacy 0.025 | -18.5 % [-24.0, -14.3], 16/16 | -42.4 % [-46.1, -35.6], 16/16 | -7.1 % [-8.1, -4.8], 16/16 |
| histogram (own bins) | -23.1 % [-28.6, -18.0], 16/16 | -50.5 % [-52.3, -45.4], 16/16 | -6.8 % [-9.0, -4.2], 13/16 |

Kernel read-out ladder (the accepted instrument, same trajectories): 0.025: dI_F -18.5 %, de_F(T) -42.4 %; 0.0125: dI_F -19.0 %, de_F(T) -47.2 %; 0.00625: dI_F -19.2 %, de_F(T) -48.3 %; raw+sigma: dI_F -19.9 %, de_F(T) -48.6 %; raw: dI_F -19.9 %, de_F(T) -48.7 %. The accepted corrected confirmation (seeds 700-715) gave -18.30 % [-26.27, -14.00] / -47.05 % at h_read*; the fresh kernel block reproduces it.

Histogram vs kernel, same arm (shared lattice init and noise stream, paired by seed): ABF I_F -1.8 % [-2.9, +3.6], 9/16; ABF e_F(T) -1.2 % [-4.4, +2.0], 9/16; ABF e_F'(T) +1.2 % [-2.2, +4.1], 7/16; ABF+FR I_F -2.4 % [-8.1, +9.4], 9/16; ABF+FR e_F(T) -3.7 % [-8.8, +0.7], 12/16.

Safety. Kernel FR: min run-long ancestor ESS/N 0.149 (floor 0.10), min windowed ESS/N 0.785, max lineage share 0.0293 (cap 0.05), floors met; replacement events 2180 (event fraction 0.00011). Histogram FR: min run-long ESS/N **0.082**, min windowed ESS/N 0.687, max lineage share **0.0566**, floors NOT met on the worst seed; replacement events 3619 (+66 %). The FR score, target and schedule are untouched; the histogram arm fires more because its own-bin bias makes the walker marginal less uniform at the FR opportunities than the kernel's smoothed bias does, and the two run-long floors are marginally crossed on one seed (0.082 vs 0.10; 0.057 vs 0.05) while the windowed statistic the gate was calibrated on stays far above 0.30. Round trips: kernel 780678 / 790149, histogram 704606 / 730940. Time to accuracy on the median curves: kernel e0/2 1.00x, e0/4 1.00x, e0/8 1.45x, abf_final 3.43x; histogram e0/2 1.00x, e0/4 1.00x, e0/8 1.91x, abf_final 3.92x.

Support and clipping. Histogram: minimum window-bin count 397009 (ABF) / 441090 (FR), no bin below the trust threshold; maximum |applied bias| before the clip 439 vs kernel 165; clip fraction (|Gamma| > 40) 4.96e-02 vs kernel 4.0e-07. The clipping is confined to the compressed-dimer edge z < -0.1 outside the evaluation window, where the physical mean force is -20 to -500; the kernel's smoothing pulls those edge values down, the histogram reports them as they are. It does not touch the window metrics.

Efficiency (sampler seconds per 120 000-step run, GPU 3): kernel abf 237 (carrying the accepted read-out bank), kernel fr_uniform 259, histogram abf 142, histogram fr_uniform 166; against the unmodified HEAD kernel without a bank (195 / 222 s) the histogram runs are 1.37x / 1.34x faster.

**Verdict (frozen rules).** Same sign True, CI rule True, magnitude I_F True, magnitude e_F(T) True: **replicates = True**. Absolute accuracy acceptable: True (histogram ABF vs kernel ABF at h_read*: I_F -1.8 %, e_F(T) -1.2 %, both CIs spanning 0). On WCA the histogram estimator reproduces the accepted ABF baseline to within noise and the accepted FR acceleration in sign, CI and magnitude (integrated -23 % vs -19 %, final -51 % vs -47 %, 16/16 both), at 1.35x the speed, with the one genealogy caveat above.


## 5. Experiment C: entropic gateway (added at the user's request after A2 and B2 closed)

Same discipline, third system: the accepted corrected-baseline gateway cell (`docs/GATEWAY_CORRECTED_BASELINE.md`:
beta 16, s 0.10, r 32, beta H = 8 kT, N 2048, dt 4e-4, 100 000 steps, T 40, save every 500, eta 0.10, fr_every 10,
ramp 0.10, score_clip 3, event cap 0.08, min_count 1, h_bias 0.07, uniform-FR gamma 1.5 inherited, inits left and
one_right, h_read* = 0.0175 from the saved fine-grid accumulators as the accepted kernel scoring). The gateway engine
(`src/gateway_core.py`) already carried an exploratory histogram branch from 2026-09-26 (`docs/GATEWAY_HISTOGRAM_ESTIMATOR.md`,
uncommitted; PMF = trapezoid of the node-sampled P0, native read-out at bin centres). For this campaign that branch
now uses the shared `eb_abffr_core.HistogramABFEstimator` (exact piecewise-linear PMF, Gauss-Legendre own F' error,
[left, right) bins, the same diagnostics), so all three systems run one implementation; the kernel branch is untouched
and remains bit-identical to `tests/fixtures/gateway_pre_transport_fixture.npz` (test added). The frozen block is the
`gateway` entry of `configs/histogram_abf/campaign.json` (an amendment; the earlier blocks are unchanged); the same
selection and verdict rules apply; fresh seeds 5200-5207 (calibration, x 2 inits = 16 ABF-only rows per width) and
5300-5315 (confirmation, x 2 inits = 32 pairs, the accepted scale). Yesterday's exploratory ladder is not used for any
choice here. Scripts: `run_histogram_abf_gateway.py`, `analyze_histogram_abf_gateway.py`.

### 5.1 Gateway, C1: deterministic floors and ABF-only calibration (seeds 5200-5207 x {left, one_right}, batch_seed 779)

| n_bins | Delta | e_F(T) median | I_F median | e_F'(T) median | floor e_F | floor e_F' | share e_F | share e_F' | to finer: d e_F(T) / d I_F / d e_F'(T) | (a) | (b) | min window count | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45 | 0.080 | 0.00624 | 0.9020 | 0.1326 | 0.00297 | 0.1315 | 0.48 | 0.99 | -5.7 % / -0.3 % / -25.9 % | FAIL | FAIL | 2010007 | 0.635 |
| 60 | 0.060 | 0.00580 | 0.8846 | 0.0982 | 0.00169 | 0.0965 | 0.29 | 0.98 | -1.6 % / +0.6 % / -31.6 % | FAIL | FAIL | 1511218 | 0.640 |
| 90 | 0.040 | 0.00585 | 0.9144 | 0.0672 | 0.00076 | 0.0645 | 0.13 | 0.96 | +0.5 % / +0.8 % / -44.2 % | FAIL | FAIL | 1003050 | 0.637 |
| 180 | 0.020 | 0.00578 | 0.9174 | 0.0374 | 0.00000 | 0.0323 | 0.00 | 0.86 | +0.4 % / +0.4 % / -33.2 % | FAIL | FAIL | 495004 | 0.640 |
| 360 | 0.010 | 0.00574 | 0.9206 | 0.0250 | 0.00000 | 0.0161 | 0.00 | 0.65 | (finest) | FAIL | n/a | 246546 | 0.643 |

Kernel ABF h = 0.07 on the same rows and noise (context only): own e_F(T) 0.01029, at the accepted h_read* = 0.0175 0.00509, own I_F 1.0347, 0.674 ms/step.

**Selection.** No width passed the frozen rule (the e_F' share is 0.65-0.99: the histogram ABF's F' error is again its discretisation floor); the predeclared fallback selected **n_bins = 45 (Delta = 0.080)**, frozen at 2026-09-27T16:08:53Z before any gateway FR result of this campaign was read. Unlike the bottleneck there is no kernel stall here (the gateway walls are not in play at beta H = 8 kT with N = 2048): the histogram ABF's own e_F(T) (0.0058-0.0062, flat across the ladder) lies between the kernel's legacy own read-out (0.0103, the closed campaign's convention) and its corrected h_read* read-out (0.0051), i.e. the ~14 % endpoint penalty of a sharper-than-0.07 online bias that the corrected-baseline audit and the 2026-09-26 exploratory ladder both measured, with no read-out audit needed to see it.


### 5.2 Gateway, C2: four-arm confirmation (seeds 5300-5315 x {left, one_right} = 32 pairs, batch_seed 780, histogram Delta = 0.080)

Kernel arms scored at the accepted h_read* = 0.0175 from the saved fine-grid accumulators (the corrected-baseline convention; legacy 0.07 = the engine's own profile alongside); histogram arms by their own bins. Medians (IQR) over 32 rows; both estimators' batches share rows, initial conditions and noise.

| arm | I_F | e_F(T) | e_F'(T) |
|---|---|---|---|
| kernel_abf | 0.825 (0.762-0.923) | 0.00516 (0.00482-0.00537) | 0.0184 |
| kernel_fr_uniform | 0.575 (0.530-0.610) | 0.00201 (0.00175-0.00239) | 0.0168 |
| hist_abf | 0.898 (0.795-0.941) | 0.00614 (0.00587-0.00656) | 0.1327 |
| hist_fr_uniform | 0.645 (0.593-0.682) | 0.00336 (0.00300-0.00379) | 0.1322 |

Kernel arms at the legacy 0.07 read-out: ABF e_F(T) 0.01019, I_F 0.981 (the corrected-baseline confirmation on seeds 400-415 measured 0.01047 at 0.07 and 0.00529 at h_read*).

| estimator | Delta I_F | Delta e_F(T) | Delta e_F'(T) |
|---|---|---|---|
| kernel @ h_read* 0.0175 | -32.9 % [-35.7, -30.6], 32/32 | -58.8 % [-64.3, -56.2], 32/32 | -9.0 % [-10.8, -6.8], 32/32 |
| kernel @ legacy 0.07 (closed convention) | -12.9 % [-16.8, -9.6], 28/32 | +10.4 % [+9.5, +12.7], 1/32 | -2.7 % [-3.4, -2.1], 27/32 |
| histogram (own bins) | -30.0 % [-31.0, -21.7], 31/32 | -47.7 % [-49.5, -40.5], 32/32 | -0.3 % [-0.4, -0.3], 32/32 |

Per init, Delta I_F: left: kernel -32.3 % (16/16), histogram -30.0 % (16/16); one_right: kernel -34.4 % (16/16), histogram -28.5 % (15/16).

Histogram vs kernel, same arm, identical noise: ABF I_F +4.8 % [+3.2, +6.2], 7/32; ABF e_F(T) +21.6 % [+17.3, +24.4], 0/32; ABF e_F'(T) +619.4 % [+611.7, +625.6], 0/32; ABF+FR I_F +10.7 % [+6.6, +20.0], 5/32; ABF+FR e_F(T) +61.0 % [+49.0, +97.1], 0/32.

Safety (gateway floors, windowed ancestor ESS/N >= 0.30, max lineage share <= 0.05, reported not gating): kernel FR min ESS/N 0.328, max share 0.0229 (met); histogram FR min ESS/N **0.270**, max share 0.0278 (ESS floor NOT met on the worst row); replacement fraction 0.0006 vs 0.0003. Support: minimum window-bin count 2020526, no untrusted bin; maximum applied bias 7.45 (kernel 6.81).

Efficiency (end to end, R = 64 rows x 100 000 steps): kernel batch 92.4 s vs histogram 87.0 s (**1.06x**; this loop carries the region / target / accumulator diagnostics of the gateway study every 500 steps and the FR marginal at every 10th step, so the estimator is a smaller share of it).

**Verdict (frozen rules).** Same sign True, CI rule True, magnitude I_F True, magnitude e_F(T) True: **replicates = True**. Absolute accuracy acceptable: True (histogram ABF vs kernel ABF at h_read*: I_F +4.8 %, e_F(T) +21.6 %, inside the predeclared +25 % margin). The FR acceleration is estimator-independent here, as the 2026-09-26 exploratory ladder found at the corrected read-out; the kernel's legacy own read-out is the only convention under which it reverses.

**Two costs to state.** (i) The fallback-selected width 0.08 is coarse for the s = 0.10 gateway feature: the histogram's own e_F'(T) is its floor, 0.133, seven times the kernel's 0.018 at h_read* (Delta I_F on e_F' is therefore ~0 under the histogram). The exact integration keeps e_F acceptable (+22 % at the end, the sharper-online-bias penalty plus discretisation), but a user who needs F' on this cell should take Delta <= 0.02 (floor 0.032) or 0.01 (0.016, the kernel's level), where the ABF-only ladder showed the same e_F and I_F. (ii) The histogram FR arm's worst row dips to windowed ESS/N 0.27 against the 0.30 gateway floor (kernel 0.33); lineage share stays well under the cap.

## 6. Conclusion

**Question.** Can online histogram ABF replace the current KDE/kernel mean-force estimator without changing the observed Fisher-Rao acceleration on the entropic bottleneck and the WCA dimer (and, added, the entropic gateway)?

**Answer.** Yes as an estimator, with one honest change of interpretation on the entropic bottleneck.

* **WCA dimer (the accepted Case IX cell): a clean replication.** The histogram ABF baseline is statistically identical to the kernel's at the accepted corrected read-out (I_F -1.8 %, e_F(T) -1.2 %, CIs spanning zero), and uniform-FR's acceleration is reproduced in sign, CI and magnitude (integrated -23 % [-29, -18] vs -19 % [-25, -15]; final -51 % vs -47 %; 16/16 both), at 1.35x the sampler speed. Caveat: the histogram FR arm fires 65 % more replacements and its worst seed marginally crosses the run-long Case IX genealogy floors (0.082 vs 0.10; 0.057 vs 0.05).
* **Entropic gateway (the accepted corrected-baseline cell): replicates.** -30 % [-31, -22] / -48 % [-50, -40] under the histogram vs -33 % / -59 % under the kernel at h_read*, 31-32/32; the kernel's legacy read-out reproduces the closed reversal (+10 % at the end) on the same trajectories, so the histogram's own numbers are the honest ones without a read-out audit. Costs: +22 % ABF endpoint (sharper online bias + the coarse fallback width) and an F' error at the P0 floor of Delta = 0.08.
* **Entropic bottleneck (Case II): the integrated acceleration replicates (-31 % [-36, -28] vs -38 %, 20/20 both), the endpoint gain does not (-14 % [-19, -4] vs -50 %), and the reason is a finding about the accepted baseline, not about FR.** The accepted kernel ABF error of 0.21 at T is a frontier stall at the steep outer walls of the evaluation window (kernel smoothing across the edge of the visited region extrapolates an underestimated wall force; walkers never climb; the region is never visited), not barrier starvation: halving the accepted bandwidth once gives 0.0135, twice gives the histogram's 0.008, doubling it gives 0.89. The histogram ABF alone ends 23x more accurate than kernel ABF and 12x more accurate than kernel ABF+FR, at its discretisation floor, so there is little left for FR to repair at the end; in the interior |x| <= 1.3 the kernel FR arm is 6 % *worse* than kernel ABF, i.e. the published Case II endpoint gain lived entirely in the outer 0.1 of the window. What survives under the histogram is a genuine earlier convergence (the e_F(t) curves separate from t ~ 2) and a small residual endpoint gain that now includes the interior.

**Absolute accuracy.** Never materially worse: EB -96 % / -79 % (much better), WCA -1 % (equal), gateway +22 % / +5 % (inside the +25 % margin; the known ~14 % sharper-online-bias penalty plus discretisation at the fallback width). The histogram's F' error is first order in Delta and equals the deterministic P0 floor on EB and the gateway, so the bin width should be chosen for the F' accuracy one needs (Delta <= 0.6 x the sharpest feature); the exact integration makes e_F insensitive to that choice.

**Efficiency (measured).** Estimator 2.3x (EB) / 3.8x (WCA) cheaper per step on the GPU, total step 1.42x / 1.38x faster; end to end 1.34x (EB), 1.35x (WCA), 1.06x (gateway loop, diagnostics-heavy); CPU 1.7x (EB) and ~1.0-1.2x (WCA, force-dominated). Memory unchanged.

**Failure modes found.** None in the sense of the brief (no NaN, no sparse bins in any window, no altered FR mechanism: the FR functions are source-identical and the marginal KDE untouched). Two genealogy flags (WCA run-long floors on one seed; gateway windowed ESS 0.27 on one row) and the F'-floor cost at coarse widths are the items to carry forward. The selection rule as frozen never passed on any system, because clause (a) assumed a Monte Carlo residue that the histogram does not have; the predeclared fallback resolved every case and is documented as such.


### 4.1 Addendum: gateway files

Modified for Experiment C: `src/gateway_core.py` (the histogram branch now instantiates
`eb_abffr_core.HistogramABFEstimator`; exact PMF; own F' error; records `C_bins`, `M_bins`, `l2_fp_nodes_t`,
`min_count_window`, `frac_untrusted_window`, `max_abs_bias_force`, `abf_estimator`; the kernel branch is untouched
and bit-identical to the accepted fixture). This file also carries the 2026-09-26 session's uncommitted
histogram/snapshot additions (`estimator`, `n_bins`, `store_snapshots`), which are committed with their tests
(`tests/test_gateway_histogram.py`, `tests/test_gateway_snapshots.py`) and scripts (`run_gateway_histogram.py`,
`analyze_gateway_histogram.py`, `make_gateway_movie.py`, `configs/gateway_histogram/`) so the committed engine is the
one that produced these results. Added: `scripts/run_histogram_abf_gateway.py`, `scripts/analyze_histogram_abf_gateway.py`,
the `gateway` block of `configs/histogram_abf/campaign.json`, `results/histogram_abf/gateway/`, one gateway test in
`tests/test_histogram_abf.py`.
