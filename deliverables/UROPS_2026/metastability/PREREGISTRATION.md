# Energetic metastability toy: histogram ABF vs ABF + uniform FR (UROPS 2026 confirmation)

*Written 2026-10-20 (UTC), before any run of the new engine. The rules below are fixed before any data exists;
the values they select are recorded, with UTC timestamps, in `FROZEN.json` as each stage completes, and that file
is committed before the first confirmation run.*

## Why a new experiment

The archived "Case I" study (`results/two_dim_xi_x/`, `report/sections/04_case_metastability.tex`) used a smoothed
binned (kernel-type) ABF estimator, 5 exploratory seeds, and FR hyperparameters selected on the same seeds by their
integrated error (best of 36 per target). It is not comparable with the histogram-ABF methodology of the other six
figures. This experiment re-runs the comparison with the final methodology and fresh seeds.

## Model (unchanged from the archived study)

* V(x, y) = 3 e^{-x²}(e^{-(y-1/3)²} − e^{-(y-5/3)²}) − 5 e^{-y²}(e^{-(x-1)²} + e^{-(x+1)²}) + 0.2x⁴ + 0.2(y−1/3)⁴,
  exactly `src/abffr/potentials.py`; ξ(x, y) = x.
* Box x ∈ [−3, 3], y ∈ [−2.5, 3.5], both reflected (as `configs/two_dim_xi_x_production_gpu.yaml`).
* β = 4 (primary and only setting). Overdamped Euler–Maruyama, unit mobility.
* N = 1000 walkers, horizon T = 200 time units, "mixed" initial conditions (half of the walkers uniform in
  [−2.4, −0.6] × [−1.9, −0.1], half in [0.6, 2.4] × [1.1, 2.9]) — the archived protocol, kept so that nothing is
  changed in FR's favour.

## Reference

Independent y-quadrature (`src/abffr/reference.compute_reference`): F(x) = −β⁻¹ log ∫ e^{−βV} dy and
F′(x) = ⟨∂ₓV⟩_{y|x} on a 6001-node x grid with a 2001-node y grid (trapezoid; the integrand is negligible at both
y walls). Convergence check: 4001 y nodes, and the archived 801-node profile.

## Metrics (archived conventions, `src/abffr/metrics.py`, applied to the histogram read-out)

* e_F(t): centred domain-RMS on [−2.5, 2.5] (trapezoid on 2001 nodes; mean over nodes removed) of F̂ − F_ref,
  F̂ = exact integral of the piecewise-constant bin force Γ_j = M_j / (C_j + 1).
* e_F′(t): domain-RMS on [−2.5, 2.5] of Γ(x) − F′_ref(x) (no centring). It contains the piecewise-constant floor
  (the error of the exact bin-averaged Γ), reported.
* e_ξ(t): ‖p_t − u‖_{L²} on [−3, 3] of the instantaneous replica histogram on the ABF bins, u = 1/6; the
  finite-N multinomial floor is reported.
* Integrated error I_F = trapezoid of e_F over [0, T] (archived `integrated_l2_over_time`); also I_F′.

## Stage 1: timestep gate (no ABF, no FR)

* Flat-bias dynamics: bias force +F′_ref(x), so the exact x-marginal is uniform on [−3, 3] and the y-conditional is
  exact. Walkers start from EXACT i.i.d. draws of this law (x uniform, y by inverse CDF of the quadrature
  conditional). 64 groups × 1000 walkers, burn-in 20 t.u., then 200 t.u. of accumulation on 120 bins.
* Read-outs per bin: conditional mean force ⟨∂ₓV⟩ (vs the bin average of F′_ref), F by the mean-force route
  (cumulative integral, centred), F by the density route (secondary), Var(y | x) (vs the quadrature value).
  Jackknife over the 64 groups; one-sided 95 % upper bound = estimate + 1.645 se.
* Gates (thresholds = one third of the ARCHIVED ABF-only final errors, median over its 5 seeds:
  e_F 0.0137, e_F′ 0.0718 — an outcome-independent error scale; the gateway rule "EM floor ≤ ABF final / 3"):
  * V1: RMS over the eval bins of the relative Var(y|x) error ≤ 2 %;
  * V2: D(F, mean-force route) 95 % upper ≤ 0.00457;
  * V3: D(F′) 95 % upper ≤ 0.0239;
  * V4: exact MALA (h = 1e-3) on the same law passes V2 and V3 (harness check).
* Candidates h ∈ {2e-3 (archived), 1e-3, 5e-4, 2.5e-4}. **Production h = the largest candidate passing V1–V3.**
  If none passes, the experiment stops and is reported as blocked.

## Stage 2: histogram resolution (ABF only)

* 8 pilot seeds 900000–900007 (never reused), ABF only, at the Stage-1 h.
* Candidates nb ∈ {60, 120, 240} bins on [−3, 3] (widths 0.1, 0.05, 0.025).
* **Rule: the nb with the smallest median I_F; candidates within 3 % of the best are tied, and a tie is broken by
  the smaller median final e_F.**

## Stage 3: FR schedule (safety only; no accuracy read-out)

Fixed by existing conventions (not tuned): uniform target 1/6; binned reflect-padded Gaussian KDE η = 0.1 (the
gateway / LTA / ITE value); score clip 2 (LTA / ITE / WCA / alanine); FR every 0.01 t.u. (the archived
fr_every 5 at h 2e-3; the WCA interval); cap max(1, ⌊0.02 N⌋) = 20 (LTA / ITE / WCA); rate ramp
r(t) = γ(1 − e^{−t/τ}), τ = 0.1 T = 20 (the archived ramp fraction 0.1, the gateway's τ = 0.1 T); FR from t = 0,
never stopped; genealogy window 10 t.u.

* The rate γ is chosen by a SAFETY-ONLY ladder γ ∈ {0.05, 0.1, 0.2, 0.5, 1, 2} on the same 8 pilot seeds at the
  Stage-1 h and Stage-2 nb. Floors (the repository's, `scripts/analyze_fr_start_timing.FLOORS`): minimum windowed
  ancestor ESS / N ≥ 0.30, maximum windowed family share ≤ 0.05, maximum realised deaths per opportunity < 0.05 N,
  finite state. The pilot script computes NO free-energy or mean-force error for FR runs.
* γ_safe = the largest rung on which every pilot seed meets every floor. **Production γ = the rung below γ_safe**
  (conservative margin); if γ_safe is the smallest rung, γ = 0.05; if no rung is safe, the experiment stops.

## Stage 4: confirmation

* 32 fresh seeds 910000–910031; both arms per seed share initial conditions and Langevin noise slot by slot (separate
  FR stream), so they are bitwise identical until the first FR death.
* Every seed is kept. Nothing is retuned after this point.
* Statistics (paired by seed): relative change (FR − ABF)/ABF of I_F, e_F(T), I_F′, e_F′(T); median with a
  10 000-resample percentile bootstrap 95 % CI of the median (seed 20261020); wins = FR < ABF.
* Verdict on I_F (primary): FR_BENEFICIAL if the CI lies below 0; FR_HARMFUL if it lies above 0; otherwise
  NEUTRAL_EQUIVALENT if the CI lies inside ±10 %, else UNRESOLVED. e_F(T) is reported with the same labels as a
  secondary.
* Expectation (a hypothesis, not a target): ABF already performs well here, so FR's effect may be small or neutral.

## Compute

Budget estimate < 10 CPU core-hours in total (numba, one core per run); cap 50 core-hours.
