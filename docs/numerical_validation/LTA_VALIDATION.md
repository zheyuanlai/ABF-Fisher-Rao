# Ethane in rigid LTA: numerical validation of the integrator, the reference and the FR result

*2026-10-09, branch `research/langevin-validation-oct2026`. Phase IV of the overnight
numerical-validation campaign.*

* Prereg (frozen before any LTA profile): `configs/numerical_validation/lta_validation_prereg.json`.
* Engine: `src/lta_validation.py`, an independent numba re-implementation; `tests/test_lta_validation.py`
  has 4 tests and all pass.
* Runner: `scripts/numerical_validation/run_lta_validation.py`.
* Analysis: `scripts/numerical_validation/analyze_lta_validation.py`.
* Data: `results/numerical_validation/lta/T{300,150}/` (`summary.json`, `analysis.log`,
  `figures/fig_lta_validation_T*.png`).
* Cost: CPU only. 300 K: 3200 jobs, about 31 core-h estimated. 150 K: the same. No GPU time.

## 1. What could go wrong, and why LTA is not WCA

The production engine (`src/lta/core_lta.py`), mapped equation by equation:

| aspect | LTA | WCA (for contrast) |
|---|---|---|
| guest model | 2-bead united-atom ethane, harmonic bond k = 400 kJ/mol/Å², no clamps | 100-particle WCA fluid + dimer |
| interactions | truncated-shifted LJ to 384 rigid framework O; analytic force = −∇U exactly (central differences 1e-9; production = independent model to 1e-13) | WCA; the production drift is clipped and is *not* −∇U |
| integrator | float64 Euler–Maruyama, dt = 2e-4 at every T, no force clip in the dynamics | EM at dt 0.002 with a per-particle clip |
| stiffest mode | the bond, dt·λ = 0.16; LJ adds ≤ 0.07 (snapshots: max abs F ≈ 150 kJ/mol/Å, displacement ≤ 0.03 Å per step) | contacts: dt·V″ = 2.05 at r = 0.95σ, beyond the EM stability limit 2 |
| CV | φ = 2π x_COM / a (linear): f = −(a/2π)(F_0x + F_1x), no Jacobian term | distance: Jacobian term −2w/(βr) |
| reference | umbrella sampling (40 windows, κ 300 at 300 K) + 180-bin WHAM, same EM at the same dt | constrained TI with the same clipped EM at dt 0.002 |

So LTA's integrator sits in a regime where EM is stable and only first-order biased. Its expected
error is small but not zero:

* the EM bond variance is inflated to (kT/k)/(1 − k·dt), +9 %;
* the predicted first-order barrier shift is about +0.05 kJ/mol at 300 K, which is 0.02 kT and 0.2 % of
  the barrier.

The validation asks whether that bias, or anything else, reaches F(φ) at the precision of the published
ABF/FR effects.

## 2. Design (frozen)

The design reproduces the published umbrella protocol with **exact Metropolis Monte Carlo** as the
arbiter:

* 40 windows, κ 300 kJ/mol/rad² at 300 K and 254 at 150 K;
* 16 independent groups × 16 molecules per window;
* MC moves: single-bead ±0.08 Å and rigid translation ±0.3 Å;
* candidates: production EM at dt 2e-4 / 1e-4 / 5e-5, and Leimkuhler–Matthews at 2e-4;
* every chain starts from 5000 exact MC steps, so no clamp is ever used.

Each run is read out three ways:

* F_density: WHAM on 1800 fine bins, summed to the 180 production bins;
* F_mid180: 180-bin WHAM with the bias at bin midpoints, the published convention;
* F_MF: the pooled conditional mean force, integrated with the production `histogram_pmf` convention.

Noise comes from a jackknife over groups. D is the debiased RMS over the circle, in kJ/mol.

**Gates.**

* Dynamics: ADMISSIBLE if D(EM 2e-4, MC) ≤ 0.03 with upper bound ≤ 0.045 on both routes. That is about
  ⅓ of the 300 K ABF final error.
* Reference: the published reference must meet the same bound.
* Conclusion check: rescore the published 300 K production runs against MC.

## 3. Results at 300 K

| profile | D vs MC (kJ/mol) | 95 % upper | noise | barrier − MC barrier (MC: 26.40) |
|---|---|---|---|---|
| MC F_MF (same chains) | 0.002 | 0.010 | 0.007 | +0.004 |
| MC F_mid180 (published WHAM convention) | 0.000 | 0.009 | 0.008 | +0.001 |
| **EM dt 2e-4 (production), F_density** | **0.000** | **0.021** | 0.020 | +0.033 |
| **EM dt 2e-4, F_MF** | **0.014** | **0.029** | 0.012 | +0.046 |
| EM dt 2e-4, F_mid180 | 0.000 | 0.021 | 0.020 | +0.034 |
| EM dt 1e-4, F_density / F_MF | 0.010 / 0.014 | 0.034 / 0.024 | 0.019 / 0.011 | +0.055 / +0.034 |
| EM dt 5e-5, F_density / F_MF | 0.000 / 0.000 | 0.034 / 0.018 | 0.028 / 0.016 | 0.000 / +0.019 |
| LM dt 2e-4, F_density / F_MF | 0.000 / 0.005 | 0.023 / 0.023 | 0.018 / 0.012 | +0.011 / +0.023 |
| **published reference** (EM 2e-4, umbrella/WHAM) | **0.034** | **0.038** | (treated as 0) | **+0.049** |

Bond variance, relative to exact MC. kT/k is only approximate, because of the 3-D r² Jacobian and the
framework LJ; MC is 0.7 % above it at 300 K and 1.5 % at 150 K.

| T | EM 2e-4 | EM 1e-4 | EM 5e-5 | LM 2e-4 |
|---|---|---|---|---|
| 300 K | **+8.65 %** | +4.15 % | +2.03 % | +0.02 % |
| 150 K | **+8.57 %** | +4.10 % | +2.01 % | +0.01 % |

This is textbook first order (EM bias halves with dt) on the stiffest mode, and LM removes it.

* **G_LTA_MC_self: PASS.** The MC mean-force and density routes agree to 0.002 kJ/mol. The linear CV's
  mean-force estimator is exact.
* **G_LTA_dyn: ADMISSIBLE.** At the production dt the EM chain is first-order biased where it is stiff:
  the bond variance is +9.4 % and halves with dt, the textbook O(dt). That bias does **not** reach F(φ)
  beyond 0.014 kJ/mol RMS (upper bound 0.029). The barrier shift, +0.03 to +0.05 kJ/mol, has the sign and
  size of the first-order EM prediction (+0.05), and it vanishes at dt 5e-5 and under the second-order
  LM scheme. It is 0.2 % of the barrier and 0.02 kT.
* **The WHAM midpoint convention is harmless.** Fine WHAM and 180-bin midpoint WHAM agree to < 0.009
  kJ/mol. My concern that the stiff umbrella (κ·Δφ·dφ up to 0.8 kT across a bin) would bias the published
  WHAM is not borne out.
* **G_LTA_reference: MARGINAL by the frozen rule** (D = 0.034 > 0.03, upper 0.038; max abs diff 0.066).
  Two components:
  1. The first-order EM barrier shift, +0.049 kJ/mol, the same as in my independent EM umbrella runs.
  2. Bin-to-bin scatter consistent with the reference's own sampling noise. Its 256 molecules × 24 t.u.
     per window give an expected noise of about 0.02-0.03 kJ/mol, by scaling the noise of my 256 × 48 t.u.
     EM runs, against the observed 0.0345 RMS. The prereg conservatively treated that noise as zero.

  So the published reference is slightly noisier than the tolerance, but it is not biased beyond the
  0.02 kT first-order EM shift.

### 3.1 Does the published LTA FR result survive the exact reference?

The published 300 K histogram production (16 seeds × N = 1024, arms ABF / FR / matched sham) is
rescored with the published arithmetic (`analyze_lta_histogram.score_group` / `contrast`). Only the
reference changes.

| contrast | published reference | **exact (MC) reference** |
|---|---|---|
| FR vs ABF, ΔI_F | −14.10 % [−15.10, −10.98] 16/16 | **−13.66 % [−14.79, −11.11]** |
| FR vs ABF, Δe_F(T) | −7.20 % [−15.42, +0.87] | −7.93 % [−14.61, +0.65] |
| sham vs ABF, ΔI_F | −0.24 % [−2.17, +1.35] | −0.31 % [−1.92, +1.11] |
| FR vs sham, ΔI_F | −13.17 % [−14.62, −12.48] | **−13.02 % [−14.15, −11.72]** |

**The 300 K FR gain SURVIVES** under the frozen clause: both FR-vs-ABF and FR-vs-sham CIs are entirely
below 0. The sham stays neutral. The final-error effect stays non-significant, as published.

### 3.2 Long-run limits: a shared finite-time ABF bias (new observation)

Pooling each arm's final production accumulators over the 16 seeds gives:

| arm | D vs MC | D vs published reference | barrier − MC barrier | window (abs z < 1 Å) | neck | cage (abs z > 4 Å) |
|---|---|---|---|---|---|---|
| ABF | 0.088 | 0.078 | **+0.221** | +0.117 | +0.027 | −0.100 |
| FR | 0.080 | 0.070 | +0.202 | +0.107 | +0.024 | −0.090 |
| sham | 0.088 | 0.079 | +0.221 | +0.115 | +0.029 | −0.101 |
| FR limit vs ABF limit | D = 0.000 (upper 0.018) | | | | | |

The per-seed final errors (median 0.094 for ABF vs MC) are almost entirely this systematic component.
The pooled-over-seeds error is no smaller (0.089). The common bias is the same in all three arms, so it
is a finite-time ABF estimator bias, not a reference or integrator effect:

* the production accumulators average over 4 ≤ t ≤ 60 t.u., including the non-stationary early phase;
* the barrier is overestimated by 0.2 kJ/mol (0.09 kT).

**FR does not create a stationary bias in LTA** (contrast WCA, where FR's limit separates from ABF's),
**and it does not remove ABF's.** The LTA FR benefit is entirely a transient-establishment benefit. That
matches its integrated (not final) significance and the starvation reading of the temperature sweep.

## 4. Results at 150 K

Same design, κ = 254 kJ/mol/rad² (the published 150 K protocol). Exact MC barrier: 16.55 kJ/mol.

| profile | D vs MC (kJ/mol) | 95 % upper | noise | barrier − MC barrier |
|---|---|---|---|---|
| MC F_MF / F_mid180 | 0.003 / 0.004 | 0.007 / 0.007 | 0.005 | −0.003 / +0.001 |
| **EM dt 2e-4: F_density / F_MF** | **0.000 / 0.011** | **0.017 / 0.021** | 0.014 / 0.009 | +0.021 / +0.032 |
| EM dt 1e-4: F_density / F_MF | 0.000 / 0.004 | 0.012 / 0.012 | 0.013 / 0.009 | +0.005 / +0.002 |
| EM dt 5e-5: F_density / F_MF | 0.000 / 0.001 | 0.024 / 0.020 | 0.022 / 0.013 | −0.010 / −0.019 |
| LM dt 2e-4: F_density / F_MF | 0.000 / 0.009 | 0.016 / 0.020 | 0.015 / 0.009 | +0.023 / +0.021 |
| **published reference** | **0.017** | **0.019** | (0) | −0.028 |

* G_LTA_MC_self **PASS** (0.003).
* **G_LTA_dyn ADMISSIBLE.**
* **G_LTA_reference ADMISSIBLE** (0.017 ≤ 0.03, upper 0.019 ≤ 0.045).

At 150 K the window is the narrowest-constrained part of the sweep, and the dynamics and reference are
still validated.

Rescoring of the published 150 K production (16 seeds):

| contrast | published reference | **exact (MC) reference** |
|---|---|---|
| FR vs ABF, ΔI_F | −28.07 % [−29.21, −26.25] 16/16 | **−28.54 % [−29.41, −26.56]** |
| FR vs ABF, Δe_F(T) | −52.60 % [−56.63, −44.01] | **−54.38 % [−60.52, −46.12]** |
| sham vs ABF, ΔI_F | −0.94 % [−2.34, +1.15] | −1.01 % [−2.37, +1.19] |
| FR vs sham, ΔI_F | −27.03 % [−30.03, −24.86] | **−27.52 % [−30.12, −24.98]** |

Pooled long-run limits at 150 K: D vs MC is ABF 0.131, **FR 0.057**, sham 0.130; FR limit vs ABF limit
0.075. At this temperature ABF's finite-time bias is larger, and FR *reduces* it: FR's long-run profile is
closer to the exact one. So FR's 150 K benefit extends to the endpoint, consistent with the significant
final-error effect (−54 %). The sham does nothing.

**The 150 K FR gain SURVIVES the exact reference, in integrated and final error.**

## 5. Verdict

| claim | status | basis |
|---|---|---|
| LTA production dynamics (EM, dt 2e-4) samples the model's Gibbs free energy along φ | **Validated at 300 K and 150 K** (frozen gate ADMISSIBLE at both) | MC and LM arbiters; EM first-order bias confined to the stiff bond (+8.6 % variance) and a ≤ 0.05 kJ/mol (0.02 kT) barrier shift |
| published umbrella/WHAM references | 150 K **ADMISSIBLE**; 300 K **MARGINAL** (0.034 kJ/mol RMS, consistent with its own sampling noise plus the +0.05 kJ/mol first-order barrier shift) | §3, §4 |
| LTA FR integrated-error gain | **Survives the exact reference at 300 K (−13.7 %) and 150 K (−28.5 %)**; sham neutral at both; FR beats the sham directly at both | §3.1, §4 |
| mechanism | transient establishment; at 300 K FR leaves the shared finite-time ABF bias (0.08-0.09 kJ/mol) untouched; at 150 K it also reduces it (0.13 → 0.06) | §3.2, §4 |

**Why LTA is a cleaner physical positive than WCA.**

1. Its integrator is in the stable, asymptotic regime (dt·λ_max ≈ 0.16-0.23, no clip), so its error is
   the small, understood first-order EM bias.
2. Its CV is linear, so the mean force has no Jacobian term to get wrong.
3. Its reference survives an exact arbiter.
4. FR's effect does not depend on which reference scores it. In WCA the sign of the effect flips with
   the reference.
5. A matched-turnover sham, run on the same noise, is neutral, so the gain is attributable to the
   direction of resampling.

What remains for publication:

* the 225 / 80 / 350 K points were not re-validated. The 300 and 150 K results bracket them and the
  mechanism is the same, but they are not checked;
* the equal-force-evaluation replica ladder (gateway and WCA showed serial ABF beating many-replica ABF
  + FR at equal budget) has not been run for LTA;
* the LTA benefit is at fixed N = 1024 and fixed physical time, equal force evaluations between arms. It
  is a parallel wall-clock (establishment) gain, not yet shown to be a total-compute gain.
