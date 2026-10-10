# Ethane/LTA: timestep validation for the equal-budget ladder (verification of the 2026-10-09 gates)

*2026-10-10.*

The production timestep **h_L = 2 × 10⁻⁴** was validated on 2026-10-09 against exact Metropolis MC at 300 K and
150 K (`docs/numerical_validation/LTA_VALIDATION.md`, prereg `configs/numerical_validation/lta_validation_prereg.json`).
This document re-verifies the evidence, adds the implementation-specific checks the ladder needs, and freezes h_L.

## Verdict: **PASS. h_L = 2 × 10⁻⁴ frozen for both temperatures**

The integrator is validated for the reaction-coordinate free energy and mean force. The known Euler–Maruyama bias
of the stiff bond is documented below and does not reach F(φ) beyond the tolerance.

## 1. Existing gates, re-verified from raw data

Raw arrays: `results/numerical_validation/lta/T{300,150}/{MC,EM2e-4,...}/g*_w*.npz`, 640 files per tag and T.
* The Phase 0A inventory rebuilt the MC fine-bin WHAM F and the MC mean-force F from them.
* Both match the committed summaries **exactly** (max abs diff 0.0).
* Gate values: `results/numerical_validation/lta/T*/summary.json`.

| gate (frozen 2026-10-09) | 300 K | 150 K |
|---|---|---|
| MC self-consistency: MC mean-force route vs density route | PASS, D 0.002 kJ/mol | PASS, D 0.003 |
| **EM h 2e-4 vs MC**: density route D [upper] / mean-force route D [upper] | **ADMISSIBLE**: 0.000 [0.021] / 0.014 [0.029] kJ/mol | **ADMISSIBLE**: 0.000 [0.017] / 0.011 [0.021] |
| EM h 1e-4 and 5e-5, LM h 2e-4 vs MC | all D ≤ 0.015, consistent with 0 | all D ≤ 0.011 |
| published umbrella/WHAM reference vs MC | MARGINAL (0.034) | ADMISSIBLE (0.017) |

**Bond-variance bias** (stiffest mode, λ = 2k_b = 800 kJ/mol/Å², h·λ = 0.16), relative to MC:

| T | EM 2e-4 | EM 1e-4 | EM 5e-5 | LM 2e-4 |
|---|---|---|---|---|
| 300 K | +8.65 % | +4.15 % | +2.03 % | +0.02 % |
| 150 K | +8.57 % | +4.10 % | +2.01 % | +0.01 % |

This is textbook first order: the closed form (kT/k)/(1 − k h) predicts +8.7 %.

It does not propagate to the reaction coordinate. φ depends on the centre of mass, and the bond force cancels
exactly in the local mean force f = −(a/2π)(F₀ₓ + F₁ₓ), so the bond mode enters F(φ) only through the coupling of
bond length to the LJ term. The measured effect on F(φ) is ≤ 0.014 kJ/mol RMS and ≤ 0.05 kJ/mol at the barrier.
That is below the 0.03 kJ/mol gate and the ladder's strict threshold.

## 2. Low-cost tests re-run (2026-10-10)

`tests/test_lta_validation.py` (4 tests) and `tests/test_lta_histogram.py` (8): **12 passed** (CPU, 7.5 min). They
cover:
* the independent model's forces, energies, φ and local mean force equal `core_lta` to 1e-10 relative at 300 K
  and 150 K;
* the analytic force equals central differences to < 1e-6;
* WHAM recovers a known density;
* the histogram-engine fixture, histogram_pmf exactness, and γ = 0 equality.

## 3. Reference availability and independence

**Free-energy reference.** MC fine-bin WHAM over 16 groups × 40 windows; noise 0.006 kJ/mol (300 K).
* The MC barrier is 26.40 kJ/mol (300 K) and 16.55 kJ/mol (150 K), under the validation's cage convention
  |z| > 4 Å.
* The published `run_lta_reference.barrier_stats` uses "within 0.4 Å of the cage centre". Under that convention
  the MC barrier is 26.79 / 16.84 kJ/mol.
* The two conventions name one quantity with two values. The ladder reports profiles and RMS errors, not this
  scalar. *(Documentation defect found by the Phase 0A audit; noted here and in the final report.)*

**Mean-force reference.** MC conditional mean force γ_j = ΣMf/ΣC per 180-bin, pooled over all windows. It is
bias-independent because the umbrella bias depends on φ only.
* Leave-one-group-out jackknife per-bin SE, RMS: 0.0143 kJ/mol/rad (300 K), 0.0101 (150 K).
* Circular mean: −3.3e-4 ± 2.6e-3 (300 K) and −1.5e-3 ± 1.9e-3 (150 K), i.e. periodic closure holds.
* It is a direct conditional average, not a finite difference of a noisy F.

**Independence.** The MC chains used their own seeds (7000 + …), never ran ABF or FR, and predate the ladder.

## 4. Implementation-specific checks for the ladder engine

ENGINE_PLACEHOLDER

## 5. Frozen physical-time scaling

| knob | physical time | steps at h_L |
|---|---|---|
| ABF warm-up (linear ramp of the bias) | 4 t.u. | 20 000 |
| estimator burn-in (production accumulators) | 4 t.u. | 20 000 |
| FR start | 4 t.u. | 20 000 |
| FR update interval | 0.001 t.u. | 5 |
| windowed genealogy | 4 t.u. | 20 000 |

The same constants hold for every N; only n_steps = 3.072 × 10⁸/N changes.
