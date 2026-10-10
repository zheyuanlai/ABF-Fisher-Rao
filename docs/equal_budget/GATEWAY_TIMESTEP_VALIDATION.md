# Entropic gateway: timestep validation for the equal-budget ladder

*2026-10-10.*

* Gate frozen before any run: `configs/equal_budget_v2/gateway_validation.json`.
* Harness: `src/gateway_validation.py`. Tests: `tests/test_gateway_validation.py` (6 pass).
* Runs: `scripts/equal_budget/run_gateway_validation.py`. Analysis: `scripts/equal_budget/analyze_gateway_validation.py`.
* Data: `results/equal_budget_v2/gateway_validation/` (`summary.json`).
* Figure: `figures/equal_budget_v2/gateway/validation/fig_gateway_timestep_validation.png`.

## Verdict: **PASS. Production timestep h_G = 2.5 × 10⁻⁵** (frozen)

h_G is the largest candidate in {4e-4, 1e-4, 2.5e-5} that passes the frozen conditional-variance, free-energy and
mean-force gates. The choice used no ABF or FR outcome.

| h | Var(Y\|x≈0) inflation, measured ± se (closed form) | D(F, exact), density route [95 % upper] | D(F, exact), mean-force route [upper] | D(mean force, exact) [upper]; closed-form EM bias | V1 (≤ 2 %) | V2 (F ≤ 0.00185) | V3 (F′ ≤ 0.0115) |
|---|---|---|---|---|---|---|---|
| 4e-4 (historical) | **+26.0 ± 0.08 %** (25.75 %) | 0.00135 [0.00145] | 0.00139 [0.00143] | **0.0149** [0.0150]; 0.0145 | FAIL | pass | **FAIL** |
| 1e-4 | **+5.45 ± 0.09 %** (5.40 %) | 0.00038 [0.00054] | 0.00031 [0.00038] | 0.0034 [0.0035]; 0.0033 | **FAIL** | pass | pass |
| **2.5e-5** | **+1.35 ± 0.04 %** (1.30 %) | 0.00036 [0.00069] | 0.00000 [0.00018] | 0.0008 [0.0010]; 0.0008 | pass | pass | pass |
| MALA, h 1e-4 (exact) | +0.03 / −0.06 ± 0.08 % (0) | 0.00000 [0.00041] | — | 0.0000 | V4 pass | | |

* **V4.** The exact MALA sampler (acceptance 0.9989) reproduces the exact Gibbs law. This validates the harness
  and the read-out.
* **V5.** The unbiased trajectories at every h had no non-finite state and no wall reflection in 200 t.u. Walkers
  stay in their wells; the βH = 8 kT barrier is not crossed without bias. The flat-bias walkers cross the gate
  repeatedly (panel e).

## G1. Analytic gradient

Implemented potential, as in `gateway_numba.py`:

* V(x, y) = H(x²−1)² + ω(x)²y²/2, with ω(x) = 1 + 31·exp(−x²/(2·0.1²)), β = 16, H = 0.5.
* x is reflected into [−1.8, 1.8]; y is unbounded.

Checks (`tests/test_gateway_validation.py`):

* **G1a.** The analytic gradient equals central differences at 80 points to < 1e-6 relative. The points cover the
  wells, the flanks and the gate (x = 0, ±0.01, ±0.05, ±0.1, ±0.12), with y up to 5 conditional standard
  deviations.
* **G1b.** The *production* force equals it to < 1e-7 relative. The force is extracted from
  `gateway_numba.simulate` itself (one step, zero external noise, bias forced to 0).
* **G1c.** The reference of `eb_abffr_core` is F = H(x²−1)² + β⁻¹log ω + const and F′ = 4Hx(x²−1) + ω′/(βω),
  identical to 1e-10. These are the exact marginal and mean force of this V, because Y | X = x ~ N(0, 1/(βω²)).
* **H1.** The harness's EM chain is **bitwise** the production `simulate` under the same noise.

A latent edge case found during the audit: `gateway_numba.simulate` with `ess_window <= 0` never initialises its
ancestor labels, and its save block then indexes garbage, which caused a segfault. Production always passes
4000, so no historical result is affected. The new ladder engine initialises the labels unconditionally.

## G2. Trajectories

See panel (e) of the figure:

* Unbiased EM from both wells and from the gate is stable at all three h, with no reflection.
* The largest one-step |Δx| is 0.036 at h 4e-4 and 0.011 at h 2.5e-5.
* A walker started at the gate falls into a well within a few t.u.
* Flat-bias walkers diffuse across the whole domain.

## G3. Transverse conditional law

The stiffest point is the gate centre, ω = 32. Exact: Var(Y|x) = 1/(βω²). The EM y-update at fixed x has variance
1/(βω²(1 − ω²h/2)).

* The measured inflation in the two central fine bins (width 0.0025) reproduces the closed form at every h:
  25.97/26.02 % vs 25.75 %, 5.45 % vs 5.40 %, and 1.35 % vs 1.30 %.
* The full profile in panel (a) follows the closed form over the gate region.
* The small excess, about 1 se, is consistent with the coupling to the moving x coordinate.
* MALA at the same location gives 0 within 0.08 %.

**The historical h = 4 × 10⁻⁴ samples a transverse fibre 26 % too wide at the gate.**

## G4. Free energy

Design: the flat-bias chain samples exp(−β(V − F_exact)), so −kT log p_h(x) = F_h(x) − F_exact(x) + const.

* At h 4e-4 the error is a gate-localised bump of +0.0065, 1.2× the anchor ABF endpoint error 0.0055 at the gate.
  It reproduces the 2026-10-04 floor of 0.00138.
* At 2.5e-5 it is below 0.0004 everywhere. Away from the gate the density-route residual at the 5e-4 level is
  incomplete x-mixing in 500 t.u., included in the jackknife noise.
* Both routes pass V2 at every h. **The historical timestep would have passed a free-energy-only gate.** Only the
  conditional-variance and mean-force checks reject it.

## G5. Mean force

The conditional mean force ⟨∂ₓV | x⟩_h follows the closed-form EM prediction F′ + (ω′/(βω))[1/(1 − ω²h/2) − 1]
(panel c): ±0.063 at h 4e-4, ±0.014 at 1e-4, and ±0.003 at 2.5e-5.

## G6. Independent invariant-measure check

* **MALA.** EM proposal plus Metropolis–Hastings on V − F_exact. Proposals leaving [−1.8, 1.8] are rejected,
  since the target is zero there. It is exact for any h.
* **Exact i.i.d. sampler.** Inverse CDF in x, Gaussian y; used for the read-out code.

Both agree with the exact law within their statistical errors (V4; tests H2).

## Consequences for the production ladder

* The production ladder uses **h_G = 2.5 × 10⁻⁵**. The anchor (N₀ = 2048, T₀ = 40) gives B_G = 2048 × 1.6 × 10⁶ =
  **3.2768 × 10⁹ walker-steps per arm per seed**.
* Physical-time FR knobs at h_G:
  * fr_every = 0.004/h = 160 steps;
  * ramp timescale 4 t.u. = 160 000 steps;
  * windowed-genealogy window 1.6 t.u. = 64 000 steps.
* Earlier gateway ladders at h 4 × 10⁻⁴ (`results/gateway_replica_ladder/`) are kept for history and engineering
  checks only. They are **not** a validated physical result under this standard.
