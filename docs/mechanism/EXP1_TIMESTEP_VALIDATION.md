# Experiments I and II: numerical-validation gate

*2026-10-10. The gate is frozen in `docs/mechanism/SCIENTIFIC_PLAN.md` §3 and Amendment 1 (bf56da4); runs
11:52–13:26 UTC; analysis `scripts/mechanism/analyze_validation.py`.*

* Summary: `results/mechanism/validation/summary.json`.
* Figure: `figures/mechanism/validation/exp1_timestep_validation.{pdf,png}`.
* Harness: `src/gateway_family_validation.py`, tested by `tests/test_gateway_family_validation.py` (57 tests).

## Verdict: **PASS for all seven dynamics at h = 2.5 × 10⁻⁵**

The seven dynamics are α = 1, 0.5, 0 and the shifted fibre (λ = 1), plus λ = 0.5, 0.25, 0.1 on V₁.
* The verdict is the same under the plan reading and the strict reading of V1.
* No V4 clause is rejected by Holm (32 tests, smallest p 0.021, which needs ≤ 0.0016).
* The ABF-bias smoke test passes for every dynamics.
* So **one common h = 2.5 × 10⁻⁵**, B = 3.2768 × 10⁹ walker-steps per arm per seed, and identical physical horizons
  T_N = 40 / 160 / 640 for N = 2048 / 512 / 128 in every cell.
* No refinement was needed, and no ABF/FR performance was read.

## 1. Analytical identities (verified before the gate)

For V_α = H(x² − 1)² + ((1 − α)/β) log ω + ½ ω^{2α} y² and V_sh = F\* + ½κ²(y − m)² with m = (√(2/β)/κ) log ω:

| identity | derivation | numerical check |
|---|---|---|
| Y \| x ~ N(0, 1/(βω^{2α})) or N(m, 1/(βκ²)) | Gaussian in y at fixed x | quadrature, 11 x values; engine `init_family` moments (tests) |
| F_model = F\* + C | ∫e^{−βV}dy = √(2π/(β·stiffness)) e^{−β(non-y part)}; the α part gives (α/β) log ω, which adds to the (1 − α)/β term | constant to 2 × 10⁻¹⁶ on a 721-point grid |
| E[∂ₓV \| x] = F\*′ | E[y²] = 1/(βω^{2α}) removes the α factor; E[y − m] = 0 | 2 × 10⁻¹⁵ |
| Var(∂ₓV \| x) = (2α²/β²)(ω′/ω)² for the α family; (2/β²)(ω′/ω)² for the shifted fibre (κ = 1) | Var(y²) = 2σ⁴; Var(κ²(y − m)m′) = κ⁴m′²/(βκ²) | 2 × 10⁻¹⁵ relative |
| analytic gradient | — | central FD, 1 × 10⁻¹⁰ |

The sign convention is the production one: the deposited sample is ∂ₓV, so the variance is sign-free.

## 2. Gate results at h = 2.5 × 10⁻⁵

16 groups × 256 walkers × 500/λ t.u. flat-bias EM, plus MALA at h 1e-4 and exact i.i.d. draws, per dynamics.
D is the debiased RMS with a group jackknife; "up" is its 95 % upper bound.

| dynamics | V1: conditional-variance inflation, measured [two central fine bins] vs closed form | V2: D(F_density) [up] | V2: D(F_MF) [up] | V3: D(mean force) [up] | V3b: Var(f\|x) rel. dev. | verdict |
|---|---|---|---|---|---|---|
| α = 1 | +1.32 %, +1.30 % (± 0.05–0.06) vs +1.30 % | 0.00008 [0.00056] | 0 [0.00020] | 0.00075 [0.00103] | 1.0 % | PASS |
| α = 0.5 | +0.09 %, +0.08 % (± 0.16) vs +0.04 % | 0.00032 [0.00085] | 0.00010 [0.00028] | 0.00032 [0.00068] | 0.4 % | PASS |
| α = 0 | −0.09 %, −0.05 % (± 0.33) vs +0.001 % | 0 [0.00039] | 0 [0.000001] | 0.000003 [0.00001] | Var/⟨f²⟩ = 0 (deterministic) | PASS |
| shifted fibre | −0.08 %, −0.08 % (± 0.34) vs +0.001 %; mean of y in the gate bins within 3 se of m(x) | 0 [0.00063] | 0 [0.00058] | 0.00079 [**0.0033**] | 0.1 % | PASS |
| λ = 0.5 | +0.67 %, +0.65 % (± 0.04) vs +0.64 % | 0 [0.00032] | 0.00012 [0.00026] | 0.00059 [0.00089] | 0.6 % | PASS |
| λ = 0.25 | +0.36 %, +0.35 % (± 0.04) vs +0.32 % | 0 [0.00026] | 0 [0.00020] | 0 [0.00050] | 0.3 % | PASS |
| λ = 0.1 | +0.13 %, +0.15 % (± 0.03) vs +0.13 % | 0.00010 [0.00031] | 0 [0.00013] | 0 [0.00033] | 0 % | PASS |

The thresholds are those of the frozen gateway gate:
* V1: closed form ≤ 2 % and measured ≤ 2 % + 2 se;
* V2: D ≤ 0.00185, upper bound ≤ 0.0028;
* V3: D ≤ 0.0115, upper bound ≤ 0.017;
* V3b: ≤ 5 %;
* V4: MALA and the exact sampler reproduce the conditional law and F; Holm family-wise 5 %;
* V5: finite states, wall reflections < 1e-6 per walker-step unbiased, the flat-bias walkers cross the gate, ABF-bias
  smoke.

**V5 ABF-bias smoke** (`scripts/mechanism/abf_smoke.py`, `results/mechanism/validation/abf_smoke.json`; N = 2048,
4 t.u., 2 seeds, both arms, Amendment A1 reading):
* Every dynamics is finite, with final max |Γ| on bins with ≥ 100 deposits = 7.9, against the limit 2 × max |F\*′| = 15.8.
* The running maximum over all bias reads is 7.9–8.5 for λ ≥ 0.5, 14.4 for λ = 0.25 and 19.6 / 30.3 for λ = 0.1.
* The cause is physical: walkers reach the gate flank still carrying well-width y and deposit large local forces into
  nearly empty bins. That transient is the subject of Experiment II.

## 3. Mechanistic numbers measured on the way (descriptive)

These are inputs for Experiments I and II; they are not part of the gate.

**Conditional-force variance** (panel e): the measured Var(f | x) profiles fall on the exact curves for every dynamics.
* Peak values: 2.06 for α = 1, the shifted fibre and every λ; 0.52 for α = 0.5; 0 for α = 0.

**Conditional relaxation** (panels g–i): integrated ACF times of z_y and z_f along flat-bias trajectories, conditioned
on x in windows around −1, −0.21 and 0.
* α family: they agree with the frozen-x OU predictions τ_y = 1/(λ·stiffness(x)) (and τ_{y²} = τ_y/2) over four
  decades.
* **Shifted fibre:** the frozen-x prediction (τ = 1/κ² = 1) does **not** describe it. At the gate its conditional
  relaxation is about 30× faster, because x and y are strongly coupled in the curved channel. Its y-ACF also goes
  negative at the gate.
  * So the shifted fibre is variance-matched to α = 1 but has a different, coupled relaxation, as §1.1 of the plan
    warned.
  * Its conditional mean-force deviation near the gate reaches ±0.008 (panel d). This is within V3's tolerance, and it
    is the visible signature of the y lag behind m(x).

**Transverse variance inflation** reproduces 1/(1 − λ·stiffness·h/2) at every λ, which confirms that drift and
noise are both scaled.

## 4. Engine equivalence (prerequisite, `tests/test_gateway_family_numba.py`, 83 tests)

* `src/gateway_family_numba.py` at α = 1, λ = 1 is **bitwise** `src/gateway_ladder_numba.py`: both arms, N 2, 128 and
  2048, all diagnostics on, including a 400 000-step production-knob run.
* The harness's unbiased EM chain is bitwise the engine's ABF arm (bias off) for all seven dynamics, so the gate
  validates the production integrator itself.
* The diagnostics D1–D4 are inert (bitwise on/off) and match an independent pure-Python reference.
* Under Amendment A4, every production α = 1 / λ = 1 job is also compared bitwise with its equal-budget counterpart
  (`results/mechanism/reuse_gate/alpha1_bitwise.json`).

## 5. Cost

245 jobs, measured 57 core-h on 120 cores, 1 h 34 min wall. Each λ = 0.1 flat-bias group took 1.6 h.
