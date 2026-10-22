# Results: alanine dipeptide with an incomplete CV (φ alone or ψ alone)

* **Data.** 16 paired seeds, N = 2048, 100 ps per arm. All values are reproduced from the existing runs with the
  original analysis code, and agree with `results/alanine_1d/analysis/summary.json` to 10⁻¹².
* **Errors** are measured against the full 1-D marginal of the 2-D MBAR reference: equilibrium-weighted,
  additive-constant aligned, on the 8 kT mask.
* **Paired changes.** Median per-seed (FR − ABF)/ABF with a BCa 95 % CI, followed by the number of seeds where FR is
  better. A negative change means FR is better.

## ABF alone

| | φ alone | ψ alone |
|---|---|---|
| final e_F vs the full marginal | **0.374 kJ/mol** (reference-noise floor 0.205) | **2.070 kJ/mol** |
| final e_F vs the φ < 0 conditional | — | **0.514** (ratio 4.0: F(ψ) converges to the conditional) |
| final e_F′ (raw bin force) | 1.35 kJ/mol/rad, **reference-limited** (reference uncertainty 1.26) | 4.03 (reference uncertainty 1.70, reliable) |
| final e_ξ (instantaneous marginal; floor 0.086) | 0.093 | 0.122 |
| C7ax first visited | **16/16 seeds**, 2.3–3.5 ps (median 3.07) | **16/16 seeds**, 3.4–5.6 ps (median 4.41) |
| D_cond by window [1,5] / [5,20] / [20,50] / [50,100] (sampling floor) | 0.061 / 0.095 / 0.075 / 0.074 (0.043 / 0.041 / 0.022 / 0.017) | 0.061 / 0.321 / 0.288 / 0.249 (0.034 / 0.021 / 0.015 / 0.012) |
| walkers at φ > 0 (target 0.321) | — | window-pooled 0.000 / 0.023 / 0.067 / **0.129**; instantaneous at 100 ps 0.159 |

* **φ alone works.** The slow coordinate is biased directly, and F(φ) reaches 0.37 kJ/mol. About 30 % of the squared
  endpoint error is the reference's own noise.
* **ψ alone does not.**
  * C7ax is first *visited* in every seed by 5.6 ps, but a first visit is not equilibration.
  * The φ > 0 mass climbs roughly linearly to 0.13–0.16 at 100 ps, against a target of 0.321. The unbiased Gibbs
    0.032 is not the target under a uniform-ψ bias.
  * D_cond stays 0.25–0.32, far above its floor.
  * F(ψ) lies on the φ < 0 conditional profile and puts the barrier between ψ ≈ −130° and −50° about 9 kJ/mol too
    high.
  * The hidden φ equilibrates on a timescale longer than the run.

## FR vs ABF (own window [1, 100] ps)

| arm | ΔI_F | Δe_F(100) | ΔD_cond [50,100] | Δe_F′(100) | Δ mean e_ξ [1,100] | events/opp, ESS min | verdict (original rules) |
|---|---|---|---|---|---|---|---|
| φ, FR 0.02 | **−1.34 % [−3.30, −0.51]** 12/16 | +5.70 % [−0.27, +7.42] 4/16 | +1.3 % [−2.2, +3.6] | +0.5 % [−1.6, +1.3] | −3.2 % [−4.6, −2.2] | 2.3, 0.91 | INCONCLUSIVE (neutral in substance) |
| φ, FR 0.15 | **+18.9 % [+12.9, +21.1]** 0/16 | +67.1 % [+55.3, +78.8] 0/16 | +11.6 % [+9.7, +13.5] | −5.5 % [−9.0, −4.8] † | −6.3 % [−8.6, −4.8] | 18.6, 0.47 | HARMFUL |
| ψ, FR 0.02 | **+0.24 % [+0.03, +0.34]** 2/16 | +0.78 % [+0.43, +0.88] 1/16 | +3.4 % [+2.4, +4.1] 1/16 | +0.7 % [+0.6, +1.3] | −5.1 % [−6.7, −4.4] | 1.4, 0.96 | NEUTRAL_SIG |
| ψ, FR 0.15 | **+1.31 % [+1.03, +1.53]** 0/16 | +3.30 % [+2.79, +3.51] 0/16 | **+14.5 % [+12.3, +16.1]** 0/16 | +2.8 % [+2.0, +3.3] | −15.7 % [−17.4, −14.9] | 8.2, 0.77 | NEUTRAL_SIG |

† The φ e_F′ is reference-limited. The *closed* variant (the derivative of the reported PMF) reverses the sign:
1.43 vs ABF 1.32. Do not read a mean-force benefit into this, especially since e_F is 67 % worse.

**Hidden coordinate under FR (ψ alone).** FR slows the equilibration of the hidden coordinate in proportion to its
dose:

| arm | φ > 0 mass, [50,100] window | φ > 0 mass at 100 ps (instantaneous) | C7ax occupancy at 100 ps |
|---|---|---|---|
| ABF | 0.129 | 0.159 | 0.144 |
| FR 0.02 | 0.105 | 0.123 | 0.113 |
| FR 0.15 | 0.064 | 0.069 | 0.063 |

## Readings

1. With an incomplete CV, ABF on the slow angle φ still works, while ABF on the fast angle ψ converges to the wrong,
   conditional profile on the 100 ps horizon.
2. Uniform FR does not repair the hidden coordinate.
   * Gentle FR is neutral on F.
   * Strong FR is harmful on φ, where it keeps resampling an already flat marginal, and on ψ it degrades the hidden
     conditional (D_cond +14.5 %).
   * This is consistent with the project's theorem that marginal reallocation leaves p(hidden | ξ) unchanged in
     expectation.
   * The proposed mechanism is that FR kills in the over-filled ψ region where the bias still lags, removing the
     walkers that carry φ > 0. It is a hypothesis from the original study, not tested here.
3. Marginal uniformity is not accuracy. FR flattens the CV marginal (Δe_ξ −3 % to −16 %) while F and the hidden
   conditional stay the same or get worse.
4. Reference limits:
   * The 1-D free-energy reference carries 0.20–0.27 kJ/mol of noise.
   * The 1-D mean-force reference is reliable for ψ, but for φ its uncertainty (1.26) is about the size of ABF's
     endpoint e_F′ (1.35), so the φ mean-force panel is informative only during the first ≈ 10 ps.
   * Alternative φ diagnostics: the seed dispersion of the mean force at 100 ps is 0.09, and the indicative
     noise-debiased error is 0.67 kJ/mol/rad.
