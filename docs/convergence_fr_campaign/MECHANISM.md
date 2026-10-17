# Mechanism: why continuous FR loses late, why stopping it works, and whether circulation matters

*Read-only diagnostics; neither algorithm was modified. Code:*
* *`scripts/convergence_fr/decay_slopes.py`;*
* *`circulation.py`;*
* *`scripts/overnight/fr_mechanism.py` (pooled late-half estimates with a seed jackknife, hidden-coordinate
  conditionals);*
* *the `secondary` block of `analyze_block.py`.*

*Data: `results/convergence_fr_campaign/mechanism/` and `analysis/`. "Pooled" means the accumulators are summed over
the 16 seeds, which removes most noise and leaves the systematic part of an estimator.*

## 1. Cumulative histogram ABF is memory-limited

| block | decay slope of e_F on [0.4T, T], ABF | same, FR arm | pooled whole-run estimate vs MC, ABF / FR | pooled post-window estimate vs MC, ABF / FR | per-seed post-window estimate, FR vs ABF |
|---|---|---|---|---|---|
| ITE 300 K, K1024, rate 2, continuous (expl.) | -0.84 | -0.69 | 0.074 / 0.084 | 0.066 / 0.077 | +17 % [+1, +24] |
| ITE 150 K, K256, rate 0.5, continuous, T200 (expl.) | -0.78 | -0.52 | 0.041 / 0.071 | 0.040 / 0.070 | +57 % [+30, +101] |
| ITE 300 K, K1024, rate 2, **stop 10** | -1.00 | -1.04 | 0.074 / 0.048 | 0.009 / 0.011 | -0 % [-21, +25] |
| ITE 150 K, K256, rate 0.5, **stop 15**, T100 | -0.87 | -0.81 | 0.075 / 0.059 | 0.017 / 0.018 | +15 % [+0, +58] |
| ITE 300 K, K256, rate 1, continuous | -0.85 | -0.54 | 0.082 / 0.104 | 0.075 / 0.098 | +11 % [-1, +36] |
| ITE 150 K, K256, rate 1, continuous | -0.98 | -0.45 | 0.118 / 0.147 | 0.115 / 0.144 | +31 % [+7, +55] |
| ERI 600 K, K256, rate 2, continuous | -0.94 | -0.55 | 0.117 / 0.261 | 0.101 / 0.247 | +126 % [+88, +280] |
| ERI 600 K, K1024, rate 2, continuous | -0.82 | -0.58 | 0.116 / 0.252 | 0.098 / 0.237 | +142 % [+123, +166] |

* The post-window is "after t_stop" for E1/E2 and "after the FR start" for the others.
* **ABF's late error falls as about 1/t in every cell** (slope −0.78 to −1.00), not as the 1/√t of sampling noise.
  * Pooled over seeds, the whole-run ABF estimate is about as far from MC as a single seed's (ITE: 0.074–0.075 pooled
    vs 0.076 per seed in E1). The error is therefore systematic.
  * The deposits made after 10–15 t.u. are nearly unbiased: E1/E2 post-stop windows, pooled 0.009–0.018 in both
    arms. For the continuous blocks the post-window starts at 1 t.u. and still contains the early deposits.
  * **Conclusion.** The end-of-run error of cumulative histogram ABF in these cells is the diluting memory of early,
    non-equilibrium deposits, made while the bias and the population were still far from stationary.
* **Early-stopping FR keeps ABF's 1/t law (−1.04, −0.81) with a smaller memory prefactor.**
  * Its pooled whole-run estimate is closer to MC: 0.048 vs 0.074 (E1) and 0.058 vs 0.074 (E2).
  * Its post-stop deposits are as unbiased as ABF's: pooled 0.011 vs 0.0095 (E1) and 0.018 vs 0.017 (E2).
  * The FR period populated the window and flank sooner (establishment 1.9 / 2.7 t.u. earlier), so fewer of the
    early deposits were made under a badly wrong bias. That advantage is then diluted like ABF's own error, and its
    ratio persists (−32 %, −19 % at T).
* **Continuous FR decays only as about t^−½ (−0.45 to −0.69).** It adds a contribution that does not dilute: its
  post-start deposits are biased.
  * ITE, per-seed estimate from the post-start deposits vs ABF's: +17 % (X1), +57 % (X2), +11 % (M300, not resolved)
    and +31 % (M150).
  * ERI: +126 % / +142 %, the pooled late-half bias being 0.13 kJ/mol against ABF's 0.03.
  * That is why its curves cross ABF's (ITE 300 K: 17–29 t.u.; ITE 150 K: 33–39; ERI: about 5).

## 2. FR-induced late bias: where it lives

| block | pooled late-half vs MC, ABF | same, FR | FR vs ABF D (noise) | η conditional RMS vs MC, ABF / FR |
|---|---|---|---|---|
| ITE 300 K, K1024, rate 2, continuous (expl.) | 0.013 | 0.035 | 0.028 (0.011) | 0.0005 / 0.0007 |
| ITE 150 K, K256, rate 0.5, continuous, T200 (expl.) | 0.010 | 0.039 | 0.031 (0.012) | 0.0004 / 0.0007 |
| ITE 300 K, K1024, rate 2, **stop 10** | 0.004 | 0.007 | 0.000 (0.011) | 0.0007 / 0.0005 |
| ITE 150 K, K256, rate 0.5, **stop 15**, T100 | 0.004 | 0.000 | 0.000 (0.016) | 0.0006 / 0.0006 |
| ERI 600 K, K256, rate 2, continuous | 0.027 | 0.130 | 0.100 (0.036) | 0.0033 / 0.0160 |
| ERI 600 K, K1024, rate 2, continuous | 0.029 | 0.127 | 0.098 (0.016) | 0.0067 / 0.0171 |

* **ITE.**
  * With continuous FR the late-half bias is 0.035 / 0.039 against ABF's 0.013 / 0.010 (X1 / X2), with clean
    hidden-coordinate conditionals.
  * With early stopping (E1, E2) the late-half FR − ABF difference is **zero** (D = 0.000).
* **ERI.**
  * Continuous FR's late-half estimate is 0.13 kJ/mol off, against ABF's 0.03; the FR − ABF difference is 3–6× its
    noise.
  * The **η conditional in the cages is distorted** by 2.5–5×. η is the position along the cage's long c axis.
* **Hypothesis for ERI (not established).** The uniform target keeps draining the cage and refilling the window, so
  long-resident cage walkers are replaced by recent arrivals whose η has not relaxed.
  * The result is a non-equilibrium steady state with a wrong conditional mean force.
  * The resulting error in the bias then keeps the natural marginal non-uniform, which keeps FR active.
* **Why this does not contradict the theorem.** The theorem (FR preserves p(y|φ) *instantaneously*) is not
  contradicted. The distortion is a property of the FR + dynamics steady state, not of a single resampling event.

## 3. Periodic circulation (Experiment 5)

* **Closure convention (checked).** For the linear periodic CV the exact mean force integrates to zero. The
  references satisfy this: ITE 300 K mean Γ_ref 0.00025 kJ/mol/rad; the 150 K smooth reference is exactly 0.
* **Read-out vs dynamics.** The read-out `histogram_pmf` projects the circular mean out. The applied bias, in both
  arms, does not: c(t) = mean_j Γ_j(t) is a non-conservative tilt of 2πc per period.
* **Definition used below.** c is the late-half average of the cumulative read-out Γ. The "late-deposit"
  definition, Γ from deposits after T/2, is also in the JSON. It reproduces the overnight numbers: X1 0.008 / 0.015,
  X2 0.011 / 0.025.

| block | c late half, ABF / FR (kJ/mol/rad) | paired FR − ABF [95 % CI] | tilt 2π\|c\|/k_BT, ABF / FR | within-arm ρ(c, e_F(T)), ABF / FR | ρ(Δc, Δe_F(T)) across seeds |
|---|---|---|---|---|---|
| ITE 300 K, K1024, rate 2, continuous (expl.) | +0.025 / +0.030 | +0.0016 [-0.0051, +0.0111] | 0.06 / 0.07 | +0.00 / +0.18 | +0.16 |
| ITE 150 K, K256, rate 0.5, continuous, T200 (expl.) | +0.031 / +0.048 | +0.0084 [+0.0017, +0.0141] | 0.16 / 0.24 | +0.81 / +0.63 | +0.15 |
| ITE 300 K, K1024, rate 2, **stop 10** | +0.029 / +0.025 | -0.0042 [-0.0091, +0.0013] | 0.07 / 0.06 | +0.29 / +0.42 | +0.27 |
| ITE 150 K, K256, rate 0.5, **stop 15**, T100 | +0.061 / +0.052 | +0.0005 [-0.0227, +0.0136] | 0.31 / 0.26 | +0.52 / +0.79 | +0.51 |
| ITE 300 K, K256, rate 1, continuous | +0.037 / +0.029 | +0.0110 [-0.0283, +0.0350] | 0.09 / 0.07 | +0.38 / +0.64 | +0.67 |
| ITE 150 K, K256, rate 1, continuous | +0.040 / +0.047 | -0.0011 [-0.0218, +0.0179] | 0.20 / 0.24 | +0.41 / +0.75 | -0.09 |
| ERI 600 K, K256, rate 2, continuous | -0.046 / -0.025 | +0.0023 [-0.0472, +0.0246] | 0.06 / 0.03 | +0.01 / +0.25 | -0.30 |
| ERI 600 K, K1024, rate 2, continuous | -0.025 / -0.022 | +0.0070 [-0.0005, +0.0132] | 0.03 / 0.03 | +0.26 / +0.02 | +0.20 |

**Observations.**
* **Circulation is present in every cell and both arms.** It is positive in ITE and negative in ERI, at 0.03–0.3
  k_BT per period.
* **The FR − ABF difference is resolved only in X2.**
* **Circulation correlates with final error, mostly within the FR arm.** ρ is 0.42–0.79 in every ITE block except
  X1 (0.18). ABF's within-arm correlation is ≥ 0.6 only in X2 (0.81). The seed-wise FR − ABF circulation difference
  tracks the seed-wise error difference only in M300 (0.67) and E2 (0.51); elsewhere ρ is −0.30 to +0.27.
* **Circulation does not separate the two FR schedules.** E1's stopped-FR arm circulates as much as continuous FR,
  0.025 vs 0.030 (different seeds), yet its final error is 32 % below ABF's instead of 14 % above. The ERI late bias,
  the largest in the campaign, has no FR − ABF circulation difference.
* **The tilt is small at 300 K** (0.06–0.09 k_BT per period, a drift of about 0.5 Å over 25 t.u.) and moderate at
  150 K (0.16–0.31).

**Answer to Q6.**
* Nonzero circulation of the applied histogram bias is real. It is a plausible *minor* contributor at 150 K, where
  the tilt is up to 0.3 k_BT per period and correlates with error.
* It is **not supported as the explanation of the late-time FR penalty**:
  1. FR does not reliably increase it;
  2. stopping FR removes the penalty while the circulation stays about the same. E1 vs X1, 0.025 vs 0.030, is a
     cross-block comparison on different seeds and was not tested;
  3. the largest FR penalty (ERI) has no circulation difference.
* **Correction experiment not run.** The frozen trigger (§6) reads: "within-arm |ρ| ≥ 0.6 between circulation and
  late error in both ITE temperatures, **and** the FR − ABF circulation difference is resolved", and it also requires
  budget.
  * **Reading 1: "late error" = the late-window (last 20 %) estimate error,** the §6 diagnostic. Every ITE |ρ| is
    ≤ 0.47 (ABF −0.40…+0.30, FR 0.07…0.47), so the trigger is **not met**.
  * **Reading 2: "late error" = e_F(T).** The FR arm reaches ρ ≥ 0.6 at both temperatures (M300 0.64; X2 0.63, E2
    0.79, M150 0.75) but not in X1 (0.18) or E1 (0.42). ABF reaches it only in X2 (0.81). The difference clause holds
    only in X2.
  * Reading 2 therefore meets the trigger only partly, and only through the FR arm.
  * The budget clause failed in any case: about 1.6 core-h remained, against the ≥ 7 core-h a two-arm corrected
    block at K ≥ 256 needs.

## 4. Why the stop works (summary)

* **ABF's early phase.** ABF spends its first 5–15 t.u. building a bias while the window is under-populated, and its
  cumulative estimator never forgets those deposits.
* **What FR adds.** During that phase FR is useful. It populates the window 1.4–3.1 t.u. sooner and lowers the
  early post-ramp integrated error by 18–37 %: E2 −18 %, M300 −20 %, M150 −28 %, X1 [1, 10] −34 %, E1 [1, 20]
  −37 %.
* **What FR costs later.** After establishment FR has nothing left to repair. It keeps resampling (X1: 90
  replacements per t.u. in the late half, 70 % of its early rate), and its late deposits are biased.
* **Stopping keeps the first part and avoids the second.** After the stop the run is plain ABF (bitwise by test),
  with a better-conditioned memory.
* **ERI is the counter-example.** ABF is never establishment-limited there (2.1 t.u.), so the useful phase is
  shorter than any stop time the frozen rule can choose, and FR's bias appears within about 5 t.u.
