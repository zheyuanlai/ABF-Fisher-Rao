# Mechanism: late-time FR error, periodic circulation of the histogram bias, and what happens after FR stops

*Read-only diagnostics. Neither algorithm is modified. Code: `scripts/convergence_fr/circulation.py` and the
`secondary` block of `analyze_block.py`. Data: `results/convergence_fr_campaign/mechanism/`.*

## 1. Closure convention (checked)

* The CV is linear and periodic, φ = 2π(X_COM − x_w)/P, so it has no Jacobian term. The exact mean force F′(φ)
  integrates to zero over one period.
* **The reference satisfies this.** The mean over bins of Γ_ref is:
  * 0.00025 kJ/mol/rad for the 300 K exact-MC 180-bin mean force;
  * exactly 0 for the smooth 150 K reference, by construction.
* **Read-out vs dynamics.**
  * The read-out PMF (`histogram_pmf`) subtracts the circular mean of Γ before integrating, so the *reported*
    F̂ never carries a circulation.
  * The *applied* bias in both arms is the raw ramp × trust × Γ_j. Its circular mean
    c(t) = (1/2π)∫Γ_t dφ = mean_j Γ_j (equal bins) is a non-conservative tilt. The work per period is 2πc.
* **"c ≠ 0" is therefore a statement about the dynamics, not the reported estimate.** Two quantities are reported:
  * *c (cumulative)*: the circular mean of the cumulative estimate that is being applied, averaged over t ≥ T/2.
  * *c (late deposits)*: the circular mean of Γ built only from deposits after T/2. This is the definition used in
    the overnight `RESULTS.md`, and it reproduces those numbers exactly.

## 2. Existing exploratory blocks X1 (ITE 300 K, K1024, rate 2.0, T 50) and X2 (ITE 150 K, K256, rate 0.5, T 200)

| quantity (median over 16 seeds) | X1 ABF | X1 FR | X2 ABF | X2 FR |
|---|---|---|---|---|
| c, cumulative, late half (kJ/mol/rad) | 0.025 | 0.030 | 0.031 | 0.048 |
| paired FR − ABF (95 % CI) | +0.0016 [−0.0051, +0.0111] | | **+0.0084 [+0.0017, +0.0141]** | |
| c, late-half deposits | 0.0079 | 0.0154 | 0.0114 | 0.0249 |
| FR > ABF, late deposits | 12/16 | | 13/16 | |
| tilt per period 2πc / k_BT (cumulative) | 0.063 | 0.075 | 0.16 | 0.24 |
| within-arm Spearman ρ(c_late, e_F(T)) | 0.00 | 0.18 | **0.81** | **0.63** |
| within-arm ρ(c_late, last-20 % estimate error) | −0.23 | 0.31 | 0.26 | 0.07 |
| across seeds ρ(Δc, Δe_F(T)), paired | 0.16 | | 0.15 | |
| last-20 % deposits estimate error (kJ/mol) | 0.042 | 0.053 | 0.044 | 0.061 |
| paired change of that error | +20 % [−9, +62] | | +40 % [−9, +116] | |
| FR replacements per t.u.: first 5 t.u. after start / late half | – | 131 / 90 | – | 16.5 / 11.2 |
| RMS of the pooled FR − ABF final profile (kJ/mol) | 0.012 | | 0.033 | |

**Observations.**

* Both arms circulate in the same direction (c > 0), in both cells. FR circulates more. The difference is resolved
  only at 150 K.
* **At 150 K, circulation tracks final error within each arm** (ρ 0.81 ABF, 0.63 FR). At 300 K it does not
  (ρ 0.00 / 0.18).
* **FR's extra error is not explained by its extra circulation.** The seed-wise FR − ABF circulation difference
  does not predict the seed-wise FR − ABF error difference (ρ 0.15–0.16).
* **The size of the tilt is small at 300 K and moderate at 150 K.**
  * At 300 K: 2πc ≈ 0.06–0.08 k_BT per period (P = 9.72 Å). The induced drift is about 0.02 Å per t.u., i.e. about
    0.5 Å over 25 t.u., against a diffusive spread of about 16 Å.
  * At 150 K: 0.16 (ABF) to 0.24 (FR) k_BT per period.
* **FR keeps resampling at a high rate after establishment.** X1 still makes 90 replacements per t.u. in the late
  half, 70 % of its early rate. It never becomes idle, because a finite-K population never matches a uniform target
  exactly.

**Reading (hypotheses, not established).**

* **Circulation.**
  * Circulation is a plausible contributor to the late error at 150 K, where the tilt is a sizeable fraction of
    k_BT and correlates with error.
  * It is an implausible main cause at 300 K, where the tilt is about 0.07 k_BT per period and uncorrelated with
    error.
  * Either way the data do not show that FR's *extra* error goes through circulation.
* **Correction experiment not triggered.** The preregistered trigger for the zero-mean correction experiment (§6:
  |ρ| ≥ 0.6 in **both** temperatures **and** a resolved FR − ABF difference) is **not met** on X1/X2: 300 K fails
  both clauses. It is re-evaluated with E1 and the matrix cells below.
* **Competing explanation.** FR's continuing resampling, about 90 events per t.u. at 300 K, keeps perturbing the
  population after establishment. FR's φ-selection then disturbs the genealogy and the local time-correlation of
  the deposits. The test is E1, which removes FR after establishment while keeping everything else.

## 3. E1: what happens after FR is switched off

(filled from `results/convergence_fr_campaign/analysis/E1.json` and `mechanism/circulation_E1.json`)

## 4. ERI and the matrix cells

(filled when those blocks complete)
