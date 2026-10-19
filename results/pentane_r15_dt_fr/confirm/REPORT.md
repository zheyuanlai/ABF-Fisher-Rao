# Confirmatory (16 fresh seeds, rng 20261201-04): selected dt 0.00025, secondary rate None

| arm | e_F(40) | I_F | e_F'(40) | I_F' | low-support | D_cond(40) | D_cond9(40) | int KL(p||U) | fam. transitions | replacements/N | ESS/N | max anc frac | clip frac | max disp | wall/seed (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abf r0 dt0.00025 | 1.820 | 113.26 | 8.61 | 479.5 | 0.136 | 0.371 | 0.339 | 13.30 | 711580 | 0.000 | nan | nan | 1.6e-08 | 0.106 | 8.1 |
| abf r0 dt0.0005 | 1.379 | 101.35 | 7.38 | 444.2 | 0.148 | 0.368 | 0.335 | 13.19 | 573622 | 0.000 | nan | nan | 1.1e-05 | 0.183 | 4.1 |
| fr r0.02 dt0.00025 | 1.860 | 114.27 | 8.76 | 482.9 | 0.136 | 0.372 | 0.340 | 12.78 | 708530 | 0.102 | 0.805 | 0.006 | 1.5e-08 | 0.106 | 8.2 |
| fr r0.02 dt0.0005 | 1.415 | 101.92 | 7.47 | 446.3 | 0.148 | 0.369 | 0.335 | 12.75 | 572796 | 0.094 | 0.828 | 0.005 | 1.1e-05 | 0.183 | 4.1 |

## Paired contrasts (median relative change [95 % bootstrap CI] wins/n; negative favours the first arm)

* **F0.02 vs A, dt 0.0005**: ΔI_F +0.7% [+0.5, +0.8] 0/16; ΔeF_end +2.7% [+1.9, +3.0] 0/16; ΔI_Fp +0.5% [+0.3, +0.6] 0/16; ΔeFp_end +1.2% [+0.7, +1.5] 0/16; ΔDcond_end +0.1% [+0.1, +0.2] 3/16; ΔDc9_end +0.1% [+0.1, +0.3] 3/16; ΔI_KL -2.9% [-3.6, -1.8] 16/16; Δlowsup_end +0.0% [-4.0, +0.0] 6/16; Δdep_bias_rms +0.1% [+0.0, +0.1] 3/16; ΔeF_compact +1.8% [+1.4, +2.1] 0/16; ΔeF_rest +3.1% [+2.1, +3.4] 0/16
* **F0.02 vs A, dt 0.00025**: ΔI_F +0.6% [+0.5, +0.7] 0/16; ΔeF_end +2.0% [+1.4, +2.4] 0/16; ΔI_Fp +0.5% [+0.4, +0.6] 0/16; ΔeFp_end +1.5% [+1.1, +1.9] 0/16; ΔDcond_end +0.2% [+0.1, +0.3] 0/16; ΔDc9_end +0.3% [+0.0, +0.4] 4/16; ΔI_KL -3.8% [-5.7, -2.1] 16/16; Δlowsup_end -4.9% [-8.3, -2.4] 12/16; Δdep_bias_rms +0.2% [+0.1, +0.3] 0/16; ΔeF_compact +1.5% [+1.1, +1.7] 0/16; ΔeF_rest +2.2% [+1.5, +2.7] 0/16
* **A dt 0.00025 vs A dt 5e-4 (unpaired)**: eF_end +32.0% [+26.4, +36.8]; I_F +11.7% [+9.8, +13.8]; eFp_end +16.7% [+13.1, +19.3]; I_Fp +7.9% [+6.3, +8.9]; lowsup_end -8.3% [-17.3, +0.0]; Dcond_end +0.8% [+0.3, +1.2]

## Classification (frozen rules)

```
{
 "A_numerical_rescue": {
  "holds": false,
  "stageA_fail_at_dt5e4": true,
  "abf_final_error_improves": false
 },
 "B_genuine_acceleration (rate 0.02, dt 0.0005)": false,
 "C_inactive (rate 0.02, dt 0.0005)": true,
 "D_harmful (rate 0.02, dt 0.0005)": true,
 "D_harmful_with_more_uniform_marginal (rate 0.02, dt 0.0005)": true,
 "equivalent_TOST_10pct (rate 0.02, dt 0.0005)": true,
 "F_inconclusive (rate 0.02, dt 0.0005)": false,
 "genealogy (rate 0.02, dt 0.0005)": {
  "ess_over_N": 0.8278097568371579,
  "max_anc_frac": 0.0048828125,
  "replacements_over_N": 0.09423828125
 },
 "B_genuine_acceleration (rate 0.02, dt 0.00025)": false,
 "C_inactive (rate 0.02, dt 0.00025)": true,
 "D_harmful (rate 0.02, dt 0.00025)": true,
 "D_harmful_with_more_uniform_marginal (rate 0.02, dt 0.00025)": true,
 "equivalent_TOST_10pct (rate 0.02, dt 0.00025)": true,
 "F_inconclusive (rate 0.02, dt 0.00025)": false,
 "genealogy (rate 0.02, dt 0.00025)": {
  "ess_over_N": 0.8050394074542884,
  "max_anc_frac": 0.005859375,
  "replacements_over_N": 0.1015625
 },
 "E_limited": {
  "sampling_dominated": true,
  "eF_end": 1.8202549769494634,
  "stageA_D_mf": 0.04116866106368933,
  "lowsup_end": 0.13580246913580246,
  "Dcond_end": 0.3711628930839186,
  "family_late_or_missing": [
   false,
   false,
   false,
   false
  ],
  "discovery_limited": true,
  "conditional_limited": false
 }
}
```
