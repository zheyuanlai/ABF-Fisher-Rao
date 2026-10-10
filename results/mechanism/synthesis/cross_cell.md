# Mechanism campaign: preregistered cross-cell contrasts

Generated 2026-10-10T14:19:27Z by scripts/mechanism/cross_cell.py (mechanism_cross_cell/1); 10000 seed-bootstrap resamples (seed 20261010).

Delta_ab = median over seeds of [log(I^FR/I^ABF)_a - log(I^FR/I^ABF)_b] (seeds paired across cells); ratio = exp(Delta): **< 1 means a has the larger FR gain**.  CI: seed bootstrap of the median; p: two-sided percentile-bootstrap p (+1 correction); Holm within the frozen family (Exp I: 12 tests, Exp II: 2); equivalent iff the 95 % CI of exp(Delta) lies in [0.9, 1.11].

## Cells

| cell | available | engine | model | summary generated | n_boot |
|---|---|---|---|---|---|
| conditional_relaxation/lam0.1 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 1.0, 'kappa': None, 'lam': 0.1} | 2026-10-10T14:19:12Z | 10000 |
| conditional_relaxation/lam0.25 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 1.0, 'kappa': None, 'lam': 0.25} | 2026-10-10T14:19:13Z | 10000 |
| conditional_relaxation/lam0.5 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 1.0, 'kappa': None, 'lam': 0.5} | 2026-10-10T14:19:15Z | 10000 |
| conditional_relaxation/lam1 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 1.0, 'kappa': None, 'lam': 1.0} | 2026-10-10T14:19:15Z | 10000 |
| matched_free_energy/alpha0.5 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 0.5, 'kappa': None, 'lam': 1.0} | 2026-10-10T14:19:19Z | 10000 |
| matched_free_energy/alpha0 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 0.0, 'kappa': None, 'lam': 1.0} | 2026-10-10T14:19:18Z | 10000 |
| matched_free_energy/alpha1 | yes | gateway_family_numba/1 | {'variant': 'alpha', 'alpha': 1.0, 'kappa': None, 'lam': 1.0} | 2026-10-10T14:19:18Z | 10000 |
| matched_free_energy/shift | yes | gateway_family_numba/1 | {'variant': 'shift', 'alpha': None, 'kappa': 1.0, 'lam': 1.0} | 2026-10-10T14:19:18Z | 10000 |

Primary reference digest (identical in every cell): `8c48b7cee8a2942d1787df5b497a7c243d5feb3ac10420121c369f8ffb207666`

## Experiment I (matched free energy)

N = [2048, 512, 128]; family size 12; cells present ['alpha0', 'alpha0.5', 'alpha1', 'shift']; missing none.

### Ibar_F (primary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| C1 alpha0 vs alpha1 | 2048 | 32 | 1.119 [1.054, 1.218] | 0.0052 | 0.0312 | 9/32 | 0.0201 | 0.00167 | differs (Holm): b has the larger FR gain |
| C1 alpha0 vs alpha1 | 512 | 32 | 1.195 [1.076, 1.412] | 0.001 | 0.007 | 7/32 | 0.0021 | 0.00193 | differs (Holm): b has the larger FR gain |
| C1 alpha0 vs alpha1 | 128 | 32 | 1.166 [0.9805, 1.646] | 0.08 | 0.4 | 11/32 | 0.11 | 0.295 | inconclusive |
| C2 shift vs alpha1 | 2048 | 32 | 1.19 [1.112, 1.269] | 0.0002 | 0.0024 | 4/32 | 1.93e-05 | 2.84e-05 | differs (Holm): b has the larger FR gain |
| C2 shift vs alpha1 | 512 | 32 | 1.799 [1.432, 1.974] | 0.0002 | 0.0024 | 1/32 | 1.54e-08 | 6.52e-09 | differs (Holm): b has the larger FR gain |
| C2 shift vs alpha1 | 128 | 32 | 2.867 [2.152, 3.374] | 0.0002 | 0.0024 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 2048 | 32 | 1.046 [0.97, 1.135] | 0.171 | 0.683 | 12/32 | 0.215 | 0.134 | inconclusive |
| C3 shift vs alpha0 | 512 | 32 | 1.518 [1.251, 1.645] | 0.0002 | 0.0024 | 3/32 | 2.56e-06 | 7.87e-08 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 128 | 32 | 2.303 [1.957, 3.157] | 0.0002 | 0.0024 | 1/32 | 1.54e-08 | 1.11e-06 | differs (Holm): b has the larger FR gain |
| C4 alpha0.5 vs alpha1 | 2048 | 32 | 1.053 [0.965, 1.124] | 0.301 | 0.903 | 13/32 | 0.377 | 0.21 | inconclusive |
| C4 alpha0.5 vs alpha1 | 512 | 32 | 0.9302 [0.8571, 1.235] | 0.444 | 0.903 | 18/32 | 0.597 | 0.443 | inconclusive |
| C4 alpha0.5 vs alpha1 | 128 | 32 | 1.058 [0.8928, 1.418] | 0.585 | 0.903 | 15/32 | 0.86 | 0.733 | inconclusive |

### Ibar_TV_half (secondary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| C1 alpha0 vs alpha1 | 2048 | 32 | 0.9313 [0.9187, 0.9423] | 0.0002 | 0.0024 | 32/32 | 4.66e-10 | 4.66e-10 | equivalent (within [0.90, 1.11]) but differs (Holm) |
| C1 alpha0 vs alpha1 | 512 | 32 | 0.8357 [0.7917, 0.8863] | 0.0002 | 0.0024 | 26/32 | 0.000535 | 3.51e-06 | differs (Holm): a has the larger FR gain |
| C1 alpha0 vs alpha1 | 128 | 32 | 0.9396 [0.8943, 1.03] | 0.114 | 0.228 | 20/32 | 0.215 | 0.35 | inconclusive |
| C2 shift vs alpha1 | 2048 | 32 | 1.232 [1.216, 1.249] | 0.0002 | 0.0024 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| C2 shift vs alpha1 | 512 | 32 | 0.4505 [0.4286, 0.4862] | 0.0002 | 0.0024 | 32/32 | 4.66e-10 | 4.66e-10 | differs (Holm): a has the larger FR gain |
| C2 shift vs alpha1 | 128 | 32 | 0.2959 [0.2709, 0.3199] | 0.0002 | 0.0024 | 32/32 | 4.66e-10 | 4.66e-10 | differs (Holm): a has the larger FR gain |
| C3 shift vs alpha0 | 2048 | 32 | 1.333 [1.309, 1.344] | 0.0002 | 0.0024 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 512 | 32 | 0.5217 [0.4967, 0.5562] | 0.0002 | 0.0024 | 32/32 | 4.66e-10 | 4.66e-10 | differs (Holm): a has the larger FR gain |
| C3 shift vs alpha0 | 128 | 32 | 0.3031 [0.284, 0.3283] | 0.0002 | 0.0024 | 32/32 | 4.66e-10 | 4.66e-10 | differs (Holm): a has the larger FR gain |
| C4 alpha0.5 vs alpha1 | 2048 | 32 | 0.9683 [0.9618, 0.978] | 0.0002 | 0.0024 | 30/32 | 2.46e-07 | 4.1e-08 | equivalent (within [0.90, 1.11]) but differs (Holm) |
| C4 alpha0.5 vs alpha1 | 512 | 32 | 0.9581 [0.9241, 0.9743] | 0.0082 | 0.0246 | 23/32 | 0.0201 | 0.00144 | equivalent (within [0.90, 1.11]) but differs (Holm) |
| C4 alpha0.5 vs alpha1 | 128 | 32 | 0.9697 [0.9128, 1.023] | 0.2 | 0.228 | 20/32 | 0.215 | 0.149 | equivalent |

### Ibar_Fp_stat (secondary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| C1 alpha0 vs alpha1 | 2048 | 32 | 0.9726 [0.9384, 1.083] | 0.667 | 1 | 17/32 | 0.86 | 0.705 | equivalent |
| C1 alpha0 vs alpha1 | 512 | 32 | 1.115 [0.9096, 1.198] | 0.304 | 1 | 13/32 | 0.377 | 0.905 | inconclusive |
| C1 alpha0 vs alpha1 | 128 | 32 | 1.147 [0.7481, 1.264] | 0.594 | 1 | 14/32 | 0.597 | 0.905 | inconclusive |
| C2 shift vs alpha1 | 2048 | 32 | 1.332 [1.245, 1.423] | 0.0002 | 0.0024 | 1/32 | 1.54e-08 | 4.98e-07 | differs (Holm): b has the larger FR gain |
| C2 shift vs alpha1 | 512 | 32 | 1.701 [1.597, 2.003] | 0.0002 | 0.0024 | 1/32 | 1.54e-08 | 9.31e-10 | differs (Holm): b has the larger FR gain |
| C2 shift vs alpha1 | 128 | 32 | 2.916 [2.632, 3.476] | 0.0002 | 0.0024 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 2048 | 32 | 1.289 [1.211, 1.42] | 0.0002 | 0.0024 | 3/32 | 2.56e-06 | 3.55e-07 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 512 | 32 | 1.779 [1.522, 2.099] | 0.0002 | 0.0024 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| C3 shift vs alpha0 | 128 | 32 | 2.707 [2.511, 3.142] | 0.0002 | 0.0024 | 1/32 | 1.54e-08 | 9.48e-07 | differs (Holm): b has the larger FR gain |
| C4 alpha0.5 vs alpha1 | 2048 | 32 | 1.049 [0.9763, 1.093] | 0.342 | 1 | 13/32 | 0.377 | 0.172 | equivalent |
| C4 alpha0.5 vs alpha1 | 512 | 32 | 0.9715 [0.909, 1.108] | 0.605 | 1 | 18/32 | 0.597 | 0.993 | equivalent |
| C4 alpha0.5 vs alpha1 | 128 | 32 | 1.028 [0.9323, 1.194] | 0.556 | 1 | 14/32 | 0.597 | 0.665 | inconclusive |

### Per cell: median log(I^FR/I^ABF) and median G = exp(L) - 1

| cell | N | Ibar_F median G (n) | Ibar_TV_half median G (n) | Ibar_Fp_stat median G (n) |
|---|---|---|---|---|
| alpha0 | 2048 | -24.0 % (32) | -73.4 % (32) | -27.0 % (32) |
| alpha0 | 512 | -31.7 % (32) | -76.4 % (32) | -36.9 % (32) |
| alpha0 | 128 | -7.2 % (32) | -71.4 % (32) | -9.5 % (32) |
| alpha0.5 | 2048 | -26.2 % (32) | -72.2 % (32) | -25.8 % (32) |
| alpha0.5 | 512 | -36.3 % (32) | -74.0 % (32) | -33.2 % (32) |
| alpha0.5 | 128 | -21.4 % (32) | -72.6 % (32) | -17.5 % (32) |
| alpha1 | 2048 | -31.4 % (32) | -71.5 % (32) | -29.9 % (32) |
| alpha1 | 512 | -41.0 % (32) | -72.7 % (32) | -32.6 % (32) |
| alpha1 | 128 | -30.3 % (32) | -70.9 % (32) | -20.0 % (32) |
| shift | 2048 | -21.0 % (32) | -64.7 % (32) | -6.1 % (32) |
| shift | 512 | +1.9 % (32) | -87.7 % (32) | +21.3 % (32) |
| shift | 128 | +104.9 % (32) | -91.6 % (32) | +131.3 % (32) |

## Experiment II (transverse mobility)

N = [2048, 512]; family size 2; cells present ['lam0.1', 'lam0.25', 'lam0.5', 'lam1']; missing none.

### Ibar_F (primary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| D1 lam0.1 vs lam1 | 2048 | 32 | 1.21 [1.054, 1.441] | 0.0334 | 0.0334 | 10/32 | 0.0501 | 0.0184 | differs (Holm): b has the larger FR gain |
| D1 lam0.1 vs lam1 | 512 | 32 | 1.975 [1.327, 3.432] | 0.0028 | 0.0056 | 8/32 | 0.007 | 0.0228 | differs (Holm): b has the larger FR gain |
| T1 lam0.25 vs lam1 (trend) | 2048 | 32 | 1.091 [1.032, 1.152] | 0.0066 | n/a | 9/32 | 0.0201 | 0.0125 | trend (unadjusted, no Holm): CI excludes 0, b has the larger FR gain; unadjusted p = 0.0066 |
| T1 lam0.25 vs lam1 (trend) | 512 | 32 | 1.108 [0.9238, 2.088] | 0.173 | n/a | 12/32 | 0.215 | 0.102 | trend (unadjusted, no Holm): CI includes 0; unadjusted p = 0.173 |
| T2 lam0.5 vs lam1 (trend) | 2048 | 32 | 1.015 [0.982, 1.071] | 0.205 | n/a | 12/32 | 0.215 | 0.379 | trend (unadjusted, no Holm): equivalent; CI includes 0; unadjusted p = 0.205 |
| T2 lam0.5 vs lam1 (trend) | 512 | 32 | 1.052 [0.9723, 1.252] | 0.122 | n/a | 12/32 | 0.215 | 0.0945 | trend (unadjusted, no Holm): CI includes 0; unadjusted p = 0.122 |

### Ibar_TV_half (secondary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| D1 lam0.1 vs lam1 | 2048 | 32 | 1.29 [1.261, 1.327] | 0.0002 | 0.0004 | 0/32 | 4.66e-10 | 4.66e-10 | differs (Holm): b has the larger FR gain |
| D1 lam0.1 vs lam1 | 512 | 32 | 0.5535 [0.5334, 0.659] | 0.0002 | 0.0004 | 27/32 | 0.000113 | 4.98e-07 | differs (Holm): a has the larger FR gain |
| T1 lam0.25 vs lam1 (trend) | 2048 | 32 | 1.211 [1.175, 1.252] | 0.0002 | n/a | 0/32 | 4.66e-10 | 4.66e-10 | trend (unadjusted, no Holm): CI excludes 0, b has the larger FR gain; unadjusted p = 0.0002 |
| T1 lam0.25 vs lam1 (trend) | 512 | 32 | 0.9038 [0.7565, 0.9535] | 0.0048 | n/a | 23/32 | 0.0201 | 0.00608 | trend (unadjusted, no Holm): CI excludes 0, a has the larger FR gain; unadjusted p = 0.0048 |
| T2 lam0.5 vs lam1 (trend) | 2048 | 32 | 1.077 [1.065, 1.093] | 0.0002 | n/a | 0/32 | 4.66e-10 | 4.66e-10 | trend (unadjusted, no Holm): equivalent; CI excludes 0, b has the larger FR gain; unadjusted p = 0.0002 |
| T2 lam0.5 vs lam1 (trend) | 512 | 32 | 1.035 [0.9572, 1.09] | 0.381 | n/a | 14/32 | 0.597 | 0.172 | trend (unadjusted, no Holm): equivalent; CI includes 0; unadjusted p = 0.381 |

### Ibar_Fp_stat (secondary)

| contrast | N | n | exp(Delta) [95 % CI] | p | p Holm | wins a / n | sign p | Wilcoxon p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| D1 lam0.1 vs lam1 | 2048 | 32 | 1.06 [0.9263, 1.399] | 0.389 | 0.389 | 14/32 | 0.597 | 0.144 | inconclusive |
| D1 lam0.1 vs lam1 | 512 | 32 | 1.543 [1.225, 3.181] | 0.0122 | 0.0244 | 9/32 | 0.0201 | 0.077 | differs (Holm): b has the larger FR gain |
| T1 lam0.25 vs lam1 (trend) | 2048 | 32 | 1.064 [1.013, 1.153] | 0.035 | n/a | 10/32 | 0.0501 | 0.0521 | trend (unadjusted, no Holm): CI excludes 0, b has the larger FR gain; unadjusted p = 0.035 |
| T1 lam0.25 vs lam1 (trend) | 512 | 32 | 1.076 [0.7947, 1.707] | 0.677 | n/a | 15/32 | 0.86 | 0.369 | trend (unadjusted, no Holm): CI includes 0; unadjusted p = 0.677 |
| T2 lam0.5 vs lam1 (trend) | 2048 | 32 | 1.037 [0.9649, 1.084] | 0.501 | n/a | 14/32 | 0.597 | 0.512 | trend (unadjusted, no Holm): equivalent; CI includes 0; unadjusted p = 0.501 |
| T2 lam0.5 vs lam1 (trend) | 512 | 32 | 1.017 [0.934, 1.125] | 0.587 | n/a | 14/32 | 0.597 | 0.705 | trend (unadjusted, no Holm): CI includes 0; unadjusted p = 0.587 |

### Per cell: median log(I^FR/I^ABF) and median G = exp(L) - 1

| cell | N | Ibar_F median G (n) | Ibar_TV_half median G (n) | Ibar_Fp_stat median G (n) |
|---|---|---|---|---|
| lam0.1 | 2048 | -23.3 % (32) | -62.8 % (32) | -21.6 % (32) |
| lam0.1 | 512 | +4.0 % (32) | -84.1 % (32) | +0.1 % (32) |
| lam0.25 | 2048 | -28.8 % (32) | -65.3 % (32) | -26.9 % (32) |
| lam0.25 | 512 | -14.3 % (32) | -75.7 % (32) | -19.1 % (32) |
| lam0.5 | 2048 | -30.1 % (32) | -69.4 % (32) | -26.2 % (32) |
| lam0.5 | 512 | -34.5 % (32) | -72.4 % (32) | -30.5 % (32) |
| lam1 | 2048 | -31.4 % (32) | -71.5 % (32) | -29.9 % (32) |
| lam1 | 512 | -41.0 % (32) | -72.7 % (32) | -32.6 % (32) |

