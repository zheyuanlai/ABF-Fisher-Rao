# Equal-budget ladder analysis: lta150

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/equal_budget_v2/confirm_best_lta150.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/equal_budget_v2`; generated 2026-10-10T08:49:40Z; eqb_metrics/2.
B = 307,200,000 walker-steps per arm per seed; h = 0.0002; seeds 32; thresholds e_F [0.03, 0.13, 0.5], e_F' [0.04, 0.15, 1.0], TV_half 0.1.
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/equal_budget_v2/references/lta_T150_reference.npz, sha256 ebfc205152a8f95e7c0ecf32601fa7cbb5d233b9b4708829cc4173044baf8ec5, T_K 150.0, F_ref_se_rms 0.0034570851297217185, gamma_se_rms 0.010076926242980365, gamma_ref_circular_mean -0.0014922169292974432, floor_definition errors at Gamma = gamma_ref (the reference's own per-bin mean force): e_Fp = 0 by construction (no deterministic floor), e_F = the mean-force-route vs density-route consistency of the reference; none is a lower bound, estimator Gamma = M/max(C,1) (0 where C=0) from (M_prod, C_prod) once C_prod has any count else (M_all, C_all); F = histogram_pmf(Gamma).
Zero-noise floors (scorer fed the exact bin-averaged reference): e_F 0.005602, e_Fp 0, e_Fp_proj 0.

## Frozen thresholds vs the zero-noise floors

| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |
|---|---|---|---|---|---|---|
| e_F_strict | 0.03 | 0.005602 | no | yes | -- |  |
| e_F_mid | 0.13 | 0.005602 | no | yes | -- |  |
| e_F_loose | 0.5 | 0.005602 | no | yes | -- |  |
| e_Fp_strict | 0.04 | 0 | no | yes | -- |  |
| e_Fp_mid | 0.15 | 0 | no | yes | -- |  |
| e_Fp_loose | 1 | 0 | no | yes | -- |  |
| e_Fp_proj_strict | 0.04 | 0 | no | yes | -- |  |
| e_Fp_proj_mid | 0.15 | 0 | no | yes | -- |  |
| e_Fp_proj_loose | 1 | 0 | no | yes | -- |  |

## Status

| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |
|---|---|---|---|---|---|
| 2 | 153,600,000 | 30720 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 30720 | 0.04201 [0.0371, 0.04954] | 0.04029 [0.03542, 0.05012] | 0.1268 [0.1222, 0.137] | 0.1262 [0.1193, 0.1375] | 0.02426 [0.02012, 0.03108] | 0.02488 [0.01995, 0.03086] | 0.06651 [0.06194, 0.07325] | 0.06731 [0.06177, 0.07439] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2 | -3.0 % [-8.7 %, +3.3 %] (19/32) | -2.8 % [-5.4 %, +1.0 %] (18/32) | -3.2 % [-17.5 %, +13.0 %] (17/32) | +2.7 % [-7.6 %, +7.4 %] (15/32) | -4.6 % [-11.4 %, +5.3 %] (20/32) | +4.0 % [-15.9 %, +14.4 %] (14/32) | 18.5 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2 | e_F_strict | 0.74 | 0.2812 | 0.6825 | 0.3125 | 18/13/1 | 0.473 | -17.5 % [-34.8 %, +27.6 %] (n 14) |
| 2 | e_F_mid | 0.035 | 0 | 0.0375 | 0 | 6/6/20 | 1 | +0.0 % [+0.0 %, +0.0 %] (n 32) |
| 2 | e_F_loose | 0.003609 | 0 | 0.003758 | 0 | 2/2/28 | 1 | +0.0 % [+0.0 %, +0.0 %] (n 32) |
| 2 | e_Fp_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2 | e_Fp_mid | 0.205 | 0 | 0.2075 | 0 | 18/13/1 | 0.473 | -4.7 % [-18.1 %, +6.2 %] (n 32) |
| 2 | e_Fp_loose | 0.01 | 0 | 0.01 | 0 | 1/0/31 | 1 | +0.0 % [+0.0 %, +0.0 %] (n 32) |
| 2 | TV_half | 0.04 | 0 | 0.035 | 0 | 12/6/14 | 0.238 | +0.0 % [-10.5 %, +0.0 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_strict | e_Fp_mid | e_Fp_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2 | abf | 0.74 [0.465, > 1] (0.2812) | 0.035 [0.02, 0.04625] (0) | 0.003609 [0.002507, 0.01] (0) | > 1 [> 1, > 1] (1) | 0.205 [0.18, 0.25] (0) | 0.01 [0.00875, 0.01] (0) | 0.04 [0.03, 0.055] (0) |
| 2 | fr | 0.6825 [0.43, > 1] (0.3125) | 0.0375 [0.02, 0.045] (0) | 0.003758 [0.002507, 0.00625] (0) | > 1 [> 1, > 1] (1) | 0.2075 [0.17, 0.2462] (0) | 0.01 [0.005, 0.01] (0) | 0.035 [0.025, 0.04625] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.56 | 0.56 | 0.000 | > 1 | **yes** (read cum. est. / first arrival) |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | abf | 0.01718 [0.0124, 0.02074] | 0.04 [0.03, 0.055] | 0.8889 [0.8889, 0.8889] | 0.8920 | > 1 [0.995, > 1] (ill-defined, N <= 2) | 0.625 | 0.000313 [0.0002259, 0.0005104] | 0.0001836 [0.000134, 0.0002583] | 507 [498.8, 517] |
| 2 | fr | 0.01503 [0.01207, 0.01947] | 0.035 [0.025, 0.04625] | 0.8889 [0.8889, 0.8889] | 0.8920 | > 1 [0.9988, > 1] (ill-defined, N <= 2) | 0.5312 | 0.000313 [0.0002259, 0.0005104] | 0.0001836 [0.000134, 0.0002583] | 508 [496, 516.2] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0/32 | 1 | 1 | 37 [34, 39.25] | 18.5 [17, 19.62] | 1.205e-06 [1.107e-06, 1.278e-06] | 6.023e-07 [5.535e-07, 6.389e-07] | 0.5 [0.5, 0.5] | 0.5 [0.5, 0.5] | 1 [0.5, 1] | 1 [1, 1] | 0.5 [0.5, 1] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2 | e_F | +16.1 % | 0.32 | -16.4 % | 0.225 | -2.6 % | 32 |
| 2 | e_Fp | +8.6 % | 0.1 | -5.6 % | 0.045 | -1.2 % | 32 |
| 2 | e_Fp_proj | +7.8 % | 0.415 | -5.5 % | 0.045 | +1.4 % | 32 |
| 2 | TV_half | +23.1 % | 0.325 | -17.5 % | 0.865 | +12.5 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 2 | 0.04201 | [0.03977, 0.04627] | 2: 1.00 | 2 | 0.04029 | [0.03671, 0.04751] | 2: 1.00 | -0.001724 [-0.006647, 0.00413] | -4.1 % [-14.6 %, +9.8 %] |
| final_e_F | 2 | 0.02426 | [0.02184, 0.02924] | 2: 1.00 | 2 | 0.02488 | [0.02093, 0.02868] | 2: 1.00 | 0.0006223 [-0.003895, 0.004591] | +2.6 % [-14.4 %, +20.5 %] |
| tau_bgrid_e_F_mid_u | 2 | 0.035 | [0.025, 0.0425] | 2: 1.00 | 2 | 0.0375 | [0.0225, 0.04] | 2: 1.00 | 0.0025 [-0.01, 0.0125] | +7.1 % [-28.6 %, +50.0 %] |
| Ibar_Fp | 2 | 0.1268 | [0.1235, 0.1352] | 2: 1.00 | 2 | 0.1262 | [0.1212, 0.133] | 2: 1.00 | -0.0005868 [-0.008041, 0.004395] | -0.5 % [-6.0 %, +3.5 %] |
| final_e_Fp | 2 | 0.06651 | [0.06398, 0.07135] | 2: 1.00 | 2 | 0.06731 | [0.06484, 0.07171] | 2: 1.00 | 0.000803 [-0.003977, 0.005755] | +1.2 % [-5.7 %, +8.8 %] |
| tau_bgrid_TV_half_u | 2 | 0.04 | [0.035, 0.045] | 2: 1.00 | 2 | 0.035 | [0.03, 0.045] | 2: 1.00 | -0.005 [-0.01, 0.005] | -12.5 % [-22.7 %, +12.5 %] |

## Secondary (periodically projected F')

| N | Ibar_Fp_proj | final_e_Fp_proj |
|---|---|---|
| 2 | ABF 0.1245, FR 0.1233; -2.2 % [-5.7 %, +1.7 %] (17/32) | ABF 0.06565, FR 0.06475; +3.0 % [-7.9 %, +7.5 %] (15/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 2 | 0.89198 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2 | abf | 321.9 | 1.048 | 246.6 | 32/32 |
| 2 | fr | 353.7 | 1.151 | 253.8 | 32/32 |
