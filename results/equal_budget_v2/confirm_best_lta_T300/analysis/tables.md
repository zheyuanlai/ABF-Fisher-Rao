# Equal-budget ladder analysis: lta300

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/equal_budget_v2/confirm_best_lta300.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/equal_budget_v2`; generated 2026-10-10T08:49:38Z; eqb_metrics/2.
B = 307,200,000 walker-steps per arm per seed; h = 0.0002; seeds 32; thresholds e_F [0.03, 0.094, 0.5], e_F' [0.04, 0.12, 1.0], TV_half 0.1.
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/equal_budget_v2/references/lta_T300_reference.npz, sha256 293888aae0af2ec936b2a752448566c85d2f6e4d2f6d62a8b67e426521a25b8d, T_K 300.0, F_ref_se_rms 0.00591559128362355, gamma_se_rms 0.014280134927751983, gamma_ref_circular_mean -0.0003305897670909669, floor_definition errors at Gamma = gamma_ref (the reference's own per-bin mean force): e_Fp = 0 by construction (no deterministic floor), e_F = the mean-force-route vs density-route consistency of the reference; none is a lower bound, estimator Gamma = M/max(C,1) (0 where C=0) from (M_prod, C_prod) once C_prod has any count else (M_all, C_all); F = histogram_pmf(Gamma).
Zero-noise floors (scorer fed the exact bin-averaged reference): e_F 0.007506, e_Fp 0, e_Fp_proj 0.

## Frozen thresholds vs the zero-noise floors

| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |
|---|---|---|---|---|---|---|
| e_F_strict | 0.03 | 0.007506 | no | yes | -- |  |
| e_F_mid | 0.094 | 0.007506 | no | yes | -- |  |
| e_F_loose | 0.5 | 0.007506 | no | yes | -- |  |
| e_Fp_strict | 0.04 | 0 | no | yes | -- |  |
| e_Fp_mid | 0.12 | 0 | no | yes | -- |  |
| e_Fp_loose | 1 | 0 | no | yes | -- |  |
| e_Fp_proj_strict | 0.04 | 0 | no | yes | -- |  |
| e_Fp_proj_mid | 0.12 | 0 | no | yes | -- |  |
| e_Fp_proj_loose | 1 | 0 | no | yes | -- |  |

## Status

| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |
|---|---|---|---|---|---|
| 16 | 19,200,000 | 3840 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 3840 | 0.05605 [0.04513, 0.06816] | 0.05662 [0.04641, 0.07681] | 0.1502 [0.1427, 0.1574] | 0.1526 [0.1437, 0.1587] | 0.03534 [0.02599, 0.04325] | 0.03484 [0.02435, 0.04825] | 0.08132 [0.07359, 0.08897] | 0.0816 [0.07606, 0.08865] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 16 | +6.2 % [-1.2 %, +22.4 %] (11/32) | -0.2 % [-3.1 %, +6.2 %] (16/32) | +6.3 % [-24.7 %, +31.4 %] (16/32) | +0.4 % [-4.6 %, +10.1 %] (16/32) | -6.5 % [-13.0 %, -1.8 %] (24/32) | -27.1 % [-35.0 %, +4.4 %] (20/32) | 114 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 16 | e_F_strict | > 1 | 0.625 | > 1 | 0.625 | 9/10/13 | 1 | +1.0 % [-26.1 %, +57.1 %] (n 5) |
| 16 | e_F_mid | 0.09 | 0 | 0.09 | 0 | 14/17/1 | 0.72 | +11.8 % [-21.2 %, +60.0 %] (n 32) |
| 16 | e_F_loose | 0.005 | 0 | 0.005 | 0 | 10/10/12 | 1 | +0.0 % [+0.0 %, +0.0 %] (n 32) |
| 16 | e_Fp_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 16 | e_Fp_mid | 0.4475 | 0 | 0.4725 | 0 | 12/17/3 | 0.458 | +3.3 % [-2.0 %, +13.5 %] (n 32) |
| 16 | e_Fp_loose | 0.01 | 0 | 0.01 | 0 | 5/3/24 | 0.727 | +0.0 % [+0.0 %, +0.0 %] (n 32) |
| 16 | TV_half | 0.02 | 0 | 0.0175 | 0 | 11/11/10 | 1 | +0.0 % [-20.0 %, +12.5 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_strict | e_Fp_mid | e_Fp_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 16 | abf | > 1 [0.8675, > 1] (0.625) | 0.09 [0.05875, 0.1313] (0) | 0.005 [0.003609, 0.01] (0) | > 1 [> 1, > 1] (1) | 0.4475 [0.3838, 0.5413] (0) | 0.01 [0.01, 0.01] (0) | 0.02 [0.015, 0.025] (0) |
| 16 | fr | > 1 [0.7875, > 1] (0.625) | 0.09 [0.0625, 0.215] (0) | 0.005 [0.003609, 0.01] (0) | > 1 [> 1, > 1] (1) | 0.4725 [0.3962, 0.5812] (0) | 0.01 [0.01, 0.01] (0) | 0.0175 [0.01, 0.02] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 16 | 3 | 0.192 | 0.192 | 0.000 | 0.99 | **yes** (read cum. est. / first arrival) |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | abf | 0.01061 [0.008823, 0.01378] | 0.02 [0.015, 0.025] | 0.3889 [0.3333, 0.4444] | 0.4007 | 0.9825 [0.96, 0.995] (UNRELIABLE: fails by chance at this N) | 0.0625 | 0.003066 [0.002604, 0.004248] | 0.0006933 [0.0005643, 0.0008398] | 1127 [1112, 1150] |
| 16 | fr | 0.009389 [0.006829, 0.01206] | 0.0175 [0.01, 0.02] | 0.3889 [0.3889, 0.4444] | 0.4007 | 0.9825 [0.96, 1] (UNRELIABLE: fails by chance at this N) | 0.1562 | 0.003066 [0.002213, 0.004248] | 0.0006933 [0.0005643, 0.0008398] | 1123 [1106, 1150] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0/32 | 1 | 1 | 1821 [1785, 1853] | 113.8 [111.5, 115.8] | 0.0004747 [0.0004653, 0.0004831] | 2.967e-05 [2.908e-05, 3.02e-05] | 0.0625 [0.0625, 0.0625] | 0.0625 [0.0625, 0.0625] | 0.5333 [0.525, 0.5714] | 1 [1, 1] | 0.25 [0.1875, 0.25] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 16 | e_F | +12.6 % | 0.9 | -18.0 % | 0.23 | +1.4 % | 32 |
| 16 | e_Fp | +5.3 % | 0.895 | -7.7 % | 0.405 | -0.3 % | 32 |
| 16 | e_Fp_proj | +5.9 % | 0.23 | -6.9 % | 0.05 | -0.4 % | 32 |
| 16 | TV_half | +29.7 % | 0.075 | -17.0 % | 0.415 | +11.5 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 16 | 0.05605 | [0.051, 0.06438] | 16: 1.00 | 16 | 0.05662 | [0.05041, 0.06537] | 16: 1.00 | 0.0005647 [-0.008245, 0.008895] | +1.0 % [-13.4 %, +16.7 %] |
| final_e_F | 16 | 0.03534 | [0.02781, 0.04102] | 16: 1.00 | 16 | 0.03484 | [0.02775, 0.04393] | 16: 1.00 | -0.000503 [-0.008673, 0.008935] | -1.4 % [-22.8 %, +28.3 %] |
| tau_bgrid_e_F_mid_u | 16 | 0.09 | [0.075, 0.12] | 16: 1.00 | 16 | 0.09 | [0.065, 0.13] | 16: 1.00 | 0 [-0.0275, 0.0375] | +0.0 % [-28.2 %, +44.4 %] |
| Ibar_Fp | 16 | 0.1502 | [0.1469, 0.1528] | 16: 1.00 | 16 | 0.1526 | [0.1465, 0.1574] | 16: 1.00 | 0.002455 [-0.004115, 0.008781] | +1.6 % [-2.7 %, +5.9 %] |
| final_e_Fp | 16 | 0.08132 | [0.07513, 0.087] | 16: 1.00 | 16 | 0.0816 | [0.07932, 0.08571] | 16: 1.00 | 0.0002746 [-0.00597, 0.006468] | +0.3 % [-6.9 %, +8.6 %] |
| tau_bgrid_TV_half_u | 16 | 0.015 | [0.015, 0.0225] | 16: 1.00 | 16 | 0.015 | [0.01, 0.015] | 16: 1.00 | 0 [-0.01, 0] | +0.0 % [-40.0 %, +0.0 %] |

## Secondary (periodically projected F')

| N | Ibar_Fp_proj | final_e_Fp_proj |
|---|---|---|
| 16 | ABF 0.145, FR 0.1481; +1.3 % [-1.4 %, +6.5 %] (15/32) | ABF 0.08012, FR 0.08048; +0.5 % [-5.0 %, +12.6 %] (14/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 16 | 0.40070 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 16 | abf | 305.5 | 0.9943 | 264.5 | 32/32 |
| 16 | fr | 318.3 | 1.036 | 263.7 | 32/32 |
