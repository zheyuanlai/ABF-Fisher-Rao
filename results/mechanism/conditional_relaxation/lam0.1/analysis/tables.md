# Equal-budget ladder analysis: gateway_family -- mechanism cell conditional_relaxation/lam0.1: II: lambda = 0.1 (transverse mobility / 10)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/conditional_relaxation/lam0.1.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/conditional_relaxation`; generated 2026-10-10T14:19:12Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/conditional_relaxation_lam0.1_reference.npz, sha256 1d0851ead12af78da95baf04c0592a818cc62f2c0cc66b32c7a75e2f348c4979, n_eval_nodes 151, em_floor_F_rms 7.86220637002992e-06, em_bias_Fp_bin_rms 8.022337980767168e-05, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
Zero-noise floors (scorer fed the exact bin-averaged reference): e_F 4.326e-16, e_Fp 0.03227 (HARD lower bound), e_Fp_stat 0 (HARD lower bound), e_F_em 1.05e-09, e_Fp_em 0.03227 (HARD lower bound), e_Fp_em_stat 0 (HARD lower bound).

## Frozen thresholds vs the zero-noise floors

| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |
|---|---|---|---|---|---|---|
| e_F_strict | 0.002 | 4.326e-16 | no | yes | -- |  |
| e_F_mid | 0.0055 | 4.326e-16 | no | yes | -- |  |
| e_F_loose | 0.02 | 4.326e-16 | no | yes | -- |  |
| e_Fp_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported) |
| e_Fp_mid | 0.035 | 0.03227 | yes | yes | 0.01356 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01356 |
| e_Fp_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |
| e_F_em_strict | 0.002 | 1.05e-09 | no | yes | -- |  |
| e_F_em_mid | 0.0055 | 1.05e-09 | no | yes | -- |  |
| e_F_em_loose | 0.02 | 1.05e-09 | no | yes | -- |  |
| e_Fp_em_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported) |
| e_Fp_em_mid | 0.035 | 0.03227 | yes | yes | 0.01356 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01356 |
| e_Fp_em_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |
| e_Fp_stat_strict | 0.004 | 0 | yes | yes | 0.004 |  |
| e_Fp_stat_mid | 0.0136 | 0 | yes | yes | 0.0136 |  |
| e_Fp_stat_loose | 0.0946 | 0 | yes | yes | 0.0946 |  |

## Status

| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |
|---|---|---|---|---|---|
| 2048 | 1,600,000 | 40 | 32/32 | 32/32 | -- |
| 512 | 6,400,000 | 160 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 40 | 0.08639 [0.07529, 0.1208] | 0.07995 [0.06346, 0.147] | 0.2363 [0.2155, 0.3413] | 0.2203 [0.1745, 0.4109] | 0.04672 [0.04149, 0.06438] | 0.04877 [0.03729, 0.09611] | 0.1136 [0.1007, 0.1656] | 0.1226 [0.09364, 0.2595] |
| 512 | 160 | 0.04619 [0.03357, 0.1192] | 0.0585 [0.03693, 0.1686] | 0.1276 [0.09688, 0.348] | 0.1554 [0.09967, 0.4947] | 0.01262 [0.01088, 0.02582] | 0.03583 [0.02116, 0.1271] | 0.04413 [0.04131, 0.07589] | 0.09455 [0.0591, 0.3666] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -23.2 % [-31.7 %, -3.4 %] (23/32) | -20.8 % [-34.2 %, -4.2 %] (23/32) | -9.1 % [-27.3 %, +11.6 %] (18/32) | -7.3 % [-25.7 %, +14.4 %] (18/32) | -62.8 % [-63.6 %, -62.3 %] (32/32) | -80.9 % [-87.0 %, -77.4 %] (32/32) | 3 deaths / walker |
| 512 | +4.1 % [-12.1 %, +98.1 %] (15/32) | -2.1 % [-14.2 %, +89.4 %] (17/32) | +125.7 % [+62.8 %, +366.3 %] (8/32) | +97.6 % [+28.3 %, +298.8 %] (8/32) | -84.1 % [-86.3 %, -80.7 %] (32/32) | -92.7 % [-94.5 %, -90.9 %] (32/32) | 19.3 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_loose | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_loose | > 1 | 0.75 | > 1 | 0.6562 | 10/6/16 | 0.454 | -8.5 % [-53.3 %, +8.5 %] (n 3) |
| 2048 | TV_half | 0.7325 | 0.375 | 0.17 | 0 | 32/0/0 | 4.66e-10 | -76.2 % [-76.9 %, -75.5 %] (n 20) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_loose (secondary) | > 1 | 0.75 | > 1 | 0.6562 | 10/6/16 | 0.454 | -8.5 % [-53.3 %, +8.5 %] (n 3) |
| 2048 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_em_loose (secondary) | > 1 | 0.75 | > 1 | 0.6562 | 10/6/16 | 0.454 | -8.5 % [-53.3 %, +8.5 %] (n 3) |
| 512 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_loose | 0.71 | 0.375 | > 1 | 0.7812 | 3/19/10 | 0.000855 | +25.7 % [-9.9 %, +85.7 %] (n 5) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_loose | 0.395 | 0.2188 | 0.8375 | 0.4688 | 14/16/2 | 0.856 | -27.7 % [-57.3 %, +22.4 %] (n 12) |
| 512 | TV_half | > 1 | 1 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -- [--, --] (n 0) |
| 512 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_loose (secondary) | 0.71 | 0.375 | > 1 | 0.7812 | 3/19/10 | 0.000855 | +25.7 % [-9.9 %, +85.7 %] (n 5) |
| 512 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_loose (secondary) | 0.395 | 0.2188 | 0.8375 | 0.4688 | 14/16/2 | 0.856 | -27.7 % [-57.3 %, +22.4 %] (n 12) |
| 512 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_em_loose (secondary) | 0.395 | 0.2188 | 0.8375 | 0.4688 | 14/16/2 | 0.856 | -27.7 % [-57.3 %, +22.4 %] (n 12) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (0.75) | 0.7325 [0.71, > 1] (0.375) |
| 2048 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [0.8175, > 1] (0.6562) | 0.17 [0.17, 0.1713] (0) |
| 512 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.71 [0.575, > 1] (0.375) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.395 [0.255, 0.81] (0.2188) | > 1 [> 1, > 1] (1) |
| 512 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (0.7812) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.8375 [0.2275, > 1] (0.4688) | 0.05 [0.045, 0.05] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.07417 [0.05629, 0.1271] | 0.7325 [0.71, > 1] | 0.1344 [0.1096, 0.2059] | 0.0363 | 0.33 [0.2887, 0.34] | 0 | 0.6725 [0.6075, 0.7112] | 0.03233 [0.02959, 0.03541] | 1716 [1694, 1738] |
| 2048 | fr | 0.01413 [0.01226, 0.02194] | 0.17 [0.17, 0.1713] | 0.03309 [0.02946, 0.03773] | 0.0363 | 0.1 [0.09875, 0.105] | 0 | 0.2025 [0.195, 0.205] | 0.02952 [0.02761, 0.03213] | 1544 [1512, 1565] |
| 512 | abf | 0.1829 [0.1359, 0.3298] | > 1 [> 1, > 1] | 0.1543 [0.1196, 0.2545] | 0.0728 | 0.08 [0.06875, 0.08625] | 0 | 0.16 [0.135, 0.1812] | 0.009282 [0.008473, 0.0104] | 1452 [1108, 1517] |
| 512 | fr | 0.01286 [0.009104, 0.02659] | 0.05 [0.045, 0.05] | 0.06098 [0.05518, 0.06923] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05, 0.055] | 0.009388 [0.007969, 0.01064] | 1410 [1352, 1459] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 6137 [5866, 6816] | 2.997 [2.864, 3.328] | 0.4842 [0.4675, 0.5315] | 0.0002997 [0.0002864, 0.0003328] | 0.00293 [0.002441, 0.00293] | 0.05822 [0.04844, 0.06632] | 0.362 [0.3403, 0.3872] | 397.5 [352, 414] | 0.01489 [0.01221, 0.0177] | 0.0003491 [0.0003348, 0.0003912] |
| 512 | 0/32 | 40 | 0 | 9874 [9596, 1.093e+04] | 19.29 [18.74, 21.36] | 0.2296 [0.2239, 0.2536] | 0.0004821 [0.0004685, 0.0005339] | 0.007812 [0.007324, 0.007812] | 0.01699 [0.01118, 0.02147] | 0.3676 [0.3389, 0.3934] | 19 [11.75, 22] | 0.03711 [0.0293, 0.04199] | 0.0005131 [0.0004991, 0.0005717] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +14.8 % | 0.08 | -13.1 % | 0.01 | -4.4 % | 32 |
| 2048 | e_Fp | +60.5 % | 0.085 | -8.0 % | 1 | -8.0 % | 32 |
| 2048 | e_F_em | +14.8 % | 0.08 | -13.1 % | 0.01 | -4.4 % | 32 |
| 2048 | TV_half | +91.5 % | 0.31 | -0.0 % | 0.005 | +80.9 % | 32 |
| 512 | e_F | +17.7 % | 0.03 | -184.0 % | 1 | -184.0 % | 32 |
| 512 | e_Fp | +56.8 % | 0.025 | -114.2 % | 1 | -114.2 % | 32 |
| 512 | e_F_em | +17.7 % | 0.03 | -184.0 % | 1 | -184.0 % | 32 |
| 512 | TV_half | +93.5 % | 0.735 | +0.4 % | 0.005 | +93.0 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 512 | 0.04619 | [0.03789, 0.07845] | 512: 0.99, 2048: 0.01 | 512 | 0.0585 | [0.0402, 0.08267] | 512: 0.89, 2048: 0.11 | 0.01232 [-0.02562, 0.03974] | +26.7 % [-34.7 %, +99.8 %] |
| final_e_F | 512 | 0.01262 | [0.01139, 0.0225] | 512: 1.00 | 512 | 0.03583 | [0.02336, 0.05079] | 512: 0.85, 2048: 0.15 | 0.02322 [0.008561, 0.0386] | +184.0 % [+41.5 %, +322.0 %] |
| tau_bgrid_e_F_mid_u | all censored | > 1 | [> 1, > 1] | all censored: 1.00 | all censored | > 1 | [> 1, > 1] | all censored: 1.00 | -- [--, --] | -- [--, --] |
| Ibar_Fp | 512 | 0.1276 | [0.1089, 0.2219] | 512: 0.98, 2048: 0.02 | 512 | 0.1554 | [0.1076, 0.2267] | 512: 0.90, 2048: 0.10 | 0.02777 [-0.07772, 0.1064] | +21.8 % [-37.8 %, +94.7 %] |
| final_e_Fp | 512 | 0.04413 | [0.04182, 0.06737] | 512: 1.00 | 512 | 0.09455 | [0.06408, 0.1323] | 512: 0.83, 2048: 0.17 | 0.05042 [0.01474, 0.0864] | +114.2 % [+24.2 %, +199.5 %] |
| tau_bgrid_TV_half_u | 2048 | 0.7325 | [0.715, > 1] | 2048: 0.89, all censored: 0.11 | 512 | 0.05 | [0.045, 0.05] | 512: 1.00 | -0.6825 [inf, -0.665] | -93.2 % [-93.9 %, -93.0 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.08639, FR 0.07995; -23.2 % [-31.7 %, -3.4 %] (23/32) | ABF 0.2363, FR 0.2203; -20.8 % [-34.2 %, -4.2 %] (23/32) | ABF 0.04672, FR 0.04877; -9.1 % [-27.3 %, +11.6 %] (18/32) | ABF 0.1136, FR 0.1226; -7.3 % [-25.7 %, +14.4 %] (18/32) | ABF 0.2333, FR 0.2172; -21.5 % [-34.7 %, -4.2 %] (23/32) | ABF 0.1089, FR 0.1183; -7.6 % [-27.3 %, +16.1 %] (18/32) |
| 512 | ABF 0.04619, FR 0.0585; +4.1 % [-12.1 %, +98.1 %] (15/32) | ABF 0.1276, FR 0.1554; -2.1 % [-14.2 %, +89.4 %] (17/32) | ABF 0.01262, FR 0.03583; +125.7 % [+62.8 %, +366.4 %] (8/32) | ABF 0.04413, FR 0.09455; +97.6 % [+28.3 %, +298.8 %] (8/32) | ABF 0.1202, FR 0.1513; +0.1 % [-14.1 %, +101.0 %] (16/32) | ABF 0.03011, FR 0.08887; +128.1 % [+56.8 %, +422.3 %] (8/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 289.8 | 0.08845 | 159.2 | 32/32 |
| 2048 | fr | 299 | 0.09125 | 159.2 | 32/32 |
| 512 | abf | 290 | 0.08851 | 159.2 | 32/32 |
| 512 | fr | 296.8 | 0.09057 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
