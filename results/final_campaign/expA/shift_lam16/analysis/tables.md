# Equal-budget ladder analysis: gateway_family -- mechanism cell expA/shift_lam16: A: shifted fibre, kappa = 1, transverse mobility lambda = 16

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/final_campaign/cells/expA/shift_lam16.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/final_campaign/expA`; generated 2026-10-11T09:25:19Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/final_campaign/references/expA_shift_lam16_reference.npz, sha256 f0416c62f2a74ab76de447d77a17209d4ce11a1f6eaa44808f75c832e2a6f12c, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 512 | 6,400,000 | 160 | 32/32 | 32/32 | -- |
| 128 | 25,600,000 | 640 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 160 | 0.01344 [0.0122, 0.01448] | 0.00879 [0.008297, 0.009623] | 0.05831 [0.05495, 0.06061] | 0.04927 [0.04742, 0.05095] | 0.003171 [0.00278, 0.003802] | 0.002373 [0.002106, 0.002967] | 0.03305 [0.03281, 0.03316] | 0.0327 [0.03261, 0.03287] |
| 128 | 640 | 0.004281 [0.003589, 0.004898] | 0.003468 [0.002706, 0.004381] | 0.03915 [0.03848, 0.04024] | 0.03718 [0.03653, 0.03786] | 0.000805 [0.0004396, 0.001367] | 0.00119 [0.0008839, 0.001468] | 0.03242 [0.03239, 0.03251] | 0.0325 [0.03245, 0.03255] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 512 | -33.7 % [-39.7 %, -29.9 %] (32/32) | -17.6 % [-19.2 %, -9.8 %] (31/32) | -26.8 % [-37.6 %, -5.7 %] (25/32) | -1.0 % [-1.2 %, -0.5 %] (24/32) | -73.8 % [-74.8 %, -72.5 %] (32/32) | -87.1 % [-89.2 %, -83.1 %] (32/32) | 18.3 deaths / walker |
| 128 | -22.9 % [-35.1 %, -7.1 %] (24/32) | -5.8 % [-7.7 %, -3.5 %] (29/32) | +45.2 % [-9.2 %, +139.3 %] (11/32) | +0.2 % [+0.1 %, +0.4 %] (9/32) | -73.7 % [-76.6 %, -71.8 %] (32/32) | -70.9 % [-76.4 %, -64.1 %] (32/32) | 149 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 512 | e_F_strict | > 1 | 0.9375 | > 1 | 0.8125 | 6/2/24 | 0.289 | -- [--, --] (n 0) |
| 512 | e_F_mid | 0.5875 | 0 | 0.285 | 0 | 31/1/0 | 1.54e-08 | -51.4 % [-65.5 %, -41.7 %] (n 32) |
| 512 | e_F_loose | 0.125 | 0 | 0.0475 | 0 | 32/0/0 | 4.66e-10 | -60.0 % [-66.7 %, -54.8 %] (n 32) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 0.9062 | 3/0/29 | 0.25 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | 0.465 | 0 | 0.2325 | 0 | 32/0/0 | 4.66e-10 | -54.6 % [-67.5 %, -40.0 %] (n 32) |
| 512 | e_Fp_stat_loose | 0.035 | 0 | 0.025 | 0 | 29/1/2 | 5.77e-08 | -28.6 % [-37.5 %, -28.6 %] (n 32) |
| 512 | TV_half | 0.21 | 0 | 0.045 | 0 | 32/0/0 | 4.66e-10 | -77.8 % [-78.6 %, -76.7 %] (n 32) |
| 512 | e_F_em_strict (secondary) | > 1 | 0.9375 | > 1 | 0.8125 | 6/2/24 | 0.289 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | 0.5875 | 0 | 0.285 | 0 | 31/1/0 | 1.54e-08 | -51.4 % [-65.5 %, -41.7 %] (n 32) |
| 512 | e_F_em_loose (secondary) | 0.125 | 0 | 0.0475 | 0 | 32/0/0 | 4.66e-10 | -60.0 % [-66.7 %, -54.8 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.465 | 0 | 0.2325 | 0 | 32/0/0 | 4.66e-10 | -54.6 % [-67.5 %, -41.0 %] (n 32) |
| 512 | e_Fp_loose (secondary) | 0.035 | 0 | 0.025 | 0 | 29/1/2 | 5.77e-08 | -28.6 % [-37.5 %, -28.6 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.465 | 0 | 0.2325 | 0 | 32/0/0 | 4.66e-10 | -54.6 % [-67.5 %, -41.0 %] (n 32) |
| 512 | e_Fp_em_loose (secondary) | 0.035 | 0 | 0.025 | 0 | 29/1/2 | 5.77e-08 | -28.6 % [-37.5 %, -28.6 %] (n 32) |
| 128 | e_F_strict | 0.5125 | 0.125 | 0.51 | 0.125 | 18/13/1 | 0.473 | -2.7 % [-22.6 %, +39.5 %] (n 25) |
| 128 | e_F_mid | 0.165 | 0 | 0.1125 | 0 | 24/8/0 | 0.007 | -47.5 % [-61.1 %, -8.5 %] (n 32) |
| 128 | e_F_loose | 0.035 | 0 | 0.02 | 0 | 24/6/2 | 0.00143 | -49.0 % [-59.8 %, -36.7 %] (n 32) |
| 128 | e_Fp_stat_strict | 0.7775 | 0.25 | 0.9875 | 0.4062 | 9/21/2 | 0.0428 | +26.5 % [+3.6 %, +37.9 %] (n 13) |
| 128 | e_Fp_stat_mid | 0.1575 | 0 | 0.12 | 0 | 23/9/0 | 0.0201 | -36.6 % [-50.7 %, -13.6 %] (n 32) |
| 128 | e_Fp_stat_loose | 0.015 | 0 | 0.01 | 0 | 22/0/10 | 4.77e-07 | -33.3 % [-35.4 %, -33.3 %] (n 32) |
| 128 | TV_half | 0.06 | 0 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -75.0 % [-76.0 %, -72.7 %] (n 32) |
| 128 | e_F_em_strict (secondary) | 0.5125 | 0.125 | 0.51 | 0.125 | 18/13/1 | 0.473 | -2.7 % [-22.6 %, +39.5 %] (n 25) |
| 128 | e_F_em_mid (secondary) | 0.165 | 0 | 0.1125 | 0 | 24/8/0 | 0.007 | -47.5 % [-61.1 %, -8.5 %] (n 32) |
| 128 | e_F_em_loose (secondary) | 0.035 | 0 | 0.02 | 0 | 24/6/2 | 0.00143 | -49.0 % [-59.8 %, -36.7 %] (n 32) |
| 128 | e_Fp_mid (secondary) | 0.1575 | 0 | 0.12 | 0 | 23/9/0 | 0.0201 | -35.0 % [-50.7 %, -9.2 %] (n 32) |
| 128 | e_Fp_loose (secondary) | 0.015 | 0 | 0.01 | 0 | 22/0/10 | 4.77e-07 | -33.3 % [-35.4 %, -33.3 %] (n 32) |
| 128 | e_Fp_em_mid (secondary) | 0.1575 | 0 | 0.12 | 0 | 23/9/0 | 0.0201 | -35.0 % [-50.7 %, -9.2 %] (n 32) |
| 128 | e_Fp_em_loose (secondary) | 0.015 | 0 | 0.01 | 0 | 22/0/10 | 4.77e-07 | -33.3 % [-35.4 %, -33.3 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 512 | abf | > 1 [> 1, > 1] (0.9375) | 0.5875 [0.4988, 0.7312] (0) | 0.125 [0.1037, 0.145] (0) | > 1 [> 1, > 1] (1) | 0.465 [0.415, 0.5613] (0) | 0.035 [0.03, 0.04] (0) | 0.21 [0.2, 0.22] (0) |
| 512 | fr | > 1 [> 1, > 1] (0.8125) | 0.285 [0.2037, 0.3912] (0) | 0.0475 [0.04, 0.05625] (0) | > 1 [> 1, > 1] (0.9062) | 0.2325 [0.165, 0.2638] (0) | 0.025 [0.02375, 0.025] (0) | 0.045 [0.045, 0.05] (0) |
| 128 | abf | 0.5125 [0.3712, 0.6713] (0.125) | 0.165 [0.1425, 0.225] (0) | 0.035 [0.03, 0.045] (0) | 0.7775 [0.6987, > 1] (0.25) | 0.1575 [0.1313, 0.19] (0) | 0.015 [0.01, 0.015] (0) | 0.06 [0.05, 0.06625] (0) |
| 128 | fr | 0.51 [0.3987, 0.6838] (0.125) | 0.1125 [0.05937, 0.1725] (0) | 0.02 [0.015, 0.025] (0) | 0.9875 [0.72, > 1] (0.4062) | 0.12 [0.07875, 0.1638] (0) | 0.01 [0.00625, 0.01] (0) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | abf | 0.03639 [0.02722, 0.0459] | 0.21 [0.2, 0.22] | 0.07639 [0.06462, 0.08968] | 0.0728 | 0.09 [0.085, 0.1013] | 0 | 0.205 [0.195, 0.2213] | 0.009343 [0.008322, 0.0101] | 1850 [1816, 1876] |
| 512 | fr | 0.004697 [0.00422, 0.005147] | 0.045 [0.045, 0.05] | 0.05469 [0.05181, 0.06858] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05, 0.055] | 0.009198 [0.008225, 0.01003] | 1852 [1824, 1879] |
| 128 | abf | 0.01459 [0.01161, 0.01996] | 0.06 [0.05, 0.06625] | 0.1419 [0.1226, 0.1651] | 0.1447 | 0.025 [0.025, 0.03] | 0 | 0.05 [0.05, 0.06] | 0.00329 [0.002574, 0.004036] | 1844 [1815, 1872] |
| 128 | fr | 0.004555 [0.004068, 0.004892] | 0.015 [0.015, 0.015] | 0.1189 [0.11, 0.1413] | 0.1447 | 0.01 [0.00625, 0.01] | 0 | 0.015 [0.015, 0.015] | 0.00265 [0.0021, 0.003254] | 1834 [1816, 1870] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0/32 | 40 | 0 | 9392 [9300, 9517] | 18.34 [18.16, 18.59] | 0.2193 [0.2175, 0.2217] | 0.0004586 [0.0004541, 0.0004647] | 0.007812 [0.005859, 0.007812] | 0.02122 [0.0188, 0.02729] | 0.3735 [0.3266, 0.4212] | 22 [20, 24] | 0.03223 [0.02734, 0.04541] | 0.0004873 [0.0004831, 0.0004942] |
| 128 | 0/32 | 10 | 0 | 1.911e+04 [1.899e+04, 1.917e+04] | 149.3 [148.4, 149.8] | 0.1157 [0.115, 0.1161] | 0.0009332 [0.0009272, 0.000936] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.01132] | 0.4325 [0.4143, 0.4476] | 1 [1, 2] | 0.07031 [0.0625, 0.08008] | 0.0009672 [0.0009617, 0.0009705] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 512 | e_F | +56.1 % | 0.025 | +2.7 % | 0.015 | +25.2 % | 32 |
| 512 | e_Fp | +89.0 % | 0.025 | +0.3 % | 0.01 | +1.0 % | 32 |
| 512 | e_F_em | +56.1 % | 0.025 | +2.7 % | 0.015 | +25.2 % | 32 |
| 512 | TV_half | +93.6 % | 0.095 | +0.4 % | 0.005 | +87.1 % | 32 |
| 128 | e_F | +43.7 % | 0.035 | -58.8 % | 0.845 | -47.9 % | 32 |
| 128 | e_Fp | +36.6 % | 0.01 | -0.3 % | 0.305 | -0.3 % | 32 |
| 128 | e_F_em | +43.7 % | 0.035 | -58.8 % | 0.845 | -47.9 % | 32 |
| 128 | TV_half | +89.6 % | 0.025 | +21.5 % | 0.005 | +68.8 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.004281 | [0.003676, 0.004763] | 128: 1.00 | 128 | 0.003468 | [0.003061, 0.003927] | 128: 1.00 | -0.0008132 [-0.001481, -6.046e-05] | -19.0 % [-31.7 %, -1.6 %] |
| final_e_F | 128 | 0.000805 | [0.0004876, 0.001207] | 128: 1.00 | 128 | 0.00119 | [0.001044, 0.001372] | 128: 1.00 | 0.0003852 [-8.236e-05, 0.0007569] | +47.9 % [-6.7 %, +145.2 %] |
| tau_bgrid_e_F_mid_u | 128 | 0.165 | [0.1525, 0.2] | 128: 1.00 | 128 | 0.1125 | [0.075, 0.1675] | 128: 1.00 | -0.0525 [-0.1025, 0.0075] | -31.8 % [-56.2 %, +4.7 %] |
| Ibar_Fp | 128 | 0.03915 | [0.03866, 0.03991] | 128: 1.00 | 128 | 0.03718 | [0.03678, 0.03766] | 128: 1.00 | -0.001972 [-0.002909, -0.001301] | -5.0 % [-7.3 %, -3.4 %] |
| final_e_Fp | 128 | 0.03242 | [0.0324, 0.03246] | 128: 1.00 | 128 | 0.0325 | [0.03246, 0.03254] | 128: 1.00 | 8.24e-05 [2.594e-05, 0.000125] | +0.3 % [+0.1 %, +0.4 %] |
| tau_bgrid_TV_half_u | 128 | 0.06 | [0.055, 0.0625] | 128: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.045 [-0.0475, -0.04] | -75.0 % [-76.0 %, -72.7 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 512 | ABF 0.01344, FR 0.00879; -33.7 % [-39.7 %, -29.9 %] (32/32) | ABF 0.05831, FR 0.04927; -17.6 % [-19.2 %, -9.8 %] (31/32) | ABF 0.003171, FR 0.002373; -26.8 % [-37.6 %, -5.7 %] (25/32) | ABF 0.03305, FR 0.0327; -1.0 % [-1.2 %, -0.5 %] (24/32) | ABF 0.03892, FR 0.02592; -34.5 % [-39.2 %, -23.1 %] (31/32) | ABF 0.007139, FR 0.005332; -24.5 % [-32.0 %, -16.2 %] (24/32) |
| 128 | ABF 0.004281, FR 0.003468; -22.9 % [-35.1 %, -7.1 %] (24/32) | ABF 0.03915, FR 0.03718; -5.8 % [-7.7 %, -3.5 %] (29/32) | ABF 0.000805, FR 0.00119; +45.2 % [-9.2 %, +139.3 %] (11/32) | ABF 0.03242, FR 0.0325; +0.2 % [+0.1 %, +0.4 %] (9/32) | ABF 0.01391, FR 0.01103; -18.4 % [-29.0 %, -9.2 %] (28/32) | ABF 0.003143, FR 0.003903; +25.1 % [+8.4 %, +38.0 %] (9/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 512 | abf | 280.7 | 0.08568 | 157.8 | 32/32 |
| 512 | fr | 286 | 0.08727 | 155.8 | 32/32 |
| 128 | abf | 283.9 | 0.08663 | 155.8 | 32/32 |
| 128 | fr | 287.4 | 0.08772 | 155.8 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
