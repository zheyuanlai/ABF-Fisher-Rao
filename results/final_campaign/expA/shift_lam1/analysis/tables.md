# Equal-budget ladder analysis: gateway_family -- mechanism cell expA/shift_lam1: A: shifted fibre, kappa = 1, transverse mobility lambda = 1

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/final_campaign/cells/expA/shift_lam1.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/final_campaign/expA`; generated 2026-10-11T09:25:19Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/final_campaign/references/expA_shift_lam1_reference.npz, sha256 31fc5dafdac7f84eec97dd6ed93d8d4f96c68c6f6a0d7d83712e03fc237652b7, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 512 | 160 | 0.09899 [0.09631, 0.1052] | 0.09492 [0.08024, 0.1007] | 0.2234 [0.2179, 0.2354] | 0.2584 [0.2392, 0.2787] | 0.04961 [0.04709, 0.05285] | 0.07505 [0.06351, 0.08089] | 0.1103 [0.1065, 0.1203] | 0.2025 [0.1868, 0.2173] |
| 128 | 640 | 0.03739 [0.031, 0.04059] | 0.07339 [0.06335, 0.0799] | 0.09348 [0.08348, 0.1005] | 0.1882 [0.1654, 0.2054] | 0.005337 [0.003745, 0.006967] | 0.05927 [0.05193, 0.06498] | 0.0347 [0.03362, 0.03687] | 0.1493 [0.1337, 0.1676] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 512 | -7.1 % [-14.0 %, -0.0 %] (21/32) | +15.0 % [+8.1 %, +18.8 %] (2/32) | +45.9 % [+31.3 %, +57.3 %] (1/32) | +73.8 % [+65.7 %, +89.0 %] (0/32) | -88.0 % [-88.3 %, -87.5 %] (32/32) | -96.1 % [-96.2 %, -95.7 %] (32/32) | 19.4 deaths / walker |
| 128 | +92.3 % [+74.4 %, +141.4 %] (1/32) | +100.2 % [+86.0 %, +115.5 %] (0/32) | +1093.7 % [+829.2 %, +1345.2 %] (0/32) | +320.2 % [+295.7 %, +362.5 %] (0/32) | -92.0 % [-92.4 %, -91.5 %] (32/32) | -89.1 % [-90.6 %, -87.6 %] (32/32) | 149 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 512 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_loose | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_loose | > 1 | 0.9688 | > 1 | 1 | 0/1/31 | 1 | -- [--, --] (n 0) |
| 512 | TV_half | > 1 | 1 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -- [--, --] (n 0) |
| 512 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_loose (secondary) | > 1 | 0.9688 | > 1 | 1 | 0/1/31 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_em_loose (secondary) | > 1 | 0.9688 | > 1 | 1 | 0/1/31 | 1 | -- [--, --] (n 0) |
| 128 | e_F_strict | > 1 | 0.9062 | > 1 | 1 | 0/3/29 | 0.25 | -- [--, --] (n 0) |
| 128 | e_F_mid | > 1 | 0.5 | > 1 | 1 | 0/16/16 | 3.05e-05 | -- [--, --] (n 0) |
| 128 | e_F_loose | 0.4625 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_mid | 0.9675 | 0.4062 | > 1 | 1 | 0/19/13 | 3.81e-06 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_loose | 0.275 | 0 | > 1 | 0.9688 | 0/32/0 | 4.66e-10 | +163.4 % [+163.4 %, +163.4 %] (n 1) |
| 128 | TV_half | > 1 | 0.5938 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -98.4 % [-98.5 %, -98.3 %] (n 13) |
| 128 | e_F_em_strict (secondary) | > 1 | 0.9062 | > 1 | 1 | 0/3/29 | 0.25 | -- [--, --] (n 0) |
| 128 | e_F_em_mid (secondary) | > 1 | 0.5 | > 1 | 1 | 0/16/16 | 3.05e-05 | -- [--, --] (n 0) |
| 128 | e_F_em_loose (secondary) | 0.4625 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_mid (secondary) | 0.9725 | 0.4062 | > 1 | 1 | 0/19/13 | 3.81e-06 | -- [--, --] (n 0) |
| 128 | e_Fp_loose (secondary) | 0.275 | 0 | > 1 | 0.9688 | 0/32/0 | 4.66e-10 | +163.4 % [+163.4 %, +163.4 %] (n 1) |
| 128 | e_Fp_em_mid (secondary) | 0.9725 | 0.4062 | > 1 | 1 | 0/19/13 | 3.81e-06 | -- [--, --] (n 0) |
| 128 | e_Fp_em_loose (secondary) | 0.275 | 0 | > 1 | 0.9688 | 0/32/0 | 4.66e-10 | +163.4 % [+163.4 %, +163.4 %] (n 1) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 512 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (0.9688) | > 1 [> 1, > 1] (1) |
| 512 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.05 [0.045, 0.05] (0) |
| 128 | abf | > 1 [> 1, > 1] (0.9062) | > 1 [0.8438, > 1] (0.5) | 0.4625 [0.415, 0.5212] (0) | > 1 [> 1, > 1] (1) | 0.9675 [0.7575, > 1] (0.4062) | 0.275 [0.24, 0.3113] (0) | > 1 [0.9675, > 1] (0.5938) |
| 128 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (0.9688) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | abf | 0.3517 [0.34, 0.3693] | > 1 [> 1, > 1] | 0.3452 [0.3346, 0.3623] | 0.0728 | 0.095 [0.09, 0.1] | 0 | 0.1875 [0.1787, 0.195] | 0.01084 [0.009275, 0.01229] | 617 [604.8, 631.5] |
| 512 | fr | 0.01392 [0.01276, 0.01589] | 0.05 [0.045, 0.05] | 0.06434 [0.05859, 0.0752] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05, 0.06] | 0.01016 [0.008967, 0.01145] | 605 [534.5, 698] |
| 128 | abf | 0.1067 [0.09185, 0.1268] | > 1 [0.9675, > 1] | 0.1454 [0.1326, 0.158] | 0.1447 | 0.025 [0.025, 0.03] | 0 | 0.05 [0.045, 0.055] | 0.003344 [0.00301, 0.004015] | 517.5 [504.8, 532.2] |
| 128 | fr | 0.0115 [0.01038, 0.01246] | 0.015 [0.015, 0.015] | 0.1168 [0.107, 0.1339] | 0.1447 | 0.01 [0.01, 0.01] | 0 | 0.015 [0.015, 0.01562] | 0.002974 [0.002443, 0.003505] | 558 [488, 601.8] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0/32 | 40 | 0 | 9932 [9808, 1.012e+04] | 19.4 [19.16, 19.77] | 0.2311 [0.228, 0.2353] | 0.000485 [0.0004789, 0.0004943] | 0.007812 [0.007812, 0.007812] | 0.01697 [0.01448, 0.01894] | 0.3211 [0.2868, 0.354] | 18 [16, 20] | 0.04102 [0.03662, 0.05322] | 0.0005172 [0.000511, 0.0005272] |
| 128 | 0/32 | 10 | 0 | 1.91e+04 [1.897e+04, 1.916e+04] | 149.2 [148.2, 149.7] | 0.1155 [0.115, 0.116] | 0.0009324 [0.0009263, 0.0009357] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.007812] | 0.4253 [0.409, 0.4414] | 1 [1, 1] | 0.07031 [0.0625, 0.07812] | 0.0009656 [0.0009595, 0.0009701] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 512 | e_F | +27.1 % | 0.08 | -51.3 % | 1 | -51.3 % | 32 |
| 512 | e_Fp | +48.4 % | 0.025 | -83.5 % | 1 | -83.5 % | 32 |
| 512 | e_F_em | +27.1 % | 0.08 | -51.3 % | 1 | -51.3 % | 32 |
| 512 | TV_half | +96.0 % | 0.94 | +0.5 % | 0.005 | +96.0 % | 32 |
| 128 | e_F | +27.2 % | 0.025 | -1010.4 % | 1 | -1010.4 % | 32 |
| 128 | e_Fp | +26.5 % | 0.01 | -330.3 % | 0.99 | -330.2 % | 32 |
| 128 | e_F_em | +27.2 % | 0.025 | -1010.4 % | 1 | -1010.4 % | 32 |
| 128 | TV_half | +95.4 % | 0.355 | +13.3 % | 0.005 | +89.2 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.03739 | [0.03328, 0.03939] | 128: 1.00 | 128 | 0.07339 | [0.06797, 0.07758] | 128: 1.00 | 0.036 [0.03028, 0.04263] | +96.3 % [+78.7 %, +126.6 %] |
| final_e_F | 128 | 0.005337 | [0.004115, 0.006304] | 128: 1.00 | 128 | 0.05927 | [0.05565, 0.06364] | 128: 1.00, 512: 0.00 | 0.05393 [0.05011, 0.0587] | +1010.4 % [+813.2 %, +1388.2 %] |
| tau_bgrid_e_F_mid_u | all censored | > 1 | [0.875, > 1] | 128: 0.42, all censored: 0.58 | all censored | > 1 | [> 1, > 1] | all censored: 1.00 | -- [inf, inf] | -- [inf, inf] |
| Ibar_Fp | 128 | 0.09348 | [0.09025, 0.09744] | 128: 1.00 | 128 | 0.1882 | [0.173, 0.1999] | 128: 1.00 | 0.09476 [0.07893, 0.1077] | +101.4 % [+82.9 %, +118.2 %] |
| final_e_Fp | 128 | 0.0347 | [0.03403, 0.03546] | 128: 1.00 | 128 | 0.1493 | [0.1402, 0.1605] | 128: 1.00 | 0.1146 [0.1051, 0.1259] | +330.2 % [+299.4 %, +364.8 %] |
| tau_bgrid_TV_half_u | all censored | > 1 | [0.995, > 1] | 128: 0.10, all censored: 0.90 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | inf [inf, -0.98] | -- [-98.5 %, -98.5 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 512 | ABF 0.09899, FR 0.09492; -7.1 % [-14.0 %, -0.0 %] (21/32) | ABF 0.2234, FR 0.2584; +15.0 % [+8.1 %, +18.8 %] (2/32) | ABF 0.04961, FR 0.07505; +45.9 % [+31.3 %, +57.3 %] (1/32) | ABF 0.1103, FR 0.2025; +73.8 % [+65.7 %, +89.0 %] (0/32) | ABF 0.2206, FR 0.2563; +15.4 % [+8.4 %, +19.5 %] (2/32) | ABF 0.1055, FR 0.1999; +79.5 % [+70.2 %, +94.3 %] (0/32) |
| 128 | ABF 0.03739, FR 0.07339; +92.3 % [+74.4 %, +141.4 %] (1/32) | ABF 0.09348, FR 0.1882; +100.2 % [+86.0 %, +115.5 %] (0/32) | ABF 0.005337, FR 0.05927; +1093.7 % [+829.2 %, +1345.2 %] (0/32) | ABF 0.0347, FR 0.1493; +320.2 % [+295.7 %, +362.5 %] (0/32) | ABF 0.08187, FR 0.1853; +128.3 % [+102.8 %, +146.3 %] (0/32) | ABF 0.01276, FR 0.1457; +1083.0 % [+812.2 %, +1373.3 %] (0/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 512 | abf | 278.4 | 0.08495 | 157.8 | 32/32 |
| 512 | fr | 283.8 | 0.0866 | 158.8 | 32/32 |
| 128 | abf | 281.4 | 0.08587 | 157.8 | 32/32 |
| 128 | fr | 285 | 0.08699 | 158.8 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
