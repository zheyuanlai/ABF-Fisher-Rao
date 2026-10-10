# Equal-budget ladder analysis: gateway_family -- mechanism cell matched_free_energy/shift: I: shifted fibre, kappa = 1 (variance-matched control; constant width, curved slow fibre)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/matched_free_energy/shift.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/matched_free_energy`; generated 2026-10-10T14:19:18Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/matched_free_energy_shift_reference.npz, sha256 bfaab9feab5ab7f524069fb4ba4b4a1dafd708fcfc63f236d86da88c72d92c21, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 128 | 25,600,000 | 640 | 32/32 | 32/32 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 40 | 0.156 [0.1533, 0.1594] | 0.1254 [0.1191, 0.1303] | 0.3829 [0.3768, 0.3878] | 0.3586 [0.3505, 0.3684] | 0.1207 [0.1196, 0.1229] | 0.09881 [0.09303, 0.1041] | 0.2492 [0.2453, 0.2559] | 0.2605 [0.2504, 0.2692] |
| 512 | 160 | 0.09556 [0.09164, 0.1008] | 0.09768 [0.08889, 0.1047] | 0.2175 [0.208, 0.2292] | 0.2594 [0.2481, 0.2749] | 0.04759 [0.04356, 0.05145] | 0.07871 [0.07057, 0.08416] | 0.1073 [0.09904, 0.1145] | 0.2028 [0.1919, 0.2136] |
| 128 | 640 | 0.03668 [0.03429, 0.0394] | 0.07326 [0.06235, 0.08834] | 0.09537 [0.0905, 0.09894] | 0.1971 [0.167, 0.2192] | 0.006674 [0.004506, 0.008005] | 0.05872 [0.05053, 0.07299] | 0.03543 [0.03459, 0.03693] | 0.158 [0.1336, 0.1745] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -21.0 % [-22.0 %, -17.9 %] (32/32) | -6.1 % [-8.2 %, -5.3 %] (31/32) | -19.0 % [-21.5 %, -15.6 %] (32/32) | +3.6 % [+1.8 %, +6.1 %] (7/32) | -64.7 % [-65.2 %, -64.2 %] (32/32) | -81.5 % [-81.7 %, -80.9 %] (32/32) | 3.22 deaths / walker |
| 512 | +2.0 % [-4.6 %, +7.8 %] (15/32) | +20.7 % [+11.0 %, +26.9 %] (4/32) | +65.7 % [+55.5 %, +75.6 %] (0/32) | +92.7 % [+74.8 %, +103.1 %] (0/32) | -87.7 % [-88.1 %, -87.3 %] (32/32) | -96.0 % [-96.3 %, -95.8 %] (32/32) | 19.4 deaths / walker |
| 128 | +105.0 % [+61.4 %, +127.9 %] (0/32) | +111.3 % [+83.6 %, +122.6 %] (0/32) | +830.7 % [+584.7 %, +1036.2 %] (0/32) | +330.7 % [+299.4 %, +357.5 %] (0/32) | -91.6 % [-91.9 %, -91.1 %] (32/32) | -87.2 % [-88.9 %, -85.8 %] (32/32) | 149 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_loose | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_loose | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | TV_half | 0.82 | 0.1562 | 0.18 | 0 | 32/0/0 | 4.66e-10 | -78.0 % [-78.5 %, -77.5 %] (n 27) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_em_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_loose | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_loose | > 1 | 0.7188 | > 1 | 1 | 0/9/23 | 0.00391 | -- [--, --] (n 0) |
| 512 | TV_half | > 1 | 1 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -- [--, --] (n 0) |
| 512 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_loose (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_loose (secondary) | > 1 | 0.7188 | > 1 | 1 | 0/9/23 | 0.00391 | -- [--, --] (n 0) |
| 512 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_em_loose (secondary) | > 1 | 0.7188 | > 1 | 1 | 0/9/23 | 0.00391 | -- [--, --] (n 0) |
| 128 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 128 | e_F_mid | > 1 | 0.6562 | > 1 | 1 | 0/11/21 | 0.000977 | -- [--, --] (n 0) |
| 128 | e_F_loose | 0.475 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_mid | > 1 | 0.5625 | > 1 | 1 | 0/14/18 | 0.000122 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_loose | 0.275 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | TV_half | > 1 | 0.5938 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -98.2 % [-98.5 %, -98.1 %] (n 13) |
| 128 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 128 | e_F_em_mid (secondary) | > 1 | 0.6562 | > 1 | 1 | 0/11/21 | 0.000977 | -- [--, --] (n 0) |
| 128 | e_F_em_loose (secondary) | 0.475 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_mid (secondary) | > 1 | 0.5938 | > 1 | 1 | 0/13/19 | 0.000244 | -- [--, --] (n 0) |
| 128 | e_Fp_loose (secondary) | 0.275 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_em_mid (secondary) | > 1 | 0.5938 | > 1 | 1 | 0/13/19 | 0.000244 | -- [--, --] (n 0) |
| 128 | e_Fp_em_loose (secondary) | 0.275 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.82 [0.7938, 0.85] (0.1562) |
| 2048 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.18 [0.1737, 0.18] (0) |
| 512 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [0.9975, > 1] (0.7188) | > 1 [> 1, > 1] (1) |
| 512 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.05 [0.05, 0.05] (0) |
| 128 | abf | > 1 [> 1, > 1] (1) | > 1 [0.8838, > 1] (0.6562) | 0.475 [0.4537, 0.525] (0) | > 1 [> 1, > 1] (1) | > 1 [0.9425, > 1] (0.5625) | 0.275 [0.26, 0.3013] (0) | > 1 [0.9612, > 1] (0.5938) |
| 128 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.015 [0.015, 0.01516] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.08942 [0.08285, 0.09236] | 0.82 [0.7938, 0.85] | 0.1731 [0.1611, 0.1835] | 0.0363 | 0.375 [0.3688, 0.395] | 0 | 0.7475 [0.7288, 0.765] | 0.03504 [0.02886, 0.04073] | 1316 [1293, 1341] |
| 2048 | fr | 0.01675 [0.01549, 0.01751] | 0.18 [0.1737, 0.18] | 0.03215 [0.02793, 0.03967] | 0.0363 | 0.105 [0.1, 0.105] | 0 | 0.21 [0.205, 0.215] | 0.03227 [0.02824, 0.03616] | 779 [724.8, 833.8] |
| 512 | abf | 0.3481 [0.3388, 0.3609] | > 1 [> 1, > 1] | 0.3359 [0.3286, 0.3599] | 0.0728 | 0.095 [0.09375, 0.1013] | 0 | 0.1875 [0.185, 0.1963] | 0.009308 [0.008174, 0.01193] | 630 [621.2, 638.5] |
| 512 | fr | 0.01408 [0.01302, 0.01447] | 0.05 [0.05, 0.05] | 0.06087 [0.05344, 0.06972] | 0.0728 | 0.03 [0.03, 0.03] | 0 | 0.055 [0.055, 0.06] | 0.01017 [0.008827, 0.0115] | 610 [547.8, 709.5] |
| 128 | abf | 0.1076 [0.09058, 0.1253] | > 1 [0.9612, > 1] | 0.1558 [0.1415, 0.1717] | 0.1447 | 0.025 [0.025, 0.03] | 0 | 0.05 [0.045, 0.05125] | 0.00367 [0.003119, 0.004694] | 508.5 [492.5, 525] |
| 128 | fr | 0.01275 [0.01172, 0.01375] | 0.015 [0.015, 0.01516] | 0.1181 [0.1035, 0.1339] | 0.1447 | 0.01 [0.01, 0.01] | 0 | 0.015 [0.015, 0.01672] | 0.00314 [0.002548, 0.003757] | 585 [470.8, 662] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 6604 [6514, 6716] | 3.224 [3.181, 3.279] | 0.5143 [0.51, 0.5199] | 0.0003224 [0.0003181, 0.0003279] | 0.00293 [0.002441, 0.00293] | 0.04815 [0.03915, 0.05635] | 0.334 [0.3002, 0.3578] | 352.5 [340.2, 365] | 0.01636 [0.01367, 0.01978] | 0.0003783 [0.0003731, 0.0003847] |
| 512 | 0/32 | 40 | 0 | 9950 [9849, 1.005e+04] | 19.43 [19.24, 19.64] | 0.2309 [0.2286, 0.2341] | 0.0004859 [0.0004809, 0.0004909] | 0.007812 [0.007324, 0.007812] | 0.0178 [0.01486, 0.02097] | 0.3459 [0.3045, 0.3889] | 18 [16, 19.25] | 0.03906 [0.03467, 0.04736] | 0.0005181 [0.0005109, 0.0005244] |
| 128 | 0/32 | 10 | 0 | 1.906e+04 [1.897e+04, 1.924e+04] | 148.9 [148.2, 150.3] | 0.1153 [0.1146, 0.1162] | 0.0009308 [0.0009261, 0.0009394] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.007812] | 0.4183 [0.3914, 0.4384] | 1 [1, 1] | 0.07812 [0.06836, 0.08594] | 0.000965 [0.0009598, 0.0009726] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +25.1 % | 0.365 | -15.7 % | 0.01 | +18.1 % | 32 |
| 2048 | e_Fp | +44.7 % | 0.08 | -4.5 % | 1 | -4.5 % | 32 |
| 2048 | e_F_em | +25.1 % | 0.365 | -15.7 % | 0.01 | +18.1 % | 32 |
| 2048 | TV_half | +92.3 % | 0.315 | -0.0 % | 0.005 | +81.3 % | 32 |
| 512 | e_F | +22.2 % | 0.11 | -65.4 % | 1 | -65.4 % | 32 |
| 512 | e_Fp | +47.7 % | 0.025 | -89.1 % | 1 | -89.1 % | 32 |
| 512 | e_F_em | +22.2 % | 0.11 | -65.4 % | 1 | -65.4 % | 32 |
| 512 | TV_half | +96.0 % | 0.995 | +0.3 % | 0.005 | +96.0 % | 32 |
| 128 | e_F | +26.3 % | 0.025 | -779.9 % | 1 | -779.9 % | 32 |
| 128 | e_Fp | +48.8 % | 0.01 | -347.0 % | 0.885 | -346.1 % | 32 |
| 128 | e_F_em | +26.3 % | 0.025 | -779.9 % | 1 | -779.9 % | 32 |
| 128 | TV_half | +95.3 % | 0.245 | +11.9 % | 0.005 | +88.2 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.03668 | [0.0346, 0.0384] | 128: 1.00 | 128 | 0.07326 | [0.06591, 0.07888] | 128: 1.00, 512: 0.00 | 0.03658 [0.0286, 0.04189] | +99.7 % [+76.6 %, +115.1 %] |
| final_e_F | 128 | 0.006674 | [0.005174, 0.007526] | 128: 1.00 | 128 | 0.05872 | [0.05298, 0.06356] | 128: 1.00, 512: 0.00 | 0.05205 [0.04613, 0.05707] | +779.9 % [+649.1 %, +1021.4 %] |
| tau_bgrid_e_F_mid_u | all censored | > 1 | [> 1, > 1] | 128: 0.02, all censored: 0.98 | all censored | > 1 | [> 1, > 1] | all censored: 1.00 | -- [inf, inf] | -- [inf, inf] |
| Ibar_Fp | 128 | 0.09537 | [0.09174, 0.09841] | 128: 1.00 | 128 | 0.1971 | [0.1763, 0.2066] | 128: 1.00 | 0.1017 [0.07996, 0.1135] | +106.6 % [+82.1 %, +120.2 %] |
| final_e_Fp | 128 | 0.03543 | [0.0347, 0.03635] | 128: 1.00 | 128 | 0.158 | [0.1402, 0.164] | 128: 1.00 | 0.1226 [0.1046, 0.1292] | +346.1 % [+293.2 %, +372.1 %] |
| tau_bgrid_TV_half_u | 2048 | 0.82 | [0.8, 0.83] | 2048: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.805 [-0.815, -0.785] | -98.2 % [-98.2 %, -98.1 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.156, FR 0.1254; -21.0 % [-22.0 %, -17.9 %] (32/32) | ABF 0.3829, FR 0.3586; -6.1 % [-8.2 %, -5.3 %] (31/32) | ABF 0.1207, FR 0.09881; -19.0 % [-21.5 %, -15.6 %] (32/32) | ABF 0.2492, FR 0.2605; +3.6 % [+1.8 %, +6.1 %] (7/32) | ABF 0.3813, FR 0.357; -6.1 % [-8.3 %, -5.3 %] (31/32) | ABF 0.2471, FR 0.2585; +3.7 % [+1.8 %, +6.2 %] (7/32) |
| 512 | ABF 0.09556, FR 0.09768; +2.0 % [-4.6 %, +7.8 %] (15/32) | ABF 0.2175, FR 0.2594; +20.7 % [+11.0 %, +26.9 %] (4/32) | ABF 0.04759, FR 0.07871; +65.7 % [+55.5 %, +75.6 %] (0/32) | ABF 0.1073, FR 0.2028; +92.7 % [+74.8 %, +103.1 %] (0/32) | ABF 0.2146, FR 0.2573; +21.3 % [+11.4 %, +27.8 %] (4/32) | ABF 0.1023, FR 0.2002; +99.8 % [+80.6 %, +111.6 %] (0/32) |
| 128 | ABF 0.03668, FR 0.07326; +105.0 % [+61.4 %, +127.9 %] (0/32) | ABF 0.09537, FR 0.1971; +111.3 % [+83.6 %, +122.6 %] (0/32) | ABF 0.006674, FR 0.05872; +830.7 % [+584.7 %, +1036.2 %] (0/32) | ABF 0.03543, FR 0.158; +330.7 % [+299.4 %, +357.5 %] (0/32) | ABF 0.08324, FR 0.1943; +131.3 % [+105.1 %, +155.6 %] (0/32) | ABF 0.01462, FR 0.1547; +893.0 % [+729.5 %, +1020.5 %] (0/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 303.4 | 0.09259 | 159.2 | 32/32 |
| 2048 | fr | 308 | 0.094 | 159.2 | 32/32 |
| 512 | abf | 305.7 | 0.09328 | 159.2 | 32/32 |
| 512 | fr | 309.5 | 0.09445 | 159.2 | 32/32 |
| 128 | abf | 309.8 | 0.09453 | 159.2 | 32/32 |
| 128 | fr | 316.1 | 0.09646 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
