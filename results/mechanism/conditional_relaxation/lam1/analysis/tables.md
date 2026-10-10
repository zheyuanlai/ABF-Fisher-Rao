# Equal-budget ladder analysis: gateway_family -- mechanism cell conditional_relaxation/lam1: II: lambda = 1 (original gateway)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/conditional_relaxation/lam1.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/conditional_relaxation`; generated 2026-10-10T14:19:15Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/conditional_relaxation_lam1_reference.npz, sha256 8a072e07ef9db009cdee6cc86dd185c8abb165706c22f6bdac276c3d79a77e51, n_eval_nodes 151, em_floor_F_rms 7.900546534699823e-05, em_bias_Fp_bin_rms 0.0008073243445745691, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em True.
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
| e_Fp_em_mid | 0.035 | 0.03227 | yes | yes | 0.01355 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01355 |
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
| 2048 | 40 | 0.02265 [0.02168, 0.02411] | 0.01561 [0.01467, 0.01665] | 0.1122 [0.106, 0.1164] | 0.0883 [0.08625, 0.09081] | 0.005436 [0.004983, 0.005926] | 0.002232 [0.001601, 0.002557] | 0.03447 [0.03415, 0.03482] | 0.03272 [0.03255, 0.03287] |
| 512 | 160 | 0.008331 [0.007759, 0.009223] | 0.005038 [0.004253, 0.005919] | 0.05458 [0.05204, 0.05802] | 0.04782 [0.04622, 0.0487] | 0.001522 [0.00134, 0.00175] | 0.0007024 [0.0004636, 0.0009962] | 0.03252 [0.03246, 0.0326] | 0.0324 [0.03235, 0.03249] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -31.4 % [-34.7 %, -27.7 %] (31/32) | -20.9 % [-23.7 %, -18.7 %] (30/32) | -61.4 % [-65.1 %, -55.2 %] (32/32) | -4.9 % [-5.7 %, -4.4 %] (32/32) | -71.5 % [-71.7 %, -71.2 %] (32/32) | -95.2 % [-96.0 %, -94.8 %] (32/32) | 2.61 deaths / walker |
| 512 | -41.0 % [-49.6 %, -33.4 %] (32/32) | -12.3 % [-17.2 %, -11.1 %] (31/32) | -50.5 % [-68.0 %, -36.4 %] (26/32) | -0.4 % [-0.5 %, -0.2 %] (24/32) | -72.7 % [-73.7 %, -71.2 %] (32/32) | -76.6 % [-79.7 %, -67.8 %] (32/32) | 18.4 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | > 1 | 0.5938 | 13/0/19 | 0.000244 | -- [--, --] (n 0) |
| 2048 | e_F_mid | 0.98 | 0.4375 | 0.3175 | 0 | 32/0/0 | 4.66e-10 | -59.7 % [-71.1 %, -49.6 %] (n 18) |
| 2048 | e_F_loose | 0.14 | 0 | 0.09 | 0 | 32/0/0 | 4.66e-10 | -38.9 % [-42.6 %, -32.7 %] (n 32) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | > 1 | 0.75 | 8/0/24 | 0.00781 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | 0.8275 | 0.1562 | 0.295 | 0 | 32/0/0 | 4.66e-10 | -63.4 % [-67.3 %, -55.4 %] (n 27) |
| 2048 | e_Fp_stat_loose | 0.095 | 0 | 0.075 | 0 | 28/3/1 | 4.65e-06 | -25.0 % [-30.0 %, -22.2 %] (n 32) |
| 2048 | TV_half | 0.9625 | 0.1875 | 0.17 | 0 | 32/0/0 | 4.66e-10 | -82.1 % [-82.5 %, -81.8 %] (n 26) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 0.5938 | 13/0/19 | 0.000244 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | 0.98 | 0.4375 | 0.3175 | 0 | 32/0/0 | 4.66e-10 | -59.7 % [-71.1 %, -49.6 %] (n 18) |
| 2048 | e_F_em_loose (secondary) | 0.14 | 0 | 0.09 | 0 | 32/0/0 | 4.66e-10 | -38.9 % [-42.6 %, -32.7 %] (n 32) |
| 2048 | e_Fp_mid (secondary) | 0.835 | 0.1562 | 0.295 | 0 | 32/0/0 | 4.66e-10 | -63.7 % [-67.8 %, -55.4 %] (n 27) |
| 2048 | e_Fp_loose (secondary) | 0.095 | 0 | 0.075 | 0 | 28/3/1 | 4.65e-06 | -25.0 % [-30.0 %, -22.2 %] (n 32) |
| 2048 | e_Fp_em_mid (secondary) | 0.8325 | 0.1562 | 0.295 | 0 | 32/0/0 | 4.66e-10 | -63.7 % [-68.1 %, -55.4 %] (n 27) |
| 2048 | e_Fp_em_loose (secondary) | 0.095 | 0 | 0.075 | 0 | 28/3/1 | 4.65e-06 | -25.0 % [-30.0 %, -22.2 %] (n 32) |
| 512 | e_F_strict | 0.7625 | 0.1875 | 0.2825 | 0.03125 | 28/4/0 | 1.93e-05 | -61.9 % [-77.8 %, -48.3 %] (n 25) |
| 512 | e_F_mid | 0.2725 | 0 | 0.085 | 0 | 32/0/0 | 4.66e-10 | -66.7 % [-75.8 %, -61.8 %] (n 32) |
| 512 | e_F_loose | 0.0425 | 0 | 0.025 | 0 | 28/1/3 | 1.12e-07 | -37.5 % [-44.4 %, -33.3 %] (n 32) |
| 512 | e_Fp_stat_strict | > 1 | 0.5312 | 0.5575 | 0.2188 | 21/7/4 | 0.0125 | -28.4 % [-51.6 %, +5.0 %] (n 12) |
| 512 | e_Fp_stat_mid | 0.245 | 0 | 0.0975 | 0 | 32/0/0 | 4.66e-10 | -62.7 % [-70.4 %, -50.0 %] (n 32) |
| 512 | e_Fp_stat_loose | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -20.0 % [-28.6 %, -16.7 %] (n 32) |
| 512 | TV_half | 0.2425 | 0 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -80.0 % [-80.9 %, -79.4 %] (n 32) |
| 512 | e_F_em_strict (secondary) | 0.7625 | 0.1875 | 0.28 | 0.03125 | 28/4/0 | 1.93e-05 | -61.9 % [-77.8 %, -48.3 %] (n 25) |
| 512 | e_F_em_mid (secondary) | 0.27 | 0 | 0.085 | 0 | 32/0/0 | 4.66e-10 | -67.5 % [-75.8 %, -64.3 %] (n 32) |
| 512 | e_F_em_loose (secondary) | 0.0425 | 0 | 0.025 | 0 | 28/1/3 | 1.12e-07 | -37.5 % [-44.4 %, -33.3 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.245 | 0 | 0.0975 | 0 | 31/0/1 | 9.31e-10 | -62.7 % [-70.4 %, -50.0 %] (n 32) |
| 512 | e_Fp_loose (secondary) | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -20.0 % [-28.6 %, -16.7 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.245 | 0 | 0.0975 | 0 | 31/0/1 | 9.31e-10 | -62.7 % [-70.0 %, -50.0 %] (n 32) |
| 512 | e_Fp_em_loose (secondary) | 0.03 | 0 | 0.025 | 0 | 27/2/3 | 1.62e-06 | -20.0 % [-28.6 %, -16.7 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | 0.98 [0.9038, > 1] (0.4375) | 0.14 [0.13, 0.155] (0) | > 1 [> 1, > 1] (1) | 0.8275 [0.765, 0.9625] (0.1562) | 0.095 [0.09, 0.105] (0) | 0.9625 [0.9375, 0.9912] (0.1875) |
| 2048 | fr | > 1 [0.8375, > 1] (0.5938) | 0.3175 [0.2487, 0.4025] (0) | 0.09 [0.08, 0.095] (0) | > 1 [> 1, > 1] (0.75) | 0.295 [0.2462, 0.36] (0) | 0.075 [0.07, 0.075] (0) | 0.17 [0.17, 0.17] (0) |
| 512 | abf | 0.7625 [0.6525, 0.9263] (0.1875) | 0.2725 [0.225, 0.3313] (0) | 0.0425 [0.035, 0.05125] (0) | > 1 [0.8725, > 1] (0.5312) | 0.245 [0.215, 0.2812] (0) | 0.03 [0.03, 0.035] (0) | 0.2425 [0.22, 0.26] (0) |
| 512 | fr | 0.2825 [0.17, 0.4662] (0.03125) | 0.085 [0.0625, 0.1188] (0) | 0.025 [0.025, 0.03] (0) | 0.5575 [0.3888, 0.905] (0.2188) | 0.0975 [0.06875, 0.13] (0) | 0.025 [0.02, 0.025] (0) | 0.05 [0.045, 0.05] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.09251 [0.08919, 0.09795] | 0.9625 [0.9375, 0.9912] | 0.0526 [0.04755, 0.06055] | 0.0363 | 0.415 [0.4025, 0.43] | 0 | 0.9425 [0.9188, 0.9712] | 0.03496 [0.02893, 0.03744] | 2168 [2143, 2192] |
| 2048 | fr | 0.004348 [0.003762, 0.004842] | 0.17 [0.17, 0.17] | 0.03204 [0.02882, 0.03564] | 0.0363 | 0.105 [0.1, 0.105] | 0 | 0.2075 [0.205, 0.21] | 0.02913 [0.0258, 0.03331] | 2232 [2211, 2255] |
| 512 | abf | 0.01781 [0.01221, 0.0237] | 0.2425 [0.22, 0.26] | 0.07183 [0.06619, 0.08507] | 0.0728 | 0.105 [0.095, 0.115] | 0 | 0.23 [0.2137, 0.2512] | 0.01024 [0.008852, 0.01211] | 2258 [2218, 2277] |
| 512 | fr | 0.00444 [0.004, 0.004875] | 0.05 [0.045, 0.05] | 0.05805 [0.05165, 0.07031] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.055, 0.055] | 0.009055 [0.007679, 0.009863] | 2245 [2220, 2275] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 5346 [5285, 5394] | 2.61 [2.581, 2.634] | 0.4282 [0.4238, 0.4323] | 0.000261 [0.0002581, 0.0002634] | 0.002686 [0.002441, 0.00293] | 0.03546 [0.02891, 0.03943] | 0.3483 [0.3178, 0.3696] | 425.5 [412, 435.2] | 0.01562 [0.0127, 0.01953] | 0.0003024 [0.0002988, 0.0003056] |
| 512 | 0/32 | 40 | 0 | 9418 [9376, 9498] | 18.39 [18.31, 18.55] | 0.2201 [0.2187, 0.2216] | 0.0004599 [0.0004578, 0.0004638] | 0.005859 [0.005859, 0.007812] | 0.02052 [0.01821, 0.0253] | 0.3497 [0.3153, 0.3857] | 22 [19.75, 24] | 0.03711 [0.0332, 0.04297] | 0.0004888 [0.0004862, 0.000492] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +76.4 % | 0.07 | -14.0 % | 0.01 | +58.9 % | 32 |
| 2048 | e_Fp | +92.0 % | 0.08 | -1.0 % | 0.01 | +5.1 % | 32 |
| 2048 | e_F_em | +76.4 % | 0.07 | -14.0 % | 0.01 | +59.1 % | 32 |
| 2048 | TV_half | +97.2 % | 0.31 | -0.0 % | 0.005 | +95.3 % | 32 |
| 512 | e_F | +76.2 % | 0.025 | -2.9 % | 0.015 | +53.8 % | 32 |
| 512 | e_Fp | +92.4 % | 0.025 | +0.4 % | 0.995 | +0.4 % | 32 |
| 512 | e_F_em | +76.2 % | 0.025 | -2.9 % | 0.015 | +55.7 % | 32 |
| 512 | TV_half | +95.1 % | 0.11 | +0.3 % | 0.005 | +75.1 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 512 | 0.008331 | [0.007875, 0.008956] | 512: 1.00 | 512 | 0.005038 | [0.004482, 0.005411] | 512: 1.00 | -0.003293 [-0.004148, -0.002628] | -39.5 % [-47.6 %, -33.0 %] |
| final_e_F | 512 | 0.001522 | [0.001393, 0.001634] | 512: 1.00 | 512 | 0.0007024 | [0.0005424, 0.0009037] | 512: 1.00 | -0.0008195 [-0.001011, -0.0005798] | -53.8 % [-64.7 %, -40.1 %] |
| tau_bgrid_e_F_mid_u | 512 | 0.2725 | [0.23, 0.3] | 512: 1.00 | 512 | 0.085 | [0.0675, 0.11] | 512: 1.00 | -0.1875 [-0.22, -0.145] | -68.8 % [-75.4 %, -57.7 %] |
| Ibar_Fp | 512 | 0.05458 | [0.05377, 0.05719] | 512: 1.00 | 512 | 0.04782 | [0.04701, 0.04834] | 512: 1.00 | -0.006766 [-0.009691, -0.005814] | -12.4 % [-16.9 %, -10.8 %] |
| final_e_Fp | 512 | 0.03252 | [0.03249, 0.03257] | 512: 1.00 | 512 | 0.0324 | [0.03237, 0.03245] | 512: 1.00 | -0.0001211 [-0.0001718, -6.445e-05] | -0.4 % [-0.5 %, -0.2 %] |
| tau_bgrid_TV_half_u | 512 | 0.2425 | [0.225, 0.2525] | 512: 1.00 | 512 | 0.05 | [0.045, 0.05] | 512: 1.00 | -0.1925 [-0.205, -0.175] | -79.4 % [-82.0 %, -78.3 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.02265, FR 0.0156; -31.4 % [-34.8 %, -27.7 %] (31/32) | ABF 0.1122, FR 0.08829; -20.9 % [-23.7 %, -18.7 %] (30/32) | ABF 0.005435, FR 0.002221; -61.4 % [-65.3 %, -55.2 %] (32/32) | ABF 0.03447, FR 0.03271; -4.9 % [-5.7 %, -4.4 %] (32/32) | ABF 0.09601, FR 0.06692; -29.9 % [-34.6 %, -26.3 %] (31/32) | ABF 0.01214, FR 0.005404; -57.3 % [-61.6 %, -50.6 %] (32/32) |
| 512 | ABF 0.008325, FR 0.005024; -41.1 % [-49.8 %, -33.4 %] (32/32) | ABF 0.05458, FR 0.04781; -12.4 % [-17.2 %, -11.1 %] (31/32) | ABF 0.00151, FR 0.0006681; -52.0 % [-68.1 %, -38.3 %] (26/32) | ABF 0.03252, FR 0.0324; -0.4 % [-0.5 %, -0.2 %] (24/32) | ABF 0.03192, FR 0.02111; -32.6 % [-38.0 %, -28.7 %] (31/32) | ABF 0.004066, FR 0.002943; -30.0 % [-38.3 %, -16.5 %] (24/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 282.6 | 0.08623 | 157.2 | 32/32 |
| 2048 | fr | 288 | 0.08789 | 159.2 | 32/32 |
| 512 | abf | 285.2 | 0.08702 | 157.2 | 32/32 |
| 512 | fr | 289.7 | 0.08842 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
