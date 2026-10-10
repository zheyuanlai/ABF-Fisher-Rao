# Equal-budget ladder analysis: gateway_family -- mechanism cell conditional_relaxation/lam0.25: II: lambda = 0.25 (transverse mobility / 4)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/conditional_relaxation/lam0.25.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/conditional_relaxation`; generated 2026-10-10T14:19:13Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/conditional_relaxation_lam0.25_reference.npz, sha256 7db51a761e51e83044edfdf1e458245737f2e3f74f02cff1b146071c107b42cf, n_eval_nodes 151, em_floor_F_rms 1.9672136389038837e-05, em_bias_Fp_bin_rms 0.00020076912401783273, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 2048 | 40 | 0.04445 [0.04125, 0.04875] | 0.0338 [0.02967, 0.03765] | 0.1478 [0.139, 0.1536] | 0.1167 [0.1089, 0.1264] | 0.0178 [0.01678, 0.01992] | 0.01162 [0.01038, 0.01481] | 0.05074 [0.04887, 0.0539] | 0.04228 [0.03967, 0.04747] |
| 512 | 160 | 0.0192 [0.01703, 0.02835] | 0.01636 [0.01368, 0.02405] | 0.07127 [0.06631, 0.09308] | 0.06054 [0.05496, 0.07609] | 0.004807 [0.004053, 0.005869] | 0.006265 [0.005011, 0.01024] | 0.03402 [0.0335, 0.03503] | 0.03563 [0.03414, 0.04074] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -28.8 % [-31.4 %, -23.3 %] (28/32) | -23.7 % [-25.8 %, -18.1 %] (28/32) | -38.2 % [-40.6 %, -27.9 %] (29/32) | -17.9 % [-21.0 %, -12.4 %] (29/32) | -65.3 % [-65.8 %, -64.2 %] (32/32) | -84.5 % [-85.8 %, -83.2 %] (32/32) | 2.7 deaths / walker |
| 512 | -14.3 % [-47.3 %, +5.2 %] (21/32) | -12.9 % [-29.3 %, -2.5 %] (22/32) | +27.6 % [-5.2 %, +76.5 %] (12/32) | +3.8 % [-0.8 %, +7.9 %] (12/32) | -75.7 % [-78.4 %, -73.6 %] (32/32) | -90.1 % [-93.1 %, -86.7 %] (32/32) | 18.5 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_loose | 0.85 | 0.2188 | 0.4225 | 0.125 | 27/3/2 | 8.43e-06 | -54.8 % [-57.8 %, -49.4 %] (n 23) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_loose | 0.2475 | 0 | 0.1475 | 0 | 29/3/0 | 2.56e-06 | -42.5 % [-49.0 %, -33.3 %] (n 32) |
| 2048 | TV_half | 0.725 | 0 | 0.17 | 0 | 32/0/0 | 4.66e-10 | -76.6 % [-77.4 %, -75.4 %] (n 32) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_loose (secondary) | 0.85 | 0.2188 | 0.4225 | 0.125 | 27/3/2 | 8.43e-06 | -54.8 % [-57.8 %, -49.4 %] (n 23) |
| 2048 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_loose (secondary) | 0.2475 | 0 | 0.1475 | 0 | 29/3/0 | 2.56e-06 | -42.5 % [-49.0 %, -33.3 %] (n 32) |
| 2048 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_em_loose (secondary) | 0.2475 | 0 | 0.1475 | 0 | 29/3/0 | 2.56e-06 | -42.5 % [-49.0 %, -33.3 %] (n 32) |
| 512 | e_F_strict | > 1 | 1 | > 1 | 0.9688 | 1/0/31 | 1 | -- [--, --] (n 0) |
| 512 | e_F_mid | 0.89 | 0.3125 | > 1 | 0.6562 | 9/17/6 | 0.169 | -34.0 % [-55.1 %, +6.1 %] (n 7) |
| 512 | e_F_loose | 0.25 | 0 | 0.155 | 0.09375 | 22/10/0 | 0.0501 | -40.4 % [-61.8 %, -26.2 %] (n 29) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | 0.8125 | 0.25 | > 1 | 0.5938 | 11/16/5 | 0.442 | -20.4 % [-53.2 %, +7.2 %] (n 10) |
| 512 | e_Fp_stat_loose | 0.085 | 0 | 0.05 | 0.0625 | 21/10/1 | 0.0708 | -37.5 % [-67.8 %, -18.3 %] (n 30) |
| 512 | TV_half | 0.2025 | 0.125 | 0.05 | 0 | 32/0/0 | 4.66e-10 | -76.5 % [-79.8 %, -74.3 %] (n 28) |
| 512 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 0.9688 | 1/0/31 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | 0.89 | 0.3125 | > 1 | 0.6562 | 9/17/6 | 0.169 | -34.0 % [-55.1 %, +6.1 %] (n 7) |
| 512 | e_F_em_loose (secondary) | 0.25 | 0 | 0.155 | 0.09375 | 22/10/0 | 0.0501 | -40.4 % [-61.8 %, -26.2 %] (n 29) |
| 512 | e_Fp_mid (secondary) | 0.815 | 0.25 | > 1 | 0.5938 | 11/16/5 | 0.442 | -20.0 % [-52.8 %, +7.1 %] (n 10) |
| 512 | e_Fp_loose (secondary) | 0.085 | 0 | 0.05 | 0.0625 | 21/10/1 | 0.0708 | -37.5 % [-67.8 %, -18.3 %] (n 30) |
| 512 | e_Fp_em_mid (secondary) | 0.815 | 0.25 | > 1 | 0.5938 | 11/16/5 | 0.442 | -20.0 % [-52.8 %, +6.8 %] (n 10) |
| 512 | e_Fp_em_loose (secondary) | 0.085 | 0 | 0.05 | 0.0625 | 21/10/1 | 0.0708 | -37.5 % [-67.8 %, -18.3 %] (n 30) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.85 [0.7913, 0.995] (0.2188) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.2475 [0.2188, 0.3075] (0) | 0.725 [0.6887, 0.775] (0) |
| 2048 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.4225 [0.37, 0.6175] (0.125) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.1475 [0.12, 0.1975] (0) | 0.17 [0.17, 0.175] (0) |
| 512 | abf | > 1 [> 1, > 1] (1) | 0.89 [0.7875, > 1] (0.3125) | 0.25 [0.215, 0.3113] (0) | > 1 [> 1, > 1] (1) | 0.8125 [0.7225, > 1] (0.25) | 0.085 [0.06, 0.1363] (0) | 0.2025 [0.18, 0.7737] (0.125) |
| 512 | fr | > 1 [> 1, > 1] (0.9688) | > 1 [0.8512, > 1] (0.6562) | 0.155 [0.105, 0.3425] (0.09375) | > 1 [> 1, > 1] (1) | > 1 [0.6925, > 1] (0.5938) | 0.05 [0.03875, 0.1013] (0.0625) | 0.05 [0.045, 0.05] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.05193 [0.04677, 0.05588] | 0.725 [0.6887, 0.775] | 0.04685 [0.03962, 0.05866] | 0.0363 | 0.345 [0.3337, 0.37] | 0 | 0.76 [0.7288, 0.805] | 0.03214 [0.03021, 0.03624] | 1952 [1894, 2000] |
| 2048 | fr | 0.007858 [0.006689, 0.009159] | 0.17 [0.17, 0.175] | 0.03106 [0.02842, 0.0341] | 0.0363 | 0.105 [0.1, 0.105] | 0 | 0.205 [0.2, 0.21] | 0.03022 [0.02751, 0.03318] | 1841 [1802, 1864] |
| 512 | abf | 0.06988 [0.04696, 0.08638] | 0.2025 [0.18, 0.7737] | 0.08453 [0.07031, 0.09961] | 0.0728 | 0.0825 [0.07375, 0.09125] | 0 | 0.1775 [0.145, 0.1963] | 0.01014 [0.008838, 0.01155] | 1850 [1804, 1890] |
| 512 | fr | 0.006119 [0.00518, 0.007758] | 0.05 [0.045, 0.05] | 0.06348 [0.05425, 0.07031] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05, 0.055] | 0.00913 [0.008024, 0.01025] | 1776 [1720, 1815] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 5538 [5474, 5704] | 2.704 [2.673, 2.785] | 0.4414 [0.4391, 0.4555] | 0.0002704 [0.0002673, 0.0002785] | 0.00293 [0.002441, 0.00293] | 0.05017 [0.03831, 0.06181] | 0.3593 [0.3312, 0.3848] | 430 [415, 436.2] | 0.01538 [0.01318, 0.01917] | 0.0003134 [0.0003098, 0.0003246] |
| 512 | 0/32 | 40 | 0 | 9474 [9390, 9642] | 18.5 [18.34, 18.83] | 0.2213 [0.2198, 0.2246] | 0.0004626 [0.0004585, 0.0004708] | 0.007812 [0.007812, 0.007812] | 0.02147 [0.01974, 0.02622] | 0.379 [0.3314, 0.4224] | 22.5 [20, 25.25] | 0.03223 [0.02734, 0.04297] | 0.0004914 [0.0004879, 0.0005005] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +42.0 % | 0.08 | -0.1 % | 0.005 | +34.7 % | 32 |
| 2048 | e_Fp | +79.0 % | 0.08 | -3.9 % | 0.015 | +16.7 % | 32 |
| 2048 | e_F_em | +42.0 % | 0.08 | -0.1 % | 0.005 | +34.7 % | 32 |
| 2048 | TV_half | +94.4 % | 0.305 | -0.0 % | 0.005 | +84.9 % | 32 |
| 512 | e_F | +45.8 % | 0.025 | -36.1 % | 0.955 | -30.3 % | 32 |
| 512 | e_Fp | +79.0 % | 0.025 | -5.1 % | 0.965 | -4.7 % | 32 |
| 512 | e_F_em | +45.8 % | 0.025 | -36.1 % | 0.955 | -30.3 % | 32 |
| 512 | TV_half | +92.0 % | 0.93 | +0.3 % | 0.005 | +91.2 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 512 | 0.0192 | [0.01842, 0.02486] | 512: 1.00 | 512 | 0.01636 | [0.01483, 0.01946] | 512: 1.00 | -0.002844 [-0.008626, 0.0004807] | -14.8 % [-35.8 %, +2.5 %] |
| final_e_F | 512 | 0.004807 | [0.004216, 0.005412] | 512: 1.00 | 512 | 0.006265 | [0.005438, 0.007687] | 512: 1.00, 2048: 0.00 | 0.001457 [0.0003916, 0.003109] | +30.3 % [+7.9 %, +69.0 %] |
| tau_bgrid_e_F_mid_u | 512 | 0.89 | [0.84, 0.985] | 512: 0.98, all censored: 0.02 | all censored | > 1 | [> 1, > 1] | 512: 0.02, all censored: 0.98 | inf [inf, inf] | inf [inf, inf] |
| Ibar_Fp | 512 | 0.07127 | [0.06772, 0.08285] | 512: 1.00 | 512 | 0.06054 | [0.05744, 0.06516] | 512: 1.00 | -0.01074 [-0.02287, -0.003883] | -15.1 % [-27.7 %, -5.5 %] |
| final_e_Fp | 512 | 0.03402 | [0.0337, 0.0345] | 512: 1.00 | 512 | 0.03563 | [0.0346, 0.03733] | 512: 1.00, 2048: 0.00 | 0.001613 [0.0003578, 0.003304] | +4.7 % [+1.1 %, +9.7 %] |
| tau_bgrid_TV_half_u | 512 | 0.2025 | [0.19, 0.48] | 512: 1.00, 2048: 0.00, ties: 0.00 | 512 | 0.05 | [0.045, 0.05] | 512: 1.00 | -0.1525 [-0.43, -0.14] | -75.3 % [-89.6 %, -73.7 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.04445, FR 0.0338; -28.8 % [-31.4 %, -23.3 %] (28/32) | ABF 0.1478, FR 0.1167; -23.7 % [-25.8 %, -18.1 %] (28/32) | ABF 0.0178, FR 0.01162; -38.2 % [-40.6 %, -27.9 %] (29/32) | ABF 0.05074, FR 0.04228; -17.9 % [-21.0 %, -12.4 %] (29/32) | ABF 0.1399, FR 0.1061; -26.9 % [-29.8 %, -21.1 %] (28/32) | ABF 0.03916, FR 0.02732; -33.2 % [-38.9 %, -23.3 %] (29/32) |
| 512 | ABF 0.0192, FR 0.01636; -14.3 % [-47.3 %, +5.2 %] (21/32) | ABF 0.07127, FR 0.06054; -12.9 % [-29.3 %, -2.5 %] (22/32) | ABF 0.004805, FR 0.006262; +27.6 % [-5.2 %, +76.5 %] (12/32) | ABF 0.03401, FR 0.03563; +3.8 % [-0.8 %, +7.9 %] (12/32) | ABF 0.05629, FR 0.04517; -19.1 % [-47.7 %, +0.5 %] (21/32) | ABF 0.01077, FR 0.01511; +32.4 % [-8.2 %, +81.0 %] (12/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 290.9 | 0.08879 | 159.2 | 32/32 |
| 2048 | fr | 294.9 | 0.08999 | 159.2 | 32/32 |
| 512 | abf | 283.1 | 0.08638 | 158.8 | 32/32 |
| 512 | fr | 288.9 | 0.08817 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
