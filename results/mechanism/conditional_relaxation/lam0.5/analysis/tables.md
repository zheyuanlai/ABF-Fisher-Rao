# Equal-budget ladder analysis: gateway_family -- mechanism cell conditional_relaxation/lam0.5: II: lambda = 0.5 (transverse mobility / 2)

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/mechanism/cells/conditional_relaxation/lam0.5.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/conditional_relaxation`; generated 2026-10-10T14:19:15Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/mechanism/references/conditional_relaxation_lam0.5_reference.npz, sha256 78bdc8932fa97eabcd85674a03b44666680f279de56e128d966b4da4a1b1f566, n_eval_nodes 151, em_floor_F_rms 3.9397219186191186e-05, em_bias_Fp_bin_rms 0.00040224302870181896, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 2048 | 40 | 0.03048 [0.02833, 0.03225] | 0.02121 [0.01997, 0.02251] | 0.1207 [0.1132, 0.1257] | 0.09667 [0.09344, 0.1002] | 0.009686 [0.008916, 0.01068] | 0.004933 [0.004421, 0.005407] | 0.03842 [0.03766, 0.03938] | 0.03425 [0.03379, 0.03456] |
| 512 | 160 | 0.01262 [0.0114, 0.0132] | 0.008912 [0.006825, 0.009775] | 0.05966 [0.05823, 0.06277] | 0.05042 [0.04895, 0.05262] | 0.002675 [0.002095, 0.003193] | 0.002048 [0.001248, 0.002716] | 0.03288 [0.03274, 0.03308] | 0.03275 [0.03255, 0.03303] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 2048 | -30.1 % [-34.4 %, -25.3 %] (30/32) | -20.0 % [-22.1 %, -16.8 %] (31/32) | -48.0 % [-53.8 %, -44.6 %] (32/32) | -10.8 % [-12.1 %, -9.7 %] (32/32) | -69.4 % [-69.9 %, -68.8 %] (32/32) | -92.3 % [-92.8 %, -91.4 %] (32/32) | 2.63 deaths / walker |
| 512 | -34.4 % [-43.3 %, -19.9 %] (30/32) | -16.4 % [-18.8 %, -12.9 %] (30/32) | -28.3 % [-39.5 %, -14.0 %] (25/32) | -0.5 % [-0.9 %, -0.0 %] (21/32) | -72.4 % [-73.3 %, -70.2 %] (32/32) | -84.1 % [-88.2 %, -79.5 %] (32/32) | 18.4 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 2048 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_mid | > 1 | 1 | 0.88 | 0.1562 | 27/0/5 | 1.49e-08 | -- [--, --] (n 0) |
| 2048 | e_F_loose | 0.345 | 0 | 0.1625 | 0 | 32/0/0 | 4.66e-10 | -51.5 % [-57.6 %, -48.3 %] (n 32) |
| 2048 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_mid | > 1 | 1 | 0.8025 | 0.09375 | 29/0/3 | 3.73e-09 | -- [--, --] (n 0) |
| 2048 | e_Fp_stat_loose | 0.1075 | 0 | 0.08 | 0 | 31/1/0 | 1.54e-08 | -27.3 % [-29.8 %, -21.1 %] (n 32) |
| 2048 | TV_half | 0.84 | 0 | 0.17 | 0 | 32/0/0 | 4.66e-10 | -79.9 % [-80.2 %, -79.5 %] (n 32) |
| 2048 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 2048 | e_F_em_mid (secondary) | > 1 | 1 | 0.88 | 0.1562 | 27/0/5 | 1.49e-08 | -- [--, --] (n 0) |
| 2048 | e_F_em_loose (secondary) | 0.345 | 0 | 0.1625 | 0 | 32/0/0 | 4.66e-10 | -51.5 % [-57.6 %, -48.3 %] (n 32) |
| 2048 | e_Fp_mid (secondary) | > 1 | 1 | 0.805 | 0.09375 | 29/0/3 | 3.73e-09 | -- [--, --] (n 0) |
| 2048 | e_Fp_loose (secondary) | 0.1075 | 0 | 0.08 | 0 | 31/1/0 | 1.54e-08 | -27.3 % [-29.8 %, -21.1 %] (n 32) |
| 2048 | e_Fp_em_mid (secondary) | > 1 | 1 | 0.805 | 0.09375 | 29/0/3 | 3.73e-09 | -- [--, --] (n 0) |
| 2048 | e_Fp_em_loose (secondary) | 0.1075 | 0 | 0.08 | 0 | 31/1/0 | 1.54e-08 | -27.3 % [-29.8 %, -21.1 %] (n 32) |
| 512 | e_F_strict | > 1 | 0.7812 | > 1 | 0.5312 | 15/1/16 | 0.000519 | -32.6 % [-52.4 %, -9.4 %] (n 6) |
| 512 | e_F_mid | 0.5125 | 0 | 0.285 | 0.03125 | 29/3/0 | 2.56e-06 | -43.0 % [-50.9 %, -29.5 %] (n 31) |
| 512 | e_F_loose | 0.1 | 0 | 0.055 | 0 | 27/4/1 | 3.4e-05 | -47.8 % [-54.4 %, -33.3 %] (n 32) |
| 512 | e_Fp_stat_strict | > 1 | 0.9375 | > 1 | 0.75 | 8/1/23 | 0.0391 | -23.7 % [-23.7 %, -23.7 %] (n 1) |
| 512 | e_Fp_stat_mid | 0.455 | 0 | 0.2875 | 0.03125 | 27/5/0 | 0.000113 | -39.7 % [-48.8 %, -22.4 %] (n 31) |
| 512 | e_Fp_stat_loose | 0.04 | 0 | 0.025 | 0 | 25/4/3 | 0.000104 | -33.3 % [-37.5 %, -16.7 %] (n 32) |
| 512 | TV_half | 0.2125 | 0 | 0.045 | 0 | 32/0/0 | 4.66e-10 | -77.4 % [-78.8 %, -76.3 %] (n 32) |
| 512 | e_F_em_strict (secondary) | > 1 | 0.7812 | > 1 | 0.5312 | 15/1/16 | 0.000519 | -32.6 % [-53.1 %, -9.4 %] (n 6) |
| 512 | e_F_em_mid (secondary) | 0.5125 | 0 | 0.285 | 0.03125 | 29/3/0 | 2.56e-06 | -43.2 % [-50.9 %, -29.5 %] (n 31) |
| 512 | e_F_em_loose (secondary) | 0.1 | 0 | 0.055 | 0 | 27/4/1 | 3.4e-05 | -47.8 % [-54.4 %, -33.3 %] (n 32) |
| 512 | e_Fp_mid (secondary) | 0.455 | 0 | 0.2875 | 0.03125 | 27/5/0 | 0.000113 | -39.7 % [-46.8 %, -22.4 %] (n 31) |
| 512 | e_Fp_loose (secondary) | 0.04 | 0 | 0.025 | 0 | 25/4/3 | 0.000104 | -33.3 % [-37.5 %, -16.7 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | 0.455 | 0 | 0.2875 | 0.03125 | 27/5/0 | 0.000113 | -39.7 % [-48.8 %, -22.4 %] (n 31) |
| 512 | e_Fp_em_loose (secondary) | 0.04 | 0 | 0.025 | 0 | 25/4/3 | 0.000104 | -33.3 % [-37.5 %, -16.7 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 2048 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.345 [0.3025, 0.3812] (0) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.1075 [0.095, 0.125] (0) | 0.84 [0.8262, 0.8775] (0) |
| 2048 | fr | > 1 [> 1, > 1] (1) | 0.88 [0.7388, 0.985] (0.1562) | 0.1625 [0.1437, 0.175] (0) | > 1 [> 1, > 1] (1) | 0.8025 [0.6975, 0.905] (0.09375) | 0.08 [0.075, 0.08] (0) | 0.17 [0.1688, 0.17] (0) |
| 512 | abf | > 1 [> 1, > 1] (0.7812) | 0.5125 [0.4337, 0.5312] (0) | 0.1 [0.08, 0.115] (0) | > 1 [> 1, > 1] (0.9375) | 0.455 [0.38, 0.48] (0) | 0.04 [0.03375, 0.045] (0) | 0.2125 [0.19, 0.23] (0) |
| 512 | fr | > 1 [0.6663, > 1] (0.5312) | 0.285 [0.205, 0.4213] (0.03125) | 0.055 [0.04375, 0.06625] (0) | > 1 [> 1, > 1] (0.75) | 0.2875 [0.2175, 0.3938] (0.03125) | 0.025 [0.025, 0.03] (0) | 0.045 [0.045, 0.05] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 2048 | 370 | 1.74e-72 | 1.74e-72 | 1.000 | 0.0001 | no |
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | abf | 0.06924 [0.06439, 0.07484] | 0.84 [0.8262, 0.8775] | 0.04262 [0.0392, 0.04739] | 0.0363 | 0.385 [0.3638, 0.3962] | 0 | 0.855 [0.835, 0.88] | 0.03195 [0.02855, 0.03578] | 2052 [2021, 2104] |
| 2048 | fr | 0.005418 [0.004879, 0.006327] | 0.17 [0.1688, 0.17] | 0.02905 [0.02679, 0.03303] | 0.0363 | 0.1 [0.1, 0.105] | 0 | 0.205 [0.205, 0.21] | 0.03107 [0.02798, 0.03481] | 2034 [2012, 2077] |
| 512 | abf | 0.02989 [0.02311, 0.03944] | 0.2125 [0.19, 0.23] | 0.07899 [0.06407, 0.08838] | 0.0728 | 0.095 [0.085, 0.1] | 0 | 0.21 [0.19, 0.2213] | 0.01016 [0.00903, 0.01187] | 2061 [2017, 2092] |
| 512 | fr | 0.004734 [0.004034, 0.005625] | 0.045 [0.045, 0.05] | 0.05599 [0.05056, 0.06559] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05375, 0.055] | 0.009356 [0.007352, 0.01047] | 2037 [2020, 2064] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0/32 | 163 | 0 | 5388 [5352, 5452] | 2.631 [2.613, 2.662] | 0.4333 [0.4296, 0.4361] | 0.0002631 [0.0002613, 0.0002662] | 0.00293 [0.002441, 0.00293] | 0.03915 [0.03203, 0.04266] | 0.3483 [0.3164, 0.3779] | 429 [418, 438.5] | 0.01514 [0.01355, 0.0177] | 0.0003053 [0.0003034, 0.0003077] |
| 512 | 0/32 | 40 | 0 | 9400 [9320, 9508] | 18.36 [18.2, 18.57] | 0.22 [0.2182, 0.222] | 0.000459 [0.0004551, 0.0004642] | 0.007812 [0.005859, 0.007812] | 0.02291 [0.0203, 0.02667] | 0.3858 [0.3635, 0.4093] | 23 [22, 24] | 0.03125 [0.0293, 0.03516] | 0.0004882 [0.000483, 0.0004928] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 2048 | e_F | +58.5 % | 0.07 | -0.1 % | 0.005 | +49.1 % | 32 |
| 2048 | e_Fp | +87.5 % | 0.08 | +0.1 % | 0.005 | +10.9 % | 32 |
| 2048 | e_F_em | +58.5 % | 0.07 | -0.1 % | 0.005 | +49.1 % | 32 |
| 2048 | TV_half | +96.3 % | 0.315 | -0.0 % | 0.005 | +92.2 % | 32 |
| 512 | e_F | +58.4 % | 0.02 | -1.9 % | 0.005 | +23.5 % | 32 |
| 512 | e_Fp | +87.0 % | 0.025 | +0.4 % | 0.96 | +0.4 % | 32 |
| 512 | e_F_em | +58.4 % | 0.02 | -1.9 % | 0.005 | +23.7 % | 32 |
| 512 | TV_half | +93.4 % | 0.105 | +0.3 % | 0.005 | +84.2 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 512 | 0.01262 | [0.01201, 0.01307] | 512: 1.00 | 512 | 0.008912 | [0.00745, 0.009607] | 512: 1.00 | -0.003708 [-0.005174, -0.002543] | -29.4 % [-40.7 %, -21.0 %] |
| final_e_F | 512 | 0.002675 | [0.002339, 0.00295] | 512: 1.00 | 512 | 0.002048 | [0.001582, 0.002525] | 512: 1.00 | -0.0006275 [-0.001069, -9.576e-05] | -23.5 % [-39.3 %, -3.8 %] |
| tau_bgrid_e_F_mid_u | 512 | 0.5125 | [0.465, 0.5225] | 512: 1.00 | 512 | 0.285 | [0.22, 0.395] | 512: 1.00 | -0.2275 [-0.285, -0.1075] | -44.4 % [-56.0 %, -21.3 %] |
| Ibar_Fp | 512 | 0.05966 | [0.05877, 0.0622] | 512: 1.00 | 512 | 0.05042 | [0.04998, 0.05151] | 512: 1.00 | -0.009234 [-0.01182, -0.008058] | -15.5 % [-19.0 %, -13.6 %] |
| final_e_Fp | 512 | 0.03288 | [0.03276, 0.033] | 512: 1.00 | 512 | 0.03275 | [0.03258, 0.03283] | 512: 1.00 | -0.0001277 [-0.0003091, -3.447e-06] | -0.4 % [-0.9 %, -0.0 %] |
| tau_bgrid_TV_half_u | 512 | 0.2125 | [0.2, 0.22] | 512: 1.00 | 512 | 0.045 | [0.045, 0.05] | 512: 1.00 | -0.1675 [-0.175, -0.155] | -78.8 % [-79.5 %, -75.9 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 2048 | ABF 0.03048, FR 0.02121; -30.1 % [-34.4 %, -25.3 %] (30/32) | ABF 0.1207, FR 0.09667; -20.0 % [-22.1 %, -16.8 %] (31/32) | ABF 0.009684, FR 0.00493; -48.0 % [-53.9 %, -44.6 %] (32/32) | ABF 0.03842, FR 0.03425; -10.8 % [-12.2 %, -9.7 %] (32/32) | ABF 0.1093, FR 0.07928; -26.2 % [-30.0 %, -22.5 %] (31/32) | ABF 0.02086, FR 0.01149; -45.4 % [-50.5 %, -41.1 %] (32/32) |
| 512 | ABF 0.01262, FR 0.008907; -34.5 % [-43.3 %, -19.9 %] (30/32) | ABF 0.05966, FR 0.05042; -16.4 % [-18.8 %, -12.9 %] (30/32) | ABF 0.002674, FR 0.002039; -28.5 % [-39.4 %, -14.2 %] (25/32) | ABF 0.03287, FR 0.03275; -0.5 % [-0.9 %, -0.0 %] (21/32) | ABF 0.0393, FR 0.02811; -30.5 % [-39.0 %, -23.9 %] (30/32) | ABF 0.006301, FR 0.005597; -15.7 % [-29.7 %, -1.0 %] (21/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 512 | 0.07281 |
| 2048 | 0.03635 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 2048 | abf | 281.2 | 0.08581 | 159.2 | 32/32 |
| 2048 | fr | 288.2 | 0.08794 | 159.2 | 32/32 |
| 512 | abf | 267 | 0.08149 | 159.2 | 32/32 |
| 512 | fr | 273.5 | 0.08347 | 159.2 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
