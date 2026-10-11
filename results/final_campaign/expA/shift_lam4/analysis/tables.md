# Equal-budget ladder analysis: gateway_family -- mechanism cell expA/shift_lam4: A: shifted fibre, kappa = 1, transverse mobility lambda = 4

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/final_campaign/cells/expA/shift_lam4.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/final_campaign/expA`; generated 2026-10-11T09:25:19Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 32; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1, e_Fp_stat [0.004, 0.0136, 0.0946].
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/final_campaign/references/expA_shift_lam4_reference.npz, sha256 64f8acb906f24e25152c47950e2a238ba3b348d3661a4815bafe054e7f51f85c, n_eval_nodes 151, em_floor_F_rms 1.0496374093956405e-09, em_bias_Fp_bin_rms 0.0, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent), em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor), secondary_definition per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family 4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) (1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly), secondary_equals_equal_budget_em None.
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
| 512 | 160 | 0.04342 [0.04138, 0.04525] | 0.04762 [0.04401, 0.05223] | 0.1021 [0.09944, 0.1074] | 0.1075 [0.1006, 0.112] | 0.01226 [0.01139, 0.01336] | 0.02934 [0.02792, 0.03291] | 0.04072 [0.03942, 0.04165] | 0.06573 [0.06206, 0.07076] |
| 128 | 640 | 0.01404 [0.01237, 0.01585] | 0.0295 [0.0264, 0.03338] | 0.05059 [0.04871, 0.05351] | 0.06865 [0.06428, 0.0744] | 0.002318 [0.001667, 0.003406] | 0.01918 [0.01665, 0.02193] | 0.03283 [0.03264, 0.03309] | 0.04936 [0.04528, 0.05305] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 512 | +10.1 % [+2.8 %, +20.6 %] (7/32) | +2.2 % [-1.2 %, +10.3 %] (12/32) | +145.6 % [+129.7 %, +154.9 %] (0/32) | +63.3 % [+57.3 %, +69.2 %] (0/32) | -84.3 % [-85.0 %, -83.9 %] (32/32) | -94.7 % [-95.1 %, -94.3 %] (32/32) | 18.7 deaths / walker |
| 128 | +125.1 % [+102.3 %, +135.4 %] (0/32) | +37.9 % [+29.4 %, +43.9 %] (1/32) | +810.7 % [+510.2 %, +845.8 %] (0/32) | +50.8 % [+42.0 %, +56.3 %] (0/32) | -85.5 % [-86.5 %, -84.1 %] (32/32) | -69.0 % [-77.8 %, -54.7 %] (30/32) | 149 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 512 | e_F_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_loose | 0.685 | 0 | > 1 | 0.9688 | 0/32/0 | 4.66e-10 | +53.3 % [+53.3 %, +53.3 %] (n 1) |
| 512 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_mid | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_stat_loose | 0.2775 | 0 | 0.2625 | 0 | 18/12/2 | 0.362 | -3.7 % [-20.0 %, +11.4 %] (n 32) |
| 512 | TV_half | > 1 | 1 | 0.045 | 0 | 32/0/0 | 4.66e-10 | -- [--, --] (n 0) |
| 512 | e_F_em_strict (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_F_em_loose (secondary) | 0.685 | 0 | > 1 | 0.9688 | 0/32/0 | 4.66e-10 | +53.3 % [+53.3 %, +53.3 %] (n 1) |
| 512 | e_Fp_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_loose (secondary) | 0.2775 | 0 | 0.2625 | 0 | 19/12/1 | 0.281 | -3.7 % [-19.3 %, +11.4 %] (n 32) |
| 512 | e_Fp_em_mid (secondary) | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 512 | e_Fp_em_loose (secondary) | 0.2775 | 0 | 0.2625 | 0 | 19/12/1 | 0.281 | -3.7 % [-19.3 %, +11.4 %] (n 32) |
| 128 | e_F_strict | > 1 | 0.6562 | > 1 | 1 | 0/11/21 | 0.000977 | -- [--, --] (n 0) |
| 128 | e_F_mid | 0.5025 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_F_loose | 0.17 | 0 | 0.8925 | 0.4375 | 0/32/0 | 4.66e-10 | +279.2 % [+176.2 %, +401.5 %] (n 18) |
| 128 | e_Fp_stat_strict | > 1 | 1 | > 1 | 1 | 0/0/32 | 1 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_mid | 0.45 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_stat_loose | 0.065 | 0 | 0.0775 | 0 | 11/19/2 | 0.2 | +21.1 % [-3.1 %, +43.6 %] (n 32) |
| 128 | TV_half | 0.4375 | 0 | 0.015 | 0 | 32/0/0 | 4.66e-10 | -96.6 % [-96.9 %, -96.1 %] (n 32) |
| 128 | e_F_em_strict (secondary) | > 1 | 0.6562 | > 1 | 1 | 0/11/21 | 0.000977 | -- [--, --] (n 0) |
| 128 | e_F_em_mid (secondary) | 0.5025 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_F_em_loose (secondary) | 0.17 | 0 | 0.8925 | 0.4375 | 0/32/0 | 4.66e-10 | +279.2 % [+176.2 %, +401.5 %] (n 18) |
| 128 | e_Fp_mid (secondary) | 0.455 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_loose (secondary) | 0.065 | 0 | 0.0775 | 0 | 11/19/2 | 0.2 | +21.1 % [-3.1 %, +44.0 %] (n 32) |
| 128 | e_Fp_em_mid (secondary) | 0.455 | 0 | > 1 | 1 | 0/32/0 | 4.66e-10 | -- [--, --] (n 0) |
| 128 | e_Fp_em_loose (secondary) | 0.065 | 0 | 0.0775 | 0 | 11/19/2 | 0.2 | +21.1 % [-3.1 %, +44.0 %] (n 32) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_stat_strict | e_Fp_stat_mid | e_Fp_stat_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 512 | abf | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.685 [0.6538, 0.7262] (0) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.2775 [0.25, 0.295] (0) | > 1 [> 1, > 1] (1) |
| 512 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (0.9688) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.2625 [0.2025, 0.3225] (0) | 0.045 [0.045, 0.05] (0) |
| 128 | abf | > 1 [0.9575, > 1] (0.6562) | 0.5025 [0.405, 0.5962] (0) | 0.17 [0.1525, 0.1913] (0) | > 1 [> 1, > 1] (1) | 0.45 [0.375, 0.5325] (0) | 0.065 [0.06062, 0.085] (0) | 0.4375 [0.3588, 0.4962] (0) |
| 128 | fr | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.8925 [0.6175, > 1] (0.4375) | > 1 [> 1, > 1] (1) | > 1 [> 1, > 1] (1) | 0.0775 [0.07, 0.1113] (0) | 0.015 [0.015, 0.015] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 512 | 93 | 1.53e-19 | 1.53e-19 | 1.000 | 0.0001 | no |
| 128 | 24 | 5.53e-06 | 5.53e-06 | 0.999 | 0.0001 | no |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | abf | 0.1746 [0.1609, 0.1839] | > 1 [> 1, > 1] | 0.1363 [0.1304, 0.1536] | 0.0728 | 0.075 [0.07, 0.08] | 0 | 0.155 [0.145, 0.165] | 0.009825 [0.008944, 0.01133] | 1085 [1072, 1104] |
| 512 | fr | 0.009437 [0.008438, 0.009863] | 0.045 [0.045, 0.05] | 0.05903 [0.05187, 0.06624] | 0.0728 | 0.03 [0.025, 0.03] | 0 | 0.055 [0.05, 0.055] | 0.009839 [0.008931, 0.01058] | 1236 [1186, 1286] |
| 128 | abf | 0.02432 [0.01507, 0.03981] | 0.4375 [0.3588, 0.4962] | 0.1424 [0.1337, 0.1721] | 0.1447 | 0.02 [0.01891, 0.025] | 0 | 0.04 [0.035, 0.045] | 0.003272 [0.002864, 0.003927] | 1048 [1031, 1068] |
| 128 | fr | 0.007701 [0.006543, 0.008473] | 0.015 [0.015, 0.015] | 0.1254 [0.1111, 0.1369] | 0.1447 | 0.01 [0.00625, 0.01] | 0 | 0.015 [0.015, 0.015] | 0.002754 [0.002352, 0.003836] | 1112 [1082, 1150] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0/32 | 40 | 0 | 9566 [9428, 9654] | 18.68 [18.42, 18.85] | 0.2229 [0.2206, 0.2246] | 0.0004671 [0.0004604, 0.0004714] | 0.007812 [0.005859, 0.007812] | 0.02056 [0.01796, 0.02498] | 0.3779 [0.3304, 0.4144] | 22 [20, 23] | 0.03613 [0.03076, 0.04688] | 0.000497 [0.0004896, 0.0005005] |
| 128 | 0/32 | 10 | 0 | 1.902e+04 [1.894e+04, 1.908e+04] | 148.6 [147.9, 149.1] | 0.1152 [0.1144, 0.1156] | 0.000929 [0.0009246, 0.0009318] | 0.02344 [0.02344, 0.02344] | 0.007812 [0.007812, 0.007812] | 0.4224 [0.3951, 0.4361] | 1 [1, 1] | 0.07031 [0.0625, 0.07812] | 0.0009625 [0.0009582, 0.000966] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 512 | e_F | +23.1 % | 0.07 | -139.2 % | 1 | -139.2 % | 32 |
| 512 | e_Fp | +63.2 % | 0.025 | -61.4 % | 0.995 | -61.4 % | 32 |
| 512 | e_F_em | +23.1 % | 0.07 | -139.2 % | 1 | -139.2 % | 32 |
| 512 | TV_half | +94.9 % | 0.865 | +0.4 % | 0.005 | +94.6 % | 32 |
| 128 | e_F | +21.8 % | 0.01 | -744.1 % | 0.975 | -727.5 % | 32 |
| 128 | e_Fp | +21.1 % | 0.01 | -73.7 % | 0.255 | -50.3 % | 32 |
| 128 | e_F_em | +21.8 % | 0.01 | -744.1 % | 0.975 | -727.5 % | 32 |
| 128 | TV_half | +92.8 % | 0.205 | +16.8 % | 0.005 | +68.3 % | 32 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 128 | 0.01404 | [0.01277, 0.01499] | 128: 1.00 | 128 | 0.0295 | [0.02874, 0.03159] | 128: 1.00 | 0.01546 [0.01396, 0.01791] | +110.1 % [+94.2 %, +134.6 %] |
| final_e_F | 128 | 0.002318 | [0.001916, 0.002875] | 128: 1.00 | 128 | 0.01918 | [0.01756, 0.02111] | 128: 1.00 | 0.01687 [0.01504, 0.01886] | +727.5 % [+534.5 %, +926.2 %] |
| tau_bgrid_e_F_mid_u | 128 | 0.5025 | [0.445, 0.5725] | 128: 1.00 | all censored | > 1 | [> 1, > 1] | all censored: 1.00 | inf [inf, inf] | inf [inf, inf] |
| Ibar_Fp | 128 | 0.05059 | [0.05004, 0.05307] | 128: 1.00 | 128 | 0.06865 | [0.06744, 0.07335] | 128: 1.00 | 0.01806 [0.01505, 0.02284] | +35.7 % [+28.6 %, +45.4 %] |
| final_e_Fp | 128 | 0.03283 | [0.03273, 0.03302] | 128: 1.00 | 128 | 0.04936 | [0.04698, 0.05147] | 128: 1.00 | 0.01653 [0.01413, 0.01871] | +50.3 % [+42.9 %, +57.0 %] |
| tau_bgrid_TV_half_u | 128 | 0.4375 | [0.3949, 0.47] | 128: 1.00 | 128 | 0.015 | [0.015, 0.015] | 128: 1.00 | -0.4225 [-0.455, -0.3799] | -96.6 % [-96.8 %, -96.2 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 512 | ABF 0.04342, FR 0.04762; +10.1 % [+2.8 %, +20.6 %] (7/32) | ABF 0.1021, FR 0.1075; +2.2 % [-1.2 %, +10.3 %] (12/32) | ABF 0.01226, FR 0.02934; +145.6 % [+129.7 %, +154.9 %] (0/32) | ABF 0.04072, FR 0.06573; +63.3 % [+57.3 %, +69.2 %] (0/32) | ABF 0.09305, FR 0.1013; +4.7 % [+1.2 %, +14.6 %] (10/32) | ABF 0.02484, FR 0.05727; +138.8 % [+126.1 %, +150.0 %] (0/32) |
| 128 | ABF 0.01404, FR 0.0295; +125.1 % [+102.3 %, +135.4 %] (0/32) | ABF 0.05059, FR 0.06865; +37.9 % [+29.4 %, +43.9 %] (1/32) | ABF 0.002318, FR 0.01918; +810.7 % [+510.2 %, +845.8 %] (0/32) | ABF 0.03283, FR 0.04936; +50.8 % [+42.0 %, +56.3 %] (0/32) | ABF 0.0308, FR 0.05927; +96.8 % [+71.6 %, +108.0 %] (0/32) | ABF 0.006053, FR 0.03735; +489.9 % [+364.9 %, +610.1 %] (0/32) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 128 | 0.14473 |
| 512 | 0.07281 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 512 | abf | 279.2 | 0.08519 | 155.8 | 32/32 |
| 512 | fr | 283.9 | 0.08663 | 157.8 | 32/32 |
| 128 | abf | 281.4 | 0.08589 | 153.8 | 32/32 |
| 128 | fr | 286 | 0.08727 | 157.8 | 32/32 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp itself is reported)
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; per plan section 6 no tau is computed at this threshold (e_Fp_em itself is reported)
