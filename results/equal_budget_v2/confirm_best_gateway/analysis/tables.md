# Equal-budget ladder analysis: gateway

Config `/home/zheyuanlai/ABF-Fisher-Rao/configs/equal_budget_v2/confirm_best_gateway.json`; results `/home/zheyuanlai/ABF-Fisher-Rao/results/equal_budget_v2`; generated 2026-10-10T08:49:47Z; eqb_metrics/2.
B = 3,276,800,000 walker-steps per arm per seed; h = 2.5e-05; seeds 64; thresholds e_F [0.002, 0.0055, 0.02], e_F' [0.012, 0.035, 0.1], TV_half 0.1.
Contrasts are paired per seed, G = (FR - ABF)/ABF: median [bootstrap 95 % CI, 10000 resamples] (wins = seeds with FR < ABF / n). tau is the persistent time-to-accuracy on the budget axis u (censored = '> 1').

Reference: path results/equal_budget_v2/references/gateway_reference.npz, sha256 9d551298568cbe9512ae908a5bd9b830cd8b0b87859a2d54d4ec74d6d6eb86c5, n_eval_nodes 151, em_floor_F_rms 7.900546534699823e-05, em_bias_Fp_bin_rms 0.0008073243445745691, floor_definition errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = <F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0, em_bias_definition em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j (the EM bias of F', NOT the e_F' floor), units reduced (energy; force per unit x), scorer scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, em secondary).
Zero-noise floors (scorer fed the exact bin-averaged reference): e_F 4.326e-16, e_Fp 0.03227 (HARD lower bound), e_Fp_stat 0 (HARD lower bound), e_F_em 1.05e-09, e_Fp_em 0.03227 (HARD lower bound), e_Fp_em_stat 0 (HARD lower bound).

## Frozen thresholds vs the zero-noise floors

| tau | eps | floor | hard bound | reachable | statistical budget sqrt(eps^2 - floor^2) | note |
|---|---|---|---|---|---|---|
| e_F_strict | 0.002 | 4.326e-16 | no | yes | -- |  |
| e_F_mid | 0.0055 | 4.326e-16 | no | yes | -- |  |
| e_F_loose | 0.02 | 4.326e-16 | no | yes | -- |  |
| e_Fp_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; tau is censored for every arm and 'both censored' is not a tie |
| e_Fp_mid | 0.035 | 0.03227 | yes | yes | 0.01356 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01356 |
| e_Fp_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |
| e_F_em_strict | 0.002 | 1.05e-09 | no | yes | -- |  |
| e_F_em_mid | 0.0055 | 1.05e-09 | no | yes | -- |  |
| e_F_em_loose | 0.02 | 1.05e-09 | no | yes | -- |  |
| e_Fp_em_strict | 0.012 | 0.03227 | yes | **NO (unreachable by construction)** | 0 | UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; tau is censored for every arm and 'both censored' is not a tie |
| e_Fp_em_mid | 0.035 | 0.03227 | yes | yes | 0.01355 | reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= 0.01355 |
| e_Fp_em_loose | 0.1 | 0.03227 | yes | yes | 0.09465 |  |

## Status

| N | n_steps | T_N | ABF complete | FR complete | missing / running / invalid |
|---|---|---|---|---|---|
| 16 | 204,800,000 | 5120 | 64/64 | 64/64 | -- |

## Absolute values (median [IQR] over seeds)

| N | T_N | ABF Ibar_F | FR Ibar_F | ABF Ibar_F' | FR Ibar_F' | ABF e_F(1) | FR e_F(1) | ABF e_F'(1) | FR e_F'(1) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 5120 | 0.0009753 [0.0008261, 0.001123] | 0.001123 [0.0008352, 0.001539] | 0.03281 [0.03274, 0.0329] | 0.03289 [0.03276, 0.03313] | 0.0004611 [0.0003255, 0.0006005] | 0.0005648 [0.0003073, 0.0009417] | 0.03235 [0.03233, 0.03238] | 0.03238 [0.03234, 0.03242] |

## Paired contrasts FR vs ABF at each N

| N | G Ibar_F | G Ibar_F' | G e_F(1) | G e_F'(1) | G Ibar_TV_half | G TV_half(1) | FR activity |
|---|---|---|---|---|---|---|---|
| 16 | +24.0 % [+1.1 %, +52.9 %] (22/64) | +0.3 % [+0.0 %, +0.6 %] (20/64) | +21.4 % [+2.0 %, +51.9 %] (24/64) | +0.1 % [-0.0 %, +0.1 %] (24/64) | -31.3 % [-34.7 %, -27.8 %] (63/64) | -7.7 % [-25.0 %, +8.2 %] (38/64) | 2.67e+03 deaths / walker |

## Persistent time-to-accuracy tau (budget fraction u; median, censored '> 1')

| N | metric | ABF median u | ABF censored | FR median u | FR censored | FR wins/losses/ties | sign p | G (both finite, n) |
|---|---|---|---|---|---|---|---|---|
| 16 | e_F_strict | 0.12 | 0 | 0.1075 | 0 | 24/39/1 | 0.0769 | +35.4 % [-2.9 %, +66.7 %] (n 64) |
| 16 | e_F_mid | 0.01953 | 0 | 0.025 | 0 | 26/34/4 | 0.366 | +27.6 % [-25.0 %, +92.0 %] (n 64) |
| 16 | e_F_loose | 0.003066 | 0 | 0.002213 | 0 | 39/19/6 | 0.0119 | -15.0 % [-33.3 %, +0.0 %] (n 64) |
| 16 | e_Fp_strict **(unreachable by construction: ties are not equivalence)** | > 1 | 1 | > 1 | 1 | 0/0/64 | 1 | -- [--, --] (n 0) |
| 16 | e_Fp_mid | 0.035 | 0 | 0.0375 | 0 | 29/31/4 | 0.897 | +0.0 % [-27.5 %, +28.6 %] (n 64) |
| 16 | e_Fp_loose | 0.003066 | 0 | 0.001953 | 0 | 48/15/1 | 3.76e-05 | -27.8 % [-45.9 %, -21.9 %] (n 64) |
| 16 | TV_half | 0.035 | 0 | 0.007812 | 0 | 62/0/2 | 4.34e-19 | -75.5 % [-82.8 %, -66.7 %] (n 64) |

### tau per arm (all N; median u [IQR], fraction censored)

| N | method | e_F_strict | e_F_mid | e_F_loose | e_Fp_strict | e_Fp_mid | e_Fp_loose | TV_half |
|---|---|---|---|---|---|---|---|---|
| 16 | abf | 0.12 [0.065, 0.1825] (0) | 0.01953 [0.01, 0.025] (0) | 0.003066 [0.002213, 0.004248] (0) | > 1 [> 1, > 1] (1) | 0.035 [0.03, 0.05] (0) | 0.003066 [0.002213, 0.004248] (0) | 0.035 [0.01953, 0.055] (0) |
| 16 | fr | 0.1075 [0.065, 0.245] (0) | 0.025 [0.01, 0.04125] (0) | 0.002213 [0.00188, 0.003609] (0) | > 1 [> 1, > 1] (1) | 0.0375 [0.03, 0.055] (0) | 0.001953 [0.001597, 0.002605] (0) | 0.007812 [0.005, 0.01] (0) |

## Establishment criterion: null calibration (exactly uniform population, Bin(N, target) per save)

| N | k_min of N | P(fail) per save | P(censored) | P(no failure, u > 1/2) | null median est. u | unreliable |
|---|---|---|---|---|---|---|
| 16 | 3 | 0.0374 | 0.0374 | 0.021 | 0.915 | **yes** (read cum. est. / first arrival) |

## Marginal and population (median [IQR])

| N | method | TV_half(1) | tau TV_half (u) | TV_inst(1) | E_N[TV] floor | establishment u | est. censored | cum. est. u | first arrival u | transitions |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | abf | 0.01487 [0.01007, 0.02061] | 0.035 [0.01953, 0.055] | 0.3889 [0.3889, 0.4583] | 0.4007 | 0.915 [0.8187, 0.98] (UNRELIABLE: fails by chance at this N) | 0.0625 | 0.007812 [0.005, 0.01] | 0.0008354 [0.0005946, 0.001049] | 2266 [2242, 2294] |
| 16 | fr | 0.01389 [0.01191, 0.01616] | 0.007812 [0.005, 0.01] | 0.3889 [0.3333, 0.4444] | 0.4007 | 0.9075 [0.8, 0.97] (UNRELIABLE: fails by chance at this N) | 0.01562 | 0.003066 [0.002605, 0.003609] | 0.0006326 [0.0005224, 0.0008693] | 2312 [2286, 2349] |

## FR genealogy and activity (FR arm, median [IQR])

Event = a REALISED death (= one replacement: death + copy) in both systems; the gateway's capped candidates kd + kc are the last column. 'FR inactive' = FR arms that realised no death (equal to ABF: their G = 0 is not equivalence).

| N | FR inactive seeds | cap | cap ext. | deaths | deaths / walker | opps with event | mean events/N per opp | max events/N per opp | final ESS run | min ESS win | final unique run | max family win | candidates/N per opp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0/64 | 1 | 0 | 4.267e+04 [4.254e+04, 4.277e+04] | 2667 [2659, 2673] | 0.03333 [0.03324, 0.03342] | 0.002083 [0.002077, 0.002089] | 0.0625 [0.0625, 0.0625] | 0.0625 [0.0625, 0.0625] | 0.2353 [0.2105, 0.2443] | 1 [1, 1] | 0.4375 [0.4375, 0.5] | 0.002134 [0.002127, 0.002138] |

## Max transient improvement of the median curves, (ABF - FR)/ABF over the 200 uniform u

| N | metric | max | at u | min | at u | at u = 1 | n seeds |
|---|---|---|---|---|---|---|---|
| 16 | e_F | +13.8 % | 0.145 | -79.3 % | 0.44 | -22.5 % | 64 |
| 16 | e_Fp | +5.6 % | 0.005 | -5.4 % | 0.015 | -0.1 % | 64 |
| 16 | e_F_em | +12.8 % | 0.145 | -90.5 % | 0.44 | -23.2 % | 64 |
| 16 | TV_half | +50.1 % | 0.005 | -7.9 % | 0.69 | +6.6 % | 64 |

## Best allocation (min over N of the seed median; bootstrap re-selects N in every resample)

| metric | ABF best N | ABF best | ABF 95 % CI | ABF N frequency | FR best N (N >= 2) | FR best | FR 95 % CI | FR N frequency | best FR - best ABF [95 % CI] | rel. [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Ibar_F | 16 | 0.0009753 | [0.0008842, 0.001058] | 16: 1.00 | 16 | 0.001123 | [0.0009857, 0.001341] | 16: 1.00 | 0.0001481 [-1.065e-05, 0.0003836] | +15.2 % [-1.0 %, +41.1 %] |
| final_e_F | 16 | 0.0004611 | [0.0003988, 0.0005314] | 16: 1.00 | 16 | 0.0005648 | [0.0004211, 0.0006801] | 16: 1.00 | 0.0001036 [-5.6e-05, 0.0002488] | +22.5 % [-11.5 %, +59.4 %] |
| tau_bgrid_e_F_mid_u | 16 | 0.02 | [0.015, 0.02] | 16: 1.00 | 16 | 0.025 | [0.015, 0.03] | 16: 1.00 | 0.005 [-0.0025, 0.015] | +25.0 % [-12.5 %, +100.0 %] |
| Ibar_Fp | 16 | 0.03281 | [0.03277, 0.03286] | 16: 1.00 | 16 | 0.03289 | [0.03283, 0.03296] | 16: 1.00 | 7.933e-05 [3.47e-06, 0.0001559] | +0.2 % [+0.0 %, +0.5 %] |
| final_e_Fp | 16 | 0.03235 | [0.03234, 0.03237] | 16: 1.00 | 16 | 0.03238 | [0.03235, 0.03239] | 16: 1.00 | 2.604e-05 [-5.036e-06, 4.546e-05] | +0.1 % [-0.0 %, +0.1 %] |
| tau_bgrid_TV_half_u | 16 | 0.035 | [0.0275, 0.04] | 16: 1.00 | 16 | 0.01 | [0.005, 0.01] | 16: 1.00 | -0.025 [-0.035, -0.02] | -71.4 % [-86.7 %, -66.7 %] |

## Secondary (EM-consistent reference; *_Fp_stat = floor-free companion of e_F' (descriptive))

| N | Ibar_F_em | Ibar_Fp_em | final_e_F_em | final_e_Fp_em | Ibar_Fp_stat | final_e_Fp_stat |
|---|---|---|---|---|---|---|
| 16 | ABF 0.000956, FR 0.001132; +17.7 % [+1.1 %, +51.7 %] (24/64) | ABF 0.03281, FR 0.03289; +0.3 % [+0.0 %, +0.6 %] (19/64) | ABF 0.0004541, FR 0.0005594; +18.5 % [-2.2 %, +53.1 %] (25/64) | ABF 0.03235, FR 0.03237; +0.1 % [-0.0 %, +0.1 %] (26/64) | ABF 0.004526, FR 0.004853; +12.0 % [+4.5 %, +17.2 %] (19/64) | ABF 0.002342, FR 0.002678; +10.9 % [-1.5 %, +22.7 %] (24/64) |

## Finite-N floor of TV_inst, E_N[TV] under multinomial(N, uniform 18)

exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial).

| N | E_N[TV] |
|---|---|
| 16 | 0.40070 |

## Cost (median per run)

| N | method | wall s | us / walker-step | peak RSS MB | force evals ok |
|---|---|---|---|---|---|
| 16 | abf | 284.9 | 0.08695 | 151.4 | 64/64 |
| 16 | fr | 289 | 0.08819 | 153.4 | 64/64 |

## Warnings

* tau_e_Fp_strict: UNREACHABLE BY CONSTRUCTION: e_Fp >= 0.03227 > eps for every estimate; tau is censored for every arm and 'both censored' is not a tie
* tau_e_Fp_em_strict: UNREACHABLE BY CONSTRUCTION: e_Fp_em >= 0.03227 > eps for every estimate; tau is censored for every arm and 'both censored' is not a tie
