# Alanine 1-D (incomplete CV) study: scoreboard

Generated 2026-09-30T13:11:21Z. Errors: equilibrium-weighted aligned L2 of the 1-D PMF vs the FULL marginal reference (8 kT mask); 'visited' = vs the phi < 0 conditional marginal. D_cond = equal-weight mean TV of p(hidden | xi) over CV bins with >= 200 walkers. Reference C7ax population 3.13 %.

## Arms

| arm | CV | method | estimator | FR start | rate | I_F W1 full | final full | final visited (psi) | full/visited | C7ax first hit (ps) | censored | C7ax @20/100 | D_cond by window (floor) | mass phi>0 by window (target = psi-uniform average of the reference conditional) | KL @1/100 | T @1 | ctx 2-D final | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi_abf | phi | abf | histogram | 0 | 0.02 | 76.296 | 0.3743 | nan | nan | 3.07 | 0/16 | 0.3069/0.1345 | 0.061(0.043) 0.095(0.041) 0.075(0.022) 0.074(0.017) | - | 1.45/0.03 | 295.8 | 0.2099 | 11.91 |
| phi_abf_kernel | phi | abf | kernel | 0 | 0.02 | 68.098 | 0.4174 | nan | nan | 3.12 | 0/16 | 0.3503/0.1616 | 0.058(0.037) 0.091(0.038) 0.069(0.022) 0.070(0.017) | - | 1.46/0.04 | 295.8 | 0.2099 | 11.37 |
| phi_u02_t0 | phi | fr_uniform | histogram | 0 | 0.02 | 75.637 | 0.3967 | nan | nan | 3.06 | 0/16 | 0.2937/0.1340 | 0.061(0.043) 0.096(0.040) 0.075(0.022) 0.074(0.017) | - | 1.44/0.03 | 295.8 | 0.2099 | 11.80 |
| phi_u15_t0 | phi | fr_uniform | histogram | 0 | 0.15 | 93.221 | 0.6257 | nan | nan | 3.12 | 0/16 | 0.2627/0.1511 | 0.055(0.042) 0.106(0.037) 0.086(0.022) 0.082(0.017) | - | 1.39/0.03 | 295.8 | 0.2099 | 11.72 |
| psi_abf | psi | abf | histogram | 0 | 0.02 | 205.166 | 2.0702 | 0.5142 | 4.03 | 4.41 | 0/16 | 0.0366/0.1440 | 0.061(0.034) 0.321(0.021) 0.288(0.015) 0.249(0.012) | 0.000 0.023 0.067 0.129 (target 0.321; equilibrium 0.032) | 0.74/0.04 | 295.8 | 0.2422 | 11.91 |
| psi_abf_kernel | psi | abf | kernel | 0 | 0.02 | 203.693 | 2.0510 | 0.6349 | 3.23 | 4.51 | 0/16 | 0.0383/0.1487 | 0.063(0.034) 0.319(0.021) 0.283(0.015) 0.240(0.012) | 0.000 0.025 0.072 0.134 (target 0.321; equilibrium 0.032) | 0.75/0.05 | 295.8 | 0.2422 | 11.67 |
| psi_u02_t0 | psi | fr_uniform | histogram | 0 | 0.02 | 205.917 | 2.0944 | 0.5266 | 3.98 | 4.54 | 0/16 | 0.0354/0.1130 | 0.061(0.034) 0.321(0.021) 0.291(0.015) 0.257(0.012) | 0.001 0.023 0.061 0.105 (target 0.321; equilibrium 0.032) | 0.74/0.03 | 295.8 | 0.2422 | 11.92 |
| psi_u15_t0 | psi | fr_uniform | histogram | 0 | 0.15 | 207.797 | 2.1375 | 0.5883 | 3.63 | 4.47 | 0/16 | 0.0312/0.0635 | 0.104(0.036) 0.323(0.021) 0.298(0.015) 0.283(0.012) | 0.001 0.020 0.048 0.064 (target 0.321; equilibrium 0.032) | 0.71/0.03 | 295.9 | 0.2422 | 11.92 |

## FR vs matched ABF (own window, full reference)

| arm | CV | rate | dI_F own (full) | dI_F W1 | final (full) | dI_F own (visited) | D_cond final | C7ax censored arm/abf | C7ax final arm/abf | events/opp | ESS min | ratio min (t) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi_u02_t0 | phi | 0.02 |  -1.34 % [ -3.30,  -0.51] 12/16 |  -1.60 % [ -4.33,  -0.70] 12/16 |  +5.70 % [ -0.27,  +7.42] 4/16 | - |  +1.27 % [ -2.16,  +3.62] 6/16 | 0/0 | 0.1340/0.1345 | 2.3 | 0.91 | 0.890 (36) | INCONCLUSIVE |
| phi_u15_t0 | phi | 0.15 | +18.88 % [+12.87, +21.11] 0/16 | +24.66 % [+15.81, +27.42] 0/16 | +67.08 % [+55.25, +78.78] 0/16 | - | +11.63 % [ +9.66, +13.46] 0/16 | 0/0 | 0.1511/0.1345 | 18.6 | 0.47 | 0.994 (9) | HARMFUL |
| psi_u02_t0 | psi | 0.02 |  +0.24 % [ +0.03,  +0.34] 2/16 |  +0.27 % [ +0.04,  +0.37] 2/16 |  +0.78 % [ +0.43,  +0.88] 1/16 |  +4.63 % [ +1.20,  +6.46] 3/16 |  +3.40 % [ +2.40,  +4.05] 1/16 | 0/0 | 0.1130/0.1440 | 1.4 | 0.96 | 0.993 (4) | NEUTRAL_SIG |
| psi_u15_t0 | psi | 0.15 |  +1.31 % [ +1.03,  +1.53] 0/16 |  +1.38 % [ +1.09,  +1.62] 0/16 |  +3.30 % [ +2.79,  +3.51] 0/16 |  +7.07 % [ -1.69, +11.42] 6/16 | +14.50 % [+12.32, +16.05] 0/16 | 0/0 | 0.0635/0.1440 | 8.2 | 0.77 | 0.974 (7) | NEUTRAL_SIG |

## Estimator check: kernel ABF vs histogram ABF (paired)

- phi: I_F W1  -9.71 % [-17.01,  -6.96] 16/16; final  +6.66 % [ -1.24, +14.49] 4/16; D_cond final  -5.41 % [ -7.65,  -1.44] 12/16
- psi: I_F W1  -0.78 % [ -0.93,  -0.53] 16/16; final  -0.96 % [ -1.45,  -0.78] 16/16; D_cond final  -3.67 % [ -4.19,  -1.99] 15/16
