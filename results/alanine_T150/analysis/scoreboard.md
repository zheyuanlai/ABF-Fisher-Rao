# Low-T alanine scoreboard (results/alanine_T150; reference T = 150.0 K, kT = 1.247 kJ/mol; raw read-out floor 0.216 kJ/mol)

Generated 2026-10-01T02:53:59Z. Reference acceptance: "see acceptance.json"

## Predictor on `abf`: **FAST**

- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median 7 ps (max 7); coverage-only t_cov 6 ps; t_F (error within 2x final): 30 ps; rule: t_est >= 30 ps and t_est > t_F (medians over seeds)
- C7ax first hit median 4.86 ps, censored 0/16; ABF raw error @1/5/20/50/100 ps: 1.341 / 3.714 / 0.699 / 0.327 / 0.240; I_F W1 52.26; KL(p||U) @1/100: 4.25/1.67; KL(p||support) @1/100: 1.93/1.67

## FR vs matched ABF (own window, raw read-out; frozen rules)

| arm | method | rate | dI_F own | dI_F W1 | final | S(e0/2) | S(e0/4) | S(e0/8) | ratio min (t) | ratio final | events/opp | ESS min | KL_U final arm/abf | KL_support final arm/abf | C7ax final arm/abf | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s02_t0 | fr_support | 0.02 |  +1.07 % [ -0.45,  +1.69] 6/16 |  +1.18 % [ -0.49,  +1.91] 6/16 |  +0.26 % [ -0.98,  +1.06] 8/16 | 0.96 | 1.01 | - | 0.993 (3) | 1.003 | 3.6 | 0.90 | 1.674/1.671 | 1.674/1.671 | 0.013/0.013 | NEUTRAL |
| s15_t0 | fr_support | 0.15 |  +1.91 % [ +1.30,  +3.15] 2/16 |  +2.79 % [ +2.27,  +4.45] 1/16 |  +3.63 % [ +2.68,  +4.18] 1/16 | 0.91 | 0.98 | - | 0.907 (3) | 1.036 | 29.5 | 0.46 | 1.681/1.671 | 1.681/1.671 | 0.014/0.013 | NEUTRAL_SIG |
| u02_t0 | fr_uniform | 0.02 |  +1.41 % [ -0.08,  +2.23] 5/16 |  +1.76 % [ -0.02,  +2.68] 4/16 |  +1.19 % [ +0.14,  +1.73] 4/16 | 0.96 | 0.99 | - | 0.995 (3) | 1.012 | 3.5 | 0.90 | 1.668/1.671 | 1.668/1.671 | 0.014/0.013 | NEUTRAL |
| u15_t0 | fr_uniform | 0.15 |  +2.87 % [ +1.42,  +4.87] 2/16 |  +4.00 % [ +2.45,  +6.78] 1/16 |  +3.28 % [ +2.62,  +4.69] 0/16 | 0.88 | 0.98 | - | 0.905 (3) | 1.033 | 29.1 | 0.47 | 1.678/1.671 | 1.678/1.671 | 0.015/0.013 | NEUTRAL_SIG |

## Support target vs torus target (paired, equal rate)

- rate 0.02: support vs torus target, own window -0.03 % [-0.90, +0.72] 8/16; final -0.36 % [-1.82, +0.11]
- rate 0.15: support vs torus target, own window -0.51 % [-1.51, +0.25] 10/16; final -1.09 % [-1.45, +0.46]
