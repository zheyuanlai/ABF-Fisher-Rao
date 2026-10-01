# Low-T alanine scoreboard (results/alanine_friction; reference T = 300.0 K, kT = 2.494 kJ/mol; raw read-out floor 0.559 kJ/mol)

Generated 2026-10-01T13:20:29Z. Reference acceptance: "see acceptance.json"

## Predictor on `g10_abf` (N = 2048, gamma = 10.0 /ps): **FAST**

- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median 7 ps (max 7); coverage-only t_cov 5 ps; t_F (error within 2x final): 16 ps; rule: FAST if t_est < 30 ps; favourable iff t_F >= t_est (amendment A1, docs/ALANINE_NLADDER.md)
- C7ax first hit median 3.54 ps, censored 0/16; ABF raw error @1/5/20/50/100 ps: 2.656 / 7.400 / 0.902 / 0.616 / 0.550; I_F W1 83.17; KL(p||U) @1/100: 2.90/1.67; KL(p||support) @1/100: 1.39/1.67

## Predictor on `g50_abf` (N = 2048, gamma = 50.0 /ps): **FAST**

- t_est (95 % of mask cells at count >= c_min, and C7ax visited): median 8 ps (max 9); coverage-only t_cov 5 ps; t_F (error within 2x final): 0 ps; rule: FAST if t_est < 30 ps; favourable iff t_F >= t_est (amendment A1, docs/ALANINE_NLADDER.md)
- C7ax first hit median 4.05 ps, censored 0/16; ABF raw error @1/5/20/50/100 ps: 3.002 / 9.721 / 6.089 / 3.486 / 1.890; I_F W1 377.24; KL(p||U) @1/100: 3.12/1.68; KL(p||support) @1/100: 1.55/1.68

