### Primary outcome

| cell | system | role | test | R2 stop, t.u.: median [IQR] (range) | S_geo [95 % CI] | 90 % CI | Bonferroni-8 CI | FR faster | verdict |
|---|---|---|---|---|---|---|---|---|---|
| R2C1 | ITE 300 K K1024 T50 | in-sample | superiority | 5.00 [5.00, 5.00] (5.00–5.50) | **1.80 [1.55, 2.04]** | [1.55, 2.00] | [1.52, 2.13] | 16/16 | **SUPPORTED** |
| R2C2 | ITE 150 K K256 T100 | in-sample | superiority | 8.50 [8.38, 9.00] (7.00–10.00) | **1.44 [1.29, 1.70]** | [1.30, 1.65] | [1.28, 1.78] | 16/16 | **SUPPORTED** |
| R2C5 | ERI 600 K K1024 T40 | in-sample; expected-negative | equivalence | 4.00 [4.00, 4.00] (4.00–4.00) | **0.92 [0.86, 1.04]** | [0.87, 1.00] | [0.83, 1.05] | 5/16 | **EQUIVALENT** |
| R2C3 | LTA 150 K K1024 T60 | held out; positive control | superiority | 18.50 [17.88, 19.12] (17.50–20.50) | **2.45 [2.26, 2.61]** | [2.27, 2.60] | [2.23, 2.74] | 16/16 | **SUPPORTED** |
| R2C4 | LTA 300 K K1024 T60 | held out | superiority | 14.50 [14.00, 15.12] (13.50–16.00) | **1.39 [1.34, 1.51]** | [1.36, 1.48] | [1.31, 1.56] | 16/16 | **SUPPORTED** |
| R2H80 | LTA 80 K K1024 T60 | held out; new strong-positive candidate | superiority | 31.25 [30.38, 32.50] (28.00–34.50) | **1.84 [1.64, 2.19]** | [1.64, 2.18] | [1.45, 2.37] | 15/16 | **SUPPORTED** |
| R2H350 | LTA 350 K K1024 T60 | held out; new borderline candidate | superiority | 14.00 [13.50, 14.50] (13.00–15.00) | **1.29 [1.25, 1.45]** | [1.26, 1.38] | [1.20, 1.49] | 16/16 | **SUPPORTED** |
| R2N256 | ERI 600 K K256 T40 | held-out allocation; expected-negative | equivalence | 4.00 [4.00, 4.00] (4.00–4.00) | **0.87 [0.81, 0.98]** | [0.81, 0.96] | [0.75, 1.02] | 4/16 | **EQUIVALENT** |

### Secondary outcomes (median per-seed FR/ABF − 1 [95 % CI], FR better; wins/16)

| cell | post-ramp Ī_F | early Ī_F [t_FR, 0.4 T] | late-window Ī_F [0.8 T, T] | e_F(T/2) | e_F(T) | e_F′(T) | post-stop-only estimate | endpoint flag |
|---|---|---|---|---|---|---|---|---|
| R2C1 | -39 % [-45, -35] (16/16) | -39 % [-42, -35] (16/16) | -40 % [-47, -32] (16/16) | -40 % [-50, -36] (16/16) | -42 % [-49, -27] (16/16) | -24 % [-40, -16] (16/16) | -14 % [-36, +3] (11/16) | gain |
| R2C2 | -25 % [-32, -18] (16/16) | -26 % [-29, -19] (16/16) | -19 % [-36, -9] (13/16) | -29 % [-37, -16] (15/16) | -20 % [-32, -8] (13/16) | -17 % [-26, -0] (12/16) | -8 % [-31, +23] (8/16) | gain |
| R2C5 | -1 % [-5, +3] (10/16) | -5 % [-8, -1] (12/16) | +11 % [-1, +17] (5/16) | +13 % [-6, +22] (5/16) | +10 % [-2, +13] (6/16) | +5 % [+3, +13] (2/16) | +24 % [-6, +49] (5/16) | unresolved |
| R2C3 | -49 % [-54, -47] (16/16) | -39 % [-43, -36] (16/16) | -59 % [-71, -56] (16/16) | -63 % [-69, -56] (16/16) | -59 % [-71, -54] (16/16) | -49 % [-54, -42] (16/16) | -53 % [-68, -39] (16/16) | gain |
| R2C4 | -26 % [-28, -21] (16/16) | -22 % [-26, -20] (16/16) | -29 % [-35, -20] (16/16) | -30 % [-33, -21] (16/16) | -24 % [-39, -16] (16/16) | -15 % [-21, -11] (15/16) | -4 % [-18, +11] (9/16) | gain |
| R2H80 | -62 % [-66, -61] (16/16) | -41 % [-45, -36] (16/16) | -83 % [-85, -81] (16/16) | -75 % [-81, -75] (16/16) | -83 % [-85, -81] (16/16) | -57 % [-62, -55] (16/16) | -78 % [-80, -74] (16/16) | gain |
| R2H350 | -26 % [-29, -22] (16/16) | -24 % [-25, -21] (16/16) | -27 % [-35, -16] (15/16) | -34 % [-39, -27] (16/16) | -24 % [-31, -14] (15/16) | -1 % [-2, -0] (12/16) | +9 % [-2, +27] (6/16) | gain |
| R2N256 | +2 % [-6, +10] (7/16) | -2 % [-4, +3] (10/16) | +16 % [-11, +26] (5/16) | +11 % [+6, +34] (3/16) | +12 % [-6, +27] (6/16) | +5 % [-11, +19] (7/16) | -10 % [-29, +11] (9/16) | unresolved |

### The online schedule actually realised

| cell | never stopped | reasons | t_est median (range) | FR replacements per run | events after stop | min windowed ESS/K | max lineage | window establishment ABF → FR (t.u.) | paired diff. | translocations ABF / FR | wall s ABF / FR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2C1 | 0 | established | 2.50 (2.50–2.75) | 582 | 0 | 0.324 | 0.019 | 4.12 → 2.38 | -1.75 [-1.81, -1.62] | 1601 / 1640 | 776 / 768 |
| R2C2 | 0 | established | 4.25 (3.50–5.00) | 116 | 0 | 0.499 | 0.031 | 7.00 → 4.12 | -2.62 [-3.50, -1.88] | 398 / 402 | 408 / 418 |
| R2C5 | 0 | established | 2.00 (2.00–2.00) | 398 | 0 | 0.671 | 0.006 | 2.05 → 1.90 | -0.20 [-0.20, -0.10] | 3627 / 3610 | 523 / 532 |
| R2C3 | 0 | established | 9.25 (8.75–10.25) | 364 | 0 | 0.603 | 0.008 | 16.75 → 9.10 | -7.50 [-7.90, -7.20] | 317 / 383 | 490 / 501 |
| R2C4 | 0 | established | 7.25 (6.75–8.00) | 198 | 0 | 0.759 | 0.005 | 8.95 → 7.20 | -1.80 [-1.90, -1.55] | 926 / 949 | 495 / 485 |
| R2H80 | 0 | established | 15.62 (14.00–17.25) | 491 | 0 | 0.716 | 0.007 | 36.40 → 15.55 | -20.85 [-24.00, -19.80] | 66 / 138 | 495 / 485 |
| R2H350 | 0 | established | 7.00 (6.50–7.50) | 178 | 0 | 0.782 | 0.005 | 8.20 → 6.95 | -1.10 [-1.40, -1.00] | 1150 / 1166 | 497 / 500 |
| R2N256 | 0 | established | 2.00 (2.00–2.00) | 158 | 0 | 0.546 | 0.023 | 2.05 → 1.90 | -0.20 [-0.20, -0.10] | 902 / 902 | 239 / 232 |

### Per-rung sustained speedups

* **R2C1** (kT ladder): ε 0.200: S 1.85 [1.51, 1.99] reach A16/B16; ε 0.160: S 1.80 [1.53, 2.09] reach A16/B16; ε 0.120: S 1.90 [1.59, 2.18] reach A16/B16; ε 0.100: S 1.85 [1.61, 2.16] reach A13/B16; ε 0.080: S 1.64 [1.38, 1.91] reach A6/B16
* **R2C2** (abf ladder): ε 0.216: S 1.52 [1.33, 1.66] reach A16/B16; ε 0.180: S 1.47 [1.33, 1.66] reach A16/B16; ε 0.144: S 1.46 [1.31, 1.54] reach A16/B16; ε 0.115: S 1.48 [1.19, 1.72] reach A15/B16; ε 0.094: S 1.32 [1.04, 1.68] reach A14/B15
* **R2C5** (kT ladder): ε 0.399: S 0.95 [0.89, 1.00] reach A16/B16; ε 0.319: S 0.95 [0.90, 0.96] reach A16/B16; ε 0.239: S 0.93 [0.84, 1.00] reach A16/B16; ε 0.200: S 0.93 [0.81, 1.02] reach A16/B16; ε 0.160: S 0.92 [0.84, 1.05] reach A16/B16
* **R2C3** (abf ladder): ε 0.399: S 2.20 [2.00, 2.49] reach A16/B16; ε 0.333: S 2.50 [2.29, 2.70] reach A16/B16; ε 0.266: S 2.47 [2.34, 2.68] reach A16/B16; ε 0.213: S 2.55 [2.35, 2.81] reach A16/B16; ε 0.173: S 2.51 [2.36, 2.88] reach A16/B16
* **R2C4** (abf ladder): ε 0.295: S 1.45 [1.38, 1.54] reach A16/B16; ε 0.246: S 1.45 [1.36, 1.58] reach A16/B16; ε 0.197: S 1.40 [1.26, 1.54] reach A16/B16; ε 0.158: S 1.36 [1.27, 1.48] reach A16/B16; ε 0.128: S 1.36 [1.30, 1.45] reach A15/B16
* **R2H80** (abf ladder): ε 0.498: S 1.00 [1.00, 1.00] reach A16/B16; ε 0.415: S 1.00 [1.00, 1.05] reach A16/B16; ε 0.332: S 1.01 [1.00, 1.84] reach A16/B16; ε 0.266: S 4.10 [2.85, 5.70] reach A16/B16; ε 0.216: S 4.35 [3.98, 5.36] reach A15/B16
* **R2H350** (abf ladder): ε 0.163: S 1.37 [1.18, 1.41] reach A16/B16; ε 0.136: S 1.26 [1.18, 1.36] reach A16/B16; ε 0.109: S 1.30 [1.25, 1.49] reach A16/B16; ε 0.087: S 1.27 [1.14, 1.56] reach A14/B16; ε 0.071: S 1.23 [1.19, 1.40] reach A13/B15
* **R2N256** (kT ladder): ε 0.399: S 0.92 [0.81, 1.01] reach A16/B16; ε 0.319: S 0.93 [0.83, 0.96] reach A16/B16; ε 0.239: S 0.85 [0.72, 0.99] reach A16/B16; ε 0.200: S 0.87 [0.78, 0.99] reach A15/B15; ε 0.160: S 0.94 [0.87, 1.00] reach A12/B12

### Integrity

* R2C1: paired bitwise before FR start 16/16; force evaluations per run 512,001,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1400000–1400015.
* R2C2: paired bitwise before FR start 16/16; force evaluations per run 256,000,256 (equal in both arms); arms differ only in ['fr_controller']; seeds 1410000–1410015.
* R2C5: paired bitwise before FR start 16/16; force evaluations per run 204,801,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1440000–1440015.
* R2C3: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1420000–1420015.
* R2C4: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1430000–1430015.
* R2H80: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1450000–1450015.
* R2H350: paired bitwise before FR start 16/16; force evaluations per run 307,201,024 (equal in both arms); arms differ only in ['fr_controller']; seeds 1460000–1460015.
* R2N256: paired bitwise before FR start 16/16; force evaluations per run 51,200,256 (equal in both arms); arms differ only in ['fr_controller']; seeds 1470000–1470015.

### Rule-level questions

* preserves_acceleration: **True**
* in_sample_reproduced: **True**
* avoids_eri_harm: **True**
* eri_harm_declared: **False**
* generalises: **True**
