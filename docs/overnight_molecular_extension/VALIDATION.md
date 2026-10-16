# Numerical validation — overnight molecular extension

The gates are frozen in `PREREGISTRATION.md`. Units are kJ/mol. D is the debiased RMS difference over 180 bins, with
noise from a 16-group jackknife and the upper bound from 2000 group-bootstrap resamples.

## ERI (600 K)

All gates pass. See `docs/eri_campaign/RESULTS.md` and `docs/eri_campaign/EXECUTION_LOG.md`:
* Gate 0: 16 tests;
* Gate 1: every check PASS, noise 0.007;
* Gate 2: EM 2e-4 admissible (D 0.022 / 0.037);
* Gate 3: frozen;
* Gate 4: 7 engine tests;
* a four-lens adversarial audit, including an independent quadrature F(φ) that agrees with MC to 0.024.

## ITE, 300 K (candidate A′)

### Gate 0 — forces, geometry, CV (`tests/test_overnight_ite.py`, 5 tests, PASS)

* The energy equals an independent brute-force sum to 1e-9 relative.
* Central-difference forces agree to 2e-6 relative at cages, at windows (axial and strongly tilted), across box faces
  and in far-unwrapped configurations.
* Guest–guest CH3–CH3 matches brute force.
* All 12 family-A windows sit at φ = 0 and all cage centres at ±π.
* The local mean force equals dU/dφ along a rigid translation.
* A sideways move to the neighbouring channel costs > 50 kJ/mol at every orientation tried. Single-bead percolation
  confirms 1-D channels.
* The engine kernels (forces, ABF/FR mechanics, pairing, checkpointing, FR selection direction, whole-replica cloning)
  are covered by `tests/test_eri_engine.py` (7 tests).
* The classifier statistics are covered by `tests/test_eri_metrics.py` (5 tests with positive and negative controls).

### Gate 1 — exact MC umbrella/WHAM reference

`results/overnight/ite/reference/ite_itq3/gate1.json`: 40 windows, κ 300, 16 groups × 16 molecules × 200k MC steps.

| check | value | verdict |
|---|---|---|
| min fine-bin count | 186 793 | PASS |
| D(F_density, F_MF) | 0.0099 [0.012] | PASS (≤ 0.03) |
| split-half | 0.0047 [0.011] | PASS |
| first vs second half | 0.000 [0.0095] | PASS |
| closure ⟨Γ⟩ | 0.0002 ± 0.0008 | PASS |
| reference noise | 0.004 | – |
| barrier | **39.70 kJ/mol = 15.92 kBT** | – |
| decomposition | ΔU = 20.61, −TΔS = 19.09 kJ/mol (48 % entropic) | – |

### Gate 2 — timestep ladder vs exact MC (tolerance 0.03, upper 0.045)

| scheme | D_density [upper] | D_MF [upper] | bond-variance error | barrier − MC | verdict |
|---|---|---|---|---|---|
| EM 2e-4 | 0.0357 [0.051] | 0.0301 [0.039] | +19.2 % | +0.13 | **NOT ADMISSIBLE** |
| **EM 1e-4** | 0.0148 [0.030] | 0.0129 [0.021] | +8.8 % | +0.03 | **ADMISSIBLE → production** |
| EM 5e-5 | 0.0069 [0.029] | 0.0034 [0.013] | +4.2 % | −0.01 | admissible |
| LM 2e-4 | 0.0081 [0.027] | 0.0095 [0.018] | +0.03 % | +0.05 | admissible |

The second clause holds: D_MF 0.0129 ≤ ⅓ × the K-1024 screen median e_F(T) of 0.072, i.e. ≤ 0.024.

### Gate 3 — histogram ABF (ABF only)

| step | result | rule |
|---|---|---|
| horizon | median e_F at K 1024 = 0.0625 / 0.0457 / 0.0344 / 0.0234 at 50 / 100 / 200 / 400 t.u. → **T = 50** | ratio ≤ 1.5 |
| ramp × pre-eq | **ramp 1 t.u., pre-eq 5 t.u.**: weighted score 0.914 vs 1.000 / 1.192 / 1.324 | – |
| bins | P0 floor at 180 bins = 0.0046 (≈ 5 % of e_F(T)) → **180 bins** (the switch needs a floor > 10 %) | – |

Bin ladder (3 seeds, T 50; smooth P0-compatible reference). It is reported but does not enter the decision.

| K | bins | median e_F(T) | median Ī_F | P0 floor |
|---|---|---|---|---|
| 1024 | 90 | 0.0615 | 0.280 | 0.0183 |
| 1024 | 180 | 0.0659 | 0.293 | 0.0046 |
| 1024 | 360 | 0.0650 | 0.296 | 0.0011 |
| 64 | 90 | 0.230 | 0.600 | – |
| 64 | 180 | 0.156 | 0.596 | – |
| 64 | 360 | 0.236 | 0.644 | – |

At K 1024 the three widths agree within 7 %, so the ladder is on its plateau. At K 64 the differences are seed
noise.

### Gate 4 — FR mechanism

Engine tests (above). FR is never run in production because no cell is eligible.

## ITE, 150 K (cell A′-150): same model, Gate 0 shared

### Gate 1 (64 windows, κ 250, 16 groups × 16 molecules × 200k MC steps): PASS

| check | value |
|---|---|
| min fine-bin count | 298 429 |
| D(F_density, F_MF) | 0.0103 [0.0115] |
| split-half | 0.000 [0.0034] |
| first vs second half | 0.000 [0.0040] |
| closure | 0.0013 ± 0.0007 |
| noise | 0.0023 |
| barrier | **29.97 kJ/mol = 24.03 kBT** |
| decomposition | ΔU 20.16, −TΔS 9.81 kJ/mol (33 % entropic) |

### Gate 2

| scheme | D_density [upper] | D_MF [upper] | bond-variance error | verdict |
|---|---|---|---|---|
| EM 2e-4 | 0.0161 [0.024] | 0.0168 [0.022] | +19.1 % | admissible, but **fails the second clause** (0.0168 > ⅓ × 0.041) |
| **EM 1e-4** | 0.0026 [0.011] | 0.0107 [0.016] | – | **production** (second clause 0.0107 ≤ 0.0132 on the screen) |

### Gate 3

| step | result |
|---|---|
| horizon (dt 2e-4) | e_F 0.140 / 0.082 / 0.049 / 0.035 at 50 / 100 / 200 / 400 → **T = 200** |
| ramp × pre-eq (dt 2e-4) | **1 t.u. / 5 t.u.** (score 0.933) |
| consistency at dt 1e-4 (Amendment 5) | 0.0390 vs 0.0411, ratio 0.95, **PASS** |
| bins (dt 1e-4) | K 1024 e_F(T) 0.0380 / 0.0390 / 0.0348 for 90 / 180 / 360 bins; floor share of 180 = 10.2 % and 360 better by 10.7 % → **360 bins** (thin margins, rule applied as written) |

### Classifier null control at the 8 t.u. block design

FP 0/300 at every K. Power for a 40 % deficit: 68 / 98 / 100 % at K 64 / 256 / 1024.
