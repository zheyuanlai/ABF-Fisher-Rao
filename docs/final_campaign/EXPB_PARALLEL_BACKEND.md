# Experiment B: walker-parallel CPU backend — implementation, validation, feasibility

**2026-10-11.** Preregistration `docs/final_campaign/PREREGISTRATION.md` §2.2–§2.4 (frozen `c5a2264f`), Amendment 2
(engineering choices before any formal run, `ef8ebc20`), Amendment 3 (gateway V-B4 record and exploratory continuation,
`ff067f4f`). Design brief `docs/final_campaign/EXPB_BACKEND_DESIGN.md`. Claims typed **[identity]** / **[observed]** /
**[hypothesis]**.

## 1. Implementation

| | gateway (α = 1, λ = 1, h 2.5e-5) | LTA 150 K (h 2e-4) |
|---|---|---|
| validated engine | `src/gateway_ladder_numba.py` (equal-budget engine; numba, serial) | `src/lta_ladder_numba.py` (numba, serial) |
| parallel engine | `src/parallel/gw_kernel.c` (C/OpenMP, `-O3 -fno-fast-math -ffp-contract=off`) driven by `src/parallel/gateway_parallel.py` | `src/parallel/lta_parallel.py` (numba `parallel=True`, OpenMP layer) |
| parallel phases | noise, force/bin, deposit, bias read + move | noise, force/CV/bin (≈ 87 % of the work), bias read + move |
| deposit | `replicated_dissem` (A2.1): per-thread table copies, every thread adds all N deposits in slot order (bitwise the serial order), one C11 dissemination barrier per step | serial in slot order |
| FR, saves | serial, the validated numba functions | serial, the validated numba functions |
| noise modes | **replay** = the validated engine's MT19937 stream (serial fill); **counter** = Philox4x64-10 keyed by SeedSequence([seed, N, 2]), counter (step, slot, word, 0), Box–Muller | replay = validated PCG64 array; counter keyed by SeedSequence([seed, N, 150, 2]) |

Tests: `tests/test_parallel_gateway.py` (36), `tests/test_parallel_lta.py` (37), harness and statistics
`tests/test_final_expB_{harness,stats,hybrid}.py`.

## 2. Correctness gates (frozen §2.3) — V-B1, V-B2, V-B3, V-B5 PASS

Run 08:34–08:47 UTC on 16 cores (766 s), `results/final_campaign/expB/validation/`.

| gate | gateway | LTA 150 K |
|---|---|---|
| **V-B1** replay identity vs the validated engine, threads 1/4/16, both arms | **bitwise** on every result array, N 16 (12 t.u.), 64, 512 (6 t.u.) | **bitwise**, N 2 (ABF), 16, 64, 512 |
| **V-B2** counter-mode thread invariance, threads 1/2/4/8/16 | bitwise, N 16/64/512, both arms | bitwise |
| **V-B3** RNG | Philox = numpy oracle; counter map injective over every production configuration; kernel noise = the Philox module | 10⁸ normals: 9-test Holm family (moments, three lag-1 correlations, Box–Muller partners, KS), 0 rejections (smallest p 0.32); counter map injective |
| **V-B5** ABF/FR pairing before FR's first realised death (counter mode) | PASS, N 16/64/512 | PASS |

Additionally (post hoc, Amendment 3): the gateway parallel engine in replay mode is **bitwise identical to the validated
engine over full-budget ABF N 16 runs** (4/4 seeds, 2.05 × 10⁸ steps each). [identity] The parallel kernel performs the
validated arithmetic exactly; counter mode differs from the validated engine only in its noise source.

## 3. Statistical equivalence (V-B4)

### 3.1 Gateway — frozen verdict **FAIL**; confirmatory gateway Experiment B stopped

Counter mode, 32 fresh seeds 8500–8531, vs the validated equal-budget production 8100–8131
(`results/final_campaign/expB/equiv/gateway/vb4.json`):

| arm | ratio of median Ī_F [95 % CI] | band | frozen status |
|---|---|---|---|
| ABF 16 | 1.30 [1.07, 1.46] | [0.85, 1.18] | **FAIL** (point outside the extension window) |
| ABF 512 | 0.94 [0.87, 1.08] | [0.85, 1.18] | PASS |
| FR 512 | 1.06 [0.91, 1.21] | [0.85, 1.18] | EXTEND |
| G difference at N 512 (counter − validated) | +0.056 [−0.034, +0.180] | ±0.10 | **FAIL** (point outside ±0.05) |

By §3 stopping rule (ii) the confirmatory gateway Experiment B stops. This verdict is not revised.

**Post-hoc diagnostic** (Amendment 3, A3.2; `results/final_campaign/expB/diagnostics/vb4_abf16/`), with the frozen
bootstrap:

| comparison, ABF N 16 | ratio [95 % CI] |
|---|---|
| counter 8500–8531 vs validated engine on the same labels | 1.05 [0.84, 1.35] |
| counter fresh 8740–8771 vs validated confirmation 8200–8263 | 1.00 [0.83, 1.20] |
| all counter (64) vs all validated (128) | 1.11 [0.99, 1.25], MW p 0.09 |
| null control: validated 8500–8531 vs validated 8100–8131 | **1.24 [0.90, 1.54]** |
| null control: validated 8200–8231 vs validated 8100–8131 | 1.15 [0.91, 1.36] |
| null control: validated 8232–8263 vs validated 8100–8131 | 1.08 [0.91, 1.25] |

[observed] No law difference is detected, but none is excluded at the frozen ±18 % either. **The frozen ABF-16
criterion fails for the validated engine against itself in 3/3 null controls**: block medians of ABF-16 Ī_F vary by up
to ×1.24 between blocks of 32 validated seeds. The gate therefore had no power to certify equivalence for this arm. That
is a design defect of the preregistered gate, recorded in Amendment 3. The gateway continues only as a labelled
**exploratory** study with a frozen law-insensitive robustness analysis (A3.3).

### 3.2 LTA 150 K — frozen verdict **PASS** (after the one pre-specified extension)

Counter mode, 32 fresh seeds 15300–15331, vs the validated engine (16 production runs 15000–15015 plus 16 new validated
runs 15340–15355); extension for ABF 2 only (counter 15360–15391, validated 15460–15491)
(`results/final_campaign/expB/equiv/lta150/vb4.json`; initial verdict `vb4_initial.json`):

| arm | initial (32 vs 32) | after extension (64 vs 64) | status |
|---|---|---|---|
| ABF 2 | 1.045 [0.849, 1.194] → EXTEND | **1.058 [0.930, 1.157]** | PASS |
| ABF 512 | 1.004 [0.946, 1.092] | — | PASS |
| FR 512 | 0.995 [0.954, 1.100] | — | PASS |
| G difference at N 512 | +0.007 [−0.026, +0.040] | — | PASS |

The A3.4 null control (validated 15340–15355 vs 15000–15015, 16 vs 16; descriptive) gives ABF 2 1.09 [0.85, 1.36],
ABF 512 0.94 [0.88, 1.03], FR 512 1.01 [0.83, 1.08]. As in the gateway, the small-N arm's block-to-block spread is
comparable to the band at the initial n, and the extension resolved it.

## 4. Feasibility benchmark (frozen §2.4)

Protocol as frozen: production configurations truncated to 5 % of B (gateway, 1.6384 × 10⁸ walker-steps) or 10 %
(LTA, 3.072 × 10⁷); every candidate (ABF N 1…512, FR N 64…512; serial engine and parallel t ∈ {1, 2, 4, 8, 16},
t ≤ N) × 3 repeats in rotated order; each timed run on CCDs of its own (t ≤ 8: one CCD; t = 16: two CCDs of one node),
with 1 Hz telemetry and the A2.4 contention rule. The descriptive t = 32 points were not measurable (A4.2).

### 4.1 Gateway (exploratory, Amendment 3) — F1 and F1b **PASS**

222/222 runs, 5 contended; the verdict is the same on uncontended runs only. Median walker-steps per second
(`results/final_campaign/expB/bench/gateway/summary.json`):

| configuration | serial engine | parallel t 1 | t 2 | t 4 | t 8 | t 16 |
|---|---|---|---|---|---|---|
| ABF 512 | 1.32e7 | 2.01e7 | 3.74e7 | 6.77e7 | **1.11e8** | 1.06e8 |
| FR 512 | 1.33e7 | 2.00e7 | — | — | 1.06e8 | **1.09e8** |
| ABF 16 | 1.32e7 | 1.98e7 | **2.92e7** | 2.41e7 | — | — |

* **F1** (best parallel ≥ 6.0 × serial at N 512): ABF 8.37× (t 8), FR 8.25× (t 16). **F1b** (FR 512 best ≥ 4.0 × its own
  t 1): 5.48×. [observed]
* Counter-mode noise alone (parallel t 1 vs the serial engine) is 1.24–1.52× faster: the validated engine's
  MT19937 + polar-method normals are a substantial part of its per-step cost.
* Scaling saturates at t 8–16 (≈ 1.1 × 10⁸ walker-steps/s at N 512, efficiency ≈ 0.5 at t 16): the per-step work
  (≈ 40 µs serial at N 512) is split into ≈ 5 µs slices separated by a barrier, and the serial FR and save phases
  remain. At small N more threads slow the run (ABF 16: best at t 2).
* **Competitor completion** (A2.3, `tau_bgrid_e_F_mid_u`): projected median wall time to the mid threshold W_proj =
  Q50(τ_mid,u) × B × c\*. Best ABF = ABF 16 (1.68 s, a core arm); best FR = **FR 64 (1.21 s), added as an arm**
  (FR 512: 2.55 s; ABF 512: 8.05 s). The selection is the same under `tau_e_F_mid_u`.
* **Implementation per arm** (fastest; ties within 3 % to fewer cores): ABF 16 parallel t 2; ABF 64 t 8; ABF 512 t 8;
  FR 512 t 8 (t 16 is within 3 %); FR 64 t 8.

### 4.2 LTA 150 K

*(pending: benchmark in progress)*
