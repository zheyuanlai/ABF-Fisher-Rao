# Inventory: gateway replica ladder and ethane/LTA at 300 K and 150 K (existing data, references, validation)

All checks were read-only and ran on CPU. Scratch scripts are `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/audit/lta_mc_mf.py` and `gw_yvar.py`. Each ran in under 2 s.

## 0. Key facts
- **Raw data exists only on this disk.** `*.npz` is gitignored (`.gitignore`, the `*.npz` line; restated in `results/numerical_validation/README.md:4`). Git tracks only summaries, logs and figures. `main` equals `origin/main` at 7bbbe23, so `OVERNIGHT_REPORT.md:3` ("not pushed") is out of date.
- **The LTA MC Gibbs reference can be rebuilt completely from raw arrays at both temperatures.** I rebuilt F_MF and the fine-bin WHAM F_density and both match the committed `summary.json` profiles exactly (max abs diff 0.0).
- **The gateway_dt files cannot measure Var(y|x) directly.** The only (x, y) samples on disk are final snapshots at dt 4e-4 (section 4).
- **Under a 2 % fibre-variance criterion, every gateway result at ω_in = 32 fails except gateway_dt at 2.5e-5.** That includes the replica ladder, all older campaigns and the movies (all at dt 4e-4), and gateway_dt at 1e-4. The limit is h ≤ 3.83e-5.

## 1. Which implementations are validated, and how

**LTA**
- **Independent float64 numba model** (`src/lta_validation.py`). Its 4 tests in `tests/test_lta_validation.py` pass:
  - L1/L3 (`:23`): forces, energies, φ and the local mean force equal the production `core_lta` to 1e-10 relative, at 300 K and 150 K.
  - L2 (`:41`): the analytic force matches finite differences to < 1e-6.
  - L4 (`:59`): WHAM recovers a known density.
  - Caveat: L4 tests `src/lta_validation.wham` (`:323`). The analysis uses its own `class Wham` (`analyze_lta_validation.py:58`), which is checked only through the MC self-consistency gate.
  - `src/lta_validation.py` was last changed in eb5698f, so every raw file was produced by the same engine, even though the per-file commits differ (eb5698f, 1614abf, 255fd78, 67ea3e3, b8ddcb6).
- **Frozen gates** (`configs/numerical_validation/lta_validation_prereg.json`; verdict logic in `analyze_lta_validation.py:156-217`). Results from `results/numerical_validation/lta/T{300,150}/summary.json`:

  | gate | 300 K | 150 K |
  |---|---|---|
  | G_LTA_MC_self | PASS (D 0.002) | PASS (D 0.003) |
  | G_LTA_dyn (EM 2e-4 vs MC) | ADMISSIBLE: D 0.000 / 0.014, upper 0.021 / 0.029 kJ/mol | ADMISSIBLE: 0.000 / 0.011, upper 0.017 / 0.021 |
  | G_LTA_reference (published reference) | MARGINAL: D 0.034, upper 0.038 | ADMISSIBLE: 0.017, upper 0.019 |

  - MC acceptance (single-bead / translation): 0.779 / 0.505 at 300 K, 0.697 / 0.391 at 150 K.
- **Production LTA engine** (torch `src/lta/core_lta.py`, float64 by default at `:143`, dt 2e-4). The ABF/FR code path was not itself run against MC. Its EM physics is validated through the independent EM at the same dt plus the L1/L3 equality.
  - `tests/test_lta_histogram.py` has 6 tests: legacy engine bit-identical to a fixture, config hash, exact histogram_pmf, the histogram branch with γ = 0, sham replay, and snapshots being bit-inert.

**Gateway**
- **numba port** `src/gateway_numba.py` (`tests/test_gateway_numba.py`):
  - v1: ABF bitwise-equal to the torch `gateway_core` under shared noise.
  - v2: KDE and score equal to round-off.
  - v3: resampling law matches in distribution.
  - v4: leave-one-out term.
  - The doc reports 12 tests passing, 15 with v4 (`GATEWAY_REPLICA_LADDER.md:41-48,176`).
  - Gates (a) and (b) passed (scorer reproduces the accepted values to 8e-16; numba vs GPU agree, `:49-57`).
- **Reference.** The analytic F is exact. The EM y-update bias has a closed form, `<y²|x> = 1/(βω²(1−ω²dt/2))` (`GATEWAY_REPLICA_LADDER.md:66-79`; `analyze_gateway_dt.py:25-34`).
  - Pooled ladder accumulators reproduce the EM floor 0.001383 to 1 % (doc §4).
  - My direct snapshot check at dt 4e-4 agrees within 2–3 % standard error (section 4).
- **dt refinement** (`gateway_dt_prereg.json`). The admissibility rule is "EM floor ≤ ABF final error / 3":
  - Floors are 0.001383 / 0.000322 / 0.000079 against ABF final ≈ 0.0055, so all three dt pass and the verdict is SURVIVES.
  - FR vs ABF on Ibar_F: −29.5 % [−32.5, −26.9] 32/32, −28.4 % 31/32, −29.7 % 30/32.
  - **Caveat (my computation):** this is an F-floor rule tied to the T = 40, N = 2048 error scale. Applied to the replica ladder at dt 4e-4, only N = 2048 passes (floor / ABF final = 0.25). N = 1024 fails at 0.40, N ≤ 512 at 0.71–0.96 (analytic reference), and the EM-reference ratios run 0.45 to 4.1.

## 2. Raw arrays on disk vs summaries only

| path | files | keys and shapes | notes |
|---|---|---|---|
| `results/numerical_validation/lta/T300/{MC,EM2e-4,EM1e-4,EM5e-5,LM2e-4}/g{00-15}_w{00-39}.npz` | 5 × 640 = 3200 | `C_fine`(1800), `C`(180), `Mf`(180), `Mf2`(180); scalars `sum_bond`, `sum_bond2`, `sum_U`, `n_dep`, `n_acc`, `n_prop`, `n_tacc`, `n_tprop`, `cell_crossings`, `max_step_component`; `meta_json` | no `.tmp`, no duplicates; MC: 16 molecules × 200k steps, sampled every 5, κ 300 |
| same under `T150/` | 3200 | same | κ 254 |
| `results/numerical_validation/gateway_dt/dt{0.0004,0.0001,2.5e-05}/s{7100-7131}.npz` | 3 × 32 = 96 | `save_at`(224), `u`(224), `M`, `C` (4 arms × 224 × 180; 45-bin arms use the first 45), `ess`, `die`, `clone` (4 × 224) | no `X_final`/`Y_final`, `wmax`, `P` or `maxbias` (`run_gateway_dt.py:44-45`); all at commit c01ce48 |
| `results/gateway_replica_ladder/production/raw/N{1..2048}_s{7100-7131}.npz` | 12 × 32 = 384 | as above plus `wmax`, `P`(4,224,3), `maxbias`(4), `X_final`, `Y_final` (4, N) | dt 4e-4, rescorable with either reference |
| `.../validation/raw/N2048_s5300-5315.npz` | 16 | 2 arms, 45 bins, 201 saves, `X_final`/`Y_final` (2, 2048) | |
| `.../score_fix/raw/N{4,16,64,256,1024}_s*.npz` | 160 | arms `abf_h180`, `fr_h180`, `fr_emp_h180`, `fr_loo_h180` | |
| `.../per_run_curves{,_emref}.npz` | 2 | derived e_F curves (3, 224) per arm, N and seed | derived, not raw |
| `results/lta_histogram/production_T{300,150}/{abf,fr_uniform,fr_sham}.npz` | 3 per T | per seed (16): `fsum_prod`, `u_counts`, `final_eff_counts`, `birth_hist`/`death_hist` (16, 180); 101 checkpoints every 0.6 t.u. of `mean_force`, `pmf`, `p_hat`, `q_target`, `eff_counts` (101, 16, 180); genealogy and region fractions; `event_counts` (56001, 16) for FR and sham | N 1024, 300k steps, dt 2e-4, 60 t.u.; **no Σf², no positions** |
| `results/uniform_campaign/lta/reference/reference_T{80,150,225,300,350}.npz` | 5 | `F`, `U`, `TS`, `p`, `F_unbiased`, `unbiased_hist` (180), `window_hist` (40, 180), barrier scalars, `protocol` | window histograms are pooled over 256 replicas: 180-bin midpoint WHAM can be redone, but there is **no group split (no jackknife), no fine histogram, no mean-force accumulators** |

Also on disk but not inspected in depth: the kernel-ABF sweep `results/uniform_campaign/lta/production_T*/{abf,fr_uniform}.npz` (17 MB each), `results/lta_movie/` (1.3 GB), and the gateway movies' `movie_data.npz`.

**Summaries only (git-tracked):**
- `gateway_replica_ladder/summary{,_emref}.json`, `scoreboard*.md`, `validation/validation.json`, `score_fix/summary.json`
- `numerical_validation/gateway_dt/summary.json`
- `lta/T*/summary.json`: profiles for every tag and read-out, `mc_se` (density-route standard error), rescoring, pooled limits; plus `analysis.log`
- `lta_histogram/summary.json`, `scoreboard.md`, `comparison_T*.csv`

## 3. Rebuilding the LTA MC reference (computed)

- **Files:** `results/numerical_validation/lta/T{300,150}/MC/gNN_wMM.npz`. 640 per T, exactly one per (group, window) for 16 × 40, one metadata tuple: `('MC', 2e-4, 16, 200000, 5000, 5, κ)`.
- **Deposits:** 640,000 per (group, window); 4.096e8 per T.
- **Pooled bin counts:** 1.89e6–2.68e6 at 300 K, 1.79e6–2.73e6 at 150 K. The smallest per-group bin count is 1.18e5 / 1.11e5.
- **WHAM inputs:** `C_fine` (1800 bins) per (group, window), with `C_fine` summed over 10 equal to `C` checked for every file.
- **Exact reproduction:** fine WHAM and F_MF (`hist_pmf`) match `summary.json` "MC density" and "MC mf" with max abs diff 0.0 at both T.

Mean force γ = ΣMf / ΣC per 180-bin, in kJ/mol/rad (f = −(a/2π)(F0x + F1x), `lta_validation.py:191`). Jackknife is leave-one-group-out over 16 groups.

| quantity | 300 K | 150 K |
|---|---|---|
| RMS / max of abs γ | 12.93 / 28.89 | 8.84 / 20.93 |
| circular mean of γ (jackknife SE) | −3.3e-4 (2.6e-3) | −1.49e-3 (1.9e-3) |
| closure 2π·mean, kJ/mol (SE) | −0.0021 (0.016) | −0.0094 (0.012) |
| **per-bin SE of γ: RMS** (median, min–max) | **0.0143** (0.0120, 0.0038–0.0345) | **0.0101** (0.0085, 0.0018–0.0218) |
| SE RMS by region: window abs z < 1 Å / neck 1–4 Å / cage > 4 Å | 0.0093 / 0.0182 / 0.0084 | 0.0052 / 0.0125 / 0.0075 |
| SE of integrated F_MF, kJ/mol: RMS / max | 0.0043 / 0.0057 | 0.0031 / 0.0042 |
| Var(f \| bin) from Mf2: median (range) | 133 (45–302), sd 11.5 | 55.7 (13–132), sd 7.5 |
| inefficiency (SE_jack / SE_iid)², median | 2.23 | 2.70 |
| barrier F_MF / F_density (validation convention) | 26.407 / 26.403 | 16.547 / 16.550 |
| EM 2e-4: γ SE RMS; RMS(γ_EM − γ_MC) | 0.0298; 0.0446 | 0.0278; 0.0330 |

Both circular means are consistent with zero.

INFERENCE: Mf2 makes Var(f|φ) available for Neyman-type allocation. The umbrella tilt within a bin is small, since the MC self-gate gives D = 0.002–0.003.

**Defect (documentation).** The docstring of `analyze_lta_validation.py:148-153` claims to reproduce `run_lta_reference.barrier_stats`, but the cage mask differs:
- validation: abs z > 4 Å;
- reference: within 0.4 Å of the cage centre (`scripts/run_lta_reference.py:79`).

Under the reference's convention the MC barrier is 26.793 / 16.843, not 26.403 / 16.550. Published minus MC is +0.063 / −0.032 under that convention, against +0.049 / −0.028 under the validation one. So one named quantity carries two values.

## 4. Gateway: can Var(y|x) be measured from existing files?

- **Directly from gateway_dt: no.** The engine deposits only `C` and `M` per bin (`gateway_numba.py:405-406`). The runner discards `X_final`/`Y_final` (`run_gateway_dt.py:44-45`), although `run_ladder_point` returns them (`gateway_numba.py:502`). No y, y² or f² is stored.
- **Indirectly through M/C, only partly.** Since f = U0′ + ωω′y² (`gateway_numba.py:396`), M/C implies `<y²|x>` wherever ω′ ≠ 0. The measured / predicted ratio of the entropic mean force in the inner flank (pooled ABF h180, 32 seeds):

  | dt | x = −0.09 | x = +0.09 | EM prediction |
  |---|---|---|---|
  | 4e-4 | 1.108 | 1.105 | 1.103 |
  | 1e-4 | 1.027 | 1.024 | 1.024 |
  | 2.5e-5 | 1.007 | 1.005 | 1.006 |

  This route has two limits:
  - **The outer flank is contaminated.** At abs x 0.17–0.29 there is an antisymmetric deviation, up to +7.7 % on the left and −7.1 % on the right, that is the same at all three dt. INFERENCE: this is left-to-right establishment flux (fibre lag) in time-averaged accumulators, so the dt part cannot be isolated there at the 2 % level.
  - **It is blind at x ≈ 0.** ω′(0) = 0, yet that is where the inflation is largest (+25.75 % at dt 4e-4).
- **Direct snapshots exist only at dt 4e-4.** Ladder production has 262,080 final walkers in the ABF arms. ⟨βω²y²⟩, which is 1 in the continuum:

  | abs x range | n | measured | EM prediction |
  |---|---|---|---|
  | < 0.02 | 2763 | 1.177 ± 0.030 | 1.253 |
  | 0.02–0.05 | — | 1.273 ± 0.028 | 1.221 |
  | 0.05–0.1 | — | 1.134 ± 0.019 | 1.137 |
  | > 0.5 | — | 1.007 ± 0.003 | 1.000 |

  This is consistent with EM but far too noisy for a 2 % test. A 0.5 % relative standard error needs about 8e4 independent samples per bin near the gate.
- **What a new run needs.** For each candidate dt (for example 4e-4, 1e-4, 5e-5, 3.83e-5, 2.5e-5, 1.25e-5):
  1. Add per-fine-x-bin accumulators of count, Σy², optionally Σy⁴, and Σf² (the LTA `Mf2` analogue) to `gateway_numba.simulate`, and save X/Y snapshots in the runner.
  2. Run in a stationary regime (frozen bias = analytic F′, or long unbiased runs) so the flux confound disappears.
  3. Sample every ~0.01 t.u.; the gate's y relaxation time is 1/ω² ≈ 1e-3.

## 5. Timestep suitability under the 2 % fibre-variance criterion, inflation 1/(1 − ω²h/2)

**Gateway.** The 2 % limit is h ≤ 2(0.02/1.02)/ω²: **3.83e-5 for ω = 32** and 1.53e-4 for ω = 16.

| result set | ω_in | dt | inflation | 2 % criterion |
|---|---|---|---|---|
| replica ladder: production, validation, score_fix (`configs/gateway_replica_ladder/design.json:3`) | 32 | 4e-4 | +25.75 % | fail |
| gateway_dt | 32 | 4e-4 / 1e-4 / 2.5e-5 | +25.75 / +5.40 / +1.30 % | fail / fail / **pass** |
| uniform_campaign/gateway (`gateway_prereg.json:10`) | 32 | 4e-4 | +25.75 % | fail |
| gateway_anchor (production, confirmatory, confirmatory_v2) | 32 | 4e-4 | +25.75 % | fail |
| histogram_abf gateway block (`campaign.json:226`) | 32 | 4e-4 | +25.75 % | fail |
| gateway_histogram, information_campaign gateway, transport gateway preregs | 32 | 4e-4 (stated, or inherited from the anchor prereg; `GatewayConfig.dt` default at `src/gateway_core.py:103`) | +25.75 % | fail |
| gateway movies (`movie_data.npz` config_json) | 32 | 4e-4 | +25.75 % | fail |
| gateway_phase map | 4 / 8 / 16 / 32 | 4e-4 | +0.32 / +1.30 / +5.40 / +25.75 % | pass / pass / fail / fail |

Aside, same EM y-update (`eb_abffr_core.py:689,712`): the entropic bottleneck runs at ω_in 25, dt 1e-3 (`histogram_abf/campaign.json:22-25`; `opes_closure/toys_closure.yaml:50-51`) carry +45.5 %.

**LTA.** Every LTA run uses h = 2e-4:
- the histogram sweep at 80–350 K (`configs/lta_histogram/campaign.json:21`);
- the kernel sweeps (`lta_sweep_prereg.json:18`, v1 `lta_prereg.json:30`);
- all references (protocol dt 0.0002);
- the movies;
- the validation EM2e-4 and LM2e-4 tags.

**Only 300 K and 150 K are validated:** dynamics ADMISSIBLE at both, and the FR gain survives rescoring against MC (−13.66 % and −28.54 % on I_F). The 80, 225 and 350 K results share the dt but have no gate.

If the 2 % rule were applied to LTA's stiffest fibre mode, the bond (λ = 2k_b = 800, limit h ≤ 4.90e-5):

| scheme | predicted | measured 300 K / 150 K |
|---|---|---|
| EM 2e-4 | +8.70 % | +8.65 / +8.57 % |
| EM 5e-5 | +2.04 % | +2.03 / +2.01 % (borderline fail) |
| LM 2e-4 | — | +0.02 % |

INFERENCE: the bond force cancels exactly in f = −(a/2π)(F0x + F1x), so this mode reaches F(φ) only through the LJ term. The validated F shift is ≤ 0.014 kJ/mol RMS, and the barrier shift ≤ 0.05 kJ/mol.

## 6. Disk usage and free space

| directory | size |
|---|---|
| `results/gateway_replica_ladder` | 638 MB (production 369, score_fix 250, validation 4.2, per_run_curves 6 + 6) |
| `results/numerical_validation` | 278 MB (gateway_dt 95 = 31 + 32 + 33; lta 52 = 26 per T, about 5.1 per tag; wca 121; wca_confirmatory 12) |
| `results/lta_histogram` | 123 MB (production_T300 22, production_T150 22, calibration 6.4) |
| `results/uniform_campaign/lta/reference` | 1.4 MB |
| `results/uniform_campaign/lta` | 97 MB |
| `results/lta_movie` | 1.3 GB |

Free space: `/` (`/dev/nvme1n1p4`, which also holds `/tmp`) is 3.5 TB with 3.0 TB used and **332 GB available (91 % full)**.