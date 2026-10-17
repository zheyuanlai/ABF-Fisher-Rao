# Experiment log — adaptive-FR campaign

Times are UTC (`date -u` or git).

## 2026-10-17

* **06:19 — inspection.**
  * main = origin/main = 0f225325. No job of ours running.
  * GPUs 0–3 belong to other users (2/3 at 98 %), so the campaign is CPU only.
  * Load about 10 on 256 cores. 165 GB disk free.
* **06:20 — worktree.** Created `/home/zheyuanlai/ABF-Fisher-Rao-adaptive` on the new branch
  `research/adaptive-fr-equal-budget`, and linked the shared inputs (`scripts/convergence_fr/link_data.sh`).
* **06:20 — EB baseline verified.** 16/16 complete.
  * Configs equal `jobs_EB.json`.
  * Engine sha256 equals the current import (`a165f0a9…`).
* **06:21 — Gate F pilot** (seed 1140000, `pilot_pairing.py`).
  * The FR configuration is bitwise equal to the historical file before FR start.
  * An ABF re-run with `fr_rate = 1.0` is bitwise equal through step 15 000.
  * 14 engine tests pass, including the new inertness test.
* **06:22:39 — PREREGISTRATION §A frozen** (bc78e255); block A launched (16 runs).
* **06:25 — A analysis code committed** (2aa9c026) before any A result. Re-analysing E1 with the extended script
  reproduces its JSON exactly.
* **06:26–06:32 — Exp B development runs:** continuous FR, ITE only, designated exploratory seeds.
  * ITE 300 K K1024 rate 2 (1200000/1), 150 K K256 rate 0.5 (1200100/1), and 300 K K256 rate 1 (1200200/1).
  * Online observables recorded every 0.25 t.u.
* **06:35:41 — A done** (3.415 core-h).
* **06:35 — blind genealogy check of the FR arm:** PASS (ESSw/K ≥ 0.535; lineage ≤ 0.029; 26 replacements per t.u.;
  0 events after the stop).
* **06:35:54 — A frozen analysis** (`analysis/A.json`). All asserts pass, with the ABF arm's `fr_rate` inert.
  * **PRIMARY S_geo 1.281 [0.904, 1.426], FR faster 11/16 → UNRESOLVED.**
  * H1 −25.5 % [−29.9, −16.3], 16/16 → SUPPORTED.
  * H2 e_F(T) −0.9 % [−14.1, +14.0] → inconclusive.
  * H4 +8.3 % [−9.1, +21.4] → inconclusive.
  * Post-ramp Ī_F −15.3 % [−22.4, −4.1].
* **06:36 — A.5 equal budget** (`analysis/A_equal_budget4.json`).
  * The best allocation is K256 for both algorithms.
  * Best FR / best ABF median geometric-mean τ_B = 0.913 [0.646, 1.086] → **Question B UNRESOLVED**; all-in 0.925
    [0.691, 1.075].
* **06:38–07:08 — Exp B development.**
  * Controller R1 implemented, then integrated in both engines with 33 tests. The default paths are unchanged: old
    suites pass, EB and X1 files re-run bitwise, and the LTA suite gives 39 passed.
  * Independent review: no blocker; design fixes applied before the freeze.
  * Weak-rate dev runs, ITE rate 0.2 (seeds 1200300/400/500).
  * Development replay: stop at 5.5–6.5 t.u. at 300 K and 10–10.5 at 150 K.
* **07:10:05 — PREREGISTRATION B (R1 frozen) + C** committed as c3ef3128.
* **07:10:16 — wave 1 launched:** C3, C4 (LTA engine, `launch_lta.sh`) and C5 (ERI).
* **07:12 — D1 bias–variance on existing blocks** (`mechanism/bias_variance.json`).
  * Continuous FR's late-only deposits are biased in every cell: |bias| 0.035–0.13 vs ABF 0.01–0.03.
  * Stopped FR's post-stop deposits are not (0.009–0.015).
* **07:15 — C3 and C4 done**; wave 2 (C1, C2) launched at 07:15.
* **07:15 — C3 LTA 150 K: SUPPORTED**, S_geo 2.52 [2.33, 2.60]; 99 % [2.21, 2.67]; 16/16.
  * R1 stops at a median 21.5 t.u. (20–24.5).
  * e_F(T) −58.6 %.
* **07:15 — C4 LTA 300 K: SUPPORTED**, S_geo 1.46 [1.37, 1.60]; 16/16.
  * R1 stops at a median 16 t.u. (14–18.5).
  * e_F(T) −22.6 %.
* **07:20 — C5 ERI 600 K: CONTRADICTED**, S_geo 0.61 [0.53, 0.71]; 99 % [0.49, 0.73]; 0/16.
  * R1 stops late, at a median 15.25 t.u. (5.5–25.5).
  * e_F(T) +55 %.
  * Controller logs: the residual directed flux after the t ≈ 2 peak stays at 30–60 % of the peak, above the 25 %
    threshold, for 5–10 t.u.
* **07:30 — D2 clone-age runs** (1.501 core-h): 32/32 bitwise identical to the historical FR arms.
  * Young deposits (walker cloned < 1 t.u. ago) are worse.
  * Dropping them leaves most of FR's late bias: ERI late-half F error 0.132 (all FR) → 0.110 (old only) vs ABF
    0.038; ITE 0.050 → 0.043 vs 0.024.
  * The bias is population-level.
* **07:32 — PREREGISTRATION E** committed (fd3b10ed). The K64 calibration rung 1.0 was launched a few seconds before
  that commit; the calibration is blind and safety-only.
* **07:32 — EA_R1 launched** (16 FR runs, R1 at K256 × T200, paired with EB).
* **07:32–07:35 — K64 blind calibration, top-down.**

  | rung | outcome |
  |---|---|
  | 1.0 | unsafe (lineage 0.078) |
  | 0.5 | unsafe (lineage 0.0625) |
  | 0.2 | **SAFE** (ESSw/K 0.82, lineage 0.047; 2 replacements per t.u.) |

  Selected rate 0.2.
* **07:35 — EB64 launched** (32 runs, K64 × T800, rate 0.2, R1).
* **07:44:45 — EA_R1 done** (3.321 core-h).
* **≈ 07:45 — amendment E-A1** (integrity handling only). The frozen `analyze_online.py` refused EA_R1 because the new
  configs carry `diag_age_steps` = 0 while the historical EB files lack the key. E-A1 allows that key to differ only
  when it is 0 or absent in every run, and asserts this.
  * It was written after EA_R1 had finished and first committed at 07:56:23 (d0929f7a).
  * It changes no statistic.
* **07:45 — EA_R1 analysed. SUPPORTED**: S_geo 1.31 [1.07, 1.47], 99 % [1.02, 1.69]; 13/16.
  * R1 stops at a median 6.5 t.u. (5.5–9).
  * Post-ramp Ī_F −17.4 %; e_F(T) −8.8 % [−21.1, +5.5].
* **07:47:47 — EB64 done** (6.566 core-h). **UNRESOLVED**: S_geo 1.11 [0.92, 1.33]; 9/16.
  * R1 stops at a median 7.5 t.u.
  * Window establishment 4.0 → 4.0 t.u. This is not resolvable: the save grid is 2 t.u.
  * Every secondary outcome unresolved.
* **07:48 — E.3 frontier** (`equal_budget_frontier.py`, six configurations).
  * Best ABF = K64 and best FR = K64. Best FR / best ABF = 1.02 [0.55, 1.46] → **UNRESOLVED** (production);
    all-in is similar.
* **07:49–07:53 — E.4 wall-clock benchmark, run 1** (K ≥ 64; about 0.5 core-h including warm-up; host load 16–25).
  * K64 is the fastest wall-clock configuration for both algorithms at every thread count.
* **07:56 — amendment E-A2** (K16 frontier cell EB16) committed as d0929f7a, before any K16 run.
* **07:56 — K16 blind calibration** (3 rungs × 3 seeds, run concurrently; 0.042 core-h).
  * Every rung was SAFE; at rates 1.0 and 0.5, lineage sits exactly at the 3/K = 0.1875 limit.
  * Largest safe rung selected: **rate 1.0**.
* **07:57 — EB16 launched** (32 runs, K16 × T3200, rate 1.0, R1).
* **08:00 — D2 addition.** The u_x² conditional by age class was in the D prereg but missing from `clone_age.py`; it
  has been added and the analysis re-run on existing data.
  * ERI: old FR 0.0068 vs ABF 0.0026 (relative RMS).
  * ITE: 0.0045 vs 0.0038.
* **08:10:32 — EB16 done** (6.825 core-h). **UNRESOLVED**: S_geo 0.93 [0.70, 1.29] (99 % [0.56, 1.51]); 7/16.
  * R1 stops at a median 11.0 t.u. (9–14).
  * Window establishment 8.0 → 8.0 t.u. This is uninformative: the save grid is 8 t.u.
  * Every secondary outcome unresolved.
* **08:10 — E.3 frontier re-run with eight configurations.**
  * Best ABF = K16 and best FR = K16. Best FR / best ABF = 1.04 [0.74, 1.33] → **UNRESOLVED** (all-in 1.04
    [0.75, 1.31]).
  * ABF K16 / ABF K64 = 0.75 [0.44, 1.13]. At the tightest rung ABF K64 is faster than ABF K16 (for FR, K16 is
    the fastest).
  * The frontier is not bracketed below K16.
* **08:10–08:15 — E.4 wall-clock benchmark, run 2** (all four K; load 28 → 19).
  * Replicates run 1: run2/run1 median 0.998, range 0.91–1.06.
  * Wall-clock time to accuracy: best ABF = K16 at P = 1, K64 at P ≥ 4. Best FR / best ABF 1.03–1.07, all
    UNRESOLVED.
* **08:15–08:20 — documents.**
  * New: MECHANISM, EFFICIENCY, FINAL_SYNTHESIS, HANDOFF.
  * Updated: ONLINE_STOPPING_RULE §6, EQUAL_BUDGET_RESULTS A.6.
  * Figures: `equal_budget_frontier`, `wallclock_ite300`.
  * Analysis reproduction script: `scripts/adaptive_fr/reproduce_analyses.sh`.
* **08:21 — reproduction check.** `reproduce_analyses.sh` re-ran every analysis and figure. Every analysis JSON and
  PNG is byte-identical to f98d3ba8; only the PDF timestamps differ. Engine and controller tests: 35 passed.
* **08:22–08:31 — independent doc-vs-JSON audit** (a read-only review agent working from the commit). The headline
  numbers were confirmed. Corrections applied 08:31–08:35:
  1. **Save-grid artefacts.** The K64/K16 establishment values are artefacts of the 2 and 8 t.u. save grids, and are
     no longer used as evidence.
  2. **ERI flux.** "ERI flux never decays" is replaced by the logged plateau (30–60 % of peak for 5–10 t.u.; all runs
     stopped).
  3. **Late-only windows.** The windows differ between arms in the stopped blocks. A matched-window ABF variant was
     added to D1: ABF bias 0.006 / 0.010 / 0.014 against stopped FR 0.009 / 0.013 / 0.014 over the same post-stop
     window.
  4. **Replacement counts.** The wrong cell was cited; the D2 counts now are 1817 per ERI run and 1120 per ITE run.
  5. **Preregistration timing.** Records corrected by a dated erratum in PREREGISTRATION.md; E-A2's header edit is
     reverted.
  6. **Overstatements.** "Neutral / parity / equivalent / not an efficiency gain" is replaced by "unresolved" or
     "not shown". The unbracketed frontier is stated as such. Circulation is caveated in LTA 150 K.
  7. **Smaller fixes.** Multiplier ranges (2.7–6.4×), 0.1 T ratios by K, overhead ranges, the 127 s entry, the HANDOFF
     test count and run time, and the aggregate core-h.
* **08:25 — figure 7b** (`mechanism_meanforce_genealogy`): per-bin late-half mean-force error by clone age (D2) and
  FR genealogy over time in the C cells.

## D4 — selection-hypothesis test (session of 2026-10-17, from 11:16 UTC; branch `claude/nifty-hellman-99b892`)

* **11:16–11:28 — instrumentation.** `diag_sel` (read-only birth/death record, ancestor-tagged deposits, fibre
  histograms) and the always-on `Sbond` / `Suz2` accumulators added to `src/eri/engine.py`. A two-arm trajectory was
  frozen from the pre-change engine (6716aefc) BEFORE editing (`tests/fixtures/eri_pre_selection_fixture.npz` +
  sha256 json); `tests/test_diag_sel.py` (6 tests) proves the default path reproduces it bitwise and that the
  diagnostic is inert. A first version segfaulted: the flag was named `dsel`, which shadowed the kernel's dying-slot
  array; renamed `selon`. 37 engine/controller/stop tests pass. Overhead measured 1.0 % at K256.
* **11:30:27 — PREREGISTRATION committed (49ae642d)**, then `launch.sh configs/adaptive_fr/jobs_D3.json 64` at
  11:30:27 (registry line in `logs/launches.tsv`; stage label D3, experiment D4 in the documents).
* **11:35:19 — ALLDONE** 3.655 core-h, 292 s wall. Identity gate: 64/64 runs bitwise equal to their historical arms
  (sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond, q_final). Raw files: `results/adaptive_fr_campaign/D3_{eri600,ite300}_K256_sel/`
  in the session worktree, copied to the same path in `/home/zheyuanlai/ABF-Fisher-Rao-adaptive` (0.9 GB).
* **11:36–12:10 — analysis** (`scripts/adaptive_fr/selection_split.py`, four passes: a keyword clash `nonlocal`, a
  degenerate-coordinate guard for ITE's η (≈ constant on the channel axis), per-region selection and net-exchange
  statistics added). Results in MECHANISM.md §2.4, figure `mechanism_selection`.

## D5 — arrival-law mechanism test (same session, from 12:00 UTC)

* **12:00–12:11 — instrumentation.** `diag_arr` (read-only provenance / bin-entry record: per deposit the last visited
  region and time since leaving it, the coarse-bin entry side and residence time; 11 quantities incl. the transverse
  COM phases and the sub-bin offset) and always-on `Scy/Ssy/Scz/Ssz`. `tests/test_diag_arr.py` (4 tests); 41 tests
  pass across the ERI suites; the frozen-fixture test still passes.
* **12:12:22 — PREREGISTRATION committed (8564ce92)**, then `launch.sh configs/adaptive_fr/jobs_D5.json 64`.
* **12:16:05 — ALLDONE** 3.109 core-h, 223 s wall. Identity gate: 64/64 bitwise equal to the historical arms. Raw files
  `results/adaptive_fr_campaign/D5_{eri600,ite300}_K256_arr/` (97 MB each), copied to `/home/zheyuanlai/ABF-Fisher-Rao-adaptive`.
* **12:17–12:40 — analysis** (`scripts/adaptive_fr/arrival_split.py`; a second pass added the exact weight/state
  decomposition of FR's excess and the fixed-bin relaxation table, both post hoc). Results in MECHANISM.md §2.5,
  figure `mechanism_arrival`.
