# HANDOFF — adaptive-FR campaign (2026-10-17)

* **Branch.** `research/adaptive-fr-equal-budget`, cut from `main` at 0f225325.
* **Worktree.** `/home/zheyuanlai/ABF-Fisher-Rao-adaptive`. Historical run files are read from the main checkout
  through `ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao`.
* **Not merged into `main`.** The campaign brief allows a merge only with explicit permission.

## 1. Where things are

| what | path |
|---|---|
| preregistration (§0 rules, A, B = frozen R1, C, D, E, amendments E-A1, E-A2) | `docs/adaptive_fr_campaign/PREREGISTRATION.md` |
| preregistration of D4 (selection test) | `docs/adaptive_fr_campaign/MECHANISM_SELECTION_PREREG.md` |
| preregistration of D5 (arrival-law test) | `docs/adaptive_fr_campaign/MECHANISM_ARRIVAL_PREREG.md` |
| results documents | `docs/adaptive_fr_campaign/{EQUAL_BUDGET_RESULTS, ONLINE_STOPPING_RULE, CROSS_SYSTEM_RESULTS, MECHANISM, EFFICIENCY, FINAL_SYNTHESIS}.md` |
| status, log, compute | `docs/adaptive_fr_campaign/{STATUS, EXPERIMENT_LOG, COMPUTE_LEDGER}.md` |
| run files (npz, one per arm × seed) | `results/adaptive_fr_campaign/<block>/{abf,fr}_N<K>_s<seed>.npz` |
| analysis outputs | `results/adaptive_fr_campaign/analysis/*.json` (+ `_curves.npz`), `results/adaptive_fr_campaign/mechanism/*.json` |
| development records of R1 (ITE only, before the freeze) | `results/adaptive_fr_campaign/dev/` |
| job files | `configs/adaptive_fr/jobs_*.json` (written by `scripts/adaptive_fr/make_jobs.py <block>`) |
| launch registry and logs | `results/adaptive_fr_campaign/logs/{launches.tsv, <stage>.log}` |
| figures (PNG + PDF) | `figures/adaptive_fr/` |

### Blocks

| block | dir | role |
|---|---|---|
| A | `A_ite300_K256_T200_stop10` (FR arm only; ABF = historical EB) | fixed stop 10, K256 × T200 |
| C1–C5 | `C{1..5}_*_R1` | frozen online rule R1, five systems |
| D2 | `D2_{eri600,ite300}_K256_age` (FR arm only) | clone-age diagnostic re-runs, bitwise equal to the historical arms |
| D4 | `D3_{eri600,ite300}_K256_sel` (BOTH arms) | selection-hypothesis test (MECHANISM_SELECTION_PREREG.md): bitwise re-runs with the read-only selection diagnostic `diag_sel`; 0.9 GB |
| D5 | `D5_{eri600,ite300}_K256_arr` (BOTH arms) | arrival-law test (MECHANISM_ARRIVAL_PREREG.md): bitwise re-runs with the read-only provenance record `diag_arr`; 0.2 GB |
| EA_R1 | `EA_ite300_K256_T200_R1` (FR arm only; ABF = EB) | R1 at K256 × T200 |
| EB64 / EB16 | `EB64_ite300_K64_T800_R1`, `EB16_ite300_K16_T3200_R1` | frontier cells |
| Ecal_K64 / Ecal_K16 | `Ecal_K{64,16}` | blind safety calibration (+ `frcal_K*.json`) |

## 2. Code added or changed

* **`src/eri/fr_controller.py`** (new). The online stopping rule R1 (`OnlineStop`, `R1` spec). Reference-free by
  construction: it imports only math and numpy, and an AST test enforces that.
* **`src/eri/engine.py`.**
  * `fr_stop_steps` (fixed stop); `fr_controller` (online stop: host-side checks via `advance_controlled`);
    `diag_age_steps` (read-only clone-age split); `diag_sel` (+ `diag_sel_band_steps`, `diag_sel_split_step`): read-only
    birth/death record with the fibre state of parent and killed walker, ancestor tag per slot, per-event deposit
    accumulators (early/late halves; time-since-birth bands × coarse bins), founders' deposits, per-bin fibre
    histograms in both arms; result keys `sel_*`.
  * `diag_arr` (+ `diag_arr_band_steps`): read-only provenance / bin-entry record; result keys `arr_*`.
  * Always-on whole-run fibre accumulators `Sbond`, `Suz2` (next to `Seta`, `Sux2`) and the transverse COM phases
    `Scy`, `Ssy`, `Scz`, `Ssz`.
  * All diagnostics default off; the default path is bitwise unchanged: `tests/test_diag_sel.py` checks it against a
    trajectory frozen from engine 6716aefc (`tests/fixtures/eri_pre_selection_fixture.npz`).
* **`src/lta_ladder_numba.py`.** `fr_stop_steps` and `fr_controller` (same semantics; stop-aware opportunity count;
  resume).
* **Tests.**
  * `tests/test_fr_stop.py` (7), `tests/test_fr_controller.py` (15), `tests/test_lta_fr_stop.py` (4),
    `tests/test_diag_age.py` (2), `tests/test_diag_sel.py` (6), `tests/test_diag_arr.py` (4).
  * These four files plus `tests/test_eri_engine.py` (7): 35 passed (re-run 08:25 UTC). Full LTA suite: 39 passed,
    1 skipped.
* **`scripts/adaptive_fr/`.**
  * Job generation and launch: `make_jobs.py`, `launch.sh`, `launch_lta.sh`, `run_lta_jobs.py`.
  * Analyses: `analyze_online.py` (frozen two-arm analysis, both engines), `equal_budget4.py` (A.5),
    `equal_budget_frontier.py` (E.3 + E-A2), `wallclock_bench.py` / `wallclock_tta.py` (E.4), `bias_variance.py`
    (D1), `clone_age.py` (D2), `selection_split.py` (D4), `arrival_split.py` (D5).
  * Figures: `fig_campaign.py`.
  * R1 development (pre-freeze): `dev_*.py`.
* **`scripts/convergence_fr/`.** `analyze_block.py` gained `--abf-dir` / `--allow-diff` (FR knobs may differ only when
  inert); `systems.py` gained `load_eri_cell(abf_dir=)`; `block_tables.py` reads `BLOCK_AN`; `link_data.sh` also links
  `cache/lta/framework.npz`.

## 3. Exact commands

Environment: conda env `abffr` (`/home/zheyuanlai/miniconda3/envs/abffr/bin/python`, Python 3.14, numba). CPU only.
Every run is single-threaded (`NUMBA_NUM_THREADS=1`).

```bash
cd /home/zheyuanlai/ABF-Fisher-Rao-adaptive
bash scripts/convergence_fr/link_data.sh /home/zheyuanlai/ABF-Fisher-Rao
```

Tests:

```bash
NUMBA_NUM_THREADS=1 /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_fr_stop.py tests/test_fr_controller.py tests/test_lta_fr_stop.py tests/test_diag_age.py tests/test_eri_engine.py
```

Re-run every analysis and figure from the saved run files (under a minute; verified 08:21 UTC to reproduce every
analysis JSON and PNG byte for byte):

```bash
bash scripts/adaptive_fr/reproduce_analyses.sh
```

Re-run a block from scratch. Example: EB16, 32 single-threaded runs of about 13 min each (770–804 s on this host). Outputs are skipped when
they already exist, so delete or move the block directory first:

```bash
/home/zheyuanlai/miniconda3/envs/abffr/bin/python scripts/adaptive_fr/make_jobs.py EB16_rate1
```

```bash
bash scripts/adaptive_fr/launch.sh configs/adaptive_fr/jobs_EB16_rate1.json 32
```

LTA blocks (C3, C4) use `bash scripts/adaptive_fr/launch_lta.sh configs/adaptive_fr/jobs_C3.json 32`.

Wall-clock benchmark (run only while nothing else of ours is running; refuses to overwrite an existing output):

```bash
NUMBA_NUM_THREADS=32 /home/zheyuanlai/miniconda3/envs/abffr/bin/python scripts/adaptive_fr/wallclock_bench.py E_wallclock_bench_run3.json
```

Using R1 on a new system: add `fr_controller=R1` (from `eri.fr_controller import R1`) to an FR config whose
`fr_start_steps` ≥ 1, and call `Run.run()`. `Run.advance()` raises when a controller is set.

## 3b. Follow-up campaign: the occupancy-triggered rule R2 (2026-10-17, branch `claude/ecstatic-bhaskara-b3b3ea`)

* Open item 1 below was addressed by a second frozen rule, **R2** (`src/eri/fr_controller.py`, `R2`): FR stops at
  2 × the run's own online window-establishment time (trailing-1 t.u. deposit share of the window bins ≥ 0.75 × the
  flat-bias share, with R1's support and safety guards; the stability statistic is logged, not gating).
* Documents: `docs/r2_campaign/{DEVELOPMENT, PREREGISTRATION, RESULTS, COMPUTE_LEDGER}.md`. Raw run files:
  `/home/zheyuanlai/ABF-Fisher-Rao/results/r2_campaign/raw` (git-ignored; symlinked as `results/r2_campaign/raw` in
  that worktree); analysis JSONs committed under `results/r2_campaign/analysis/`.
* Outcome (eight cells, 16 fresh paired seeds each): every positive cell SUPPORTED at the Bonferroni-8 level, including
  the four held-out LTA cells (150 / 300 / 80 / 350 K: S_geo 2.45 / 1.39 / 1.84 / 1.29) and the in-sample ITE cells
  (1.80, 1.44); ERI stops at 4.0 t.u. in every run and goes from R1's 0.61 to 0.92 [0.86, 1.04] (K1024) and
  0.87 [0.81, 0.98] (K256): EQUIVALENT by the preregistered TOST margin [0.80, 1.25], but a small residual slowdown,
  resolved at K256. "R2 generalises by the preregistered definition": yes. See `docs/r2_campaign/RESULTS.md`.

## 4. Open items and suggested next steps

1. **The ERI failure of R1** (CROSS_SYSTEM_RESULTS.md §5). **Addressed by R2 (§3b)**; the residual ERI slowdown
   (8–13 %) is the FR-period memory of the 1–4 t.u. deposits and is the remaining open item there.
   * After its early peak, the directedness statistic plateaus at 30–60 % of that peak for 5–10 t.u., so R1 stops
     late (median 15.25 t.u.).
   * A rule that would stop early in ERI, or never start FR there, must be designed and frozen on development data.
   * ERI 600 K is now spent as a held-out test, so it cannot be that development data.
2. **The late-FR bias mechanism** (MECHANISM.md §2.3–2.4). D4 tested the selection hypothesis directly: births are
   not fibre-selected within their bin; the late error is carried by lineages according to their BIRTH REGION
   (cage-born lineages carry the pooled pattern, window-born its mirror image) while the carried fibre state relaxes
   over 1–3 t.u. (ERI). The supported reading is the arrival-law (net CV flux) mechanism of
   MECHANISM_SELECTION_PREREG.md §7.2. D5 then measured it with a per-deposit provenance record (MECHANISM.md §2.5):
   the premise holds with a large effect (a flank walker's deposit depends on the region it last visited by
   ±1–16 kJ/mol/rad, through its fibre state), the two arrival directions balance exactly in ABF, and FR's excess
   decomposes exactly into a provenance-weight shift and a shift of the cage arrivals' own state. The sub-bin
   alternative is excluded. The ITE carrier is a joint fibre displacement (η, u_x², u_z² all differ by provenance),
   not a single coordinate.
3. **Equal-budget claims.** No configuration of ABF + FR was shown to beat the best tested ABF allocation, in force
   evaluations or in wall-clock (EFFICIENCY.md: CIs wide, point estimates on ABF's side). The frontier is not
   bracketed below K16. Any further "accelerator" claim must be made against the best ABF allocation, not against a
   fixed K.
4. **Compute.** The adaptive-FR campaign used about 44 core-h; the 2026-10-17 follow-ups added E-A3 21.8, R2 36.7 and D4/D5 7.1, so the combined total is about 109.7 of the 100 authorised core-h (COMPUTE_LEDGER.md, final section).

## 5. Amendment E-A3 (2026-10-17, branch `claude/priceless-rubin-9602df`, worktree `.claude/worktrees/priceless-rubin-9602df`)

* **What it adds.** (1) A selection-aware nested bootstrap of best FR / best ABF in `equal_budget_frontier.py`
  (new keys only; EFFICIENCY.md §7). (2) The frontier bracket below K16: blocks EB8 (K8 × T6400, both arms, rate 1.0
  from the blind K8 calibration), EB4 (K4 × T12800, ABF only) and EB1 (K1 × T51200, ABF only), all saving every
  1 t.u.; PREREGISTRATION.md E-A3 has the cells, the K4 FR criterion (not met: R1's backstops cannot fire at K4),
  the verdict rules and the pilot-measured cost.
* **Data layout from a worktree.** The run files live only in the adaptive worktree's
  `results/adaptive_fr_campaign/<block>/`. Link everything with

  ```bash
  bash scripts/convergence_fr/link_data.sh /home/zheyuanlai/ABF-Fisher-Rao /home/zheyuanlai/ABF-Fisher-Rao-adaptive
  ```

  which symlinks each block directory (and the npz files of `Ecal_K*`, whose calibration JSONs are tracked) and adds
  the symlinks to the worktree's `.git/info/exclude`. The historical EB and reference files are still read through
  `ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao`. New blocks are created as real directories in the adaptive
  worktree and symlinked here before launching, so `run_jobs.py` writes through the link.

| block | dir | role |
|---|---|---|
| EB8 | `EB8_ite300_K8_T6400_R1` | K8 × T6400 frontier cell, both arms, R1, rate 1.0, saves every 1 t.u. |
| EB4 | `EB4_ite300_K4_T12800_abf` | K4 × T12800, ABF only, saves every 1 t.u. |
| EB1 | `EB1_ite300_K1_T51200_abf` | K1 × T51200, ABF only, saves every 1 t.u. |
| Ecal_K8 / Ecal_K4 | `Ecal_K{8,4}` | blind safety calibration rungs 1.0 / 0.5 / 0.2 (JSONs copied to `analysis/frcal_K{8,4}.json`) |

* **New analysis outputs.** `analysis/E_frontier.json` (eight configurations; new keys `nested_bootstrap`,
  `question_B_verdict_nested`, `save_resolution_force_evals`), `analysis/E_frontier_ext.json` (+ `_curves.npz`; the 12
  configurations K1..K1024), `analysis/EB8.json`, `analysis/E_establishment.json`
  (`scripts/adaptive_fr/establishment_frontier.py`), figures `equal_budget_frontier_ext` (3c) and
  `establishment_frontier` (9). `reproduce_analyses.sh` runs all of them.
* **Job files.** `configs/adaptive_fr/jobs_{EB8_rate1, EB4_abf, EB1_abf, Ecal_K8_ladder, Ecal_K4_ladder}.json` from
  `make_jobs.py {EB8_rate1, EB4_abf, EB1_abf, Ecal_K8_rate*, Ecal_K4_rate*}`.
* **Results (EFFICIENCY.md §7–8; FINAL_SYNTHESIS.md).** Nested best-vs-best CI on the eight configurations 1.04
  [0.78, 1.28] (fixed [0.74, 1.33]). EB8: S_geo 1.03 [0.72, 1.42], UNRESOLVED; establishment 4.5 → 4.5 t.u. at 1 t.u.
  resolution; R1 stopped at a median 16.5 t.u. Extended frontier: best ABF = K1 (0.170 × 10⁸), best FR = K16 (0.188);
  best FR / best ABF 1.10, fixed [0.75, 1.51], nested [0.87, 1.43], UNRESOLVED; ABF flat from K1 to K64; no K4 FR arm
  (R1's backstops cannot fire at K4). Cost 21.8 core-h; campaign ≈ 65.9 / 100.
* **Open items added by E-A3.**
  1. The kT ladder is reached in the first 3–5 % of the budget for K ≤ 16, so the small-K frontier compares early
     transients, and the end-of-budget error orders the allocations differently (ABF K1 worst of K ≤ 64). A tighter
     ladder for small K would need a fresh preregistration.
  2. Below K64 the allocations differ by ≤ 1.4× while 16 unpaired seeds resolve ≈ ±35 %; resolving the flat minimum
     would need paired-across-K designs or several times the seeds.
  3. R1 stops 3–4× past establishment at K8 because its directedness statistic sees ≈ 12 events per run; R1 is not
     validated below K256 and should not be used there without a small-K variant.
