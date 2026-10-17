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

## 4. Open items and suggested next steps

1. **The ERI failure of R1** (CROSS_SYSTEM_RESULTS.md §5).
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
4. **Compute.** About 51 of the 100 authorised campaign core-h were used (COMPUTE_LEDGER.md).
