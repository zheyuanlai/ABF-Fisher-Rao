# Equal-budget ladders: where the data lives and how to regenerate it

*2026-10-10. Sizes and counts were measured on this date. The content hashes of the raw runs and references are in
`results/equal_budget_v2/DATA_MANIFEST.json`, written by `scripts/equal_budget/data_manifest.py`.*

Every command below assumes this environment:

```bash
cd /home/zheyuanlai/ABF-Fisher-Rao
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_eqb
```

## 0. Overview

`*.npz` and `*.log` are gitignored repo-wide, and so is `docs/*` (tracked docs are force-added). In practice the
**arrays are never in git**. Their JSON summaries, tables, figures, configs and code are.

| class | location | in git | size | files |
|---|---|---|---|---|
| raw production runs | `results/equal_budget_v2/{gateway,lta_T300,lta_T150}/N*/s<seed>_<abf\|fr>.npz` | **no** | 1163.3 MB (431.5 + 366.0 + 365.9) | 1408 (736 + 336 + 336) |
| per-run ledgers | `results/equal_budget_v2/<sys>/ledger.csv` | yes | 142 kB | 3 |
| frozen references | `results/equal_budget_v2/references/*.npz` | **no** | 33 kB | 3 |
| reference summary | `results/equal_budget_v2/references/references_summary.json` | yes | 0.6 kB | 1 |
| analysis summaries | `results/equal_budget_v2/<sys>/analysis/{summary.json,tables.md}` | yes | small | 6 |
| per-run metric cache, JSON | `results/equal_budget_v2/<sys>/analysis/run_metrics/N*/s*_*.json` | yes | (part of 55 MB) | 1408 |
| per-run metric cache, arrays | `results/equal_budget_v2/<sys>/analysis/run_metrics/N*/s*_*.npz` | **no** | (part of 55 MB) | 1408 |
| median curves | `results/equal_budget_v2/<sys>/analysis/median_curves.npz` | **no** | 2.1 MB | 3 |
| completeness audit, mechanism tests | `results/equal_budget_v2/{completeness_audit,mechanism}.json` | yes | 1.7 MB | 2 |
| resource ledger summary | `results/equal_budget_v2/ledger_summary.json` | no (untracked, not ignored) | 0.9 kB | 1 |
| run / analysis / plot logs | `results/equal_budget_v2/logs/*.log` (+ `launch_confirm.sh`, which **is** tracked) | **no** (logs) | ~120 kB | 26 + 1 |
| per-configuration figures A–F | `figures/equal_budget_v2/<sys>/N<N>/` | yes | 413 MB in total (397 MiB on disk) | 1370 (PNG + PDF + MANIFEST) |
| synthesis S1–S8, cross-system X1 | `figures/equal_budget_v2/<sys>/synthesis/`, `figures/equal_budget_v2/cross_system/synthesis/` | yes | (included above) | 90 |
| gateway timestep validation, arrays | `results/equal_budget_v2/gateway_validation/<tag>/g*.npz` | **no** | 41 MB | 67 |
| gateway timestep validation, summary and figure | `results/equal_budget_v2/gateway_validation/summary.json`, `figures/equal_budget_v2/gateway/validation/` | yes | 1.5 MB (summary 0.2 MB) | 3 |
| LTA timestep validation (2026-10-09), arrays | `results/numerical_validation/lta/T{300,150}/{MC,EM2e-4,EM1e-4,EM5e-5,LM2e-4}/g*_w*.npz` | **no** | 7.8 MB (52 MB on disk: 6400 small files) | 6400 |
| LTA timestep validation, summaries | `results/numerical_validation/lta/T*/summary.json` (+ figures) | yes | small | |
| **LTA statistical-equivalence runs** | `/tmp/claude-1008/.../scratchpad/engines/{fullrun,fix_lta/tdraw}/` (§7) | **no; volatile scratch** | 291 MB + 36 MB (279 + 35 MiB) | 128 + 48 runs |
| CUDA production that §7 compared against | `results/lta_histogram/production_T{300,150}/{abf,fr_uniform}.npz` | **no** | 30.6 MB | 4 |
| best-allocation confirmation | `results/equal_budget_v2/confirm_best_{gateway,lta_T300,lta_T150}/` | arrays no; ledgers, configs yes | 203 MB of run files (§8) | 256 runs |

Sizes are decimal (1 MB = 10⁶ bytes) of the files' apparent size unless marked MiB or "on disk".

## 1. Raw production runs

* **Layout.** One file per (system, N, seed, method), `results/equal_budget_v2/<out_dir>/N<N>/s<seed>_<method>.npz`.
  `meta_json` and `cfg_json` are stored as strings. N = 1 has an ABF arm only.
* **Counts** (all `status: complete`, none missing against the production configs; checked by
  `data_manifest.py`):

| system | out_dir | N ladder | seeds | runs | bytes | engine (meta) | engine sha256 | commit at run time |
|---|---|---|---|---|---|---|---|---|
| gateway | `gateway` | 2048 … 1 (12) | 8100–8131 (32) | 736 | 431 452 907 | `gateway_ladder_numba/2` | `bdeaa40a…25074ce` | 55a4e78 (clean) |
| LTA 300 K | `lta_T300` | 1024 … 1 (11) | 30000–30015 (16) | 336 | 366 000 064 | `lta_ladder_numba/2` | `d2d3c431…25a804b` | badfbab (clean) |
| LTA 150 K | `lta_T150` | 1024 … 1 (11) | 15000–15015 (16) | 336 | 365 872 599 | `lta_ladder_numba/2` | `d2d3c431…25a804b` | badfbab (clean) |

* The engine sha256 recorded in every file equals the sha256 of the committed `src/gateway_ladder_numba.py` and
  `src/lta_ladder_numba.py` at HEAD on 2026-10-10. Production therefore ran the committed engines.
* **Configs** (in git): `configs/equal_budget_v2/{gateway,lta_300K,lta_150K}_production.json`.
* **Cost** (from the ledgers, one single-threaded process per run, so wall seconds are core-seconds): LTA 300 K
  26.23 core-h, LTA 150 K 26.47, gateway 54.73. No GPU.
* **Regenerate:**

  ```bash
  WORKERS=110 bash scripts/equal_budget/launch_production.sh     # LTA 300 K, then 150 K, then gateway
  # or per system:
  python -u scripts/equal_budget/run_ladder.py --system lta300  --order coarse_first --workers 110
  python -u scripts/equal_budget/run_ladder.py --system lta150  --order coarse_first --workers 110
  python -u scripts/equal_budget/run_ladder.py --system gateway --order ladder       --workers 110
  ```

  `run_ladder.py` skips any complete result already at the output path. It always writes to
  `results/equal_budget_v2/<out_dir>/`, and has no results-root option. To regenerate **without touching** the
  stored files, copy the production config, change only `"out_dir"` (for example to `regen_lta_T300`), and pass it
  with `--config`, as the confirmation configs do.
* **Reproducibility, tested 2026-10-10.** Two production jobs were re-run through `run_ladder`'s own code path
  into scratch: LTA 300 K N = 64 seed 30000 FR, and gateway N = 128 seed 8100 FR, each pinned to one core. Every
  array was bitwise identical to the stored file (43 and 38 array keys; the provenance and cost keys
  `meta_json`, `cfg_json`, `wall_s`, `peak_rss_mb`, `sessions_json`, … were excluded).
  * **File** sha256 values will **not** match the manifest, because the meta records wall time, pid and
    timestamps. Compare arrays, not file hashes.
  * Bitwise identity holds per numba compile target. The LTA engine records it in `meta.compile_target`
    (znver4 here) and refuses a cross-target resume.

## 2. Frozen references

* **Files** (not in git):
  * `results/equal_budget_v2/references/lta_T300_reference.npz`;
  * `results/equal_budget_v2/references/lta_T150_reference.npz`;
  * `results/equal_budget_v2/references/gateway_reference.npz`.
* `references_summary.json` is tracked. The sha256 of each file is in the manifest.
* **Sources:**
  * LTA: the exact Metropolis-MC data of the 2026-10-09 validation,
    `results/numerical_validation/lta/T{300,150}/MC/g*_w*.npz` (16 groups × 40 windows = 640 files per T; the
    builder asserts these counts).
  * Gateway: analytic, via `scripts/analyze_gateway_replica_ladder.py`'s `Scorer` (needs torch, CPU is enough).
* **Regenerate:** `python scripts/equal_budget/build_references.py`. It takes no options, overwrites the three files
  and the summary, and costs ≤ 0.5 core-h.
* The analysis keys its cache on the reference sha256 (`eqb_metrics.run_cache_key`). A rebuilt reference that
  differs in any bit therefore invalidates every cached run metric.

## 3. Analysis outputs and caches

* `results/equal_budget_v2/<sys>/analysis/` contains:
  * `summary.json` and `tables.md`, tracked;
  * `median_curves.npz`, gitignored;
  * `run_metrics/N*/s*_*.{json,npz}`: the JSON is tracked, the npz is gitignored.
* Sizes: gateway 31.4 MB, each LTA temperature 12.0 MB.
* A cache entry is reused only if its key matches. The key covers the run file's content sha256, the metrics
  version, the config digest, and the sha256 of the scoring code and of the reference.
* **Regenerate** (seconds to a few minutes per system):

  ```bash
  python scripts/equal_budget/analyze_ladder.py --system lta300     # also lta150, gateway
  python scripts/equal_budget/mechanism_analysis.py                 # -> results/equal_budget_v2/mechanism.json
  python scripts/equal_budget/audit_completeness.py --system all    # -> results/equal_budget_v2/completeness_audit.json
  python scripts/equal_budget/ledger_summary.py --json results/equal_budget_v2/ledger_summary.json
  python scripts/equal_budget/data_manifest.py                      # -> results/equal_budget_v2/DATA_MANIFEST.json
  ```

## 4. Figures

* **In git:** all 1462 files under `figures/equal_budget_v2/` (712 PNG, 712 PDF, 38 MANIFEST.json), 413 MB in total
  (412 939 148 bytes; `du -sh` shows 397M).
  * gateway: 499 files, 150 MB, including the 2 validation figures;
  * LTA 300 K: 480 files, 124 MB;
  * LTA 150 K: 480 files, 139 MB;
  * cross-system: 3 files.
* **Regenerate:**

  ```bash
  python scripts/equal_budget/plot_config.py --system lta300 [--only-N 1024]   # per-(N) figures A-F; also lta150, gateway
  python scripts/equal_budget/plot_synthesis.py --system all                  # S1-S8 per system + X1
  python scripts/equal_budget/analyze_gateway_validation.py                   # gateway/validation figures (+ summary.json)
  ```

* `plot_synthesis.py` refuses a summary that is stale relative to the run files unless `--refresh-analysis` is
  given.

## 5. Logs, ledgers and the workflow record

* **Production and analysis logs** (gitignored): `results/equal_budget_v2/logs/`, holding `production.log`,
  `analysis_*.log`, `plot_*.log`, `synthesis_*.log`, `audit_all.log` and `confirm_*.log`. The launcher
  `launch_confirm.sh` in the same directory is tracked (force-added).
* **Per-run ledgers** (tracked): `results/equal_budget_v2/<sys>/ledger.csv`.
* **Workflow transcripts:** `/home/zheyuanlai/.claude/projects/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/`.
  * The main session is `b25e8a63-….jsonl` (next to that directory).
  * The workflow subagents are in `subagents/workflows/wf_*/`:
    * `wf_5f0ce4cd-458`: Phase 0A audit;
    * `wf_672afd95-8cd`: engines;
    * `wf_ec09632e-d4e`: analysis pipeline;
    * `wf_42c726cc-39b`: report verification;
    * `wf_4a0aa34b-865`: gap closing.
  * These transcripts are outside the repository and are not backed up by git.

## 6. Validation runs

**Gateway timestep validation** (`GATEWAY_TIMESTEP_VALIDATION.md`)
* 67 jobs, 3.50 core-h.
* Tags: `em_flat_h{4e-4,1e-4,2.5e-5}` (16 groups each), `mala_flat_h1e-4` (16) and `em_unbiased_h*` (1 each).
* Arrays: `results/equal_budget_v2/gateway_validation/<tag>/g*.npz`, 41 MB, not in git. `run.log` is not in git.
  `summary.json` is tracked.
* Regenerate:

  ```bash
  python scripts/equal_budget/run_gateway_validation.py --workers 64    # default h list 4e-4 1e-4 2.5e-5
  python scripts/equal_budget/analyze_gateway_validation.py
  ```

**LTA timestep validation, 2026-10-09** (`LTA_TIMESTEP_VALIDATION.md` §1; `docs/numerical_validation/LTA_VALIDATION.md`)
* Prereg: `configs/numerical_validation/lta_validation_prereg.json`.
* 3200 jobs per T. 640 files per tag per T, 6400 files and 7.8 MB in total (52 MB on disk), not in git. Summaries,
  figures and `analysis.log` are tracked.
* Estimated cost about 31 core-h per T (from that doc).
* Regenerate:

  ```bash
  python scripts/numerical_validation/run_lta_validation.py --T 300 --workers 64   # and --T 150; skips existing files
  python scripts/numerical_validation/analyze_lta_validation.py --T 300            # and --T 150
  ```

**LTA smoke test under the live ABF bias, 2026-10-10 ~06:25 UTC** (`LTA_TIMESTEP_VALIDATION.md` §4)
* **No output file exists.** It was an inline script in the main session that printed its numbers and saved
  nothing.
* The script and its printed output are in the main transcript, as the tool call at 2026-10-10T06:24:40Z (in the
  same command as the 65af1be engine commit).
* Runs: N = 1024 for 60 000 steps (ABF and FR) and N = 1 ABF for 2 × 10⁶ steps, at both T, seed 99. About 4 core-min.
* Regenerate by re-running that script. It uses `make_cfg(T, N, 99, n_steps=n)` and the save points
  `[1, n/4, n/2, n]`.

**Engine tests and analysis fixtures.**
* Engine tests: `tests/test_{gateway,lta}_ladder_numba.py`. Their scratch outputs are temporary.
* Analysis fixtures: `scripts/equal_budget/make_fixtures.py` writes `tempfile.gettempdir()/eqb_fixtures`
  (currently `/tmp/eqb_fixtures`, 5.3 MB), regenerated on demand by `tests/test_eqb_metrics.py`.

## 7. LTA statistical-equivalence runs (numba ladder engine vs the CUDA production)

### 7.1 What the reports quote

* `IMPLEMENTATION_VALIDATION.md` §2 and `LTA_TIMESTEP_VALIDATION.md` §4 state:
  * the paired FR-vs-ABF ΔI_F of the numba engine is **−12.3 % at 300 K and −28.9 % at 150 K**, inside the
    published CIs [−15.1, −11.0] and [−29.2, −26.3];
  * I_F equivalence at ±5 % is not shown;
  * window occupancy differs by about 1 % between CPU and CUDA;
  * numba driven by torch's own CPU random draws agrees with numba's own draws.
* The "published CIs" are the 2026-10-03 CUDA production of the LTA histogram campaign, recorded in
  `results/lta_histogram/README.md` (tracked), lines 27 and 29: −28.1 % [−29.2, −26.3] at 150 K and −14.1 %
  [−15.1, −11.0] at 300 K.

### 7.2 Where the outputs are: they still exist, but only in volatile scratch

Base directory `S = /tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines`.
None of it is in the repository or in git.

**(a) Native numba runs: `$S/fullrun/`**
* Produced by the engine implementer (`implement:lta`, agent ab42b538d86cfcb67).
* **128 runs:** T ∈ {300, 150} × {abf, fr} × seeds 1–32.
* Every run: N = 1024, `make_cfg` defaults (the frozen production knobs), n_steps = 300 000 (60 t.u.), saves every
  3000 steps (101 saves).
* Files: `T{T}_{abf,fr}_s{seed}.npz` (128 files, 97.4 MB), `.npz.ckpt` (128, 193.7 MB) and `.log` (128); 291 MB
  in all (`du -sh` 279M).
* Scripts: `run_one.py` (runner), `score.py`, `score_multi.py`, and `score_multi2.py` (the scorer behind the
  quoted numbers).
* **Engine:** `lta_ladder_numba/1`, the implementer's pre-review version. The file was untracked then, so the
  `git_commit` ea14f11 in meta does not identify the code. That code is not preserved, but see §7.5.
* Run 2026-10-10 05:07–05:30 UTC, unpinned, up to ~64 concurrent processes.

**(b) Torch-draw discriminator: `$S/fix_lta/tdraw/`**
* Produced by the LTA fixer (`fix:lta`, agent a527a5a525a80b461).
* **48 runs:** 300 K, ABF only, seeds 1–48, N = 1024, 300 000 steps.
* The numba dynamics are driven by torch's CPU MT19937 draws: initial conditions from
  `LTASystem.initial_conditions`, noise from `torch.randn`, with the generator seeded 7 000 000 + seed.
* Files: `T300_abf_s{seed}.{npz,log}`, 36.4 MB (`du -sh` 35M).
* Scripts: `$S/fix_lta/run_torchdraws.py`, `score_tdraw.py` and `tost_tdraw.py`.
* The engine is the snapshot copy `$S/fix_lta/lta_ladder_numba_snap.py` (sha256 `8984ffa6…`, `ENGINE =
  lta_ladder_numba/1`), taken mid-fix at 06:00 UTC. Run 06:00–06:14 UTC.

**(c) TOST re-scoring: `$S/review_lta/tost_fullrun.py`**
* Produced by the LTA spec reviewer (`review:lta:spec`, agent a5dd312acbbadae8b).
* It applies the Audit 0A §8C gate to (a): bootstrap CI of the median ratio, 10 000 resamples, 90 %.
* No new runs.

**Comparison inputs** (in the repository, npz gitignored)
* CUDA production: `results/lta_histogram/production_T300/{abf,fr_uniform}.npz` (16 seeds 1060–1075, rng_seed
  20260904) and `results/lta_histogram/production_T150/{abf,fr_uniform}.npz` (seeds 1020–1035, rng_seed 20260902).
  Both at git_rev 56ab774, from 2026-10-03.
* The 4-seed CUDA ABF calibration: `results/lta_histogram/calibration/width_ladder_T300/n180.npz` (seeds
  1100–1103).
* **Reference used for all of §7:** `results/uniform_campaign/lta/reference/reference_T{300,150}.npz`, the umbrella
  reference that the published CIs also use (gitignored, not in git; sha256 `14641b0c…` at 300 K, `41746d17…` at
  150 K on 2026-10-10). It is **not** the equal-budget MC reference of §2, so the §7 numbers
  are not comparable digit-for-digit with the ladder's own I_F values.

### 7.3 The numbers as recorded (scorer output in the transcripts)

**`score_multi2.py`** (numba n = 32 per arm vs CUDA n = 16; mean ± SE; Welch p)

| T | quantity | numba | CUDA | p |
|---|---|---|---|---|
| 300 K | ABF I_F | 24.12 ± 0.33 | 23.52 ± 0.50 | 0.32 |
| 300 K | FR I_F | 21.02 ± 0.35 | 20.35 ± 0.47 | 0.25 |
| 300 K | ABF window occupancy | 0.2531 ± 0.0008 | 0.2489 ± 0.0010 | 0.003 |
| 300 K | FR deaths in cages | 0.4185 ± 0.0061 | 0.4517 ± 0.0069 | 0.001 |
| 300 K | **paired ΔI_F, median** | **−12.25 %** | −14.10 % | MW 0.65 |
| 150 K | ABF I_F | 28.58 ± 0.25 | 28.03 ± 0.40 | 0.25 |
| 150 K | FR I_F | 20.48 ± 0.30 | 20.31 ± 0.38 | 0.73 |
| 150 K | **paired ΔI_F, median** | **−28.89 %** | −28.07 % | MW 0.44 |

**TOST** (`tost_fullrun.py`, ±5 % on I_F)
* I_F equivalence is NOT SHOWN in any of the 4 cells. At 300 K the ABF ratio is 1.058 [1.002, 1.108].
* Lineage cage crossings and FR event counts are EQUIVALENT.

**Torch-draw** (`score_tdraw.py` / `tost_tdraw.py`), 300 K ABF window occupancy:

| sample | occupancy | n |
|---|---|---|
| numba, torch draws | 0.2516 ± 0.0007 | 48 |
| numba, own draws | 0.2531 ± 0.0008 | 32 |
| CUDA production | 0.2489 ± 0.0010 | 16 |
| CUDA calibration | 0.2532 ± 0.0036 | 4 |

* Torch draws vs numba's own draws: p 0.18.
* All 80 CPU-engine runs vs all 20 CUDA runs: Welch p 0.056, MW p 0.041.
* The I_F median ratio numba / torch-draws is 1.015 [0.969, 1.062]: NOT SHOWN at ±5 %.

### 7.4 Cost

* **CPU:**
  * (a) 128 runs, 35 038 core-s = **9.73 core-h** (mean 274 s, max 288 s);
  * (b) 48 runs, 13 800 core-s = **3.83 core-h**;
  * scoring and TOST: minutes.
  * Total **≈ 13.6 core-h**. None of it appears in the `EXPERIMENT_LOG.md` resource ledger or its bounded
    estimates, so that log's "≤ 150 core-h" total is low: 133.0 ledgered + ≤ 12.5 bounded + 13.6 ≈ 159 core-h,
    before the FR-overhead benchmarks.
* **GPU: none was used.** The CUDA side is pre-existing data from the 2026-10-03 histogram campaign. Its
  `runtime_seconds` for one 16-seed arm is 1306 s (ABF) and 1418–1419 s (FR) per T, so the four arms used total
  ≈ 1.5 GPU-h. That cost belongs to that campaign.

### 7.5 Regeneration, and a check that it works

```bash
S=/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines
# (a) native numba: 128 single-core jobs of ~4.6 min each, writing into any scratch directory OUT
OUT=/path/to/scratch/fullrun; mkdir -p $OUT; cd $OUT
cp $S/fullrun/run_one.py .        # or recreate it: 11 lines, quoted in agent ab42b538d86cfcb67's transcript at 05:07:37Z
for T in 300 150; do for m in abf fr; do for s in $(seq 1 32); do
  python run_one.py $T $m $s T${T}_${m}_s${s}.npz > T${T}_${m}_s${s}.log 2>&1 &
done; done; done; wait
python $S/fullrun/score_multi2.py   # run from $OUT; reads T*_s*.npz in the cwd
# (b) torch-draw: 48 jobs (needs torch, CPU only); run_torchdraws.py imports the snapshot from its own directory
mkdir -p tdraw
for s in $(seq 1 48); do python $S/fix_lta/run_torchdraws.py 300 abf $s tdraw/T300_abf_s$s.npz > tdraw/T300_abf_s$s.log 2>&1 & done; wait
```

Throttle the loops to the free cores; the originals ran ~64 at once. `run_one.py` calls
`E.make_cfg(T, 1024, seed)` and `E.run_arm(c, method, np.arange(0, n_steps + 1, 3000), checkpoint_path=out + ".ckpt",
checkpoint_every_s=120)`.

Scoring (b) and the TOST: `cd $OUT && python $S/fix_lta/tost_tdraw.py` (it imports `score_tdraw.py` from its own
directory and prints that table too) and `python $S/review_lta/tost_fullrun.py`. Both scorers hard-code the native
sample's directory (`FULL` / `D` = `$S/fullrun`), so edit that path if (a) was regenerated elsewhere; every scorer
reads the CUDA arrays from `results/lta_histogram/` and the reference `results/uniform_campaign/lta/reference/`.
`run_torchdraws.py` also needs `cache/lta/framework.npz` (present, gitignored, not in git) and the tracked torch
engine `src/lta/core_lta.py`.

**Check, 2026-10-10 08:41–08:51 UTC.** Three of these runs were re-run against the **current committed**
`src/lta_ladder_numba.py` (`lta_ladder_numba/2`), each pinned to one free core:
* T300 ABF seed 1;
* T300 FR seed 1;
* the torch-draw T300 ABF seed 1, with the snapshot import replaced by the committed engine.

All 43 array keys of each were **bitwise identical** to the saved scratch files. The only cfg difference is the
added key `deposit_clip_override`. So:
* the pre-review `/1` engine and the production `/2` engine give the same dynamics at the frozen knobs, at both
  temperatures (150 K: see the reviewer check below);
* the equivalence results can be regenerated with the committed engine;
* 150 K was not re-checked by this pass. **Reviewer check, 2026-10-10 08:58–09:03 UTC:** T150 FR seed 1 and T150 ABF
  seed 2 were re-run the same way (same `regen_one.py`, one pinned core each, 0.15 core-h) into
  `scratchpad/reviewer_dataloc/`; all 43 array keys of both are bitwise identical to `$S/fullrun/`, and the only cfg
  difference is again `deposit_clip_override`.

The check cost 0.43 core-h, including the two production re-runs of §1. Outputs and scripts:
`/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/data_locations/`.

### 7.6 If the scratch directory is wiped

`/tmp` is 95 % full and this is a session scratchpad, so the ~327 MB above (314 MiB on disk) can disappear at any time. Copying it
into `results/equal_budget_v2/equivalence/` (gitignored npz) would preserve it. That was **not done here**, per
instruction.

If the scratch directory is gone, this evidence remains:
* **Transcripts** under
  `/home/zheyuanlai/.claude/projects/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/subagents/workflows/wf_672afd95-8cd/`:
  * `agent-ab42b538d86cfcb67.jsonl` (1.8 MB): every launch command at 05:07:37, 05:12:59, 05:18:10 and 05:25:20Z,
    and the full `score_multi2.py` output;
  * `agent-a5dd312acbbadae8b.jsonl` (1.1 MB): the TOST table at 05:52:16Z;
  * `agent-a527a5a525a80b461.jsonl` (1.3 MB): the torch-draw launches at 06:00:43 and 06:09:06Z, the
    `score_tdraw.py` output at 06:14:18Z and the TOST output at 06:14:36Z;
  * `journal.jsonl`: the three agents' final reports, with the numbers in §7.3.
  * The source of every script is recoverable from the same transcripts: each was written by a `cat > … <<EOF`
    heredoc in a Bash call, `run_one.py` at 05:07:37Z and `score_multi2.py` at 05:25:39Z (agent ab42b538…),
    `tost_fullrun.py` at 05:52:16Z (a5dd312a…), `run_torchdraws.py` at 06:00:43Z (with the snapshot `cp`),
    `score_tdraw.py` at 06:08:45Z and `tost_tdraw.py` at 06:14:36Z (a527a5a5…).
* **The CUDA comparison data**, still in `results/lta_histogram/` (npz gitignored).
* **The runs themselves are deterministic** on a znver4 compile target (§7.5), so they can be rebuilt bit for bit
  for ≈ 13.6 core-h.

### 7.7 Differences between the reports and this record

* **Seed count.** `LTA_TIMESTEP_VALIDATION.md` §4 says "16+ seeds". The record is 32 numba seeds per (T, arm), 48
  torch-draw seeds at 300 K ABF, and 16 CUDA seeds per (T, arm).
  `IMPLEMENTATION_VALIDATION.md` §2's "not demonstrable with 16 seeds" refers to the 16-seed CUDA side.
* **What the ±5 % ratio compares.** "CPU torch vs numba gives 1.015 [0.969, 1.062]" is the I_F median ratio of
  numba's own draws over the torch-draw sample, with a 90 % CI, at 300 K ABF only.
* **Smoke-test max |Γ|.** `LTA_TIMESTEP_VALIDATION.md` §4 quotes max |Γ| = 29 / 21 kJ/mol/rad (300 / 150 K). Those
  are the N = 1024 values (29.2 ABF and FR at 300 K; 21.2 ABF, 21.3 FR at 150 K). The N = 1 ABF run reached 31.2 at
  300 K (above the reference max |dF/dφ| 28.9) and 20.2 at 150 K; all are well under the clip of 60.
* **Size of the occupancy shift.** "About 1 %" is the pooled CPU vs CUDA shift. numba's own draws vs the CUDA
  production is 1.7 % (p 0.003).
* **Which engine.** The equivalence runs predate the reviewed engine. §7.5 shows the arrays are identical at 300 K and 150 K
  (5 runs in all).

## 8. Best-allocation confirmation (`CONFIRM_BEST_ALLOCATION.md`)

* 256 runs: LTA 300 K N = 16 with seeds 30100–30131, LTA 150 K N = 2 with seeds 15100–15131, gateway N = 16 with
  seeds 8200–8263.
* Configs: `configs/equal_budget_v2/confirm_best_*.json`. Outputs:
  `results/equal_budget_v2/confirm_best_{lta_T300,lta_T150,gateway}/N<N>/` and `…/analysis/`.
* Cost 22.1 core-h.
* That tree was being written when this document was started. It is **not** in `DATA_MANIFEST.json`.
* Measured after it finished (reviewer, 2026-10-10 ~09:00 UTC, read-only): 256 run files `N*/s*_*.npz`, 203.2 MB
  in total (gateway 79, LTA 300 K 67, LTA 150 K 59 MiB per directory on disk), no checkpoints left. In git (commit
  f3dc7e2): each `ledger.csv`, `analysis/{summary.json,tables.md}` and the `analysis/run_metrics/` JSON; the run
  npz are not.
* Regenerate with `bash results/equal_budget_v2/logs/launch_confirm.sh`, or with `run_ladder.py --system <s>
  --config configs/equal_budget_v2/confirm_best_<s>.json`.

## 9. Manifest

```bash
python scripts/equal_budget/data_manifest.py           # (re)write results/equal_budget_v2/DATA_MANIFEST.json
python scripts/equal_budget/data_manifest.py --check   # re-hash and compare; exit 1 on any changed / missing / new file
```

**Contents** (2026-10-10T08:42Z): 1411 files, 1 163 358 703 bytes.
* Per file: sha256, bytes, mtime, `in_git`, (system, N, seed, method), and engine / engine_sha256 / git_commit /
  git_dirty / status from `meta_json`. A run whose N, seed or method in `meta_json` disagrees with its path gets a
  `PATH/META … MISMATCH` status and is not counted as complete (none does).
* Per system: counts, bytes, engines, statuses, and the planned jobs that are missing (none).
* Run time is about 2 s; the files were in the page cache.


---

*Update 2026-10-10 ~09:45 UTC (main session).* The LTA equivalence runs were **copied** for preservation, because
the scratchpad `/tmp` was 98 % full:
* `fullrun/*.npz` (128 runs) and `fix_lta/tdraw/*.npz` (48 runs), 133.7 MB with no checkpoint files;
* the runner and scorer scripts.

They are now in `results/equal_budget_v2/equivalence/`, with a `README.md` and a sha256 `MANIFEST.json`. The npz
files are gitignored, and the scripts, README and manifest are tracked. The scripts still carry their original
scratchpad paths.
