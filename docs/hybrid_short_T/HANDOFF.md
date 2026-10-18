# Hybrid short-T wall-clock study: handoff

Preregistration: `docs/hybrid_short_T/PREREGISTRATION.md`. Config: `configs/hybrid_short_T/campaign.json`. Results:
`docs/hybrid_short_T/RESULTS.md`. Validation: `VALIDATION.md`. Benchmark: `BENCHMARK_RESULTS.md`.

## Environment (every command)

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate abffr
cd ~/ABF-Fisher-Rao
export CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=$HOME/.cache/numba_hst OMP_NUM_THREADS=1
```

Drivers and operator commands run on CCD 15 (`taskset -c 120-127`). The scheduler never allocates that CCD
(`hst_sched.RESERVED_CCDS`). Every timed run is a fresh process pinned by the scheduler to its own exclusive CCD set,
with the 1 Hz telemetry monitor of `scripts/final_campaign/expB_alloc.py`. The CCD lock registry is
`/tmp/claude-1008/final_campaign_alloc` and is shared with the other campaigns of this user. **No GPU is used.**

Long stages are launched detached:

```
setsid nohup taskset -c 120-127 python -u <script> ... > results/hybrid_short_T/logs/<name>.log 2>&1 < /dev/null &
```

Every stage is **resumable**: complete records with an unchanged signature are skipped; records left `ran` by a dead
driver are finalised; a job whose CCD lock is still held by a live orphan is never launched twice.

## Stages, in order

| stage | command | output |
|---|---|---|
| unit tests | `taskset -c 120-127 python -m pytest -q tests/test_hybrid_short_T.py` | 19 tests |
| seeds | `taskset -c 120-127 python scripts/hybrid_short_T/check_seeds.py` | `results/hybrid_short_T/seed_check.json` |
| I validation | `python -u scripts/hybrid_short_T/validate.py run --max-ccds 8` (re-runs only missing / ERROR checks; `--only V3_lta` etc.) | `results/hybrid_short_T/validation/` |
| II benchmark | `python -u scripts/hybrid_short_T/benchmark.py bench --max-ccds 6` | `results/hybrid_short_T/bench/<system>/<config>/r<r>_s<seed>.json` |
| II long-window supplement (A2.5) | `python -u scripts/hybrid_short_T/benchmark.py bench --long --max-ccds 6 --cap 300` | `bench/<system>/<config>_L/...` |
| II summary | `python scripts/hybrid_short_T/benchmark.py summary` and `... tables` | `bench/summary.json`, `bench/tables.md` |
| II routing freeze (ONCE; refuses to overwrite) | `python scripts/hybrid_short_T/benchmark.py freeze` | `results/hybrid_short_T/routing/routing.json` |
| III smoke | `python -u scripts/hybrid_short_T/run_campaign.py prod --max-seeds 2 --cap 300` then `python scripts/hybrid_short_T/smoke_check.py` | `results/hybrid_short_T/prod/<system>/<arm>/s<seed>.{json,npz}` |
| III production | `python -u scripts/hybrid_short_T/run_campaign.py prod --cap 300` | same; plan in `results/hybrid_short_T/plans/prod_plan.json` |
| IV exploration analysis + confirmation selection (ONCE) | `python scripts/hybrid_short_T/analyze.py --select` | `results/hybrid_short_T/analysis/{summary.json,tables.md,confirm_selection.json,curves_*.npz}` |
| III confirmation | `python -u scripts/hybrid_short_T/run_campaign.py confirm --cap 300` (design fixed in `plans/confirm_plan.json` on first run) | `results/hybrid_short_T/confirm/...` |
| IV final analysis | `python scripts/hybrid_short_T/analyze.py` | same files, with the confirmation |
| figures | `python scripts/hybrid_short_T/plot.py` | `figures/hybrid_short_T/{A..F}_*.{png,pdf}`, `legibility.json` |
| status | `python scripts/hybrid_short_T/run_campaign.py status` | counts and used core-h |
| manifest | `python scripts/hybrid_short_T/manifest.py` | `results/hybrid_short_T/SHA256SUMS.txt` |

**Compute ledger.** `results/hybrid_short_T/compute_ledger.csv` has one row per run process: threads × process wall
= used core-h. The scheduler refuses new launches once used + projected > the cap. The cap is **300 core-h**:
Amendment A3, a user instruction; campaign.json keeps its frozen 100 for the record. Pass `--cap 300`.

## Gotchas

* `scripts/mechanism/parallel_benchmark.py` sets `NUMBA_NUM_THREADS=1` at import. Never import it in a process that
  will start the LTA parallel engine with more than one thread. `validate.py one` restores the environment.
* The LTA parallel engine needs `NUMBA_NUM_THREADS >= threads`. `expB_alloc.omp_env` sets it to the allocation size.
* Other users' unpinned processes roam over every CCD, so a fully idle 16-core set is rare. Allocation takes the
  least-loaded set after 10 s (Amendment A1), and contention is judged by each run's telemetry (repeat if
  contended, at most 3 repeats).
* `.npz` files are gitignored. The JSON records (with the sha256 of their npz) and the manifest are tracked.

## State at hand-off (2026-10-18 15:40 UTC)

**Every stage is COMPLETE.**

| stage | outcome |
|---|---|
| Validation | 49/49 PASS |
| Benchmark | 1026 + 198 jobs; routing frozen 09:54:52 UTC |
| Production | 1408/1408 jobs |
| Confirmation | 576/576 jobs (design 3 arms × 32 seeds) |
| Analysis and figures A–F | final; legibility checks pass |
| Compute | 169.2 used core-h of the 300 cap; 0 H200-h |

**Result.** Best ABF + FR vs best ABF in wall-clock to the mid threshold is UNRESOLVED on all three systems in the
fresh-seed confirmation:

| system | R (W) | R (W_e2e) |
|---|---|---|
| gateway | 0.78 [0.49, 1.20] | 0.90 [0.74, 1.10] |
| LTA 300 K | 1.11 [0.85, 1.59] | 1.10 [0.85, 1.52] |
| LTA 150 K | 0.95 [0.76, 1.33] | 0.95 [0.79, 1.29] |

The gateway's exploratory advantage (0.72 [0.46, 0.95]) did not replicate. See `RESULTS.md`.

**Re-running the analysis only** (no new runs; about 1 min):
* `python scripts/hybrid_short_T/analyze.py`
* `python scripts/hybrid_short_T/plot.py`
* `python scripts/hybrid_short_T/manifest.py`
* `python scripts/hybrid_short_T/manifest.py --check` verifies the data.

`analyze.py --select` refuses to run again: the confirmation selection is written once.

**Open follow-ups, not started.**
* An H200 arm with a GPU kernel that is not launch-bound (graph capture or a fused kernel), validated against replay
  identity like V3, once a GPU is idle and authorised.
* A larger confirmation (≥ 128 seeds per arm) to resolve differences of about 20 %.
