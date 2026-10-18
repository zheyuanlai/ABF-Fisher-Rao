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
| II summary | `python scripts/hybrid_short_T/benchmark.py summary` and `... tables` | `bench/summary.json`, `bench/tables.md` |
| II routing freeze (ONCE; refuses to overwrite) | `python scripts/hybrid_short_T/benchmark.py freeze` | `results/hybrid_short_T/routing/routing.json` |
| III smoke | `python -u scripts/hybrid_short_T/run_campaign.py prod --max-seeds 2` | `results/hybrid_short_T/prod/<system>/<arm>/s<seed>.{json,npz}` |
| III production | `python -u scripts/hybrid_short_T/run_campaign.py prod` | same |
| IV exploration analysis + confirmation selection (ONCE) | `python scripts/hybrid_short_T/analyze.py --select` | `results/hybrid_short_T/analysis/{summary.json,tables.md,confirm_selection.json,curves_*.npz}` |
| III confirmation | `python -u scripts/hybrid_short_T/run_campaign.py confirm` | `results/hybrid_short_T/confirm/...` |
| IV final analysis | `python scripts/hybrid_short_T/analyze.py` | same files, with the confirmation |
| figures | `python scripts/hybrid_short_T/plot.py` | `figures/hybrid_short_T/{A..F}_*.{png,pdf}`, `legibility.json` |
| status | `python scripts/hybrid_short_T/run_campaign.py status` | counts and used core-h |
| manifest | `python scripts/hybrid_short_T/manifest.py` | `results/hybrid_short_T/SHA256SUMS.txt` |

**Compute ledger.** `results/hybrid_short_T/compute_ledger.csv` has one row per run process: threads × process wall
= used core-h. The scheduler refuses new launches once used + projected > the cap of 100 core-h (`--cap` overrides
it; that is an amendment).

## Gotchas

* `scripts/mechanism/parallel_benchmark.py` sets `NUMBA_NUM_THREADS=1` at import. Never import it in a process that
  will start the LTA parallel engine with more than one thread. `validate.py one` restores the environment.
* The LTA parallel engine needs `NUMBA_NUM_THREADS >= threads`. `expB_alloc.omp_env` sets it to the allocation size.
* Other users' unpinned processes roam over every CCD, so a fully idle 16-core set is rare. Allocation takes the
  least-loaded set after 10 s (Amendment A1), and contention is judged by each run's telemetry (repeat if
  contended, at most 3 repeats).
* `.npz` files are gitignored. The JSON records (with the sha256 of their npz) and the manifest are tracked.
