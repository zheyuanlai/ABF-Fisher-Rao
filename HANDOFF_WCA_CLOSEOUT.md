# WCA closeout — Part H: serial one-walker ABF at equal force-evaluation budget

Additive closeout on top of the WCA follow-up (Parts B–G). It answers the advisor/reviewer
fairness concern: an `N`-replica ABF/mFR run at time `T` spends `N×` the force evaluations of
ordinary single-walker ABF at time `T`; the fair control is a **serial one-walker ABF run at
the same total force-evaluation budget** `budget = n_replicas · n_steps`. For the base parallel
run `N_parallel=1024, nsteps_base=120000`, the exact equal-budget serial control is
`N_serial=1, nsteps_serial = 1024·120000 = 122,880,000` steps.

Nothing existing was modified: no OPES, no new physical systems, no notebook edits, and no
existing `results/**` raw dir or TI cache was touched. All new code is command-line runnable,
resumable/idempotent, and manifest-writing. **OPES may start only after this closeout is
reviewed.**

Environment: `conda activate abffr` (source `~/miniconda3/etc/profile.d/conda.sh` first).
GPU policy (enforced by the launcher): use ONLY GPUs 4–7, never 0–3, never more than two,
one process per GPU; `nvidia-smi` + the selected GPU list are printed before every launch.

---

## 1. What changed (all additive)

New files:
- `configs/wca_serial_abf_equal_budget.yaml` — base ABF knobs (identical to the equal-compute
  study), `serial_abf` method (N=1), per-cell target steps, ladder fractions, checkpoint
  cadence; stages `production`, `smoke`, `benchmark`.
- `src/wca_serial_abf.py` — dedicated serial engine. A batched per-trajectory ABF estimator
  (`BatchedKernelABFEstimator`, `num/den` of shape `(G, n_grid)`) advances `G` independent
  one-walker ABF trajectories (the seeds of one physics cell) together, each with its OWN
  accumulators, reusing `wca_abffr_core`'s validated force / geometry / mean-force / ABF-force
  / wall / region / reference / metric code. No Fisher–Rao / birth–death / oracle / adaptive.
  Chunked + checkpointed (`run_serial_abf_batched`); a checkpoint stores step, `q`, both ABF
  accumulators, per-trajectory crossings / region-time / z-marginal, torch CPU+CUDA RNG state,
  and the budget-ladder snapshot series. Partial-emit at every checkpoint writes per-seed npz
  (`complete=False`) so a still-running multi-day job is analyzable up to its accumulated
  budget; the final emit is `complete=True`.
- `scripts/run_wca_serial_abf.py` — resumable runner (`--dry-run`, `--benchmark`, `--smoke` via
  `--stage smoke`, `--cells/--seeds`, `--shard/--num-shards`, `--checkpoint-every`,
  `--overwrite`, `--max-steps`); one batched job per (cell, target); writes a manifest.
- `scripts/run_wca_serial_abf_h200.sh` — launcher; refuses GPUs outside 4–7 and >2 GPUs,
  prints `nvidia-smi` + selected GPUs, one process per GPU, shards cells across GPUs.
- `scripts/analyze_wca_serial_abf.py` — `serial_abf_run_summary.csv`, `serial_abf_summary.csv`,
  `serial_equal_compute_merged.csv` (serial + parallel ABF/mFR on one budget axis), and
  `report/tables/wca_serial_abf{,_numbers}.tex`.
- `scripts/plot_wca_serial_abf.py` — `report/figures/fig_wca_serial_abf_equal_budget.png`
  (per-cell panels; serial curve over budget + parallel ABF/mFR points).
- `results/wca_serial_abf/{raw,checkpoints,summaries,logs,figures}/`.

Modified (additive): `report/sections/06_case_wca.tex` (new subsection + Case-finding update),
`report/main.tex` (`\input` the serial numbers macros), and stub table/macros/figure so the
report always compiles.

---

## 2. Validation (all PASS, before production)

- **G=1 parity** vs `core.run_sampler_gpu("abf", n_replicas=1)`: identical to ~1e-7 relative on
  `l2_f`/`l2_fp`, barrier crossings 45=45 — the batched per-trajectory estimator reproduces the
  reference ABF engine.
- **Resume continuity**: a run stopped at step 6000 and resumed to 12000 is **bitwise identical**
  (every metric `d=0.00e+00`, `l2_f(t)` series `max|diff|=0`) to a straight 12000-step run — the
  checkpoint fully restores RNG + ABF accumulators + diagnostics.
- **Partial-emit**: checkpoint emits produce analyzable but non-`run_is_valid` (`complete=False`)
  npz, so the group still resumes; the final emit is `complete=True` and valid.
- **Smoke** (2 cells × 2 seeds, GPU 4): full runner → checkpoint (every 5k) → per-seed npz →
  analyze (3 CSVs) → plot → report compile round-trip.
- **Launcher guards**: GPU 0 refused; 3 GPUs refused.
- **Benchmark** (GPU 4, 300k steps, G=5): **1.428 ms/step** ⇒ exact NT (122.88M) ≈ **48.8 h**
  wall / batched job; 1/4 ladder (30.72M) ≈ **12.2 h** / job.

---

## 3. Production status

Launched detached (`setsid`, survives the session), resumable, two GPUs (both verified free at
launch: GPUs 0–3 busy, 4–7 idle):

| GPU | cells | seeds | target steps | est. wall | status |
| --- | --- | --- | --- | --- | --- |
| 4 | starved `b1,h2` | 0–9 (10) | 122,880,000 (exact NT) | ≈ 49–55 h | RUNNING |
| 5 | intermediate `b2,h6`, easy `b4,h1` | 0–4 (5 each) | 30,720,000 (1/4 ladder) | ≈ 24 h total | RUNNING |

Logs: `results/wca_serial_abf/logs/production_gpu{4,5}_*.log`. Checkpoints every 1,000,000 steps
under `results/wca_serial_abf/checkpoints/`; a per-seed partial npz (`complete=False`) is emitted
at each checkpoint, so the budget ladder is analyzable while the run is still going. Re-running
the same runner command (see §4) resumes from the last checkpoint without losing ABF
accumulators (validated bitwise).

**Exact serial NT completed?** Not within this session — the exact 122.88M-step starved control
is a ~2-day run (benchmark: 1.428 ms/step). It is running and resumable; the intermediate/easy
cells are deliberately capped at the 1/4-budget ladder cutoff (per the agreed scope) and can be
extended to full NT later by raising their `n_steps` in the config and re-running the runner
(which resumes). To continue/monitor, re-issue the §4 production commands (idempotent) and
re-run analyze/plot as partial data accrues.

---

## 4. Reproduce / continue

```bash
conda activate abffr

# dry-run + benchmark
python scripts/run_wca_serial_abf.py --config configs/wca_serial_abf_equal_budget.yaml --stage production --dry-run
CUDA_VISIBLE_DEVICES=4 python scripts/run_wca_serial_abf.py --config configs/wca_serial_abf_equal_budget.yaml --stage benchmark --benchmark

# production (resumable; re-running the SAME command continues from the last checkpoint):
CUDA_VISIBLE_DEVICES=4 python -u scripts/run_wca_serial_abf.py \
  --config configs/wca_serial_abf_equal_budget.yaml --stage production --cells b1_h2      # starved, exact 122.88M
CUDA_VISIBLE_DEVICES=5 python -u scripts/run_wca_serial_abf.py \
  --config configs/wca_serial_abf_equal_budget.yaml --stage production --cells b2_h6,b4_h1  # 1/4 ladder cutoff

# aggregate + figure + report (works on partial or complete data):
python scripts/analyze_wca_serial_abf.py --config configs/wca_serial_abf_equal_budget.yaml --stages production \
  --equal-compute results/wca_equal_compute/summaries/equal_compute_summary.csv
python scripts/plot_wca_serial_abf.py --config configs/wca_serial_abf_equal_budget.yaml --stages production \
  --equal-compute results/wca_equal_compute/summaries/equal_compute_summary.csv --report-figdir report/figures
cd report && tectonic -X compile main.tex
```

---

## 5. Scientific conclusion

The equal-budget serial one-walker ABF control tests whether ordinary ABF, run for the full
`N·T` force-evaluation budget by a single walker, closes the gap to mFR in the starved cell. If
it does not, the starved-regime mFR gain is the birth–death variance reduction, not raw compute;
if it does, mFR is a parallel finite-time accelerator rather than an asymptotic advantage. The
central WCA thesis stays conditional either way.

**Early partial reading (starved anchor, budget reached ≈ 1e6 of 1.2288e8 target).** The serial
walker descends quickly: at ~1/122 of the target budget its median `L2(F)≈0.085` already matches
base parallel ABF (`≈0.087` at the full 1.23e8 budget) and beats the `2048×60k` ABF shape
(`0.106`), but it remains ~2× above base-budget mFR (`0.041`). So at this point the serial
control has NOT closed the gap to mFR, but a large budget span (1e6 → 1.23e8) is still to be
covered; the verdict is deferred to the completed exact run. Re-run analyze/plot (§4) as the
budget accrues to refresh the report. The intermediate cell shows the same shape; the easy cell's
serial data begins after the intermediate cell finishes on GPU 5 (≈12 h).
