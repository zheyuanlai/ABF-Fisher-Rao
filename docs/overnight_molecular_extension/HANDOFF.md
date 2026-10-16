# Handoff — overnight molecular extension

## Read first

`STATUS.md` (current stage), then `RESULTS.md`, then `CANDIDATES.md`.

## What exists

| system | code | data | decision |
|---|---|---|---|
| ethane / ERI 600 K | `src/eri/*`, `scripts/eri_campaign/*`, `tests/test_eri_*.py` | `results/eri_campaign/` (npz local only) | ABF-sufficient; no FR run |
| ethane / ITE 300 K | `scripts/overnight/*`, `configs/overnight/ite/`, `tests/test_overnight_ite.py` | `results/overnight/ite/` | ABF-sufficient; no FR run |
| ethane / ITE 150 K | `configs/overnight/ite150/` | `results/overnight/ite150/` | in progress |

## Reproduce

```
# environment: conda env abffr, CPU only
export CUDA_VISIBLE_DEVICES="" NUMBA_NUM_THREADS=1
P=/home/zheyuanlai/miniconda3/envs/abffr/bin/python
$P -m pytest -q tests/test_eri_model.py tests/test_eri_engine.py tests/test_eri_metrics.py tests/test_overnight_ite.py
$P scripts/eri_campaign/build_frameworks.py            # ERI frameworks
$P scripts/overnight/build_ite.py && $P scripts/overnight/ite_channels.py && $P scripts/overnight/save_ite.py
# Gate 1 / Gate 2 (resumable):
ERI_REFERENCE_CFG=configs/overnight/ite/reference.json $P scripts/eri_campaign/run_reference.py --variant ite_itq3 --scheme MC --groups 0-15 --procs 32
ERI_REFERENCE_CFG=configs/overnight/ite/reference.json $P scripts/eri_campaign/run_reference.py --variant ite_itq3 --scheme EM --dt 1e-4 --groups 0-15 --procs 32
$P scripts/eri_campaign/analyze_reference.py --variant ite_itq3 --base results/overnight/ite/reference --T 300 --tol 0.03 --tolup 0.045
$P scripts/eri_campaign/reference_fine.py --base results/overnight/ite/reference/ite_itq3 --T 300
# Gate 3 + screen (job lists; the runner is resumable and checkpointed):
$P scripts/overnight/ite_jobs.py screen --dt 1e-4 --T 50 --ramp 1 --eq 5 --ng 180
NUMBA_NUM_THREADS=8 $P scripts/eri_campaign/run_jobs.py configs/overnight/ite/jobs_screen.json --procs 24
$P scripts/eri_campaign/classify_screen.py results/overnight/ite/screen --refbase results/overnight/ite/reference --variant ite_itq3 --hidden-tag EM_1e-04
```

## Unresolved / caveats

* No FR production run exists in this extension: no cell was eligible. Do not cite ERI or ITE as "FR-neutral".
* ITE is a labelled model. The literature model (flexible BKS framework + all-atom Oie ethane) is unrecoverable.
* Overdamped Langevin with unit mobility gives the correct equilibrium. Its time unit is not physical time, so
  diffusion coefficients are not comparable with the MD literature.
* The ERI loading study (2 and 5 molecules/cage) needs a GPU engine. CPU cost is about 6 core-h per N 256 run.

## Next experiments (ranked)

1. Finish A′-150 (ITE at 150 K): Gates 1–3 → screen → FR only if establishment-limited.
2. If nothing is eligible: run the ERI loading study on GPU 3 (a new torch engine, validated against the CPU engine),
   or a labelled CH4-in-rigid-DDR model (Dubbeldam 2004 CH4–O).
3. Candidate D (CH4 pair in a water-filled pore) only with an explicit smooth-wall surrogate and a rigid-water
   integrator. It is expected to be ABF-sufficient.
