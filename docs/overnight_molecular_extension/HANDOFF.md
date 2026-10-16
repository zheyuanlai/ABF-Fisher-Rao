# Handoff — overnight molecular extension

## Read first

`STATUS.md` (current stage), then `RESULTS.md`, then `CANDIDATES.md`.

## What exists

| system | code | data | decision |
|---|---|---|---|
| ethane / ERI 600 K | `src/eri/*`, `scripts/eri_campaign/*`, `tests/test_eri_*.py` | `results/eri_campaign/` (npz local only) | ABF-sufficient; no FR run |
| ethane / ITE 300 K | `scripts/overnight/*`, `configs/overnight/ite/`, `tests/test_overnight_ite.py` | `results/overnight/ite/` | ABF-sufficient; no FR run |
| ethane / ITE 150 K | `configs/overnight/ite150/`, `scripts/overnight/fig_ite150_reference.py` | `results/overnight/ite150/` | ABF-sufficient; no FR run |

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

## Next experiments (ranked; each needs a fresh preregistration)

1. **Early-phase FR test (recommended).**
   * In all three cells, 34–57 % of ABF's integrated error accrues before the window population is established at
     3–11 t.u.
   * Audit F3 correction: the share addressable AFTER the ramp is only 16–26 %, and ITE 300 K has the smallest
     (16–20 %). A meaningful test needs FR from t = 0, i.e. during the ramp, as a separately preregistered variant.
   * Design:
     * cell: ITE 150 K or ERI;
     * K 1024; two arms only;
     * FR start at the end of the ramp (t = 1 t.u.), or at t = 0 as a second preregistered variant, but not both
       without Holm;
     * FR rate by the frozen safety-only rule;
     * primary endpoint: Ī_F over [0, T] and over [0, 10 t.u.];
     * fresh seeds.
   * Eligibility must be declared in advance as an exploratory question ("does FR shorten the bias build-up phase?"),
     not by changing the frozen classifier after the fact.
   * Cost: about 3 core-h for 8 paired seeds plus about 5 for the rate calibration.
2. **ERI loading study** (2 and 5 molecules/cage) on GPU 3: a new torch engine, validated against the CPU engine.
3. **A labelled CH4-in-rigid-DDR model** (Dubbeldam 2004 CH4–O), as a new independently documented system.
4. **Candidate D** only with an explicit smooth-wall surrogate and a rigid-water integrator. It is expected to be
   ABF-sufficient.
