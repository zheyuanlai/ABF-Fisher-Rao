# UROPS 2026 convergence-figure data: ABF vs ABF + marginal Fisher–Rao birth–death

This package holds the numerical data for the seven three-panel convergence figures of the UROPS report
*Accelerating Adaptive Biasing Force with Marginal Fisher–Rao Birth–Death Dynamics*. Each figure has three panels:

* (a) the free-energy error e_F(t);
* (b) the mean-force error e_F′(t);
* (c) the reaction-coordinate marginal error e_ξ(t) = ‖p_t − u‖_{L²}.

Each panel compares ABF with ABF + uniform FR on the same paired seeds.

| figure | system | data status |
|---|---|---|
| `gateway` | entropic gateway, validated h = 2.5e-5 | original runs, post-processed |
| `lta150`, `lta300` | ethane / rigid LTA, 150 K and 300 K, exact MC reference | original runs, post-processed |
| `ite300` | ethane / Si-ITE 300 K, confirmatory early-stopping FR (E1) | original runs, post-processed |
| `meta` | energetic metastability toy (2-D Gaussian–quartic, ξ = x, β = 4) | **new preregistered experiment** |
| `alanine` | alanine dipeptide, (φ, ψ), adaptive-box histogram ABF | original runs, post-processed |
| `wca` | WCA dimer at the validated dt* = 1.25e-4 (negative control) | original runs, post-processed |

## Contents

| path | what |
|---|---|
| `figs/data/*.csv` | **the 21 deliverable CSVs** (`t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75`) |
| `figs/data_supplementary/` | two clearly labelled diagnostic marginals (alanine KDE, WCA interior) |
| `figs/previews/*.pdf` | matplotlib previews of the seven figures (checking only) |
| `per_seed/` | per-seed error curves (the unit of inference) behind every CSV |
| `meta/<fig>.json` | machine-readable configuration, provenance, definitions, units and reproduced statistics |
| `FIGURE_DATA.md` | CSV format, units, time grids, floors, panel-(c) types, genuine features |
| `FIGURE_PROVENANCE.md` | one entry per figure: runs, commits, configs, seeds, knobs, reference, definitions, statistics, caveats |
| `RESULTS_SUMMARY.md` | the validated paired findings (integrated and final errors) |
| `VALIDATION_REPORT.md` | every automated check (PASS / FAIL) and the manual audit notes |
| `INSTRUCTIONS_FOR_CLAUDE.md` | integration guidance for the manuscript (caption qualifications) |
| `manifest.json` | provenance summary + sha256 of every package file |
| `metastability/` | preregistration, frozen selections and stage results of the new experiment |
| `scripts/` | exporters, validation, previews, and the new experiment's engine, tests and stage runners |

The archive `deliverables/UROPS_2026_figure_data.zip` contains this whole directory (`UROPS_2026/…`).

## Reproduce

The raw simulation files (`*.npz`) are git-ignored and live in the main research checkout. Point
`ABFFR_DATA_ROOT` at it; the scorers are imported from its `src/` and `scripts/`. All runs used the conda env
`abffr` (Python 3.14, numpy 2.4, numba 0.68).

```bash
cd deliverables/UROPS_2026/scripts && ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python export_all.py
```

That command:

1. runs the seven exporters;
2. runs `validate_exports.py`, `plot_previews.py` and the manifest;
3. builds the ZIP.

It exits non-zero if any validation check fails. Individual figures: `python export_<name>.py`.

The metastability experiment, from scratch (≈ 6 CPU core-hours; each stage writes `metastability/FROZEN.json`):

```bash
python -m pytest -q test_meta_engine.py
python meta_validate_dt.py --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/validation --procs 64
python meta_pilot.py --stage resolution --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/pilot_resolution
python meta_pilot.py --stage rate --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/pilot_rate
python meta_confirm.py --out $ABFFR_DATA_ROOT/results/urops_2026/metastability/confirm --procs 64
```

`FROZEN.json` is committed, and every stage script refuses to overwrite a frozen stage. To repeat Stages 1–3,
delete `FROZEN.json` (it is rewritten stage by stage); to repeat only the confirmation, keep it and use an empty
output directory (existing run files are never overwritten).

## Integrate into the report

Copy `figs/data/*.csv` into `UROPS-report/figs/data/`. The file names match the report's `\ConvergenceFigure`
interface (`<fig>_eF.csv`, `<fig>_eFp.csv`, `<fig>_exi.csv`). Caption qualifications that must accompany each
figure are in `INSTRUCTIONS_FOR_CLAUDE.md`.
