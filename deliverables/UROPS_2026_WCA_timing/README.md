# WCA dimer: Fisher–Rao activation time and early stopping under validated ABF

**Question.** In the validated Phase-V WCA confirmation, uniform marginal Fisher–Rao (FR) birth–death started at
t = 40 makes histogram ABF worse. Does FR help if it is switched on earlier? And can switching it off after the
early phase keep an early gain while avoiding the late penalty?

The setup keeps the validated physics:

* dt = 1.25 × 10⁻⁴, K = 256 replicas, T = 240;
* the exact Metropolis-MC reference;
* 16 fresh paired seeds, 5600–5615.

Seven arms are run:

* ABF;
* FR from t = 0, 5, 20 and 40;
* FR from t = 0 stopped at t = 20 or t = 40.

Design, rules and hypotheses are in `PREREGISTRATION.md` and `frozen_config.json`, committed (e25edb3e) and pushed
before any confirmation run. Results: `RESULTS_SUMMARY.md`. Checks: `VALIDATION_REPORT.md`. Figure-by-figure
provenance: `FIGURE_PROVENANCE.md`. Integration guidance for the report: `INSTRUCTIONS_FOR_CLAUDE.md`.

## Contents

| path | what |
|---|---|
| `data/threepanel/<prefix>_{eF,eFp,exi}.csv` | UROPS-compatible three-panel CSVs (`t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75`), one set per fixed FR schedule vs ABF: `wca_start0`, `wca_stop20`, `wca_stop40`, `wca_start40`, `wca_start5`, `wca_start20` |
| `data/per_seed/<arm>__<series>.csv` | per-seed series at the 133 saved checkpoints (e_F, e_F′, marginal full / interior, edge fraction, region fractions, replacements, genealogy, clip fraction) |
| `data/per_seed/scalars.csv` | per (arm, seed): window integrals, finals, time to accuracy, genealogy and safety scalars |
| `data/summary/analysis.json`, `contrasts.csv` | every paired contrast (median, 95 % and 97.5 % CIs, wins), verdicts, time to accuracy, genealogy, run checks |
| `data/summary/master_series.csv` | median / IQR series of every arm and series |
| `data/summary/reference_mc.json` | the frozen exact-MC reference |
| `data/summary/regression/`, `data/summary/tests/` | bitwise regression against the stored Phase-V trajectories; test-suite logs |
| `figures/*.pdf` | main three-panel figure, timing comparison, marginal vs free energy, two supplementary figures |
| `scripts/` | engine glue, campaign runner, analysis, export, figures, validation, packaging |

The ZIP `deliverables/UROPS_2026_WCA_timing.zip` contains this directory. Raw `.npz` files (≈ 0.1 GB) are not in git;
they are in `results/urops_2026_wca_timing/raw/` of the main research checkout.

## Reproduce

The engine is `src/wca_numba.py` on this branch, extended with `fr_stop_steps` / `fr_stream_ids`. Raw data and the
exact MC chains live in the main checkout (`ABFFR_DATA_ROOT`, default `/home/zheyuanlai/ABF-Fisher-Rao`). Use the
conda env `abffr` (Python 3.14, numba 0.68). Tests:

```bash
python -m pytest -q tests/test_wca_numba.py tests/test_wca_numba_sham.py tests/test_wca_numba_stop.py
```

Run from `deliverables/UROPS_2026_WCA_timing/scripts`. The commands below cover the regression against the
validated trajectories, the 112 confirmation runs (≈ 99 CPU core-h; resumable, refuses a dirty tree), and the
analysis, export, figure, validation and packaging steps:

```bash
python regression_validated_prefix.py --arm abf --out ../data/summary/regression/abf.json
```

```bash
python regression_validated_prefix.py --arm fr_start40 --out ../data/summary/regression/fr_start40.json
```

```bash
python run_campaign.py --workers 112
```

```bash
python analyze_campaign.py && python export_figures.py && python plot_figures.py && python validate_campaign.py && python build_package.py
```
