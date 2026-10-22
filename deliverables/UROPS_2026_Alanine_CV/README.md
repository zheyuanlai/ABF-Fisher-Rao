# Alanine dipeptide with an incomplete CV: UROPS figure data

This package contains figure data, previews and documentation for the alanine incomplete-CV experiments: φ alone
and ψ alone, histogram ABF vs ABF + uniform Fisher–Rao at rates 0.02 and 0.15, 16 seeds, N = 2048, 100 ps.

* Everything is post-processing of the six existing runs in `results/alanine_1d/` (`docs/ALANINE_1D_CV.md`). No new
  simulation was run.
* The only new computation is the copy-level bootstrap of the project's MBAR reference, propagated to the 1-D
  references, which took 185 s on an idle GPU.

| path | content |
|---|---|
| `figs/data/{phi,psi}_{eF,eFp,exi}.csv` | the 2 × 3 convergence figure, ABF vs FR 0.02 (UROPS 7-column schema) |
| `figs/data_supplementary/` | FR 0.15 pairs; ψ vs the φ < 0 conditional; mean-force variants (closed, dispersion, debiased); hidden-coordinate data (learned F(ψ), instantaneous and windowed φ > 0 mass, D_cond) |
| `figs/previews/*.pdf` | 2 × 3 figure (both rates), hidden-coordinate figure, mean-force diagnostics |
| `per_seed/<stage>__<metric>.csv` | per-seed series behind every curve |
| `data/reference/reference_1d.json` | 1-D references, central-difference mean force, bootstrap standard errors, reliability inputs |
| `meta/export.json`, `meta/validation_results.json` | statistics, provenance, validation |
| `PROVENANCE.md` | sources and definitions |
| `RESULTS_SUMMARY.md` | results |
| `VALIDATION_REPORT.md` | checks |
| `INSTRUCTIONS_FOR_CLAUDE.md` | caption and integration guidance |
| `scripts/` | reference propagation, export, previews, validation, packaging |

The archive `deliverables/UROPS_2026_Alanine_CV.zip` contains this directory without the raw files.

## Reproduce

Run from `deliverables/UROPS_2026_Alanine_CV/scripts`, with the raw files and reference in
`ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao` and the conda env `abffr`. Reference propagation first (≈ 3 min on
a GPU, slower on CPU):

```bash
CUDA_VISIBLE_DEVICES=3 python reference_1d.py --out ../data/reference
```

Then export, previews, validation and packaging:

```bash
python export_alanine_cv.py && python plot_previews.py && python validate_exports.py && python build_package.py
```
