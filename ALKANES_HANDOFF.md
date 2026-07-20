# Alkane torsion benchmark (Cases IV & V) — HANDOFF

Additive extension of the ABF vs marginal-Fisher–Rao (mFR) study to two united-atom
alkane torsion benchmarks: **butane** (one dihedral, easy control) and **pentane**
(two coupled dihedrals, hidden second coordinate). Nothing in the existing WCA / OPES
/ toy / report pipelines was modified except additive `\input`s and section files in
`report/`. Plan: [`docs/ALKANES_PLAN.md`](docs/ALKANES_PLAN.md).

## Environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate abffr
# torch 2.12+cu130, numpy 2.4, python 3.14, pytest 9
```

## GPU policy & compliance
- Exactly ONE physical GPU from {4,5,6,7}, never 0–3, never >1. Log:
  [`results/alkanes/GPU_ALLOCATION.txt`](results/alkanes/GPU_ALLOCATION.txt).
- Initial selection GPU 7; **re-selected to GPU 5** when another user (`yifanchen`)
  launched on GPU 7 mid-run (recorded with timestamp + reason). All GPU runs use
  `CUDA_VISIBLE_DEVICES=5` and `cuda:0`; production runner asserts
  `torch.cuda.device_count()==1` (`--require-single-gpu`).

## New files
```
src/alkanes/{__init__,geometry,potentials,cv,periodic,core,opes,reference,metrics,jobs}.py
configs/alkanes/{smoke,tuning,production}.yaml
scripts/run_alkanes.py            resumable runner (--config --stage --dry-run --overwrite --only-method --require-single-gpu)
scripts/run_alkanes_reference.py  B0/P0 gates + full references + convergence ladder
scripts/run_alkanes_frozen.py     Part-F frozen-bias validation
scripts/analyze_alkanes.py        summaries + matched-seed paired stats + equivalence
scripts/plot_alkanes.py           manuscript figures from artifacts
scripts/make_alkanes_report_assets.py  LaTeX tables + \def number macros
tests/test_alkanes_{geometry,periodic,cloning,noleak}.py
report/sections/{10_alkane_model,11_case_butane,12_case_pentane}.tex
docs/ALKANES_PLAN.md, ALKANES_HANDOFF.md
```
Modified (additive): `report/main.tex` (title, package, section + macro `\input`s),
`report/sections/{00_abstract,01_introduction,07_synthesis,08_discussion,09_conclusion}.tex`,
`report/refs.bib` (3 entries), `report/tables/alkanes_*.tex` (generated), `README.md`.

## Tests (all pass, CPU)
```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_alkanes_*.py -q
```
Cover: signed-dihedral range/continuity/invariances/reflection/known-conformers;
`place_chain` round-trip + equilibrium geometry; V4 minima/barriers; generalized-force
grad + `div(v)` vs finite differences (1e-11) and vs `torch.func.hessian` (1e-15);
periodic KDE/estimator/integration; cloning copies full config + fixed population +
genealogy + matched-seed dynamics; no-reference-leakage guards.

## Model (documented in code + manuscript §\ref{sec:alkane_model})
United-atom bonds (k0=1000,d0=1) + angles (kθ=208, θ0=1.187 bend ⇔ 112° interior) +
Ryckaert–Bellemans torsion (c1=1.18,c2=−0.23,c3=2.64) + LJ with RB exclusion (≤3 bonds
excluded). **Butane: no LJ pair ⇒ F(φ1)=V4+C exactly. Pentane: one 1–5 LJ pair,
σ=2.3 (physical united-atom ratio), giving the pentane effect** (g+g−/g−g+ disfavoured
~8 kT). Signed dihedral via atan2 (trans=0 RB convention); correct generalized mean
force `f_loc = ∇V·v − β⁻¹∇·v`, `v=∇φ/|∇φ|²`, geometric term by exact autodiff.

## Reference generation (evaluation only)
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes_reference.py --betas 0.5 1.0 2.0
```
Independent internal-coordinate FEP; B0/P0 decoupled gates recover V4(+V4)+C to <1e-9;
pentane 1-D marginal converged to ~1e-4 over the sample ladder. Report:
`results/alkanes/references/reference_validation.json`. NOTE: reference SEM is large
(~1 kT) only in the forbidden g+g− region (FEP overlap problem there), which has ~0
Boltzmann weight and does not affect metrics.

## Commands (per stage)
```bash
# Stage 0 smoke (pipeline check, all 5 methods, ~10 min)
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes.py --config configs/alkanes/smoke.yaml --stage smoke
# Stage 1/2 tuning (mFR rate ladder + OPES grid; disjoint tuning seeds 101-103)
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes.py --config configs/alkanes/tuning.yaml --stage tuning
python scripts/analyze_alkanes.py --config configs/alkanes/tuning.yaml
# Stage 3 production (FROZEN hyperparameters; seeds 1-16; rng_seed distinct from tuning)
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes.py --config configs/alkanes/production.yaml --stage b1 --require-single-gpu
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes.py --config configs/alkanes/production.yaml --stage b2 --require-single-gpu
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes.py --config configs/alkanes/production.yaml --stage p1 --require-single-gpu
# Frozen-bias validation
CUDA_VISIBLE_DEVICES=5 python scripts/run_alkanes_frozen.py --config configs/alkanes/production.yaml --stage b1
# Analysis + figures + report assets
python scripts/analyze_alkanes.py --config configs/alkanes/production.yaml
python scripts/plot_alkanes.py    --config configs/alkanes/production.yaml --stage production --report-figdir report/figures
python scripts/make_alkanes_report_assets.py
cd report && tectonic -X compile main.tex
```

## Tuning winners & selection criterion
Selection rule (pre-registered): smallest **median integrated L2(F)** on tuning seeds
(101–103), subject to stability + genealogy. **WINNERS (frozen before production):**
- mFR rate = **0.02** (the gentlest). The pentane rate ladder is monotone: integrated
  L2(F), the conditional TV, AND ancestor ESS all worsen as the rate rises (ESS
  353→139 from 0.02→0.40) — mFR has **no useful window** on either alkane. Best-final-L2
  selection agrees (0.02). A labelled `fr_active` (rate 0.20) probe is added to pentane
  production to show the harm at scale.
- OPES = **barrier=4, sigma=0.15, pace=400** (min median integrated L2(F) over the grid).
  Full ladder in `results/alkanes/tuning/summaries/` (`scripts/select_alkane_tuning.py`).

## Production seeds / budget
Production seeds 1–16 (B1/P1), 1–12 (B2); tuning seeds 101–103 — disjoint via distinct
`rng_seed` streams (production 20260719, tuning 77777777). N=384 replicas, dt=5e-4.
Per-step ~24 ms (H200, batch-flat); butane jobs ~15 min, pentane ~25–35 min.

## Headline numbers (16 production seeds; medians)
- **Butane β=1 (easy control):** mFR is **practically EQUIVALENT** to ABF — final L2(F)
  0.275 (mFR) vs 0.275 (ABF), relative change ~0.1 %, matched-seed bootstrap 95 % CI
  well inside the pre-registered ±10 % margin, win rate ≈0.5 (both trans and dispersed,
  both final and integrated L2). mFR event fraction ≈6e-6 (essentially off). **OPES is
  HARMFUL**: 0.353, +17–29 %, CI entirely above the margin, 0/16 seeds — its deposited
  bias converges slower than ABF's direct mean-force estimate at matched force budget.
- **Pentane β=1 (hidden torsion):** tuned mFR **EQUIVALENT** on BOTH the marginal
  (L2(F(φ1)) 0.287 vs ABF 0.284) and the hidden conditional (weighted TV p(φ2|φ1) 0.027
  vs 0.028); ancestor ESS stays ~90 %. φ1 is not starved, so mFR has nothing to fix and
  the coupling difficulty lives in the unbiased φ2. `fr_active` (rate 0.2) **harms**:
  marginal 0.301, ESS→180, conditional TV→0.038 (worse on every axis). OPES worse on the
  marginal (0.393) and at β=2 visits only 6/9 basins. Cold β=2 has a much harder φ2
  conditional (TV≈0.17) which no method relieves.
- **Frozen-bias (Part F):** freezing the learned B(φ1) and reconstructing
  F=B−β⁻¹log p_B recovers F far more accurately (recon L2 butane 0.05, pentane
  0.10/0.05 at β1/β2) than the online mean-force integral (0.27–0.29), and ABF≈mFR
  recon — so the equivalence is a property of the learned bias, and the ~0.27 online
  L2 is a mean-force-ESTIMATOR resolution floor (kernel bandwidth + grid) common to
  all methods, NOT a systematic error (the FEP reference matches V4 to 1e-9; barrier
  89% recovered with correct shape). Ran on CPU (GPUs 4-7 all taken by other users at
  that point); `results/alkanes/production/{b1,p1}_frozen/frozen_summary.csv`.
- Artifact paths: `results/alkanes/production/{raw,summaries,figures_production}`,
  `results/alkanes/tuning/summaries/`.

## Failed / descoped
- P2 (2-D (φ1,φ2) CV torus) descoped: the mandatory 1-D closure is prioritised and the
  1-D results already answer the healthy-vs-false question; documented, does not block
  closure. The periodic 2-D machinery (joint reference, torus grid) is implemented and
  tested; only the 2-D biased sampler is not run.
- B2 β=0.5 uses a shorter horizon (25k steps) since the hot, low-barrier cell converges
  quickly; β=2 uses 55k.

## Report build & git
`cd report && tectonic -X compile main.tex` → `report/main.pdf`. Git: additive; no
existing OPES/WCA/toy results modified. Raw `.npz` and logs are git-ignored (summaries/
tables tracked), consistent with the repo convention.

## Limitations
- Per-step cost is dominated by the exact autodiff Hessian of the dihedral (~24 ms/step
  on one H200, batch-flat); mitigated by seed-batching. `torch.compile` does not support
  the `vmap(hessian)` transform.
- mFR acts only on the biased coordinate φ1; it cannot directly drive the hidden φ2
  conditional in pentane (this is the scientific point, not a bug).
