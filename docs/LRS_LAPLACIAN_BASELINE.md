# LRS 2007 Laplacian selection: one published baseline on existing FR-positive cells

**Status (2026-10-31): implementation and run specifications committed, not yet executed or validated on HPC.**
This is a **new, opt-in Laplacian-only** extension. Historical ABF/FR runs, frozen results, references and
their analyses must not be overwritten.

## Scope

Exactly one additional selection algorithm: the Laplacian fitness of Lelièvre,
Rousset and Stoltz (2007), and Chapter 6 of *Free Energy Computations: A Mathematical Perspective*:

```text
book fitness      S(z) = c(t) p''(z) / p(z)       (positive = birth)
implemented score score(z) = -a p''(z) / p(z)    (positive = death)
effective c(t)    = original rate(t) * a
```

The finite-population code uses the **existing** smoothed instantaneous CV marginal,
replaces the score while retaining the same selection opportunity, event cap,
random-number streams, full-replica cloning, ABF estimator and active stopping
controller as the corresponding historical FR arm. The amplitude `a` is
`laplacian_scale` (units: CV-coordinate squared). The score is centered,
clipped and re-centered, as in the original FR implementation.

The modified algorithm is therefore an **LRS Laplacian-score baseline under
our common selection/resampling implementation**, not an exact reproduction of
the paper's separate particle-clock discretization. No other score or baseline
has been added.

Boundary conditions matter:
- **Gateway**: reflecting [-1.8, 1.8], Neumann ghost cells on the
  existing reflected KDE grid.
- **LTA and ITE**: 2π-periodic collective-variable angle; periodic
  centered second differences on the same KDE as FR.

The constant-density score must be zero, and a local density maximum must have
positive *implemented* score (increased death probability): both are tested.

## Existing positive benchmarks to extend

| Case | System | K | Seeds | Historical positive |
|---|---|---:|---|---|
| Gateway 512 | Entropic gateway | 512 | 8100–8131 | Equal-budget gateway FR positive at fixed K |
| Gateway 2048 | Entropic gateway | 2048 | 8100–8131 | Equal-budget gateway FR positive at fixed K |
| R2C3 | Ethane/LTA 150 K | 1024 | 1420000–1420015 | R2 positive |
| R2C4 | Ethane/LTA 300 K | 1024 | 1430000–1430015 | R2 positive |
| R2H80 | Ethane/LTA 80 K | 1024 | 1450000–1450015 | R2 positive |
| R2H350 | Ethane/LTA 350 K | 1024 | 1460000–1460015 | R2 positive |
| R2C1 | Ethane/ITE 300 K | 1024 | 1400000–1400015 | R2 positive |
| R2C2 | Ethane/ITE 150 K | 256 | 1410000–1410015 | R2 positive |

**No WCA, deca-alanine, entropic bottleneck, neutral ITE/ERI, or other arms** are
added here. The eight cells are existing FR-positive examples selected before
looking at any Laplacian data.

The gateway runs preserve the exact ABF+FR number of walker-steps, save grid
and per-cell force-evaluation budget from
`configs/equal_budget_v2/gateway_production.json`. The LTA/ITE jobs clone only
the *FR* entry of the corresponding `configs/adaptive_fr/jobs_R2*.json`,
substituting `method=laplacian`. The original online R2 stop applies to the
Laplacian arm as a common modern comparator, not as part of LRS 2007.

## Requirements before production / scientific interpretation

**The configured `laplacian_scale = 0.01` is an UNVALIDATED initial value**, not
a tuned or evidence-backed selection intensity. It must not be reported as
the best LRS setting. A curvature score can have a much larger numerical range
than the FR log ratio; setting the two algorithmic rate parameters equal is
not a fair dose match.

1. Run `pytest -q tests/test_lrs_laplacian.py` on the production environment
   (plus relevant existing Numba/ERI/LTA validation suites). Smoke each arm
   and check force-evaluation parity with the matching historical pair,
   unchanged pre-selection dynamics and valid score/event signs.
2. Use **fresh, nonproduction pilot seeds** to examine a *small*
   `laplacian_scale` ladder (e.g. 0.003, 0.01, 0.03, 0.10), while fixing the
   published `p''/p` rule, KDE width, score clip, selection cadence, R2
   controller and ABF dynamics. Choose the scale with a predeclared
   safety/activity criterion, **without using free-energy errors** (avoid
   choosing the baseline's rate in its favour or against it).
3. Report realised replacements per replica, per-opportunity turnover,
   windowed ESS/K and largest family share. A baseline with negligible
   events or collapsed genealogy is not an interpretable head-to-head.
4. Freeze the selected scale *and* the comparison metric before producing
   any new Laplacian outcome; store the pilot results and config SHA.
5. Compare all arms at fixed K and identical physical horizon / force
   evaluations; only after that consider a best-allocation comparison.

Gate/score coupling caveat: the discrete centered/clip score only
approximates the continuum `c(t) p''/p` mean field. Check the
effective rate and stability when the KDE is small.

## NSCC execution (2026-11-01): prerequisites, pilot rule, job plan

**Where the data are.** The GitHub tree carries no `*.npz` (gitignored), so on the NSCC scratch clone the archived
raw runs of the eight cells (`results/r2_campaign/raw/*`, `results/equal_budget_v2/gateway/N*/`), the ITE framework
(`cache/eri/framework_ite_itq3.npz`), the exact-MC ITE references (`results/overnight/ite*/reference/ite_itq3/
reference_profile.npz`, `reference_fine.npz`) and the 80 / 350 K LTA references are all absent.  Only the tracked
analysis JSONs, the gateway and LTA 150 / 300 K references and the LTA framework exist.  Everything missing is
regenerated **from the frozen configs and seeds, before any Laplacian production result is looked at**:

| item | how | check against the archive |
|---|---|---|
| ABF and FR arms, 8 cells | `make_jobs.py rerun` -> `configs/laplacian_baselines/rerun_*.json`, archived seeds and configs through the unchanged `run_lta_jobs.job` / `run_jobs.job` / gateway `run_job`, outputs under `results/laplacian_baselines/raw/` | per-seed e_F(T), replacements and S_geo vs `results/r2_campaign/analysis/*.json` and `results/equal_budget_v2/gateway/analysis/run_metrics/` (`reproducibility.py`) |
| ITE framework | `scripts/overnight/save_ite.py` from the tracked CIF | geometry (P, x_window, cages, windows) equals the tracked `framework_ite_itq3.json` |
| ITE references 300 / 150 K | frozen `configs/overnight/ite*/reference.json` protocol (seed bases 8100000 / 8200000), `run_reference.py` MC -> `analyze_reference.py` -> `reference_fine.py`, into `results/laplacian_baselines/references/ite300|ite150/` | Gate 1 verdicts, noise RMS and barrier vs the tracked `gate1.json` |
| LTA 80 / 350 K references | sweep protocol (`run_lta_reference.py`, 40 windows, kappa = kT/0.07^2, steps scaled by 300/T) on one A100, converted to the `lta_T{T}_reference.npz` layout with gamma_ref by central differences (the archive's 80 / 350 K references were finite-difference-referenced, PREREGISTRATION.md section 4) | same protocol at 150 / 300 K vs the exact-MC references (RMS) |

The rebuilt arms are *re-runs*, not the archived files: the engines are deterministic given seed and code, but the
host differs, so the re-run vs archive comparison is reported rather than assumed.  The three-arm comparison itself
is internal to NSCC (same host, same code, same seeds, identical force evaluations).

**Pilot calibration (rule fixed before any pilot ran; `scripts/laplacian_baselines/pilot_decide.py`).**  Fresh seeds
(gateway 8300-8307; LTA / ITE 195x000-195x007, never used by any cell), the archived FR configuration (rate, R2
controller) unchanged, FR arm plus the Laplacian arm at the ladder {0.003, 0.01, 0.03, 0.1, 0.3}, full production
budget.  Read per run, reference-free: realised replacements R, minimum windowed ESS/K, maximum windowed
largest-family share, controller stop reason.  Per cell (the two gateway N are two cells):

1. *safety*: median over the 8 seeds of min windowed ESS/K >= 0.15 and of max family share <= 0.10 (the frozen R2
   safety thresholds, `essw_min` / `maxlin_min`);
2. *activity*: among the scales passing 1, the one whose median R is closest in log-ratio to the FR arm's median R on
   the same seeds (dose matching on realised selection activity; the FR log-ratio score and the curvature score have
   different units, so equal nominal rates are not a fair match);
3. fallbacks: no scale passes 1 -> the smallest scale, flagged `NO_SAFE_SCALE`; the FR arm itself fails 1 -> rule 2
   alone, flagged `FR_UNSAFE`; ties -> the smaller scale.

No free-energy error enters the choice.  The decision, the pilot table and the config SHA are stored in
`results/laplacian_baselines/pilot/decision.{json,md}`; `make_jobs.py apply` freezes the scale into
`configs/laplacian_baselines/jobs_*.json` (per job cfg) and `gateway.json` (`laplacian_scale_by_N`).

**Jobs** (`scripts/laplacian_baselines/nscc/`, `docs/experiments.md`): `validate.sh` (tests, ITE framework, smokes)
-> `pilot_rerun.sh` (re-runs + pilot ladder, one pool of 128 single-threaded processes) -> decision ->
`production.sh` (Laplacian arms) -> `analysis.sh`; `ite_reference.sh` and the GPU `lta_reference` task supply the
missing references in parallel.

## Run commands (after the pilot calibration and validations)

**Do not treat these commands as already executed.**

```bash
# One Laplacian-only gateway smoke
NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --dry-run
NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --only-N 512 --seeds 8100 --workers 1

# Two gateway positive replica counts, 32 matched archived seeds each
NUMBA_NUM_THREADS=1 python scripts/laplacian_baselines/run_gateway.py --workers 16

# Four positive LTA temperature cells, 16 archived paired seeds each
for b in R2C3 R2C4 R2H80 R2H350; do
  NUMBA_NUM_THREADS=1 python scripts/adaptive_fr/run_lta_jobs.py \
    configs/laplacian_baselines/jobs_${b}.json --procs 16
done

# Two positive ITE cells, 16 paired archived seeds each
for b in R2C1 R2C2; do
  NUMBA_NUM_THREADS=1 python scripts/eri_campaign/run_jobs.py \
    configs/laplacian_baselines/jobs_${b}.json --procs 16
done

# Pair the new Laplacian runs with original ABF/FR runs; validated references
python scripts/laplacian_baselines/analyze.py --case all

# Original R2 geometric sustained time-to-accuracy metric (LTA/ITE cases)
bash scripts/laplacian_baselines/analyze_r2.sh
```

All output under `results/laplacian_baselines/`. The historical files in
`results/r2_campaign/` and `results/equal_budget_v2/` remain read-only.

The analysis reports:
- Per-seed and seed-bootstrap uncertainty for time-integrated `e_F`
  (200 uniformly spaced budget fractions), final `e_F(T)`,
  integrated `e_{F'}`, and final `e_{F'}(T)`;
- Pairwise effects Laplacian/ABF, FR/ABF and Laplacian/FR at the exact
  same seeds and force-evaluation counts;
- Realised replacement counts and genealogy safety measures.

The shell wrapper `scripts/laplacian_baselines/analyze_r2.sh` calls the **unchanged** R2 analyzer on the new Laplacian arm (using its `--fr-glob` override), producing the original geometric sustained time-to-accuracy speedup against archived ABF. Its internal `fr` label denotes the *second arm*, which in these runs is **Laplacian**, not Fisher–Rao. Files go to a new analysis path, never `results/r2_campaign/analysis/`.

## Code locations

- `src/gateway_ladder_numba.py`: `laplacian_scores`
- `src/lta_ladder_numba.py`: `_laplacian_score`
- `src/eri/engine.py`: reuse LTA periodic Laplacian score
- `configs/laplacian_baselines/`: new positive-cell job specifications only
- `scripts/laplacian_baselines/run_gateway.py`: gateway runner
- `scripts/laplacian_baselines/analyze.py`: historical ABF/FR vs new Lap paired analysis
- `scripts/laplacian_baselines/analyze_r2.sh`: unchanged R2 sustained-speedup metric against ABF
- `tests/test_lrs_laplacian.py`: numerical sign / BC / zero-rate checks

## Interpretation limits

A successful Laplacian benchmark tests whether **the log-ratio FR score**
outperforms the older LRS curvature score under our common resampling
implementation. It does **not** establish superiority over every published
selection protocol, nor an equal-aggregate-cost or wall-clock advantage.
No numerical FR-versus-Laplacian comparison is available until the jobs run.

## NSCC results (2026-11-01): primary comparison, mechanism diagnostic, exploratory bandwidth ladder

The primary three-arm tables (predeclared protocol, frozen scales) are in
`results/laplacian_baselines/analysis/SUMMARY.md` (sections 1-7, written by `analysis.sh`, final run: job 26075544,
2026-10-31 21:42 UTC) and are summarised first.  Two further analyses were done on NSCC the same day and are reported
after it because they bear on how the primary result should be read.

### Primary three-arm comparison (predeclared protocol, frozen scales; `analysis.sh` -> `analysis/SUMMARY.md`)

Three arms at identical seeds, budgets and save grids: ABF, ABF + Fisher-Rao (archived configuration), ABF + LRS
Laplacian selection at the scale frozen by the fresh-seed pilot rule (0.01 everywhere except 0.03 at LTA 80 K and
350 K).  The ABF and FR arms are the NSCC re-runs of the archived seeds, not the archived files (see the reproducibility
note below); 16 paired seeds per LTA / ITE cell, 32 per gateway cell.  Errors come from the frozen `eqb_metrics`
scorers against the regenerated references (ITE 300 K and 150 K exact MC, Gate 1 PASS and equal to the archived
`gate1.json` to 1e-10 relative; LTA 80 / 150 / 350 K umbrella sweeps; gateway analytic).  G = (second - first) /
first per seed, median with seed-bootstrap 95 % CI, (wins / n) = seeds where the second arm has the lower error;
negative favours the second arm.  The last two columns of each row are the PR's main comparison (bold).

| cell | scale | I_F: FR vs ABF | I_F: Laplacian vs ABF | I_F: Laplacian vs FR | e_F(T): FR vs ABF | e_F(T): Laplacian vs ABF | e_F(T): Laplacian vs FR |
|---|---:|---|---|---|---|---|---|
| gateway K512 | 0.01 | -41.6 % [-49.0, -34.7] (32/32) | +4.9 % [-5.3, +11.9] (12/32) | **+76.5 % [+65.8, +85.2] (0/32)** | -47.0 % [-64.3, -26.2] (28/32) | +22.0 % [+12.5, +33.9] (7/32) | **+161.1 % [+73.0, +217.6] (5/32)** |
| gateway K2048 | 0.01 | -31.5 % [-34.4, -28.0] (31/32) | -4.4 % [-9.3, -1.3] (23/32) | **+43.1 % [+33.2, +48.3] (1/32)** | -64.1 % [-68.6, -57.3] (32/32) | +2.7 % [-8.2, +6.4] (15/32) | **+167.7 % [+134.4, +232.2] (0/32)** |
| LTA 80 K K1024 | 0.03 | -33.6 % [-35.5, -31.7] (16/16) | +2.0 % [-0.5, +3.7] (5/16) | **+52.3 % [+49.6, +58.9] (0/16)** | -82.6 % [-84.6, -78.5] (16/16) | +0.4 % [-6.3, +10.3] (7/16) | **+468.9 % [+385.5, +504.4] (0/16)** |
| LTA 150 K K1024 | 0.01 | -30.2 % [-32.0, -28.1] (16/16) | +0.5 % [-1.3, +1.6] (7/16) | **+42.9 % [+39.2, +49.1] (0/16)** | -57.9 % [-68.0, -54.7] (16/16) | -1.2 % [-4.0, +3.7] (9/16) | **+149.6 % [+114.4, +213.4] (0/16)** |
| LTA 300 K K1024 | 0.01 | -15.9 % [-17.9, -12.0] (16/16) | +0.0 % [-1.2, +1.0] (8/16) | **+18.2 % [+12.3, +22.7] (0/16)** | -24.1 % [-40.4, -12.2] (16/16) | +0.1 % [-5.8, +2.6] (8/16) | **+31.8 % [+13.0, +57.3] (0/16)** |
| LTA 350 K K1024 | 0.03 | -14.3 % [-16.3, -11.7] (16/16) | -0.0 % [-1.7, +1.7] (8/16) | **+15.3 % [+11.4, +19.1] (0/16)** | -21.9 % [-29.0, -14.1] (15/16) | -0.9 % [-13.2, +18.9] (8/16) | **+26.1 % [+18.1, +34.9] (2/16)** |
| ITE 300 K K1024 | 0.01 | -28.4 % [-30.9, -26.0] (16/16) | -0.0 % [-0.9, +1.6] (9/16) | **+40.5 % [+34.6, +45.9] (0/16)** | -40.9 % [-48.6, -29.5] (16/16) | -0.7 % [-8.4, +4.9] (9/16) | **+66.8 % [+42.8, +88.7] (1/16)** |
| ITE 150 K K256 | 0.01 | -22.0 % [-26.8, -15.7] (16/16) | -0.3 % [-3.0, +2.8] (8/16) | **+29.6 % [+21.3, +37.7] (1/16)** | -28.2 % [-36.0, -16.0] (14/16) | -4.9 % [-13.4, +12.4] (9/16) | **+38.4 % [+9.7, +52.0] (3/16)** |

Mean-force errors (I_F', e_F'(T)) tell the same story and are in `SUMMARY.md` section 2.

| cell | arm | replacements | min windowed ESS/K | max family share | established | t_stop |
|---|---|---:|---:|---:|---:|---:|
| gateway K512 | FR | 9454 | 0.350 | 0.037 | n/a | n/a |
| gateway K512 | Laplacian | 9990 | 0.655 | 0.014 | n/a | n/a |
| gateway K2048 | FR | 5349 | 0.348 | 0.016 | n/a | n/a |
| gateway K2048 | Laplacian | 4466 | 0.819 | 0.002 | n/a | n/a |
| LTA 80 K K1024 | FR | 491 | 0.716 | 0.007 | 16/16 | 31.25 (T = 60) |
| LTA 80 K K1024 | Laplacian | 698 | 0.894 | 0.004 | 16/16 | 74.25 (> T = 60 in 16/16) |
| LTA 150 K K1024 | FR | 362 | 0.603 | 0.008 | 16/16 | 18.50 (T = 60) |
| LTA 150 K K1024 | Laplacian | 252 | 0.923 | 0.003 | 16/16 | 33.75 (T = 60) |
| LTA 300 K K1024 | FR | 197 | 0.759 | 0.005 | 16/16 | 14.50 (T = 60) |
| LTA 300 K K1024 | Laplacian | 119 | 0.933 | 0.003 | 16/16 | 18.25 (T = 60) |
| LTA 350 K K1024 | FR | 178 | 0.782 | 0.005 | 16/16 | 14.00 (T = 60) |
| LTA 350 K K1024 | Laplacian | 330 | 0.820 | 0.004 | 16/16 | 16.50 (T = 60) |
| ITE 300 K K1024 | FR | 582 | 0.324 | 0.019 | 16/16 | 5.00 (T = 50) |
| ITE 300 K K1024 | Laplacian | 643 | 0.695 | 0.005 | 16/16 | 8.50 (T = 50) |
| ITE 150 K K256 | FR | 116 | 0.499 | 0.031 | 16/16 | 8.50 (T = 100) |
| ITE 150 K K256 | Laplacian | 140 | 0.696 | 0.018 | 16/16 | 14.75 (T = 100) |

Archived R2 metric (sustained time-to-accuracy speedup S_geo, unchanged analyzer; > 1 favours the second arm):

| cell | FR vs ABF, archive (2026-10-17) | FR vs ABF, NSCC re-run | Laplacian vs ABF, NSCC |
|---|---|---|---|
| LTA 80 K K1024 | 1.84 [1.64, 2.19] SUPPORTED (15/16 faster) | 1.84 [1.64, 2.11] SUPPORTED (15/16 faster) | 0.93 [0.83, 1.00] UNRESOLVED (4/16 faster) |
| LTA 150 K K1024 | 2.45 [2.26, 2.61] SUPPORTED (16/16 faster) | 2.45 [2.26, 2.61] SUPPORTED (16/16 faster) | 0.99 [0.97, 1.02] UNRESOLVED (7/16 faster) |
| LTA 300 K K1024 | 1.39 [1.34, 1.51] SUPPORTED (16/16 faster) | 1.39 [1.30, 1.52] SUPPORTED (16/16 faster) | 0.99 [0.98, 1.02] UNRESOLVED (7/16 faster) |
| LTA 350 K K1024 | 1.29 [1.25, 1.45] SUPPORTED (16/16 faster) | 1.34 [1.23, 1.42] SUPPORTED (16/16 faster) | 1.01 [0.95, 1.05] UNRESOLVED (10/16 faster) |
| ITE 300 K K1024 | 1.80 [1.55, 2.04] SUPPORTED (16/16 faster) | 1.79 [1.57, 2.06] SUPPORTED (16/16 faster) | 1.00 [0.97, 1.06] UNRESOLVED (9/16 faster) |
| ITE 150 K K256 | 1.44 [1.29, 1.70] SUPPORTED (16/16 faster) | 1.56 [1.36, 1.72] SUPPORTED (16/16 faster) | 1.01 [0.97, 1.05] UNRESOLVED (10/16 faster) |

Scale sensitivity on the fresh-seed pilot (8 seeds per cell, Laplacian vs FR, e_F(T); the best of the five scales is shown):

| cell | scales | best scale by e_F(T) | G e_F(T): Laplacian vs FR at that scale | G I_F at that scale |
|---|---|---:|---|---|
| gateway K512 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.01 | +79.0 % [+46.2, +90.4] (1/8) | +77.4 % [+68.8, +175.2] (0/8) |
| gateway K2048 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.003 | +104.1 % [+78.1, +213.4] (0/8) | +23.9 % [-2.7, +70.2] (2/8) |
| LTA 80 K K1024 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.003 | +422.6 % [+326.4, +1078.2] (0/8) | +52.8 % [+50.4, +55.7] (0/8) |
| LTA 150 K K1024 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.003 | +158.2 % [+92.5, +260.9] (0/8) | +43.5 % [+39.4, +48.5] (0/8) |
| LTA 300 K K1024 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.03 | +38.1 % [+17.3, +81.3] (0/8) | +20.9 % [+14.4, +26.1] (0/8) |
| LTA 350 K K1024 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.1 | +10.7 % [+3.9, +14.0] (1/8) | +12.8 % [+8.9, +19.9] (0/8) |
| ITE 300 K K1024 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.01 | +49.1 % [+21.8, +92.1] (0/8) | +45.5 % [+31.3, +47.8] (0/8) |
| ITE 150 K K256 | 0.003, 0.01, 0.03, 0.1, 0.3 | 0.003 | +41.7 % [+10.1, +155.3] (0/8) | +41.0 % [+17.9, +55.1] (0/8) |

Re-run vs archive (ABF / FR arms regenerated on NSCC from the archived seeds; `reproducibility.md`):

- ITE 300 K K1024: S_geo median re-run 1.793 vs archive 1.801 (-0.5%); FR replacements median 582 vs 582; e_F(T) FR median re-run 0.04866 vs 0.04471 (+8.8%)
- ITE 150 K K256: S_geo median re-run 1.558 vs archive 1.437 (+8.4%); FR replacements median 116 vs 116; e_F(T) FR median re-run 0.05374 vs 0.05582 (-3.7%)
- LTA 150 K K1024: S_geo median re-run 2.450 vs archive 2.450 (+0.0%); FR replacements median 362 vs 364; e_F(T) FR median re-run 0.05135 vs 0.05362 (-4.2%)
- LTA 300 K K1024: S_geo median re-run 1.386 vs archive 1.388 (-0.2%); FR replacements median 197 vs 198; e_F(T) FR median re-run 0.08425 vs 0.08134 (+3.6%)
- LTA 80 K K1024: S_geo median re-run 1.838 vs archive 1.842 (-0.2%); FR replacements median 491 vs 491; e_F(T) FR median re-run 0.03015 vs 0.02803 (+7.6%)
- LTA 350 K K1024: S_geo median re-run 1.338 vs archive 1.293 (+3.5%); FR replacements median 178 vs 178; e_F(T) FR median re-run 0.05273 vs 0.03817 (+38.1%)
- gateway (128 runs): median relative difference re-run vs archive, Ibar_F -0.00% (max |rel| 19.90%), final e_F -0.00% (max |rel| 245.81%); bitwise-identical runs: 0

How to read the primary tables:

- **Laplacian vs ABF: no effect.**  In seven of eight cells the Laplacian arm is indistinguishable from plain ABF
  (I_F within -4 / +5 %, e_F(T) within -5 / +3 %, wins 5-9 of 16; the only CI excluding zero in its favour is
  I_F at gateway K2048, -4.4 % [-9.3, -1.3]).  At gateway K512 it is worse than ABF on the final error (+22 %
  [+12.5, +33.9], 7/32).  The archived sustained time-to-accuracy metric agrees: S_geo 0.93-1.01, UNRESOLVED in all
  six LTA / ITE cells (0.93 [0.83, 1.00] at 80 K, where the arm selected until the end of the run).
- **FR vs ABF reproduces the archive in all eight cells** (I_F -14 to -42 %, e_F(T) -22 to -83 %, 14-16 of 16 or
  28-32 of 32 seeds; S_geo 1.34-2.45 SUPPORTED in all six, within 0-8 % of the archived values).
- **FR vs the published Laplacian rule (the PR's question):** FR is better in every cell, by the same margin as over
  ABF: Laplacian I_F +15 % (LTA 350 K) to +77 % (gateway K512) and e_F(T) +26 % to +469 % (LTA 80 K) relative to
  FR, with 0-5 Laplacian wins out of 16 or 32 seeds.  No scale of the pilot ladder changes this (best of five per
  cell still +11 % to +423 % in e_F(T) relative to FR, 0-1 wins of 8).
- **Activity.**  The pilot rule matched the replacement counts to FR's in log-ratio (0.6-1.9 x FR), yet the Laplacian
  arm's minimum windowed ESS/K stays higher (0.66-0.93 vs 0.32-0.78) and its largest family smaller: its deaths are
  spread uniformly over the marginal, as the noise diagnostic below predicts, rather than concentrated where the
  marginal is high.  Under the R2 stop rule (stop at 2 t1, t1 = establishment time) the Laplacian arm establishes
  1.2-2.4 x later than FR; at LTA 80 K its stop (68-92 t.u.) fell after the 60 t.u. run in 16/16 seeds, so that arm
  never stopped selecting.  (The archived analyzer's secondary "post-stop estimate" indexed past the save grid in
  that case, a branch the archive never executed; it now falls back to its documented "never stopped" case, `kref`
  = FR start.  No archived number changes.)
- **Reproducibility of the re-runs (`reproducibility.md`).**  The ABF arms reproduce the archive per run to
  platform floating-point drift (gateway: median |rel. diff| of Ibar_F 0.001 %, max 0.4 %; LTA / ITE S_geo medians
  within 0-8 %).  The FR arms do not reproduce per run (resampling amplifies the drift: gateway K512 median |rel.
  diff| 3.8 % in Ibar_F and 43 % in the final e_F), but their population medians do (Ibar_F 0.00507 vs 0.00504;
  e_F(T) medians within 9 % except LTA 350 K, +38 % on a small final error).  Hence every contrast above is between
  NSCC runs of the same seed, never NSCC Laplacian against archived ABF / FR.

Taken with the two analyses below, the fair summary is: implemented inside FR's own density estimate and calibrated
by the predeclared reference-free rule, the LRS Laplacian rule selects on estimation noise and does nothing for ABF,
so FR's improvement over it equals FR's improvement over ABF in all eight cells; given its own, much wider bandwidth
(exploratory, two LTA cells, fresh seeds) the rule becomes a real competitor at 150 K and trails FR at 80 K.  Whether
the paper reports the predeclared comparison alone, adds the smoothed Laplacian arm to the eight production cells, or
re-tunes FR's bandwidth as well, is the author's decision.

### What the Laplacian score acts on (`score_noise_diagnostic.py`, `analysis/score_noise.md`)

The implemented score is `-scale * p''/p` on the same KDE marginal that FR uses (bandwidth 0.1 rad on a 180-bin
periodic grid, K = 1024 walkers).  An offline emulation of that field from the saved instantaneous histograms,
checked against the engine's own saved score standard deviation (emulated 0.11-0.39 vs saved 0.11-0.32), shows:

| LTA cell | std of instantaneous p''/p (rad^-2) | std of p''/p of the time-averaged marginal | correlation | noise variance fraction |
|---|---:|---:|---:|---:|
| 150 K | 11.8 | 0.75 | 0.04 | 0.996 |
| 300 K | 11.4 | 0.50 | 0.01 | 0.998 |
| 80 K | 13.1 | 1.61 | 0.10 | 0.985 |
| 350 K | 11.5 | 0.59 | 0.07 | 0.997 |

At the shared bandwidth the Laplacian score is 98.5-99.8 % finite-K estimation noise.  Consistently, in the pilot and
production runs the Laplacian arm kills at a relative marginal density of 1.00-1.02 (uniformly at random with respect
to the marginal) and leaves the time-averaged marginal exactly as flat as ABF, whereas FR kills at 1.17-1.21 at 80 K
and lowers the marginal's TV from uniform from 0.20 to 0.11.  At 150 / 300 / 350 K the time-averaged ABF marginal is
already flat (TV 0.013-0.04), so no marginal-flattening rule has anything to act on there; 80 K is the one LTA cell
with structure.  A second-difference rule needs a far smoother density estimate than a log-ratio rule: at 0.3-0.5 rad
the emulated correlation rises to 0.4-0.9, but |p''/p| shrinks 10-40x, so the scale has to move up with it.

### Exploratory: Laplacian with a wider KDE bandwidth (`make_explore.py`, `explore_analyze.py`, `analysis/explore_bw.md`)

Outside the predeclared protocol, on 8 FRESH seeds per cell (1963000+, 1965000+) with ABF, FR (unchanged) and the
Laplacian arm at `kde_bandwidth` x `laplacian_scale` ladders (three rungs, 2026-10-31 19:28-20:02 UTC, 50 core-h),
LTA 150 K and 80 K only.  Paired contrasts of e_F(T) (median G with seed-bootstrap 95 % CI, wins / 8):

| cell | ABF e_F | FR e_F (vs ABF) | best Laplacian (bandwidth, scale) | e_F | vs ABF | vs FR | min ESS/K | replacements |
|---|---:|---|---|---:|---|---|---:|---:|
| LTA 150 K | 0.133 | 0.055 (-61.5 %, 8/8) | (1.2 rad, 10) | 0.032 | -75 % [-81, -72], 8/8 | **-33 % [-55, -23], 8/8** | 0.36 | 533 (FR 354) |
| LTA 150 K | | | (0.8 rad, 10) | 0.032 | -78 % [-85, -72], 8/8 | -44 % [-52, -32], 8/8 | 0.34 | 688 |
| LTA 80 K | 0.174 | 0.023 (-86.8 %, 8/8) | (0.8 rad, 1) | 0.029 | -83 % [-85, -80], 8/8 | **+21 % [+8, +44], 1/8** | 0.75 | 456 (FR 492) |
| LTA 80 K | | | (1.2 rad, 3) | 0.032 | -82 % [-90, -76], 8/8 | +60 % [-28, +66], 2/8 | 0.67 | 453 |

Both ladders have turned over (at 150 K bandwidth 2.0 rad and scale 30 are no better than 1.2 rad / 10; at 80 K every
bandwidth peaks at scale 1-3), so the exploration was closed there.  With a smoothed density estimate the LRS rule
does what it is meant to do (deaths at relative density 1.2-1.4 at 80 K, marginal TV 0.08-0.11 vs ABF 0.18) and gains
75-83 % over ABF in both cells; it then beats the archived FR configuration at 150 K and remains behind it at 80 K.

How to read this against the primary result: the predeclared protocol fixed the density estimate to FR's and let the
pilot choose only the scale, so the primary tables compare FR with a Laplacian rule that is selecting on noise.  A
reviewer will ask for the comparison at each rule's own best bandwidth; the exploratory numbers above are the first
such data (8 fresh seeds, 2 cells, Laplacian bandwidth tuned on the same seeds it is scored on, FR not re-tuned).
Promoting a smoothed Laplacian arm to the 8 production cells, and whether FR itself should be re-tuned in bandwidth,
are decisions for the author, not for this run.

## Reference-file availability

The LTA 150 K and 300 K references are tracked under `results/equal_budget_v2/references/`.
The 80 K and 350 K R2 reference files were built from umbrella/WHAM profiles before R2 production
(`docs/r2_campaign/PREREGISTRATION.md`, §5); they are **not tracked** in GitHub. The corresponding
`lta_T80_reference.npz` and `lta_T350_reference.npz` files must already be available in the
original results directory (or rebuilt according to the original frozen protocol) before either
analyzer can score those temperatures. They are **not** the exact-Metropolis-MC references used
at 150 K and 300 K. Never substitute a reference after examining new selection results.
