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

## Reference-file availability

The LTA 150 K and 300 K references are tracked under `results/equal_budget_v2/references/`.
The 80 K and 350 K R2 reference files were built from umbrella/WHAM profiles before R2 production
(`docs/r2_campaign/PREREGISTRATION.md`, §5); they are **not tracked** in GitHub. The corresponding
`lta_T80_reference.npz` and `lta_T350_reference.npz` files must already be available in the
original results directory (or rebuilt according to the original frozen protocol) before either
analyzer can score those temperatures. They are **not** the exact-Metropolis-MC references used
at 150 K and 300 K. Never substitute a reference after examining new selection results.
