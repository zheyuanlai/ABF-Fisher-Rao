# reaction-coordinate-wfr

**RC-WFR-TI** — a bias-free free-energy method that evolves the replica population by a
**Wasserstein–Fisher–Rao** flow on the low-dimensional reaction-coordinate marginal,
keeps physical conditional sampling on the reaction-coordinate fibers, and reconstructs
`F` by thermodynamic integration. No ABF, no OPES/SHUS, no learned bias.

    conditional MD on Sigma(z)  ->  W transport of z  ->  lift  ->  FR reallocation  ->  TI

The campaign asks one question:

> **Can a bias-free reaction-coordinate WFR sampler compute free energies faster than
> adaptive biasing — and than classical stratified thermodynamic integration?**

## Answer

**Yes against adaptive biasing, in a regime you can identify in advance. Against
classical stratification it depends on the fiber — and on getting two things right
that the campaign had to discover.**

* Use the **deterministic probability-flow** Wasserstein step, not the SDE one: its
  velocity vanishes as `p -> u`, so its hysteresis self-annihilates.
* Deterministic transport plus Fisher-Rao resampling needs a small **resample-move
  jitter**, or clones never separate and the ensemble collapses.

With both, at matched force evaluations (32 fresh confirmation seeds, every baseline
screened at least as hard, 95% bootstrap CIs excluding zero on the bold entries):

| on a fiber with a hidden slow channel | |
|---|---|
| vs cold-start Hamiltonian replica-exchange TI | **-50.1%** |
| vs cold-start stratified TI | **-70.5%** |
| vs ABF | **-82.4%** |

| elsewhere | |
|---|---|
| vs ABF, long torsional CV (L = 24) | **-89%** |
| vs fixed-window TI, long torsional CV (L = 24) | **-12%** |
| vs ABF, easy fiber | **-62.6%** |
| vs fixed-window TI, easy fiber | +40.5% (TI wins) |
| vs ABF, short torsional CV (L = 3) | +191% (ABF wins) |

Removing the Fisher-Rao term from the winning arm turns a 50% win over replica exchange
into a 27% loss: the birth-death half is the difference between winning and losing, and
it is the only half that carries no hysteresis.

The hard limitation is structural and does not go away with compute: an unconditional
move in `xi` cannot be Metropolis-corrected without knowing `F`, so RC-WFR trades a
convergence problem for a **bias** problem. Its Fisher-Rao half is free of this (it
copies walkers whole and drags nothing) and does most of the useful work — removing it
costs a factor 2.4-2.5.

Full account: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).
Every measurement in the order it was taken: [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md).

## Layout

```
src/rcwfr/
  grid.py          1D CV grid primitives (reflecting + periodic), KDE, quadrature
  wasserstein.py   W transport of the labels: SDE form and probability-flow form
  fisher_rao.py    finite-theta FR toward uniform; count-balancing and sham controls
  resampling.py    exact-N systematic resampling, ancestry accounting
  estimators.py    the shared binned mean-force / TI estimator (all arms use it)
  shus.py          mollified SHUS (the ABP baseline)
  engines.py       every arm; the force-evaluation cost invariant lives here
  systems/base.py  test-system family with quadrature-exact F, F' and conditionals
  registry.py      frozen systems: EB, SLOWFIB, CHANNEL, TORSION_L*
  campaign.py      arm dispatch, scoring, estimator floor, paired bootstrap
scripts/           phase0, calibrate, sweep_wfr, compare, confirm, torsion/mspec scaling
figures/           publication figures + the script that makes them
docs/              METHOD, PREREGISTRATION, RESULTS_LOG, TECHNICAL_REPORT, PROVENANCE
tests/             20 engineering tests (~40 s on one GPU)
```

## Reproducing

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python scripts/phase0_marginal.py
```

```bash
python scripts/confirm.py --system CHANNEL --steps 100000 --seeds 32
```

Every arm in a comparison runs the same `N` and `n_steps`, so the total number of
force evaluations is matched by construction; replica-exchange energy evaluations are
charged explicitly. Results are compared against a measured **estimator floor** and no
claim is made about differences at or below it.
