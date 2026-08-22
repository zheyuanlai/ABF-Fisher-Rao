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

**Partly yes against adaptive biasing, no against stratification.** See
[`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for the full account and
[`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) for every measurement in order.

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
