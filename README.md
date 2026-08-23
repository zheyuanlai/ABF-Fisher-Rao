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

## Follow-up: the manifold reformulation

The campaign above uses a **linear** reaction coordinate, `xi(q) = x`. Reading it
back against Chapter 3 of Lelievre-Rousset-Stoltz suggests recasting the method on
the level-set manifolds `Sigma(z) = {xi(q) = z}` and replacing the ad-hoc lift with
the minimum-norm horizontal one, `grad xi G^-1 dz`. On a linear coordinate `G = 1`
and **that lift is the identity lift the campaign already used**, so none of it was
testable. A nonlinear-coordinate test family was built (`xi = x + a sin k y`, exact
by quadrature, `a = 0` reproduces the frozen systems) and the reformulation was
audited against it:

* the geometry is right — Fixman factor, mean-force divergence term, all verified;
* the minimum-norm lift is **not** the statistically correct one, and its benefit
  vanishes exactly at the barrier top;
* the lift that is correct solves a continuity equation on the fiber, and there is
  a closed-form law for the error of every other lift.

Full account: [`docs/MANIFOLD_FORMULATION.md`](docs/MANIFOLD_FORMULATION.md).

## Molecular phase: does any of it survive on a real molecule?

The toy campaigns above use a synthetic potential.  The molecular phase asks
whether the lift result transfers, on united-atom TraPPE **butane** and
**pentane** and on **alanine dipeptide** (AMBER ff14SB, vacuum, OpenMM
parameters, torch inner loop verified to 1e-9 against OpenMM), with the
reaction coordinate a torsion and the reference `F(z) = -beta^-1 log p(z)` from
2e11 force evaluations of unbiased Brownian dynamics.

It does, and it sharpens.  On 32 fresh confirmation seeds at 1.07e8 matched
force evaluations (pentane, hidden torsion `phi2`, estimator floor 0.0127):

| | `e_F` (kcal/mol) | vs naive lift |
|---|---|---|
| RC-WFR + **Metropolis conditional move, learned proposal** | **0.0215** | **-53.6%** [-57.7, -50.1] |
| ...the same arm with an oracle proposal | 0.0196 | **+1.2%** [-2.6, +15.3] -- indistinguishable |
| ...the same learned conditional, applied UNCORRECTED as a refresh | 0.2917 | **+506.9%** |
| RC-WFR, naive internal-coordinate rotation lift | 0.0475 | - |
| RC-WFR, Chapter-3 minimum-norm horizontal lift | 0.1003 | **+116%** |

and against baselines screened at least as hard: **-31.4%** vs multiple-walker
ABF, **-47.1%** vs cold stratified constrained TI, **-25.1%** vs stratified TI
warm-started from the ORACLE conditional.

Four things this phase establishes that the toy phase could not:

* the **minimum-norm horizontal lift is actively harmful** on a molecule, not
  merely useless.  Its conditional error in the slow mode is the same as a plain
  rotation's; the extra damage is in the FAST modes it bends to buy the
  constraint, and it grows **12.6x** across a 64-fold transport-rate sweep;
* the **transport-rate pathology belongs to that lift, not to RC-WFR**.  For the
  corrected method `e_F` and `D_cond` are flat to seed noise over the whole 64x
  sweep: the transport rate stops being a hyper-parameter;
* a lift learned from the run's own samples and applied **without correction is
  self-reinforcing and much worse than doing nothing**, and an ORACLE
  uncorrected refresh -- the toy phase's best arm -- is a disaster on alanine
  (+180%), because drawing the slow torsion from its marginal conditional is
  right about that torsion and wrong about the sixty coordinates it is coupled
  to;
* **corrected, the learned lift matches the oracle.**  A rigid rotation about a
  torsion axis preserves the internal-coordinate Jacobian and `det G`, so an
  independence Metropolis move on the slow torsion is *exact whatever the
  proposal is* -- the learned conditional sets only the acceptance rate.

The honest caveat is unchanged from the toy phase and is stated in the report:
the baselines are unbiased and RC-WFR is not, so this is a **speed** result at
practical budgets.  Extrapolating the fitted convergence rates one decade puts
the crossover with ABF at ~3.4x the largest budget run here.

Full account: [`docs/MOLECULAR_REPORT.md`](docs/MOLECULAR_REPORT.md). **Start there.**

## Documentation

| file | what it is |
|---|---|
| [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) | the standalone account: question, protocol, verdict, mechanism, figures, regime map. **Start here.** |
| [`docs/MANIFOLD_FORMULATION.md`](docs/MANIFOLD_FORMULATION.md) | audit of the Chapter-3 manifold reformulation on a nonlinear reaction coordinate: what survives, what does not, and the lift-lag law |
| [`docs/MANIFOLD_TABLES.md`](docs/MANIFOLD_TABLES.md) | every manifold-phase table with IQRs (machine-generated by `scripts/make_manifold_tables.py`) |
| [`docs/TABLES.md`](docs/TABLES.md) | full confirmation and scaling tables with bootstrap CIs (machine-generated by `scripts/make_tables.py`) |
| [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) | every measurement in the order it was taken, including the ones that were wrong and why |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | hypotheses and decision rules as frozen, with outcomes appended |
| [`docs/METHOD.md`](docs/METHOD.md) | the construction, the cost model, and the structural obstruction |
| [`docs/MOLECULAR_REPORT.md`](docs/MOLECULAR_REPORT.md) | the molecular phase's standalone account: question, verdict, mechanism, caveat |
| [`docs/MOLECULAR_PLAN.md`](docs/MOLECULAR_PLAN.md) | molecular campaign: systems, gates, arms, preregistered predictions |
| [`docs/MOLECULAR_METHOD.md`](docs/MOLECULAR_METHOD.md) | the molecular construction: lifts, the Metropolis conditional move, estimators, cost |
| [`docs/MOLECULAR_RESULTS.md`](docs/MOLECULAR_RESULTS.md) | every molecular measurement in the order it was taken |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | what was ported from the closed ABF/ABP campaigns and the closed alanine campaign |
| `figures/fig[1-6]*.png`, `*.pdf` | campaign figures, regenerated by `figures/make_figures.py` |
| `figures/figM*.png`, `*.pdf` | manifold-phase figures, inlined in `MANIFOLD_FORMULATION.md`, regenerated by `figures/make_manifold_figures.py` |

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
  systems/graph.py the SAME potentials with a NONLINEAR xi = x + a sin(k y);
                   exact conditionals, exact F and F_rgd, the three lifts
  manifold.py      Chapter-3 geometry: G, tangent projector, Fixman potential,
                   SHAKE projection, local mean force, constrained Langevin
  adaptive_lift.py the conditional lift built from the run's own samples
  registry.py      frozen systems: EB, SLOWFIB, CHANNEL, TORSION_L*
  mol/ff.py        batched molecular mechanics: TraPPE-UA alkanes, exact torsion rotation
  mol/geom.py      mass-metric Chapter-3 geometry: G, SHAKE, den Otter-Briels mean force
  mol/dynamics.py  free and constrained Brownian steps (torch.compile'd)
  mol/lift.py      conditional CDF map / refresh / Metropolis proposal, oracle and learned
  mol/joint.py     p(y_S | z) for a promoted SUBSET of fiber modes
  mol/engines.py   molecular arms: stratified TI, RC-WFR with every lift, ABF
  mol/systems.py   frozen molecular systems: BUT, PEN, HEX, ALA
  mol/alanine.py   Ace-Ala-Nme ff14SB from OpenMM, torch inner loop
  campaign.py      arm dispatch, scoring, estimator floor, paired bootstrap
scripts/           phase0, calibrate, sweep_wfr, compare, confirm, torsion/mspec scaling
                   manifold phase: validate_manifold, exp_divergence_term, exp_fixman,
                   exp_fixman_dynamic, exp_lift, exp_timescale, exp_arms,
                   exp_estimator_variance; analyze_timescale, analyze_burnin,
                   make_manifold_tables; run_*_campaign.sh drivers
figures/           publication figures + the script that makes them
docs/              METHOD, PREREGISTRATION, RESULTS_LOG, TECHNICAL_REPORT, PROVENANCE
tests/             20 engineering tests + 16 manifold tests (~2.5 min on one GPU)
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

The molecular phase, in order (about 9 GPU-hours end to end on one H200; the
references are the expensive part and only have to be built once):

```bash
bash scripts/run_mol_reference.sh
```

```bash
python scripts/mol_gate1.py --system BUT
```

```bash
bash scripts/run_mol_screen.sh && bash scripts/run_mol_phase2.sh
```

```bash
bash scripts/run_mol_phase3a.sh && bash scripts/run_mol_phase3.sh
```

```bash
bash scripts/make_mol_tables.sh && python figures/make_mol_figures.py
```

`openmm` is needed only to build the alanine dipeptide parameters; the alkanes
are self-contained.

Every arm in a comparison runs the same `N` and `n_steps`, so the total number of
force evaluations is matched by construction; replica-exchange energy evaluations are
charged explicitly. Results are compared against a measured **estimator floor** and no
claim is made about differences at or below it.
