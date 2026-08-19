# ABP-Fisher-Rao

Adaptive Biasing **Potential** (mollified SHUS) with a temporary marginal
**Fisher-Rao** birth-death correction. Fresh repository for the ABP-FR project; the
mFR-ABF campaign (`zheyuanlai/ABF-Fisher-Rao`) is a closed experimental record whose
mature numerics are selectively ported (see `docs/PROVENANCE.md`).

**Hypothesis under test (falsifiable):** a temporary, moderate, uniform-target
Fisher-Rao resampling step can damp the post-discovery *establishment* transient of an
accumulating SHUS bias — improving integrated free-energy error `I_F` and
time-to-accuracy `tau_eps` — without suppressing the occupation signal SHUS learns
from. Persistent/strong FR is predicted to hurt (estimator-resampling feedback).
Both accuracy (`I_F`) and speed (`S_eps = tau^SHUS/tau^FR`) are preregistered
co-primary endpoints: `docs/PREREGISTRATION_GATEWAY.md`.

## Layout

```
src/abpfr/
  grid.py          1D grid primitives (ported, validated numerics)
  shus.py          mollified SHUS accumulator (the ABP)
  fisher_rao.py    finite-theta FR step toward the uniform target, ESS backoff
  resampling.py    systematic resampling, matched-turnover sham, ancestry stats
  metrics.py       e_F (gauge-optimal L2), I_F, tau_eps, S_eps, cosine modes, bootstrap
  diagnostics.py   T_hit / T_est, KDE noise floor (the FR eligibility gate)
  io.py            run-record schema (full pmf_t + marginal_t, hard-asserted)
  systems/gateway.py  entropic gateway: analytic F_ref + batched SHUS(+FR) engine
docs/              SPEC (frozen conventions), PROVENANCE, PREREGISTRATION
scripts/           smoke + (later) stage runners
tests/             Stage-0 engineering validation (45 tests)
```

## Quickstart

```bash
pip install -e ".[dev]"      # or just make src/ importable
python -m pytest             # Stage-0 validation, ~10 s
python scripts/smoke_gateway_shus.py            # GPU smoke: SHUS + FR sanity arm
```

Production runs batch all `(seed x arm)` rows into one `simulate_batch` call on a
single GPU (target: one H200, float64); arms of a seed share initial conditions and
Langevin noise, so every comparison is paired.

## Status

Stages 0-4 and the applicability map (gateway, WCA, torus, alanine) are closed; all
outcomes are recorded in `docs/PREREGISTRATION_*.md` and are not edited.

**Where the project stands (Phase F, 2026-08-19).** MARGINAL Fisher-Rao is redundant
for an ABP and ties coarse count balancing, in four replications across two campaigns
— because both realize the same flow and differ only in the estimator of
`log p(xi)`, and because the ABP's own bias already owns the xi-marginal. The one
deficit an ABP cannot repair is the conditional structure of a coordinate it does not
bias (Type C), and marginal FR is blind to it by construction. Phase F built a system
that exhibits Type C (`src/abpfr/systems/bichannel.py`) and a step that can see it —
fiber-wise Fisher-Rao, birth-death inside xi-strata with the xi-marginal left
invariant (`src/abpfr/fisher_rao_cond.py`):

- an eightfold adaptation-gain increase moves the deficit by <= 3% (wrong sign);
- conditional FR: **-15 / -13 / -31% integrated error**, beating its matched-turnover
  sham by the same margin, while marginal FR is exactly null;
- stratified COUNT balancing ties conditional FR — at one hidden dimension the
  active ingredient is which coordinate you condition on, not the FR estimator;
- `tau_clone` is < one event stride in the CV and 160-290 strides in the hidden
  channel, in the same runs: the mechanism, and a pre-run diagnostic for which
  descriptor a reallocation should be conditioned on.

Phase G (open): the FR-specific claim, which by the lemma above can only be settled
on a reallocation descriptor of dimension >= 3.
