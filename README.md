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

- Stage 0 (engineering validation): **done** — all conventions frozen in
  `docs/SPEC_SHUS_FR.md`, tests green, GPU smoke run reproduces the predicted
  FR/SHUS feedback qualitatively.
- Stage 1 (plain-SHUS calibration + establishment diagnosis on the gateway): next.
