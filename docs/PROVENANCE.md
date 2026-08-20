# PROVENANCE

This repository is a fresh start (the ABP-FR project), separate from the closed
mFR-ABF campaign in `zheyuanlai/ABF-Fisher-Rao`. We reuse *code*, not repository
history.

**Hosting (2026-08-20).** The ABP-FR history is published as branch `abp-fisher-rao` of
that same GitHub repository, so both campaigns live in one place. This is a hosting
decision only: the branches have no common ancestor, nothing from this campaign merges
into `main`, and the mFR-ABF record on `main` (tag `v1-regime-map-final`, commit
`662f2fc`) is unchanged by it. Ported material and its origin (old repo at tag `v1-regime-map-final`,
commit `662f2fc`):

| New location | Origin (ABF-Fisher-Rao) | Notes |
|---|---|---|
| `src/abpfr/grid.py` | `src/eb_abffr_core.py:45-261` | grid, Gaussian kernel (radius `4bw/dx`, `sum*dx` norm), reflect-pad smoothing, binned density, interp, cumtrapz/trapz, reflect_into. Hard-coded domain replaced by `Grid1D`; numerics identical. `central_diff` is new. |
| `src/abpfr/systems/gateway.py` (physics) | `src/eb_abffr_core.py` (`omega_of`, `domega_of`, `U_of`, `dU_of`, `reference_profiles`) and `src/gateway_core.py` (`X_BASIN`, regions, `bias_aware_target`, `init_conditions`, `run_frozen_bias`, paired-noise batching pattern) | Entropic gateway potential, analytic reference, Euler-Maruyama with reflecting walls, `omega_in^2 dt < 2` stability guard. |
| `src/abpfr/resampling.py::ancestor_stats` | `src/gateway_core.py:433-444` | windowed ancestor ESS + `w_max`. |
| `src/abpfr/diagnostics.py::first_persistent` | `src/gateway_core.py:257-269` | persistence convention for `T_hit`/`T_est`. |
| Sham design (matched turnover, partner schedule) | `src/gateway_core.py:350-430` | concept ported; mechanics rewritten for exact-K systematic resampling. |
| `src/abpfr/metrics.py` conventions | `scripts/analyze_convergence_atlas.py` (`tau` with 0.2T persistence, threshold ladder, censoring kept visible, paired bootstrap), `src/wca_abffr_core.py:832-847` (gauge-optimal L2) | reimplemented, same definitions. |

**New in this repo (no ABF-FR precedent):** the mollified SHUS accumulator
(`src/abpfr/shus.py`; structurally inspired by the old `src/opes_core.py` grid
accumulation but a different update law), the finite-theta Fisher-Rao step with
systematic resampling and ESS backoff (`src/abpfr/fisher_rao.py`,
`src/abpfr/resampling.py`), cosine-mode projection of the bias error, and the run
record schema (`src/abpfr/io.py`).

**Deliberately NOT ported:** old result trees, ABF-specific decision logic
(mean-force accumulators, FR rate/ramp/clip machinery), regime labels from the ABF
campaign (a cell's old "establishment-limited" label says nothing about SHUS), and the
superseded WCA references (when the WCA stage opens, only the `hp_v3` high-precision
reference with its `meta_json` will be accepted, enforced in code).
