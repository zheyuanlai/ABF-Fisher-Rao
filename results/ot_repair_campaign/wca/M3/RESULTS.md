# WCA capped-OT confirmatory M3 — results

Preregistration `docs/WCA_OT_CONFIRMATORY_M3.md` (c7d9ebb, frozen before any run); analyzer
`scripts/analyze_wca_ot_m3.py` (97ed391, fixed on the first calibration seed's marginal-only
fields before the chain reached it); chain `scripts/launch_wca_ot_m3.sh`; GPU 1 only.

## M3-A — blind marginal-action calibration (seeds 880–883; CLOSED 04:17 UTC; `calibration/alpha_star.json`, figure M3-1)

J_KL = ∫_{t ≥ 40} KL(p̂_t ∥ U) dt from the stored walker marginal only (no error field read):

| arm | J_KL (median of 4 seeds) | ratio to FR | capped fraction | mean \|Δz\| per event |
|---|---|---|---|---|
| ABF | 8.26 | 1.36 | – | – |
| uniform FR (accepted arm) | 6.07 | 1 | – | – |
| OT α 0.03 | 1.28 | 0.210 | 0.000 | 0.0009 |
| OT α 0.05 | 0.99 | 0.163 | 0.000 | 0.0011 |
| OT α 0.10 | 0.56 | 0.092 | 0.000 | 0.0015 |
| OT α 0.20 | 0.35 | 0.058 | 0.005 | 0.0022 |

**α\* = 0.03** by the frozen fallback (closest in log-ratio; the [0.9, 1.1] band is unreachable).
Finding: capped OT toward the uniform quantiles is a 5–17× stronger marginal flattener than uniform
FR at every α in the ladder, while moving each walker by only 0.0009–0.0022 per event (a tenth to a
quarter of a bin) and never hitting the cap below α = 0.2.  FR's marginal action is dominated by a
persistent KL floor (≈ 0.03 per time unit, vs ABF's 0.04); OT's by its onset transient alone.
Marginal-action matching between the two allocators is therefore not possible with this operator
pair; the gentlest ladder point is used, which also makes M3-B a *harder* test for OT than the
α = 0.1 pilot.

## M3-B — A / F / T on 16 fresh seeds (900–915)

_(pending)_

## M3-C — R / F+R / T+R (only on GO)

_(pending)_
