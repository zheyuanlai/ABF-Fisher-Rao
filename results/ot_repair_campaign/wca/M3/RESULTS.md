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

## M3-B — A / F / T(α = 0.03) on 16 fresh seeds 900–915 (CLOSED 08:20 UTC; `core/analysis.json`, `core/go_nogo.json`, figures M3-2…6)

Primary = compute-normalised I_F^(C) on the common budget C\* = 122.88 M force evaluations at
h\*\* = 0.00625 (for these three arms identical to the physical-time integral); paired per seed,
10 000-resample bootstrap.

| contrast | ΔI_F^(C) median [CI95] (90 % CI) | wins | Δe_F(C\*) median [CI95] | wins | verdict |
|---|---|---|---|---|---|
| **T vs A** (H-B1) | **−14.0 %** [−22.1, −7.0] (−18.9, −8.4) | 16/16 | **−31.6 %** [−36.0, −27.0] | 16/16 | **superior; H-B1 holds** |
| T vs F (H-B2) | +3.4 % [+1.3, +11.2] (+1.6, +10.7) | 4/16 | **+26.8 %** [+21.2, +32.8] | 0/16 | equivalence **fails** (90 % CI upper 10.7 > 10); FR ahead |
| F vs A | −18.2 % [−22.5, −15.7] | 16/16 | −44.5 % [−47.8, −43.2] | 16/16 | superior |

Read-out sensitivity (raw / 0.00625 / 0.0125): T vs A −13.7 / −14.0 / −13.5 %; T vs F +7.1 / +3.4 /
+3.6 %; F vs A −19.9 / −18.2 / −19.3 % — no sign change anywhere.  Compute to ABF's final accuracy
(persistence 2 saves): F 0.29×, T 0.35× of A's.  Dose delivered: mean |Δz| 0.0009 per event
(a tenth of a bin), max ≤ 0.013, nothing capped, no NaN.

**Reading.**  H-B1 replicates the pilot on 16 fresh seeds at a *gentler* dose than the pilot's:
capped Wasserstein reallocation alone is a WCA accelerator (−14 % integrated, −32 % at the end,
16/16, read-out-stable, 0.35× compute to ABF's accuracy).  H-B2 does not hold: uniform FR is the
better raw allocator, clearly at the endpoint (+27 % for OT, 0/16) and marginally integrated
(+3 %, CI excluding 0 but inside the ±10 % band on the median).  Go/no-go: **GO** by H-B1.

## M3-C — R / F+R / T+R (only on GO)

_(pending)_
