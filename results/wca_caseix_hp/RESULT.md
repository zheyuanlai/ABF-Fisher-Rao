# C2 — Case IX recalibrated against the high-precision reference

**The WCA positive survives. `-22.83 %` becomes `-17.97 %`.**

Fresh dynamics, 16 paired seeds (400-415), 5 arms, the frozen Case IX cell with **nothing
retuned** -- 120 k steps, N = 1024, `fr_rate 0.10`, `target_ema 0.005`, `max_event_fraction
0.02`, `fr_every 5`, `fr_start 20000`, `score_clip 2.0`. The only change is the reference:
`cache/phase_hp_v3` (unsmoothed, 4 independent preparations, acquisition grid == evaluation
grid so interpolation is the identity).

## Time-integrated endpoint -- the headline

| contrast | HP v3 | 95 % CI | seeds | v1 (cached) |
|---|---|---|---|---|
| **mFR vs ABF** | **-17.97 %** | [-22.73, -17.16] | **16/16** | -22.83 % |
| mFR vs its own sham | -20.96 % | [-25.04, -11.54] | 15/16 | -26.38 % |
| sham vs ABF | -3.19 % | [-6.20, +2.89] | 9/16 | +2.60 % |
| mFR-oracle vs ABF | -19.54 % | [-25.45, -15.98] | 16/16 | -- |

Median integrated `L2(F)`: ABF 42.21, mFR 33.05, mFR-oracle 32.93, sham 41.43.

**Every preregistered check passes**: median <= -10 %, CI95 upper < 0, wins >= 12/16, and the
attribution requirement that the mFR-vs-sham CI excludes zero.

## What changed and what did not

* **Magnitude fell 21 % relative** (-22.83 -> -17.97). The corrected reference resolves a real
  mean-force trough at `z ~ 0.26` that the cached one missed by up to 67 sigma, and that trough
  sits inside the transition region where the arms differ most -- exactly where a shared
  reference error fails to cancel.
* **Direction, consistency and attribution are unchanged.** 16/16 seeds, CI excluding zero, and
  a sham that remains null.
* **The sham moved from +2.60 % to -3.19 %.** Under the cached reference the sham looked
  *harmful*; under the corrected one it is indistinguishable from ABF (CI spans zero, 9/16).
  That is a cleaner control than v1 had -- `CLOSURE_v1.md` caveat 3 noted the adverse-direction
  sham as informative but "not a neutral procedural control". It is now neutral.
* **mFR-oracle (-19.54 %) and practical mFR (-17.97 %) are close**, so the estimated target
  captures nearly all of the available gain on this system.

## Endpoint caveat

The two endpoints differ substantially: **-45.35 % final-time vs -17.97 % integrated**. mFR's
advantage grows through the run, so the time-integral dilutes it. `-22.83 %` was the integrated
endpoint, so `-17.97 %` is the like-for-like replacement. The final-time figure is reported as a
secondary quantity and must not be quoted as the headline.

## Reproducibility

Every run stores `pmf_t` (49 x 160) and the reference it was scored against. The reason C2
needed fresh dynamics at all -- Case IX having retained only scored scalars -- cannot recur.
