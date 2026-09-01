# Does a mean-force information-adequacy clock explain the campaign?

**No. Definition and prediction were frozen first (`configs/information_campaign/
prediction_v1.md`, commit `0d56426`); this is the result.** Exploratory by
construction — every cell was already closed and its ΔI_F already known.

## The proposal

After ethane/ZIF-8 came back HARMFUL in the regime the marginal clock selected,
the diagnosis was that the campaign had conflated *marginal establishment* with
*estimator establishment*. ABF estimates a conditional mean force whose accuracy
is governed by `SE[F'(z)] ~ sqrt(Var(f|z)/n_eff(z))`, not by how uniform
`p^xi(z)` is. The proposed replacement predictor was `T_info`: the time at which
a reference-free, block-bootstrapped PMF uncertainty `U_F(t)` reaches the same
relative convergence criterion already used for `T_marg`. FR would then be
gated to the window where the estimator is genuinely information-starved.

## Result: the predictor fails, for two independent reasons

Nine closed cells spanning the full outcome range, all computed from their own
saved accumulators with no new simulation:

| cell | ΔI_F | T_info | headroom H |
|---|---|---|---|
| lta_T150 | −31.92 % | 52.2 | +0.803 |
| cha_ethene_450 | −5.96 % | 57.6 | +0.657 |
| lta_T225 | −21.28 % | 43.2 | +0.653 |
| lta_T300 | −14.84 % | 43.2 | +0.653 |
| cha_propene_600 | −5.96 % | 49.6 | +0.557 |
| lta_v1_late | −0.21 % | 40.2 | +0.537 |
| cha_propene_450 | −5.72 % | 45.6 | +0.508 |
| **lta_T80** | **−35.14 %** | 31.2 | **+0.453** |
| **zif8_T300** | **+3.67 %** | 150.0 | **+0.400** |

**Spearman(H, ΔI_F) = −0.437, p = 0.240, n = 9.** The sign is as predicted and
ZIF-8 does have the least headroom, but the relation is not significant and —
decisively — it fails to separate the *strongest positive* (LTA T80, H = 0.453)
from the *harmful* cell (ZIF-8, H = 0.400). The frozen prediction
`H(LTA) > H(CHA) > H(ZIF-8)` is false: CHA overlaps LTA throughout.

### Reason 1 — it is a VARIANCE clock, and these endpoints are BIAS

| | U_F (bootstrap uncertainty) | e_F (true error) | U_F/e_F |
|---|---|---|---|
| ZIF-8 300 K, final | 0.0266 | 0.2747 | **0.097** |
| CHA ethene, final | 0.2872 | 1.8182 | **0.158** |

**99 % (ZIF-8) and 98 % (CHA) of the final squared error is not variance.** Of
ZIF-8's residual, reference noise explains only 5.7 %, leaving ~93 % as genuine
estimator bias. `U_F` falls 3.4× over the run while `e_F` falls 1.1× — the
uncertainty keeps improving long after the error has stopped, because the error
is set by a bias that more sampling does not touch. A clock built on `SE[F']`
therefore cannot say when the estimator is finished.

This is not a new discovery so much as the fifth independent arrival at the same
place. The mechanism campaign found η_bias never below 0.121 on any system and
concluded kernel-ABF free-energy endpoints are bias-dominated *by construction*;
the fibre-horizon audit found 96–97 % of FR's damage was bias; and the ZIF-8
result itself found 98 %.

### Reason 2 — it is not robust to an arbitrary analysis knob

| block length | 2 ps | 5 ps | 10 ps | 20 ps |
|---|---|---|---|---|
| U_F final | 0.0234 | 0.0259 | 0.0282 | 0.0300 |
| **headroom H** | **+0.260** | **+0.400** | **+0.567** | **+0.733** |

The *variance* estimate is stable (28 % across a 10× change), so bias dominance
is not an artifact. But `T_info` swings **3×**, because the relative-convergence
rule is applied to a curve whose shape depends on how many blocks exist at each
t. A predictor that moves 3× on a choice with no physical content is not a
property of the run.

## Instrument validation, and a bug it caught

Before any claim, the single-run block bootstrap was required to track the
across-seed spread (the honest uncertainty, since seeds are independent runs).
It initially disagreed by **28–54×** on the LTA cells. That was not the
hypothesis failing — it was two errors in my own cell table: LTA uses a
**periodic** CV and has **no `min_count` regularizer** (a plain
`smooth(f)/smooth(c)` ratio), so the reconstructed block moments were garbage.
After correcting both, every cell sits at 0.66–1.77. Without the pre-committed
instrument check this would have been reported as a result.

## Conclusion, per the frozen falsifier

The frozen prediction stated: *"If the relation fails, STOP — do not invent a
third predictor."* It failed. So:

**Do not build the information-gated FR campaign.** Its gate is a variance clock
on a bias-dominated endpoint, and it is not robust.

More constructively, the recurring diagnosis now has five independent
confirmations, and it points somewhere specific. If ~95 % of the endpoint error
is kernel-smoothing bias, then *where the walkers are* is close to irrelevant and
the dominant lever is the estimator's **bandwidth** — which the mechanism
campaign already measured as a **58× endpoint-MSE win** (h 0.07 → 0.005), against
the 1.4–1.7× that allocation was ever worth. A consistency check on ZIF-8: the
ABF arm underestimates the barrier by 0.335 kJ/mol, and Gaussian kernel bias at
h = 0.20 Å with the observed peak curvature predicts ≈ 0.8 kJ/mol — the same
order.

The licensed next experiment is therefore **not** another allocation predictor,
and **not** ZIF-8 at 250/350 K. It is a direct bandwidth sweep on a cell where
the reference is already trusted.

---

# The bandwidth sweep — the lever that was there all along

One ABF-only ZIF-8 trajectory (8 seeds × 384 replicas, 150 ps), with the **raw
binned accumulators** saved so the bandwidth is swept entirely offline at fixed
dynamics. That isolates the read-out from any change in sampling.

| h (Å) | h (bins) | e_F | bias | seed sd | barrier vs ref |
|---|---|---|---|---|---|
| 0.40 | 2.69 | 0.8373 | 0.8434 | 0.0731 | −2.91 % |
| 0.30 | 2.01 | 0.5369 | 0.5368 | 0.0674 | −1.91 % |
| **0.20 (production)** | **1.34** | **0.3018** | 0.2951 | 0.0634 | −1.17 % |
| 0.15 | 1.01 | 0.2186 | 0.2084 | 0.0622 | −0.89 % |
| 0.10 | 0.67 | 0.1620 | 0.1507 | 0.0614 | −0.69 % |
| 0.07 | 0.47 | 0.1374 | 0.1272 | 0.0611 | −0.60 % |
| **0.05** | **0.34** | **0.1266** | 0.1170 | 0.0610 | −0.54 % |
| 0.03 | 0.20 | 0.1251 | 0.1157 | 0.0609 | −0.54 % |
| 0.02 | 0.13 | 0.1251 | 0.1157 | 0.0609 | −0.54 % |

**The error IS the bias**: `bias/e_F` = 1.007, 1.000, 0.978, … 0.925 across the
whole sweep. Nothing else is happening.

**There is no bias–variance tradeoff in this regime.** The seed-to-seed spread
*also* falls with smaller h (0.0731 → 0.0609). This independently reproduces the
mechanism campaign's finding that integrated variance *rises* with bandwidth
because smoothing lengthens correlation and the endpoint integrates F′ — now
confirmed on a molecular flexible-framework system it was never derived from.
So smaller h improves both terms and there is no interior optimum; the error
saturates once h drops below about a third of the bin width, i.e. once the
kernel stops doing anything and the binning itself sets the resolution.

## The two levers, same endpoint, same system

| lever | cost | effect on endpoint MSE |
|---|---|---|
| bandwidth 0.20 → 0.05 Å | **free** — re-analysis of data already on disk | **5.8× better** |
| uniform mFR at its safety-calibrated rate | a full second production run | **1.07× worse** |

**A factor of 6.3 separates them, and the free one wins.** The production
bandwidth was 1.34 bins; the entire ZIF-8 result — reference, screen,
calibration, two 10-hour arms — was spent moving an endpoint that a one-line
change to the read-out improves by 5.8×.

The residual at h → 0 is 0.1251 kJ/mol, of which 27 % of the squared error is
the reference's own noise. So even the floor is not sampling-limited.

## What this changes

The campaign's premise was that a better *allocation* of samples accelerates
ABF. On this system, allocation was worth −7 % and the read-out was worth
+480 %. The honest reframing is that kernel-ABF endpoints are bias-dominated,
the bias is set by the estimator's bandwidth, and bandwidth is free.

**Caveat, stated because it bounds the claim:** this sweep changes only the
*analysis*. The ABF bias force during the run still used h = 0.20 Å, so this is
exactly the win available by re-reading existing data. Whether *running* at
small h helps or hurts is a separate question — a noisier bias force could
degrade sampling — and is untested. That, not another allocation predictor, is
the experiment worth doing next.

---

# Experiment 1 closed: the online bandwidth matters too (Outcome B)

Four ABF-only arms, 8 seeds, shared init pool, all scored at the **frozen**
`h_read = 0.05 Å`. `min_count` held fixed throughout.

| h_bias | e_F | ref-corrected | barrier err | ref-noise share | force roughness / truth |
|---|---|---|---|---|---|
| 0.200 | 0.1262 | 0.1079 | −0.54 % | 27 % | 0.93 |
| 0.100 | 0.1053 | 0.0823 | +0.04 % | 39 % | 0.99 |
| 0.050 | 0.0909 | 0.0630 | +0.17 % | 52 % | 1.02 |
| 0.025 | 0.1041 | 0.0808 | +0.20 % | 40 % | 1.02 |

## What is resolved, and what is not

| step | paired median | sem | seeds better | resolved |
|---|---|---|---|---|
| 0.20 → 0.10 | **−21.6 %** | 8.1 | 6/8 | **yes** |
| 0.10 → 0.05 | −10.4 % | 14.8 | 5/8 | no |
| 0.05 → 0.025 | −2.6 % | 11.1 | 4/8 | no |

**Only the first halving is resolved.** The apparent minimum at h_bias = 0.05 and
the apparent turnover at 0.025 are both inside the noise, so neither an optimum
nor a turnover is claimed. The barrier error does improve monotonically and
dramatically (−0.54 % → +0.04 %) at the first step, which is the same story.

## The predicted failure mode does not occur

A small online bandwidth is supposed to make the adaptive force noisy. Measured
on the force the dynamics actually felt: roughness 0.93 → 0.99 → 1.02 → 1.02
relative to the true profile, with **zero clipping at every arm**. The force is
not becoming noisy, it is becoming accurate — **h_bias = 0.20 was OVER-smoothing
the bias force by 7 %, under-resolving the very barrier it was meant to
flatten.** That is the mechanism behind Outcome B, and it was measured rather
than assumed.

This contradicts the prediction recorded before the runs (A or C). One bandwidth
was doing two jobs and doing both badly — but not for the reason expected.

## The experiment is now reference-limited

The share of the squared error that is the reference's own split-half
uncertainty runs 27 % → 39 % → 52 %. Because the reference enters every arm
identically its error is common-mode, `E[e²] = MSE_true + σ_ref²`, and
subtracting it gives the corrected column — which strengthens the result
(2.9× MSE from 0.20 to 0.05). But by h_bias = 0.05 the correction is half the
signal, and the three sub-0.10 arms are mutually indistinguishable.

**Resolving the small-bandwidth end would need ~10× more umbrella sampling
(≈19 h), since reference noise falls as 1/√t.** Until that is spent, "which
bandwidth below 0.10 is best" is not a question this system can answer, and the
honest recommendation is only:

> Halve the online bandwidth from the value this project has been using, and
> read out at or below the bin width. Both are free.
